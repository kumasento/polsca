//===- VhlsLLVMRewriter.cc --------------------------------------*- C++ -*-===//
//
// This file implements a pass that transforms LLVM-IR for Vitis HLS input.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

static cl::opt<std::string>
    XlnTop("xlntop", cl::desc("Specify the top function for Xilinx HLS."),
           cl::value_desc("topname"));
static cl::opt<std::string>
    XlnNames("xlnnames", cl::desc("Specify the top function param names."),
             cl::value_desc("paramname"));
static cl::opt<std::string> XlnTBTclNames(
    "xlntbtclnames",
    cl::desc(
        "Specify the file name of the tcl script for test bench generation."),
    cl::value_desc("tbname"));
static cl::opt<std::string> XlnTBSources(
    "xlntbfilesettings",
    cl::desc(
        "Specify the file settings for the test bench, e.g. \"add_files ...\""),
    cl::value_desc("tbfiles"));

namespace {

/// InsExtSequence holds a sequence of `insertvalue` and `extractvalue`
/// instructions that operate on the same aggregated value, which correpsonds to
/// a `memref` value in MLIR.
/// For more information on how the `memref` type is organized:
/// https://mlir.llvm.org/docs/ConversionToLLVMDialect/#default-convention
class InsExtSequence {
public:
  Value *ptr = nullptr;                  /// aligned pointer.
  Value *offset = nullptr;               /// offset.
  SmallVector<Value *, 4> dim_values;    /// dimensionality.
  SmallVector<Value *, 4> stride_values; /// stride in each dim.
  SmallVector<int64_t, 4> dims;          /// dimensionality.
  SmallVector<int64_t, 4> strides;       /// stride in each dim.

  /// Sequences of `insertvalue` and `extractvalue` instructions.
  SmallVector<Instruction *, 4> insertInsts;
  SmallVector<Instruction *, 4> extractInsts;

  Type *getRankedArrayType(ArrayRef<int64_t> dims) const {
    Type *newType =
        ArrayType::get(ptr->getType()->getPointerElementType(), dims.back());
    for (size_t i = 1; i < dims.size(); i++)
      newType = ArrayType::get(newType, dims[dims.size() - i - 1]);
    return newType;
  }

  /// Create a new type from the ptr and dims.
  Type *getRankedArrayType() const { return getRankedArrayType(dims); }

  /// Initialize the data fields.
  bool initialize(Type *type) {
    if (!type->isAggregateType() || !isa<StructType>(type))
      return false;

    StructType *structType = dyn_cast<StructType>(type);
    if (structType->getNumElements() != 5)
      return false;

    // The sizes of each dim is the 4th field.
    ArrayType *dimsType = dyn_cast<ArrayType>(structType->getElementType(3));
    if (!dimsType)
      return false;
    // Unset the dims array.
    dims.assign(dimsType->getNumElements(), -1);
    dim_values.assign(dimsType->getNumElements(), nullptr);

    // Gather strides.
    ArrayType *stridesType = dyn_cast<ArrayType>(structType->getElementType(4));
    if (!stridesType ||
        stridesType->getNumElements() != dimsType->getNumElements())
      return false;
    strides.assign(stridesType->getNumElements(), -1);
    stride_values.assign(stridesType->getNumElements(), nullptr);

    return true;
  }

  bool insertInstsCompleted() const {
    return insertInsts.size() == dims.size() * 2 + 3;
  }

  bool checkAggregatedType(Type *type) const {
    if (!type->isAggregateType() || !isa<StructType>(type))
      return false;

    StructType *structType = dyn_cast<StructType>(type);
    if (structType->getNumElements() != 5)
      return false;

    ArrayType *dimsType = dyn_cast<ArrayType>(structType->getElementType(3));
    if (!dimsType || dimsType->getNumElements() != dims.size())
      return false;

    ArrayType *stridesType = dyn_cast<ArrayType>(structType->getElementType(4));
    if (!stridesType || stridesType->getNumElements() != strides.size())
      return false;

    return true;
  }

  /// Will abort if the Value is not a ConstantInt.
  int64_t getI64Value(Value *value) {
    assert(isa<ConstantInt>(value));

    ConstantInt *CI = dyn_cast<ConstantInt>(value);
    assert(CI->getBitWidth() == 64);

    return CI->getSExtValue();
  }

  /// From the scalar offset to a set of offset for each dim.
  /// There should be in total dimSize * 2 expressions.
  /// Dim size will be dims.size() * 2 since the dims in this Seq only has the
  /// tile dims.
  void processOffset(Function &F) {
    // Won't process constant offsets.
    if (isa<ConstantInt>(offset))
      return;

    SmallVector<Value *> offsets;
    SmallVector<int64_t> strides;

    Value *curr = offset;
    BinaryOperator *binOp;
    unsigned dimSize = dims.size() * 2;
    for (unsigned i = 0; i < dimSize; ++i) {
      binOp = cast<BinaryOperator>(curr);
      // binOp->dump();
      assert(binOp->getOpcode() == BinaryOperator::Add);

      Value *lhs = binOp->getOperand(0), *rhs = binOp->getOperand(1);
      BinaryOperator *lhsBinOp = dyn_cast<BinaryOperator>(lhs);
      BinaryOperator *rhsBinOp = dyn_cast<BinaryOperator>(rhs);
      BinaryOperator *mul =
          (lhsBinOp && lhsBinOp->getOpcode() == BinaryOperator::Mul) ? lhsBinOp
                                                                     : rhsBinOp;
      BinaryOperator *add =
          (lhsBinOp && lhsBinOp->getOpcode() == BinaryOperator::Mul) ? rhsBinOp
                                                                     : lhsBinOp;

      assert(mul && mul->getOpcode() == BinaryOperator::Mul);
      assert(!add || add->getOpcode() == BinaryOperator::Add);

      // Offsets and strides are both from the mul operands, since we basically
      // multiply offset by stride.
      // Strides should be constant values.
      offsets.push_back(mul->getOperand(0));
      strides.push_back(getI64Value(mul->getOperand(1)));

      if (add)
        curr = cast<Value>(add);
    }

    assert(!offsets.empty());
    assert(offsets.size() == strides.size());
    assert(strides[0] == 1);

    SmallVector<int64_t> partialDims;
    for (unsigned i = 1; i < strides.size(); ++i)
      partialDims.push_back(strides[i] / strides[i - 1]);
    assert(partialDims.size() == dimSize - 1);

    std::reverse(partialDims.begin(), partialDims.end());
    std::reverse(offsets.begin(), offsets.end());

    Type *rankedArrType = getRankedArrayType(partialDims);
    Type *restoredType =
        PointerType::get(PointerType::get(rankedArrType, F.getAddressSpace()),
                         F.getAddressSpace());

    // Below is a sequence of instructions that -
    // 1. Cast the original pointer to a pointer that points to an array;
    // 2. Load the array
    // 3. Get the subarray based on the offsets using GEP.
    // 4. Cast the subarray into raw pointer, so that it can be accepted by the
    // original callers.
    BitCastInst *bitCastInst = new BitCastInst(
        ptr, restoredType, Twine(""), cast<Instruction>(offset)->getNextNode());
    LoadInst *load = new LoadInst(
        cast<PointerType>(restoredType)->getElementType(), bitCastInst,
        Twine(""), cast<Instruction>(bitCastInst->getNextNode()));
    GetElementPtrInst *gep = GetElementPtrInst::Create(
        rankedArrType, load, {offsets[0], offsets[1]}, Twine(""),
        cast<Instruction>(load->getNextNode()));
    ptr = new BitCastInst(gep, ptr->getType(), Twine(""),
                          cast<Instruction>(gep)->getNextNode());
  }

  /// Append insInst to the insertInsts list, and gather the value to be
  /// inserted into the members of this class.
  bool addInsertInst(InsertValueInst *insInst) {
    Value *agg = insInst->getAggregateOperand();
    if (!checkAggregatedType(agg->getType()))
      return false;

    ArrayRef<unsigned> indices = insInst->getIndices();
    assert(indices.size() >= 1);

    if (indices[0] == 1) {
      setMemberOnce(ptr, insInst->getInsertedValueOperand());
      assert(ptr->getType()->isPointerTy() && "ptr should be a pointer.");
    } else if (indices[0] == 2) {
      setMemberOnce(offset, insInst->getInsertedValueOperand());
    } else if (indices[0] == 3) {
      assert(dims.size() > indices[1]);
      setMemberOnce(dims[indices[1]],
                    getI64Value(insInst->getInsertedValueOperand()), -1);
      setMemberOnce(dim_values[indices[1]], insInst->getInsertedValueOperand());
    } else if (indices[0] == 4) {
      assert(strides.size() > indices[1]);
      setMemberOnce(strides[indices[1]],
                    getI64Value(insInst->getInsertedValueOperand()), -1);
      setMemberOnce(stride_values[indices[1]],
                    insInst->getInsertedValueOperand());
    }

    insertInsts.push_back(insInst);
    return true;
  }

  /// Erase all insertvalue and extractvalue instructions associated with this
  /// instance. Be careful that here we erase from the last instruction in the
  /// sequence to the first.
  void eraseAllInsts() {
    while (!extractInsts.empty()) {
      Instruction *inst = extractInsts.pop_back_val();
      if (!inst->use_empty())
        inst->dump();
      assert(inst->use_empty());
      inst->eraseFromParent();
    }

    while (!insertInsts.empty()) {
      Instruction *inst = insertInsts.pop_back_val();
      if (!inst->use_empty())
        inst->dump();
      assert(inst->use_empty());
      inst->eraseFromParent();
    }
  }

  /// Replace the uses of values defined by `extractvalue` with the original
  /// values.
  void replaceExtractValueUses() {
    for (Instruction *inst : extractInsts) {
      ExtractValueInst *extInst = dyn_cast<ExtractValueInst>(inst);
      if (extInst->getNumIndices() == 1) {
        if (extInst->getIndices()[0] == 1)
          extInst->replaceAllUsesWith(ptr);
        if (extInst->getIndices()[0] == 2)
          extInst->replaceAllUsesWith(offset);
      } else if (extInst->getNumIndices() == 2) {
        if (extInst->getIndices()[0] == 3) {
          extInst->replaceAllUsesWith(dim_values[extInst->getIndices()[1]]);
        } else if (extInst->getIndices()[0] == 4) {
          extInst->replaceAllUsesWith(stride_values[extInst->getIndices()[1]]);
        }
      }
    }
  }

private:
  /// The member `lhs` can only be set if it currently is the default value.
  template <typename T, typename U>
  void setMemberOnce(T &lhs, T rhs, U defaultValue) {
    assert(lhs == defaultValue);
    lhs = rhs;
  }

  template <typename T> void setMemberOnce(T *&lhs, T *rhs) {
    assert(lhs == nullptr);
    lhs = rhs;
  }
};

} // namespace

static void
findInsertExractValueSequences(Function &F,
                               SmallVectorImpl<InsExtSequence> &seqs) {
  SmallPtrSet<Instruction *, 4> visited;
  SmallDenseMap<Value *, unsigned> StructToSeqId;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      // An InsExtSequence starts at an `insertvalue` instruction
      // instance with its 1st operand undefined (`undef`).
      if (InsertValueInst *insInst = dyn_cast<InsertValueInst>(&I)) {
        if (isa<UndefValue>(insInst->getOperand(0))) {
          assert(insInst->getAggregateOperand());

          InsExtSequence seq;
          // If initialize fails, then we should know that `insertvalue` doesn't
          // operate on a Value of `memref` type.
          if (!seq.initialize(insInst->getAggregateOperand()->getType()))
            continue;

          bool isValidSeq = true;

          // Now we trace through the use-def chain of the value defined by the
          // first `insertvalue`.
          Instruction *inst = &I;
          while (isa<InsertValueInst>(inst)) {
            // If we find a visited instruction in the use-def chain, it is a
            // case we cannot handle, and therefore, invalid.
            // Otherwise, we try to append the instruction.
            if (visited.contains(inst) ||
                !seq.addInsertInst(cast<InsertValueInst>(inst))) {
              isValidSeq = false;
              break;
            }
            visited.insert(inst);

            if (seq.insertInstsCompleted())
              break;

            // The only user of a `insertvalue` applied to memref should be a
            // `insertvalue` inst.
            if (inst->getNumUses() != 1 ||
                !isa<InsertValueInst>(*inst->user_begin())) {
              isValidSeq = false;
              break;
            }
            inst = dyn_cast<InsertValueInst>(*inst->user_begin());
          }

          // At this point, `inst` should be the last `insertvalue` applied on
          // the memref value. It's users should all be `extractvalue`.
          Instruction *lastInst = inst;
          for (User *user : lastInst->users()) {
            inst = dyn_cast<Instruction>(user);
            if (!isa<ExtractValueInst>(inst) || visited.contains(inst)) {
              isValidSeq = false;
              break;
            }
            visited.insert(inst);
            seq.extractInsts.push_back(inst);
          }

          // Every condition is matched.
          if (isValidSeq) {
            seq.processOffset(F);
            seq.replaceExtractValueUses();
            seqs.push_back(seq);
          }
        }
      }
    }
  }
}

/// After this pass, a new function cloned from `F` will have ranked array
/// arguments at the end, which are duplicated from those unranked.
/// The mapping from unranked arrays to their resp. ranked counterparts will be
/// stored in `RankedArrVMap`.
static Function *
duplicateFunctionsWithRankedArrays(Function *F,
                                   SmallVectorImpl<InsExtSequence> &Seqs,
                                   ValueToValueMapTy &RankedArrVMap) {
  // Resolve parameter types. The first part should be the same as `F`, and the
  // second part should let every Seq create a new Array in their order.
  SmallVector<Type *, 4> ParamTypes;
  FunctionType *FuncType = F->getFunctionType();

  for (unsigned i = 0; i < FuncType->getFunctionNumParams(); ++i)
    ParamTypes.push_back(FuncType->getFunctionParamType(i));

  SmallPtrSet<Value *, 4> ExistArgs;
  for (unsigned i = 0; i < F->arg_size(); ++i)
    ExistArgs.insert(F->getArg(i));

  // Map an argument to the new ranked array type.
  SmallDenseMap<Value *, Type *> ArgToArrType;
  for (InsExtSequence &Seq : Seqs) // Each Seq has a new ArrayType arg.
    // Note that here we only set the array type ONCE. It is based on the
    // understanding that the first sequence will give the full info of the
    // target memref.
    if (ExistArgs.count(Seq.ptr) && !ArgToArrType.count(Seq.ptr))
      ArgToArrType[Seq.ptr] = Seq.getRankedArrayType();

  for (InsExtSequence &Seq : Seqs) { // Each Seq has a new ArrayType arg.
    AllocaInst *I = dyn_cast<AllocaInst>(Seq.ptr);
    if (!I) { // Will resolve the function arguments later.
      // If the source ptr for the Sequence is a function argument, we extend
      // the function signature.
      if (ExistArgs.count(Seq.ptr))
        ParamTypes.push_back(
            PointerType::get(ArgToArrType[Seq.ptr], F->getAddressSpace()));
    } else // Create a new AllocaInst.
      new AllocaInst(Seq.getRankedArrayType(), F->getAddressSpace(), Twine(""),
                     I);
  }

  FunctionType *NewFuncType =
      FunctionType::get(F->getReturnType(), ParamTypes, F->isVarArg());

  // Instantiate the new Function instance. Its name extends the original one
  // with .new, and possibly we won't need this feature.
  Function *NewFunc =
      Function::Create(NewFuncType, F->getLinkage(), F->getAddressSpace(),
                       F->getName() + ".dup_ranked", F->getParent());

  // For cloning, we should map the original parameters to the new ones in the
  // new function.
  ValueToValueMapTy VMap;
  for (unsigned i = 0; i < F->arg_size(); i++)
    VMap[F->getArg(i)] = NewFunc->getArg(i);

  // We also map the raw pointers to the new ranked arrays. Note that the key to
  // this map is the new argument in the `NewFunc`. The value of the mapping is
  // the newly appended arguments at the end of `NewFunc`.
  unsigned j = 0;
  for (unsigned i = 0; i < Seqs.size(); i++)
    if (VMap.count(Seqs[i].ptr))
      RankedArrVMap[VMap[Seqs[i].ptr]] = NewFunc->getArg(F->arg_size() + (j++));

  // Finally, clone the content from the old function into the new one.
  SmallVector<ReturnInst *, 4> Returns;
  CloneFunctionInto(NewFunc, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns);

  // Map the alloca pairs. It is based on the previous code that the new static
  // alloca will always be one instruction before the one it is mapped from.
  for (BasicBlock &BB : *NewFunc)
    for (Instruction &I : BB) {
      AllocaInst *Curr = dyn_cast<AllocaInst>(&I);
      if (!Curr)
        continue;
      AllocaInst *Next = dyn_cast<AllocaInst>(Curr->getNextNode());
      if (!Next)
        continue;
      if (Curr->isStaticAlloca())
        RankedArrVMap[Next] = Curr;
    }

  return NewFunc;
}

/// Get the size of each array dimension. If the array type is like [2 x [3 x
/// float]], then the sizes returned will be {3, 1}.
static void getArraySizes(Type *Ty, SmallVectorImpl<Value *> &Sizes) {
  SmallVector<int64_t, 4> IntDims;
  while (Ty->isArrayTy()) {
    IntDims.push_back(Ty->getArrayNumElements());
    Ty = Ty->getArrayElementType();
  }

  int64_t IntSize = 1;
  Sizes.resize(IntDims.size());
  for (unsigned i = 0; i < IntDims.size(); i++) {
    Sizes[IntDims.size() - i - 1] =
        ConstantInt::get(IntegerType::get(Ty->getContext(), 64), IntSize);
    IntSize *= IntDims[IntDims.size() - i - 1];
  }
}

/// Duplicate the given GEP instruction, with its operand replaced by its
/// ranked array counterpart.
static Instruction *duplicateGEPWithRankedArray(Instruction *I,
                                                ValueToValueMapTy &RankedArrMap,
                                                unsigned &NumNewGEP) {
  assert(isa<GetElementPtrInst>(I) &&
         "Instruction given should be a GetElementPtrInst type.");

  GetElementPtrInst *GEP = cast<GetElementPtrInst>(I);
  assert(RankedArrMap.count(GEP->getPointerOperand()) &&
         "The original operand of the GEP should be in the RankedArrMap");

  // New pointer operand for the new GEP.
  Value *RankedArrayPtr = RankedArrMap[GEP->getPointerOperand()];
  SmallVector<Value *, 4> ArrSizes;
  getArraySizes(RankedArrayPtr->getType()->getPointerElementType(), ArrSizes);

  // The address to be accessed. We will recover the addresses to each dimension
  // from it.
  Value *Addr = *(GEP->idx_begin() + 0);

  // Here we iteratively resolve the index for the new GEP at each level of the
  // ranked array.
  SmallVector<Value *, 4> IdxList;

  // The first index should always be 0.
  ConstantInt *Zero =
      ConstantInt::get(IntegerType::get(I->getContext(), 64), 0);
  IdxList.push_back(Zero);

  // The reset of the indices will be calculated iteratively, based on the
  // formula: Addr[i + 1] = Idx[i] * ArrSizes[i] + Addr[i]. Here:
  //
  // - Addr[i] is the flattened address for level i;
  // - Idx[i] is the exact index at the i-th level;
  // - and ArrSizes[i] is the i-th level's size.
  //
  // For instance, given an array of dims [2][3][4] and its flattened address
  // 10, we can calculate its each dim's address by:
  //
  // Addr[0] = 10;
  // Idx[0] = 10 / 12 = 0, Addr[1] = 10 % 12 = 10;
  // Idx[1] = 10 / 4 = 2,  Addr[2] = 10 % 4 = 2;
  // Idx[2] = 2 / 1 = 2.
  //
  // You may need -instcombine to clean-up duplicated arithmetic operations.
  for (unsigned i = 0; i < ArrSizes.size() - 1; i++) {
    Value *Idx = BinaryOperator::Create(
        BinaryOperator::BinaryOps::UDiv, Addr, ArrSizes[i],
        "gep" + Twine(NumNewGEP) + "idx" + Twine(i), GEP);
    IdxList.push_back(Idx);

    Addr = BinaryOperator::Create(
        BinaryOperator::BinaryOps::URem, Addr, ArrSizes[i],
        "gep" + Twine(NumNewGEP) + "addr" + Twine(i + 1), GEP);
  }
  IdxList.push_back(Addr);

  GetElementPtrInst *NewGEP = GetElementPtrInst::CreateInBounds(
      RankedArrayPtr, IdxList, "gep" + Twine(NumNewGEP++), GEP->getNextNode());

  return NewGEP;
}

static bool shouldSkipArgument(Value *arg) {
  return arg->use_empty() && arg->getType()->isPointerTy();
}

/// Clone F2 into a new function without duplicated parameters, and replace F1's
/// uses with the new function being created.
static Function *replaceFunction(Function *F1, Function *F2) {
  // Create the new function interface.
  SmallVector<Type *, 4> ParamTypes;
  FunctionType *FuncType = F2->getFunctionType();
  // Skip no-use args, i.e., original pointers.
  for (unsigned i = 0; i < FuncType->getFunctionNumParams(); ++i)
    if (!shouldSkipArgument(F2->getArg(i)))
      ParamTypes.push_back(FuncType->getFunctionParamType(i));

  FunctionType *NewFuncType =
      FunctionType::get(F2->getReturnType(), ParamTypes, F2->isVarArg());

  // Build a trimmed copy of the F2 interface, using the same name as F1, and
  // erase F1. After this step, we have a clean state F1 with the updated
  // argument types.
  std::string Name = F1->getName().str();
  F1->setName(Name + Twine(".origin"));
  Function *NewFunc =
      Function::Create(NewFuncType, F2->getLinkage(), F2->getAddressSpace(),
                       Name, F2->getParent());

  // Create a mapping from the original parameter list to the new one.
  DenseMap<unsigned, unsigned> ArgMap;
  unsigned NumArgs = 0;
  FunctionType *F1Type = F1->getFunctionType();
  // First gather those unaffected indices.
  for (unsigned i = 0; i < F1Type->getFunctionNumParams(); ++i)
    if (!shouldSkipArgument(F2->getArg(i)))
      ArgMap[NumArgs++] = i;
  // Then put every other indices at the end.
  for (unsigned i = 0; i < F1Type->getFunctionNumParams(); ++i)
    if (shouldSkipArgument(F2->getArg(i)))
      ArgMap[NumArgs++] = i;

  // Prepare parameter mapping for cloning. Note that we also skip no-use
  // argument here.
  ValueToValueMapTy VMap;
  unsigned argIdx = 0;
  for (unsigned i = 0; i < F2->arg_size(); i++) {
    if (!shouldSkipArgument(F2->getArg(i)))
      VMap[F2->getArg(i)] = NewFunc->getArg(argIdx++);
    else // CloneFunctionInto requires every F2 arg exists as a VMap key.
      VMap[F2->getArg(i)] = UndefValue::get(F2->getArg(i)->getType());
  }

  // Clone.
  SmallVector<ReturnInst *, 4> Returns;
  llvm::CloneFunctionInto(NewFunc, F2, VMap,
                          CloneFunctionChangeType::LocalChangesOnly, Returns);

  return NewFunc;
}

static SmallVector<Function *> TopologicalSort(ArrayRef<Function *> funcs) {
  SmallPtrSet<Function *, 4> Avail{funcs.begin(), funcs.end()};

  DenseMap<Function *, SmallPtrSet<Function *, 4>> graph;
  for (Function *F : funcs)
    graph[F] = {};

  for (Function *F : funcs)
    for (BasicBlock &BB : F->getBasicBlockList())
      for (Instruction &I : BB)
        if (isa<CallInst>(I) &&
            Avail.count(cast<CallInst>(I).getCalledFunction()))
          graph[F].insert(cast<CallInst>(I).getCalledFunction());

  SmallVector<Function *> sorted;
  while (true) {
    SmallPtrSet<Function *, 4> to_remove;
    for (auto &it : graph)
      if (it.second.empty())
        to_remove.insert(it.first);

    for (Function *F : to_remove) {
      graph.erase(graph.find(F));
      sorted.push_back(F);
    }

    for (auto &it : graph)
      for (Function *F : to_remove)
        it.second.erase(F);

    if (to_remove.empty())
      break;
  }

  assert(sorted.size() == funcs.size() &&
         "The input list of functions cannot form an acyclic graph.");

  return sorted;
}

/// This helper function convert the MemRef value represented by an
/// aggregated type to a ranked N-d array. The function interface, as well
/// as the internal usage of GEP will be updated.
///
/// The overall workflow is:
/// 1. Gather insertvalue/extractvalue sequences that identify aggregated
/// MemRef types.
/// 2. Duplicate the original function with extra ranked array types.
/// 3. Replace GEPs in the duplicated function.
/// 4. Clone the content back from the duplicated one. Clean up.
///
/// If ranked is passed as false, this function stops at step 1.
///
static void convertMemRefToArray(Module &M, bool ranked = false) {
  DenseMap<Function *, SmallVector<InsExtSequence, 4>> FuncToSeqs;

  /// First, we gather the mapping from Functions to all the InsExtSequences
  /// that they should rewrite on.
  for (auto &F : M) {
    SmallVector<InsExtSequence, 4> Seqs;
    findInsertExractValueSequences(F, Seqs);

    if (Seqs.empty())
      continue;

    FuncToSeqs[&F] = Seqs;

    // Clean up the sequences in the current function in reversed order.
    std::reverse(Seqs.begin(), Seqs.end());
    for (InsExtSequence &Seq : Seqs)
      Seq.eraseAllInsts();
  }

  // If it is ok to keep the array unranked, we can just return.
  if (!ranked)
    return;

  // Topological sort the functions.
  SmallVector<Function *, 4> Funcs;
  for (auto &it : FuncToSeqs)
    Funcs.push_back(it.first);
  Funcs = TopologicalSort(Funcs);

  // Map to the new version.
  DenseMap<Function *, Function *> FuncToNew;

  // Next, we iterate these Function, Sequence pairs and create new
  // candidate functions. The new function at this stage looks almost the
  // same as the original one, just have additional arguments that are
  // ranked arrays.
  for (Function *F : Funcs) {
    ValueToValueMapTy RankedArrVMap;
    auto &Seqs = FuncToSeqs[F];

    Function *NewFunc =
        duplicateFunctionsWithRankedArrays(F, Seqs, RankedArrVMap);

    SmallVector<Instruction *, 4> GEPList;
    for (BasicBlock &BB : *NewFunc)
      for (Instruction &I : BB)
        if (isa<GetElementPtrInst>(&I) && isa<Argument>(I.getOperand(0)))
          GEPList.push_back(&I);

    // Create new GEPs that use the ranked arrays and remove the old ones.
    unsigned NumNewGEP = 0;
    for (Instruction *I : GEPList) {
      Instruction *NewGEP =
          duplicateGEPWithRankedArray(I, RankedArrVMap, NumNewGEP);
      I->replaceAllUsesWith(NewGEP);
      I->eraseFromParent();
    }

    // If there is any caller.
    SmallVector<CallInst *> Callers;
    for (BasicBlock &BB : *NewFunc)
      for (Instruction &I : BB)
        if (CallInst *CI = dyn_cast<CallInst>(&I))
          if (FuncToNew.count(CI->getCalledFunction()))
            Callers.push_back(CI);

    for (CallInst *Caller : Callers) {
      Function *Callee = Caller->getCalledFunction();

      // Initial arguments.
      SmallVector<Value *> Args;
      unsigned NumArg = 0;
      for (Value *Arg : Caller->args())
        if (Arg->getType() == FuncToNew[Callee]->getArg(NumArg)->getType()) {
          Args.push_back(Arg);
          NumArg++;
        }

      // Duplicated arguments (new memref).
      SmallVector<Instruction *> toErase;
      for (Value *Arg : Caller->args()) {
        // If it is a newly mapped memref argument -
        if (RankedArrVMap.count(Arg))
          Args.push_back(RankedArrVMap[Arg]);
        else if (isa<BitCastInst>(Arg)) {
          // Or it is a result from a bitcast expression chain.
          // This chain is based on the instructions generated by the
          // processOffset function.
          /// TODO:  can we make this a pattern?
          BitCastInst *bitCastInst = cast<BitCastInst>(Arg);
          GetElementPtrInst *gep =
              cast<GetElementPtrInst>(bitCastInst->getOperand(0));
          LoadInst *loadInst = cast<LoadInst>(gep->getOperand(0));
          BitCastInst *src = cast<BitCastInst>(loadInst->getOperand(0));

          // Now we should replace these instructions by a single GEP.
          // This GEP directly calculates the address from the input ranked
          // array pointer.
          Value *newArg = RankedArrVMap[src->getOperand(0)];
          SmallVector<Value *> indices;
          // Start with 0.
          indices.push_back(
              ConstantInt::get((*gep->idx_begin())->getType(), 0));
          // Borrow the indices from the original GEP.
          for (Value *val : gep->indices())
            indices.push_back(val);
          // Construct the new GEP.
          GetElementPtrInst *newGEP = GetElementPtrInst::Create(
              cast<PointerType>(newArg->getType())->getElementType(), newArg,
              indices, Twine(""),
              cast<Instruction>(bitCastInst->getNextNode()));

          // The result from this newGEP will be the new argument, i.e., the
          // subview pointer.
          Args.push_back(newGEP);

          // Prepare what to erase in the end. They should be in the reversed
          // order.
          toErase.append({bitCastInst, gep, loadInst, src});
        }
      }

      // New caller.
      CallInst::Create(FuncToNew[Callee], Args, Twine(), Caller);
      // Erase the original caller.
      Caller->eraseFromParent();
      for (Instruction *inst : toErase)
        inst->eraseFromParent();
    }

    FuncToNew[F] = replaceFunction(F, NewFunc);
    FuncToNew[F]->addFnAttr(Attribute::NoInline);

    // FuncToNew[F]->dump();

    // Finally, delete the duplicate.
    NewFunc->eraseFromParent();
  }

  // Erase the original functions backward.
  std::reverse(Funcs.begin(), Funcs.end());
  for (Function *F : Funcs)
    F->eraseFromParent();
}

namespace {

struct ConvertMemRefToArray : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  ConvertMemRefToArray() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    convertMemRefToArray(M);

    return false;
  }
};
} // namespace

namespace {

struct ConvertMemRefToRankedArray : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  ConvertMemRefToRankedArray() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    convertMemRefToArray(M, /*ranked=*/true);

    return false;
  }
};

} // namespace

/// Rename the name of basic blocks, function arguments, and values defined
/// by instructions with string prefixes.
static void
renameBasicBlocksAndValues(Module &M,
                           llvm::ArrayRef<llvm::StringRef> ParamNames) {
  // Rename BB and I
  size_t BBCnt = 0, ValCnt = 1, ArgCnt = 0;
  for (Function &F : M) {
    // Rename arguments
    if (F.getName() == XlnTop) {
      for (size_t i = 0; i < ParamNames.size(); i++)
        F.getArg(i)->setName(ParamNames[i]);
      for (size_t i = ParamNames.size(); i < F.arg_size(); i++)
        F.getArg(i)->setName("arg_" + Twine(ArgCnt++));
    } else {
      for (Argument &arg : F.args()) {
        arg.setName("arg_" + Twine(ArgCnt++));
      }
    }

    for (BasicBlock &BB : F) {
      // Rename basic blocks
      BB.setName("bb_" + Twine(BBCnt++));

      for (Instruction &I : BB) {
        // Rename variables
        Value *V = &I;
        if (V && !V->getType()->isVoidTy())
          V->setName("val_" + Twine(ValCnt++));
      }
    }
  }
}

namespace {

struct RenameBasicBlocksAndValues : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  RenameBasicBlocksAndValues() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    llvm::SmallVector<llvm::StringRef, 4> ParamNames;
    llvm::SplitString(XlnNames, ParamNames, ",");

    renameBasicBlocksAndValues(M, ParamNames);
    return false;
  }
};

} // namespace

static void annotateXilinxAttributes(Module &M) {
  assert(!XlnTop.empty() &&
         "-xlntop should be specified to annotate properties.");

  Function *F = M.getFunction(XlnTop);
  assert(F != nullptr && "Top function should be found.");

  // Top function annotation.
  F->addFnAttr("fpga.top.func", XlnTop);
}

namespace {

struct AnnotateXilinxAttributes : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  AnnotateXilinxAttributes() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    annotateXilinxAttributes(M);
    return false;
  }
};

} // namespace

namespace {

struct StripInvalidAttributes : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  StripInvalidAttributes() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    // Here is the list of all supported attributes. Note that not all the
    // differences are covered.
    // https://github.com/llvm/llvm-project/blob/release%2F3.9.x/llvm/include/llvm/IR/Attributes.td
    for (auto &F : M) {
      F.removeFnAttr(Attribute::AttrKind::NoFree);
      F.removeFnAttr(Attribute::AttrKind::NoSync);
      F.removeFnAttr(Attribute::AttrKind::Speculatable);
      F.removeFnAttr(Attribute::AttrKind::WillReturn);
    }

    return false;
  }
};

} // namespace

/// Rewrite fneg to fsub, e.g., %1 = fneg double %0 will be rewritten to
/// %1 = fsub double -0.000000e+00, %0
static Instruction *rewriteFNegToFSub(Instruction &I) {
  assert(I.getOpcode() == Instruction::FNeg && "OpCode should be FNeg.");

  Value *Operand = I.getOperand(0);
  Type *OperandTy = Operand->getType();
  assert(OperandTy->isFloatingPointTy() &&
         "The operand to fneg should be floating point.");

  // NOTE: The zero created here is negative.
  Value *NegZero = ConstantFP::get(
      I.getContext(),
      APFloat::getZero(OperandTy->getFltSemantics(), /*Negative=*/true));

  std::string NIName = I.getName().str() + ".sub";
  Instruction *NI = BinaryOperator::Create(Instruction::BinaryOps::FSub,
                                           NegZero, Operand, "", &I);
  I.replaceAllUsesWith(NI);

  return NI;
}

namespace {

/// Rewrite some math instructions to work together with Vitis.
struct XilinxRewriteMathInstPass : public ModulePass {
  static char ID;
  XilinxRewriteMathInstPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    SmallVector<Instruction *, 4> ToErase;

    for (auto &F : M)
      for (auto &BB : F)
        for (auto &I : BB) {
          if (isa<UnaryInstruction>(I) && I.getOpcode() == Instruction::FNeg) {
            rewriteFNegToFSub(I);
            ToErase.push_back(&I);
          }
        }

    for (Instruction *I : ToErase) {
      assert(I->use_empty() && "Inst to be erased should have empty use.");
      I->eraseFromParent();
    }

    return false;
  }
};

} // namespace

static void unrollLoop(Loop *loop) {
  LLVMContext &Context = loop->getHeader()->getContext();
  MDNode *EnableUnrollMD =
      MDNode::get(Context, MDString::get(Context, "llvm.loop.unroll.full"));
  MDNode *LoopID = loop->getLoopID();
  MDNode *NewLoopID = makePostTransformationMetadata(
      Context, LoopID, {"llvm.loop.unroll."}, {EnableUnrollMD});
  loop->setLoopID(NewLoopID);

  if (!loop->isInnermost())
    for (auto &subloop : loop->getSubLoops())
      unrollLoop(subloop);
}

namespace {

/// Unroll all the loops in a specified function for Xilinx Vitis.
struct XilinxUnrollPass : public ModulePass {
  static char ID;
  XilinxUnrollPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {

    for (auto &F : M)
      if (F.getName() == XlnTop) {
        auto DT = llvm::DominatorTree(F);
        LoopInfo LI(DT);

        for (auto &loop : LI)
          unrollLoop(loop);
      }

    return false;
  }
};

} // namespace

/// Return a set of <dimension, size> as the partition information for the
/// current array type. The function only extacts the first half dimensions as
/// the others are the dimensions for the tilied units
std::vector<std::pair<unsigned, unsigned>>
getPartitionInfo(ArrayType *arrayTy) {
  std::vector<std::pair<unsigned, unsigned>> partitions;
  unsigned d = 0;
  do {
    partitions.push_back(
        std::pair<unsigned, unsigned>(d + 1, arrayTy->getNumElements()));
    arrayTy = dyn_cast<ArrayType>(arrayTy->getElementType());
    d++;
  } while (arrayTy);

  // The dimension number of arrays after Polymer should be a even number
  assert(d % 2 == 0);

  partitions.resize(d / 2);
  return partitions;
}

namespace {

/// Partition arrays in the top-level function arguments for Xilinx Vitis.
/// This pass partitions the array in first half dimensions completely to
/// parallelise the transformed arrays from Polymer. For instance, an array of
/// size 2x3x32x32 will be partitioned into 6 blocks of size 32x32
struct XilinxArrayPartitionPass : public ModulePass {
  static char ID;
  XilinxArrayPartitionPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {

    // Declare array partition APIs in Vitis HLS LLVM frontend
    auto mod = &M;
    auto voidTy = Type::getVoidTy(mod->getContext());
    mod->getOrInsertFunction("llvm.sideeffect",
                             FunctionType::get(voidTy, {}, false));
    auto arrayPartitionFunc = mod->getFunction("llvm.sideeffect");
    arrayPartitionFunc->addFnAttr(llvm::Attribute::InaccessibleMemOnly);
    arrayPartitionFunc->addFnAttr(llvm::Attribute::NoUnwind);
    // Remove unsupported attributes by Vitis HLS
    arrayPartitionFunc->removeFnAttr(llvm::Attribute::NoFree);
    arrayPartitionFunc->removeFnAttr(llvm::Attribute::NoSync);
    arrayPartitionFunc->removeFnAttr(llvm::Attribute::WillReturn);

    for (auto &F : M)
      if (F.getName() == XlnTop) {
        auto &BB = F.getEntryBlock();
        IRBuilder<> builder(&BB, BB.begin());
        for (unsigned i = 0; i < F.arg_size(); i++) {
          auto arg = F.getArg(i);
          if (arg->getType()->isPointerTy() &&
              arg->getType()->getPointerElementType()->isArrayTy()) {
            auto arrayTy =
                dyn_cast<ArrayType>(arg->getType()->getPointerElementType());
            auto partitions = getPartitionInfo(arrayTy);
            for (auto partition : partitions) {
              auto int32ty = Type::getInt32Ty(mod->getContext());
              OperandBundleDef bd = OperandBundleDef(
                  "xlx_array_partition",
                  (std::vector<Value *>){
                      arg, ConstantInt::get(int32ty, partition.first),
                      ConstantInt::get(int32ty, partition.second),
                      ConstantInt::get(int32ty, 1) /* block scheme*/});
              builder.CreateCall(arrayPartitionFunc, {}, {bd});
            }
          }
        }
      }

    return false;
  }
};

} // namespace

namespace {

/// Generate test bench tcl script for Xilinx Vitis. This pass parses the LLVM
/// IR and generates compatible test bench for the design in LLVM IR.
struct XilinxTBTclGenPass : public ModulePass {
  static char ID;
  XilinxTBTclGenPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    std::error_code ec;
    llvm::raw_fd_ostream XlnTBTcl(XlnTBTclNames, ec);

    XlnTBTcl << "open_project -reset tb\n"
             << XlnTBSources << "set_top " << XlnTop << "\n"
             << "open_solution -reset solution1\n"
             << "set_part \"zynq\"\n"
             << "create_clock -period \"100MHz\"\n"
             << "config_bind -effort high\n";

    for (auto &F : M)
      if (F.getName() == XlnTop) {
        for (unsigned i = 0; i < F.arg_size(); i++) {
          auto arg = F.getArg(i);
          if (arg->getType()->isPointerTy() &&
              arg->getType()->getPointerElementType()->isArrayTy()) {
            auto arrayTy =
                dyn_cast<ArrayType>(arg->getType()->getPointerElementType());
            auto partitions = getPartitionInfo(arrayTy);
            for (auto partition : partitions)
              XlnTBTcl << "set_directive_array_partition -dim "
                       << partition.first << " -factor " << partition.second
                       << " -type block \"" << XlnTop << "\" " << arg->getName()
                       << "\n";
          }
        }
      }

    XlnTBTcl << "csim_design\n"
             << "csynth_design cosim_design\n"
             << "exit\n";
    return false;
  }
};

} // namespace

char ConvertMemRefToArray::ID = 0;
static RegisterPass<ConvertMemRefToArray>
    X1("mem2ptr",
       "Convert MemRef structure to unranked array, i.e., raw pointer.");

char ConvertMemRefToRankedArray::ID = 1;
static RegisterPass<ConvertMemRefToRankedArray>
    X2("mem2arr", "Convert MemRef structure to ranked array.");

char RenameBasicBlocksAndValues::ID = 2;
static RegisterPass<RenameBasicBlocksAndValues>
    X3("xlnname", "Rename entities in the model with string prefixes to follow "
                  "Xilinx tool guidence.");

char AnnotateXilinxAttributes::ID = 3;
static RegisterPass<AnnotateXilinxAttributes>
    X4("xlnanno", "Annotate attributes for Xilinx HLS.");

char StripInvalidAttributes::ID = 4;
static RegisterPass<StripInvalidAttributes>
    X5("strip-attr",
       "Strip invalid function attributes not compatible with Clang 3.9.");

char XilinxRewriteMathInstPass::ID = 5;
static RegisterPass<XilinxRewriteMathInstPass>
    X6("xlnmath", "Rewrite math instructions for Xilinx Vitis.");

char XilinxUnrollPass::ID = 6;
static RegisterPass<XilinxUnrollPass>
    X7("xlnunroll",
       "Unroll all the loops in a specified function for Xilinx Vitis.");

char XilinxArrayPartitionPass::ID = 7;
static RegisterPass<XilinxArrayPartitionPass> X8(
    "xlnarraypartition",
    "Partition arrays in the top-level function arguments for Xilinx Vitis.");

char XilinxTBTclGenPass::ID = 8;
static RegisterPass<XilinxTBTclGenPass>
    X9("xlntbgen", "Generate test bench tcl script for Xilinx Vitis.");
