//===- VhlsLLVMRewriter.cc --------------------------------------*- C++ -*-===//
//
// This file implements a pass that transforms LLVM-IR for Vitis HLS input.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
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

namespace {

/// InsExtSequence holds a sequence of `insertvalue` and `extractvalue`
/// instructions that operate on the same aggregated value, which correpsonds to
/// a `memref` value in MLIR.
/// For more information on how the `memref` type is organized:
/// https://mlir.llvm.org/docs/ConversionToLLVMDialect/#default-convention
class InsExtSequence {
public:
  Value *ptr = nullptr;            /// aligned pointer.
  int64_t offset = -1;             /// memref offset.
  SmallVector<int64_t, 4> dims;    /// dimensionality.
  SmallVector<int64_t, 4> strides; /// stride in each dim.

  /// Sequences of `insertvalue` and `extractvalue` instructions.
  SmallVector<Instruction *, 4> insertInsts;
  SmallVector<Instruction *, 4> extractInsts;

  /// Create a new type from the ptr and dims.
  Type *getRankedArrayType() const {
    Type *newType =
        ArrayType::get(ptr->getType()->getPointerElementType(), dims.back());
    for (size_t i = 1; i < dims.size(); i++)
      newType = ArrayType::get(newType, dims[dims.size() - i - 1]);
    return newType;
  }

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

    // Gather strides.
    ArrayType *stridesType = dyn_cast<ArrayType>(structType->getElementType(4));
    if (!stridesType ||
        stridesType->getNumElements() != dimsType->getNumElements())
      return false;
    strides.assign(stridesType->getNumElements(), -1);

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
      setMemberOnce(offset, getI64Value(insInst->getInsertedValueOperand()),
                    -1);
    } else if (indices[0] == 3) {
      assert(dims.size() > indices[1]);
      setMemberOnce(dims[indices[1]],
                    getI64Value(insInst->getInsertedValueOperand()), -1);
    } else if (indices[0] == 4) {
      assert(strides.size() > indices[1]);
      setMemberOnce(strides[indices[1]],
                    getI64Value(insInst->getInsertedValueOperand()), -1);
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
      assert(inst->use_empty());
      inst->eraseFromParent();
    }

    while (!insertInsts.empty()) {
      Instruction *inst = insertInsts.pop_back_val();
      assert(inst->use_empty());
      inst->eraseFromParent();
    }
  }

  /// Replace the uses of values defined by `extractvalue` with the original
  /// values.
  void replaceExtractValueUses() {
    for (Instruction *inst : extractInsts) {
      ExtractValueInst *extInst = dyn_cast<ExtractValueInst>(inst);
      if (extInst->getNumIndices() == 1 && *extInst->idx_begin() == 1)
        extInst->replaceAllUsesWith(ptr);
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
          if (isValidSeq)
            seqs.push_back(seq);
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
  for (InsExtSequence Seq : Seqs) // Each Seq has a new ArrayType arg.
    ParamTypes.push_back(
        PointerType::get(Seq.getRankedArrayType(), F->getAddressSpace()));
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
  for (unsigned i = 0; i < Seqs.size(); i++)
    RankedArrVMap[VMap[Seqs[i].ptr]] = NewFunc->getArg(F->arg_size() + i);

  // Finally, clone the content from the old function into the new one.
  SmallVector<ReturnInst *, 4> Returns;
  CloneFunctionInto(NewFunc, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns);

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

/// Clone F2 into a new function without duplicated parameters, and replace F1's
/// uses with the new function being created.
static Function *replaceFunction(Function *F1, Function *F2) {
  // Create the new function interface.
  SmallVector<Type *, 4> ParamTypes;
  FunctionType *FuncType = F2->getFunctionType();
  // Skip no-use args, i.e., original pointers.
  for (unsigned i = 0; i < FuncType->getFunctionNumParams(); ++i)
    if (!F2->getArg(i)->use_empty())
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
    if (!F2->getArg(i)->use_empty())
      ArgMap[NumArgs++] = i;
  // Then put every other indices at the end.
  for (unsigned i = 0; i < F1Type->getFunctionNumParams(); ++i)
    if (F2->getArg(i)->use_empty())
      ArgMap[NumArgs++] = i;

  // Prepare parameter mapping for cloning. Note that we also skip no-use
  // argument here.
  ValueToValueMapTy VMap;
  unsigned argIdx = 0;
  for (unsigned i = 0; i < F2->arg_size(); i++) {
    if (!F2->getArg(i)->use_empty())
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
  DenseMap<Function *, SmallPtrSet<Function *, 4>> graph;
  for (Function *F : funcs)
    graph[F] = {};

  for (Function *F : funcs)
    for (BasicBlock &BB : F->getBasicBlockList())
      for (Instruction &I : BB)
        if (isa<CallInst>(I))
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
  DenseMap<Function *, SmallVector<InsExtSequence>> FuncToSeqs;

  /// First, we gather the mapping from Functions to all the InsExtSequences
  /// that they should rewrite on.
  for (auto &F : M) {
    SmallVector<InsExtSequence> Seqs;
    findInsertExractValueSequences(F, Seqs);

    if (Seqs.empty())
      continue;

    // Clean up the current function.
    for (InsExtSequence Seq : Seqs) {
      Seq.replaceExtractValueUses();
      Seq.eraseAllInsts();
    }
    FuncToSeqs[&F] = Seqs;
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
    SmallVector<InsExtSequence> &Seqs = FuncToSeqs[F];

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

      // Duplicated arguments.
      for (Value *Arg : Caller->args())
        if (RankedArrVMap.count(Arg))
          Args.push_back(RankedArrVMap[Arg]);

      // New caller.
      CallInst *NewCaller =
          CallInst::Create(FuncToNew[Callee], Args, Twine(), Caller);
      // Erase the original caller.
      Caller->eraseFromParent();
    }

    FuncToNew[F] = replaceFunction(F, NewFunc);

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
