//===- Utils.cc - Utility functions ------------------ C++-===//

#include "phism/mlir/Transforms/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace llvm;
using namespace phism;

/// Copied from AffineToStandard.cpp
namespace {
/// Visit affine expressions recursively and build the sequence of operations
/// that correspond to it.  Visitation functions return an Value of the
/// expression subtree they visited or `nullptr` on error.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value> {
public:
  /// This internal class expects arguments to be non-null, checks must be
  /// performed at the call site.
  AffineApplyExpander(OpBuilder &builder, ValueRange dimValues,
                      ValueRange symbolValues, Location loc)
      : builder(builder), dimValues(dimValues), symbolValues(symbolValues),
        loc(loc) {}

  template <typename OpTy> Value buildBinaryExpr(AffineBinaryOpExpr expr) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    auto op = builder.create<OpTy>(loc, lhs, rhs);
    return op.getResult();
  }

  Value visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<arith::AddIOp>(expr);
  }

  Value visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<arith::MulIOp>(expr);
  }

  /// Euclidean modulo operation: negative RHS is not allowed.
  /// Remainder of the euclidean integer division is always non-negative.
  ///
  /// Implemented as
  ///
  ///     a mod b =
  ///         let remainder = srem a, b;
  ///             negative = a < 0 in
  ///         select negative, remainder + b, remainder.
  Value visitModExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (modulo by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "modulo by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value remainder = builder.create<arith::RemSIOp>(loc, lhs, rhs);
    Value zeroCst = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value isRemainderNegative = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, remainder, zeroCst);
    Value correctedRemainder =
        builder.create<arith::AddIOp>(loc, remainder, rhs);
    Value result = builder.create<SelectOp>(loc, isRemainderNegative,
                                            correctedRemainder, remainder);
    return result;
  }

  /// Floor division operation (rounds towards negative infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///        a floordiv b =
  ///            let negative = a < 0 in
  ///            let absolute = negative ? -a - 1 : a in
  ///            let quotient = absolute / b in
  ///                negative ? -quotient - 1 : quotient
  Value visitFloorDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (division by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value zeroCst = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value noneCst = builder.create<arith::ConstantIndexOp>(loc, -1);
    Value negative = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, lhs, zeroCst);
    Value negatedDecremented = builder.create<arith::SubIOp>(loc, noneCst, lhs);
    Value dividend =
        builder.create<SelectOp>(loc, negative, negatedDecremented, lhs);
    Value quotient = builder.create<arith::DivSIOp>(loc, dividend, rhs);
    Value correctedQuotient =
        builder.create<arith::SubIOp>(loc, noneCst, quotient);
    Value result =
        builder.create<SelectOp>(loc, negative, correctedQuotient, quotient);
    return result;
  }

  /// Ceiling division operation (rounds towards positive infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///     a ceildiv b =
  ///         let negative = a <= 0 in
  ///         let absolute = negative ? -a : a - 1 in
  ///         let quotient = absolute / b in
  ///             negative ? -quotient : quotient + 1
  Value visitCeilDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(loc) << "semi-affine expressions (division by non-const) are "
                        "not supported";
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value zeroCst = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value oneCst = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value nonPositive = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, lhs, zeroCst);
    Value negated = builder.create<arith::SubIOp>(loc, zeroCst, lhs);
    Value decremented = builder.create<arith::SubIOp>(loc, lhs, oneCst);
    Value dividend =
        builder.create<SelectOp>(loc, nonPositive, negated, decremented);
    Value quotient = builder.create<arith::DivSIOp>(loc, dividend, rhs);
    Value negatedQuotient =
        builder.create<arith::SubIOp>(loc, zeroCst, quotient);
    Value incrementedQuotient =
        builder.create<arith::AddIOp>(loc, quotient, oneCst);
    Value result = builder.create<SelectOp>(loc, nonPositive, negatedQuotient,
                                            incrementedQuotient);
    return result;
  }

  Value visitConstantExpr(AffineConstantExpr expr) {
    auto valueAttr =
        builder.getIntegerAttr(builder.getIndexType(), expr.getValue());
    auto op = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                valueAttr);
    return op.getResult();
  }

  Value visitDimExpr(AffineDimExpr expr) {
    assert(expr.getPosition() < dimValues.size() &&
           "affine dim position out of range");
    return dimValues[expr.getPosition()];
  }

  Value visitSymbolExpr(AffineSymbolExpr expr) {
    assert(expr.getPosition() < symbolValues.size() &&
           "symbol dim position out of range");
    return symbolValues[expr.getPosition()];
  }

private:
  OpBuilder &builder;
  ValueRange dimValues;
  ValueRange symbolValues;

  Location loc;
};
} // namespace

/// Create a sequence of operations that implement the `expr` applied to the
/// given dimension and symbol values.
namespace phism {
mlir::Value expandAffineExpr(OpBuilder &builder, Location loc, AffineExpr expr,
                             ValueRange dimValues, ValueRange symbolValues) {
  return AffineApplyExpander(builder, dimValues, symbolValues, loc).visit(expr);
}
} // namespace phism

static bool hasPeCaller(FuncOp f) {
  bool ret = false;
  f.walk([&](CallOp caller) {
    if (caller->hasAttr("phism.pe"))
      ret = true;
  });
  return ret;
}

namespace phism {

FuncOp getTopFunction(ModuleOp m) {
  FuncOp top = nullptr;
  m.walk([&](FuncOp f) {
    if (hasPeCaller(f)) {
      if (!top) {
        m.dump();
      }
      assert(!top && "There should be only one top function.");
      top = f;
    }
  });
  return top;
}

AffineMap filterExtraConstantResults(AffineMap affMap) {
  if (affMap.isSingleConstant())
    return affMap;

  SmallVector<AffineExpr> results;
  for (AffineExpr result : affMap.getResults()) {
    if (result.isa<AffineConstantExpr>())
      continue;
    results.push_back(result);
  }

  return AffineMap::get(affMap.getNumDims(), affMap.getNumSymbols(), results,
                        affMap.getContext());
}

FuncOp findPhismTop(ModuleOp m) {
  FuncOp top = nullptr;
  m.walk([&](FuncOp f) {
    if (f->hasAttr("phism.top")) {
      assert(!top && "There can only be one function with phism.top.");
      top = f;
    }
  });
  return top;
}

/// Put all the functions from the module that not the top or being called by
/// the top into the keep set.
void getFunctionsToKeep(ModuleOp m, FuncOp top, SmallPtrSetImpl<FuncOp> &keep) {
  m.walk([&](FuncOp f) { keep.insert(f); });

  keep.erase(top);
  top.walk([&](CallOp caller) {
    keep.erase(cast<FuncOp>(m.lookupSymbol(caller.getCallee())));
  });
}

static void getArgs(ArrayRef<Operation *> ops, llvm::SetVector<Value> &args) {
  args.clear();

  llvm::SetVector<Operation *> internalOps;
  for (Operation *parentOp : ops) {
    internalOps.insert(parentOp);
    parentOp->walk([&](Operation *op) { internalOps.insert(op); });
    parentOp->walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          if (!internalOps.contains(defOp))
            args.insert(operand);
        } else if (BlockArgument bArg = operand.dyn_cast<BlockArgument>()) {
          if (!internalOps.contains(bArg.getOwner()->getParentOp()))
            args.insert(operand);
        } else {
          llvm_unreachable("Operand cannot be handled.");
        }
      }
    });
  }
}

static std::pair<FuncOp, BlockAndValueMapping>
createCallee(MutableArrayRef<Operation *> ops, StringRef calleeName, ModuleOp m,
             OpBuilder &b) {
  assert(!ops.empty());

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(m.getBody(), std::prev(m.getBody()->end()));

  FunctionType calleeType = b.getFunctionType(llvm::None, llvm::None);
  FuncOp callee =
      b.create<FuncOp>(ops.front()->getLoc(), calleeName, calleeType);

  // Initialize the entry block and the return operation.
  Block *entry = callee.addEntryBlock();
  b.setInsertionPointToStart(entry);
  b.create<mlir::ReturnOp>(callee.getLoc());
  b.setInsertionPointToStart(entry);

  // Grab arguments from the top forOp.
  llvm::SetVector<Value> args;
  getArgs(ops, args);

  // Argument mapping for cloning. Also intialize arguments to the entry block.
  BlockAndValueMapping mapping;
  for (Value arg : args)
    mapping.map(arg, entry->addArgument(arg.getType()));

  callee.setType(b.getFunctionType(entry->getArgumentTypes(), llvm::None));

  for (Operation *op : ops)
    b.clone(*op, mapping);

  return {callee, mapping};
}

static CallOp createCaller(MutableArrayRef<Operation *> ops, FuncOp callee,
                           BlockAndValueMapping vmap, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);

  // Inversed mapping from callee arguments to values in the source function.
  BlockAndValueMapping imap = vmap.getInverse();

  SmallVector<Value> args;
  transform(callee.getArguments(), std::back_inserter(args),
            [&](Value value) { return imap.lookup(value); });

  // Get function type.
  assert(!ops.empty());
  b.setInsertionPoint(ops.front());
  CallOp caller = b.create<CallOp>(ops.front()->getLoc(), callee, args);

  return caller;
}

std::pair<Operation *, Operation *>
outlineFunction(MutableArrayRef<Operation *> ops, StringRef funcName,
                ModuleOp m) {
  FuncOp callee;
  BlockAndValueMapping vmap;

  OpBuilder b(m.getContext());
  std::tie(callee, vmap) =
      createCallee(MutableArrayRef<Operation *>(ops), funcName, m, b);
  CallOp caller =
      createCaller(MutableArrayRef<Operation *>(ops), callee, vmap, b);

  return {callee, caller};
}

} // namespace phism
