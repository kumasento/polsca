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
    if (caller->hasAttr("scop.pe"))
      ret = true;
  });
  return ret;
}

namespace phism {

FuncOp getTopFunction(ModuleOp m) {
  FuncOp top = nullptr;
  m.walk([&](FuncOp f) {
    if (hasPeCaller(f)) {
      assert(!top && "There should be only one top function.");
      top = f;
    }
  });
  return top;
}

} // namespace phism
