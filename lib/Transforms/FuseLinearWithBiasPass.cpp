#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "vortex/Dialect/Vortex/IR/VortexDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_FUSELINEARWITHBIAS
#include "vortex/Transforms/Passes.h.inc"

namespace {

struct LinearWithBiasPattern {
  linalg::FillOp fill;
  linalg::GenericOp contraction;
  linalg::GenericOp bias;
  Value lhs;
  Value rhs;
  Value output;
  Value fillValue;
  Value biasValue;
  int64_t m;
  int64_t n;
  int64_t k;
};

static bool hasIteratorTypes(linalg::GenericOp generic,
                             ArrayRef<utils::IteratorType> expected) {
  SmallVector<utils::IteratorType> iterators = generic.getIteratorTypesArray();
  return llvm::equal(iterators, expected);
}

static bool hasIndexingMaps(linalg::GenericOp generic,
                            ArrayRef<AffineMap> expected) {
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  return llvm::equal(maps, expected);
}

static SmallVector<AffineMap> getLinearContractionMaps(MLIRContext *ctx) {
  AffineExpr i = getAffineDimExpr(0, ctx);
  AffineExpr j = getAffineDimExpr(1, ctx);
  AffineExpr k = getAffineDimExpr(2, ctx);
  return {
      AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, {i, k}, ctx),
      AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, {j, k}, ctx),
      AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, {i, j}, ctx),
  };
}

static SmallVector<AffineMap> getBiasMaps(MLIRContext *ctx) {
  AffineExpr i = getAffineDimExpr(0, ctx);
  AffineExpr j = getAffineDimExpr(1, ctx);
  return {
      AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0, {i, j}, ctx),
      AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0, {j}, ctx),
      AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0, {i, j}, ctx),
  };
}

static bool isMulOf(Value value, Value lhs, Value rhs) {
  auto mul = value.getDefiningOp<arith::MulFOp>();
  if (!mul)
    return false;
  return mul.getLhs() == lhs && mul.getRhs() == rhs;
}

static bool matchContractionBody(linalg::GenericOp generic) {
  Block &body = generic.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;

  auto add = yield.getValues().front().getDefiningOp<arith::AddFOp>();
  if (!add)
    return false;

  Value lhsArg = body.getArgument(0);
  Value rhsArg = body.getArgument(1);
  Value accArg = body.getArgument(2);

  return add.getRhs() == accArg && isMulOf(add.getLhs(), lhsArg, rhsArg);
}

static bool matchBiasBody(linalg::GenericOp generic) {
  Block &body = generic.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;

  auto add = yield.getValues().front().getDefiningOp<arith::AddFOp>();
  if (!add)
    return false;

  Value valueArg = body.getArgument(0);
  Value biasArg = body.getArgument(1);
  return add.getLhs() == valueArg && add.getRhs() == biasArg;
}

static bool hasStaticF32LinearShapes(linalg::GenericOp contraction, Value lhs,
                                     Value rhs, Value output, int64_t &m,
                                     int64_t &n, int64_t &k) {
  auto lhsType = dyn_cast<MemRefType>(lhs.getType());
  auto rhsType = dyn_cast<MemRefType>(rhs.getType());
  auto outType = dyn_cast<MemRefType>(output.getType());
  if (!lhsType || !rhsType || !outType)
    return false;
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !outType.hasStaticShape())
    return false;
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      outType.getRank() != 2)
    return false;
  if (!isa<Float32Type>(outType.getElementType()) ||
      lhsType.getElementType() != outType.getElementType() ||
      rhsType.getElementType() != outType.getElementType())
    return false;

  m = outType.getShape()[0];
  n = outType.getShape()[1];
  k = lhsType.getShape()[1];
  if (lhsType.getShape()[0] != m || rhsType.getShape()[0] != n ||
      rhsType.getShape()[1] != k)
    return false;

  return true;
}

static bool matchOptionalBias(linalg::GenericOp bias, Value output,
                              int64_t n) {
  if (!bias || !bias.hasPureBufferSemantics())
    return false;
  if (bias.getInputs().size() != 2 || bias.getOutputs().size() != 1)
    return false;
  if (bias.getInputs()[0] != output || bias.getOutputs()[0] != output)
    return false;

  auto biasType = dyn_cast<MemRefType>(bias.getInputs()[1].getType());
  auto outType = dyn_cast<MemRefType>(output.getType());
  if (!biasType || !outType || !biasType.hasStaticShape())
    return false;
  if (biasType.getRank() != 1 || biasType.getShape()[0] != n)
    return false;
  if (biasType.getElementType() != outType.getElementType())
    return false;

  MLIRContext *ctx = bias.getContext();
  if (!hasIteratorTypes(bias, {utils::IteratorType::parallel,
                               utils::IteratorType::parallel}))
    return false;
  if (!hasIndexingMaps(bias, getBiasMaps(ctx)))
    return false;
  return matchBiasBody(bias);
}

static FailureOr<LinearWithBiasPattern>
matchLinearWithBias(linalg::GenericOp contraction) {
  if (!contraction.hasPureBufferSemantics())
    return failure();
  if (contraction.getInputs().size() != 2 ||
      contraction.getOutputs().size() != 1)
    return failure();

  MLIRContext *ctx = contraction.getContext();
  if (!hasIteratorTypes(contraction, {utils::IteratorType::parallel,
                                      utils::IteratorType::parallel,
                                      utils::IteratorType::reduction}))
    return failure();
  if (!hasIndexingMaps(contraction, getLinearContractionMaps(ctx)))
    return failure();
  if (!matchContractionBody(contraction))
    return failure();

  auto fill = dyn_cast_or_null<linalg::FillOp>(contraction->getPrevNode());
  if (!fill || fill.getInputs().size() != 1 || fill.getOutputs().size() != 1)
    return failure();

  Value lhs = contraction.getInputs()[0];
  Value rhs = contraction.getInputs()[1];
  Value output = contraction.getOutputs()[0];
  if (fill.getOutputs()[0] != output)
    return failure();
  if (output == lhs || output == rhs)
    return failure();

  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  if (!hasStaticF32LinearShapes(contraction, lhs, rhs, output, m, n, k))
    return failure();

  LinearWithBiasPattern pattern;
  pattern.fill = fill;
  pattern.contraction = contraction;
  pattern.lhs = lhs;
  pattern.rhs = rhs;
  pattern.output = output;
  pattern.fillValue = fill.getInputs()[0];
  pattern.m = m;
  pattern.n = n;
  pattern.k = k;

  auto bias = dyn_cast_or_null<linalg::GenericOp>(contraction->getNextNode());
  if (matchOptionalBias(bias, output, n)) {
    pattern.bias = bias;
    pattern.biasValue = bias.getInputs()[1];
  }

  return pattern;
}

static LogicalResult rewriteLinearWithBias(LinearWithBiasPattern pattern) {
  Location loc = pattern.contraction.getLoc();
  OpBuilder builder(pattern.fill);

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value cM = builder.create<arith::ConstantIndexOp>(loc, pattern.m);
  Value cN = builder.create<arith::ConstantIndexOp>(loc, pattern.n);
  Value cK = builder.create<arith::ConstantIndexOp>(loc, pattern.k);

  auto outerI = builder.create<scf::ForOp>(loc, c0, cM, c1);
  OpBuilder iBuilder = OpBuilder::atBlockBegin(outerI.getBody());
  Value i = outerI.getInductionVar();

  auto outerJ = iBuilder.create<scf::ForOp>(loc, c0, cN, c1);
  OpBuilder jBuilder = OpBuilder::atBlockBegin(outerJ.getBody());
  Value j = outerJ.getInductionVar();

  auto reduction = jBuilder.create<scf::ForOp>(
      loc, c0, cK, c1, ValueRange{pattern.fillValue},
      [&](OpBuilder &bodyBuilder, Location bodyLoc, Value k,
          ValueRange iterArgs) {
        Value acc = iterArgs.front();
        Value lhs = bodyBuilder.create<memref::LoadOp>(
            bodyLoc, pattern.lhs, ValueRange{i, k});
        Value rhs = bodyBuilder.create<memref::LoadOp>(
            bodyLoc, pattern.rhs, ValueRange{j, k});
        Value prod = bodyBuilder.create<arith::MulFOp>(bodyLoc, lhs, rhs);
        Value next = bodyBuilder.create<arith::AddFOp>(bodyLoc, prod, acc);
        bodyBuilder.create<scf::YieldOp>(bodyLoc, next);
      });

  Value result = reduction.getResult(0);
  if (pattern.bias) {
    Value bias = jBuilder.create<memref::LoadOp>(loc, pattern.biasValue,
                                                 ValueRange{j});
    result = jBuilder.create<arith::AddFOp>(loc, result, bias);
  }
  jBuilder.create<memref::StoreOp>(loc, result, pattern.output,
                                   ValueRange{i, j});

  if (pattern.bias)
    pattern.bias->erase();
  pattern.contraction->erase();
  pattern.fill->erase();
  return success();
}

struct FuseLinearWithBias
    : public impl::FuseLinearWithBiasBase<FuseLinearWithBias> {
  using impl::FuseLinearWithBiasBase<
      FuseLinearWithBias>::FuseLinearWithBiasBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (!func->hasAttr(VortexDialect::getKernelAttrName()))
      return;

    SmallVector<linalg::GenericOp> worklist;
    func.walk([&](linalg::GenericOp generic) { worklist.push_back(generic); });

    for (linalg::GenericOp generic : llvm::reverse(worklist)) {
      if (!generic || !generic->getParentRegion())
        continue;

      FailureOr<LinearWithBiasPattern> pattern =
          matchLinearWithBias(generic);
      if (failed(pattern))
        continue;
      if (failed(rewriteLinearWithBias(*pattern))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::vortex
