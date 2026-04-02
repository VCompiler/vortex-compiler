#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_TILEMATMULFORPREVORTEX
#include "vortex/Transforms/Passes.h.inc"

namespace {

struct FrontendMatmulPattern {
  linalg::FillOp fill;
  linalg::MatmulOp matmul;
  Value lhs;
  Value rhs;
  Value accumBuffer;
  Value outputBuffer;
  memref::AllocOp tempAlloc;
  memref::CopyOp copyOut;
  SmallVector<memref::DeallocOp> deallocs;
};

struct StaticMatmulShape {
  int64_t m;
  int64_t n;
  int64_t k;
};

static FailureOr<StaticMatmulShape> getStaticMatmulShape(linalg::MatmulOp matmul,
                                                         Value outputBuffer,
                                                         int64_t tileSize) {
  auto lhsType = dyn_cast<MemRefType>(matmul.getInputs()[0].getType());
  auto rhsType = dyn_cast<MemRefType>(matmul.getInputs()[1].getType());
  auto outType = dyn_cast<MemRefType>(outputBuffer.getType());

  if (!lhsType || !rhsType || !outType)
    return matmul.emitOpError() << "requires memref operands/results";
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !outType.hasStaticShape()) {
    return matmul.emitOpError()
           << "currently only static-shape matmul is supported";
  }
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || outType.getRank() != 2)
    return matmul.emitOpError() << "currently only rank-2 matmul is supported";
  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != outType.getElementType()) {
    return matmul.emitOpError()
           << "requires matching element types across A/B/C";
  }

  int64_t m = lhsType.getShape()[0];
  int64_t k = lhsType.getShape()[1];
  int64_t rhsK = rhsType.getShape()[0];
  int64_t n = rhsType.getShape()[1];
  if (rhsK != k || outType.getShape()[0] != m || outType.getShape()[1] != n) {
    return matmul.emitOpError()
           << "requires compatible static matmul shapes";
  }
  if (m % tileSize != 0 || n % tileSize != 0 || k % tileSize != 0) {
    return matmul.emitOpError()
           << "requires all static dimensions to be divisible by tile size "
           << tileSize;
  }

  return StaticMatmulShape{m, n, k};
}

static FailureOr<FrontendMatmulPattern>
matchFrontendMatmulPattern(linalg::MatmulOp matmul) {
  if (!matmul.hasPureBufferSemantics()) {
    return matmul.emitOpError()
           << "requires buffer semantics before tiling to pre-vortex";
  }
  if (matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1) {
    return matmul.emitOpError()
           << "requires exactly two inputs and one output";
  }

  auto fill = dyn_cast_or_null<linalg::FillOp>(matmul->getPrevNode());
  if (!fill || fill.getOutputs().size() != 1 ||
      fill.getOutputs().front() != matmul.getOutputs().front()) {
    return matmul.emitOpError()
           << "requires an immediately preceding linalg.fill on the same "
              "accumulator buffer";
  }
  if (fill.getInputs().size() != 1) {
    return fill.emitOpError()
           << "requires exactly one scalar fill value for MVP tiling";
  }

  FrontendMatmulPattern pattern;
  pattern.fill = fill;
  pattern.matmul = matmul;
  pattern.lhs = matmul.getInputs()[0];
  pattern.rhs = matmul.getInputs()[1];
  pattern.accumBuffer = matmul.getOutputs()[0];
  pattern.outputBuffer = pattern.accumBuffer;

  auto tempAlloc = pattern.accumBuffer.getDefiningOp<memref::AllocOp>();
  if (!tempAlloc)
    return pattern;

  memref::CopyOp copyOut;
  SmallVector<memref::DeallocOp> deallocs;
  for (Operation *user : tempAlloc->getUsers()) {
    if (user == fill.getOperation() || user == matmul.getOperation())
      continue;

    if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      if (copy.getSource() != tempAlloc.getResult()) {
        return matmul.emitOpError()
               << "requires temp alloc to appear only as memref.copy source";
      }
      if (copyOut) {
        return matmul.emitOpError()
               << "requires at most one memref.copy from the temp alloc";
      }
      copyOut = copy;
      continue;
    }

    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      deallocs.push_back(dealloc);
      continue;
    }

    return matmul.emitOpError()
           << "temp accumulator alloc has unsupported user '"
           << user->getName() << "'";
  }

  if (!copyOut)
    return pattern;
  if (copyOut->getBlock() != matmul->getBlock() ||
      !matmul->isBeforeInBlock(copyOut)) {
    return matmul.emitOpError()
           << "requires temp-alloc memref.copy to stay after matmul in the "
              "same block";
  }
  if (copyOut.getTarget() == pattern.lhs || copyOut.getTarget() == pattern.rhs) {
    return matmul.emitOpError()
           << "refuses to rewrite when output buffer aliases an input SSA value";
  }

  pattern.tempAlloc = tempAlloc;
  pattern.copyOut = copyOut;
  pattern.outputBuffer = copyOut.getTarget();
  pattern.deallocs = std::move(deallocs);
  return pattern;
}

static memref::SubViewOp createStaticTileSubview(OpBuilder &builder,
                                                 Location loc, Value base,
                                                 Value offset0, Value offset1,
                                                 int64_t tileSize) {
  SmallVector<OpFoldResult> offsets{offset0, offset1};
  SmallVector<OpFoldResult> sizes{builder.getIndexAttr(tileSize),
                                  builder.getIndexAttr(tileSize)};
  SmallVector<OpFoldResult> strides{builder.getIndexAttr(1),
                                    builder.getIndexAttr(1)};
  return builder.create<memref::SubViewOp>(loc, base, offsets, sizes, strides);
}

static LogicalResult tileFrontendMatmulPattern(FrontendMatmulPattern pattern,
                                               int64_t tileSize) {
  FailureOr<StaticMatmulShape> shape =
      getStaticMatmulShape(pattern.matmul, pattern.outputBuffer, tileSize);
  if (failed(shape))
    return failure();

  Location loc = pattern.matmul.getLoc();
  OpBuilder builder(pattern.fill);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value cTile = builder.create<arith::ConstantIndexOp>(loc, tileSize);
  Value cM = builder.create<arith::ConstantIndexOp>(loc, shape->m);
  Value cN = builder.create<arith::ConstantIndexOp>(loc, shape->n);
  Value cK = builder.create<arith::ConstantIndexOp>(loc, shape->k);

  auto outerI = builder.create<scf::ForOp>(loc, c0, cM, cTile);
  OpBuilder iBuilder = OpBuilder::atBlockBegin(outerI.getBody());
  Value ii = outerI.getInductionVar();

  auto outerJ = iBuilder.create<scf::ForOp>(loc, c0, cN, cTile);
  OpBuilder jBuilder = OpBuilder::atBlockBegin(outerJ.getBody());
  Value jj = outerJ.getInductionVar();

  auto cTileView = createStaticTileSubview(jBuilder, loc, pattern.outputBuffer,
                                           ii, jj, tileSize);
  // 这里保留 tile 级 linalg.fill + linalg.matmul，而不是立刻再往标量循环打散，
  // 这样后半段现有的 pre-vortex/Vortex pass 还能继续利用结构化算子语义。
  jBuilder.create<linalg::FillOp>(loc, pattern.fill.getInputs().front(),
                                  cTileView.getResult());

  auto innerK = jBuilder.create<scf::ForOp>(loc, c0, cK, cTile);
  OpBuilder kBuilder = OpBuilder::atBlockBegin(innerK.getBody());
  Value kk = innerK.getInductionVar();

  auto aTileView =
      createStaticTileSubview(kBuilder, loc, pattern.lhs, ii, kk, tileSize);
  auto bTileView =
      createStaticTileSubview(kBuilder, loc, pattern.rhs, kk, jj, tileSize);
  kBuilder.create<linalg::MatmulOp>(
      loc, ValueRange{aTileView.getResult(), bTileView.getResult()},
      ValueRange{cTileView.getResult()});

  if (pattern.copyOut)
    pattern.copyOut->erase();
  pattern.matmul->erase();
  pattern.fill->erase();
  for (memref::DeallocOp dealloc : pattern.deallocs)
    dealloc.erase();
  if (pattern.tempAlloc)
    pattern.tempAlloc->erase();

  return success();
}

struct TileMatmulForPreVortex
    : public impl::TileMatmulForPreVortexBase<TileMatmulForPreVortex> {
  using impl::TileMatmulForPreVortexBase<
      TileMatmulForPreVortex>::TileMatmulForPreVortexBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (tileSize <= 0) {
      func.emitOpError() << "requires positive tile size";
      signalPassFailure();
      return;
    }

    SmallVector<linalg::MatmulOp> worklist;
    func.walk([&](linalg::MatmulOp matmul) { worklist.push_back(matmul); });

    for (linalg::MatmulOp matmul : llvm::reverse(worklist)) {
      if (!matmul || !matmul->getParentRegion())
        continue;

      FailureOr<FrontendMatmulPattern> pattern =
          matchFrontendMatmulPattern(matmul);
      if (failed(pattern) ||
          failed(tileFrontendMatmulPattern(*pattern, tileSize))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::vortex
