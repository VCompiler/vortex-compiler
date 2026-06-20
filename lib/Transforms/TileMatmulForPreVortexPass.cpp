#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_TILEMATMULFORPREVORTEX
#include "vortex/Transforms/Passes.h.inc"

namespace {

static constexpr llvm::StringLiteral kLoopMappingAttrName = "vortex.mapping";
static constexpr llvm::StringLiteral kMatmulScheduleAttrName =
    "vortex.matmul_schedule";
static constexpr llvm::StringLiteral kPromoteToLocalAttrName =
    "vortex.promote_to_local";

struct MatmulSchedule {
  int64_t blockM;
  int64_t blockN;
  int64_t blockK;
  int64_t numSubgroups;
  int64_t numThreads;
};

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

struct LaneState {
  Value linearTid;
  Value laneCount;
};

static FailureOr<StaticMatmulShape>
getStaticMatmulShape(linalg::MatmulOp matmul, Value outputBuffer) {
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

  return StaticMatmulShape{m, n, k};
}

static LogicalResult verifySchedule(linalg::MatmulOp matmul,
                                    StaticMatmulShape shape,
                                    MatmulSchedule schedule) {
  if (schedule.blockM <= 0 || schedule.blockN <= 0 || schedule.blockK <= 0)
    return matmul.emitOpError() << "requires positive matmul block sizes";
  if (schedule.numSubgroups <= 0 || schedule.numThreads <= 0)
    return matmul.emitOpError()
           << "requires positive subgroup and thread counts";
  if (shape.m % schedule.blockM != 0 || shape.n % schedule.blockN != 0 ||
      shape.k % schedule.blockK != 0) {
    return matmul.emitOpError()
           << "requires static dimensions to be divisible by matmul block "
              "sizes";
  }

  auto lhsType = cast<MemRefType>(matmul.getInputs()[0].getType());
  if (!isa<FloatType>(lhsType.getElementType())) {
    return matmul.emitOpError()
           << "currently requires floating-point element type for scheduled "
              "per-lane matmul";
  }

  return success();
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

static Value createIndexConstant(OpBuilder &builder, Location loc,
                                 int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

static memref::SubViewOp createStaticTileSubview(
    OpBuilder &builder, Location loc, Value base, Value offset0, Value offset1,
    int64_t size0, int64_t size1, bool promoteToLocal = false) {
  SmallVector<OpFoldResult> offsets{offset0, offset1};
  SmallVector<OpFoldResult> sizes{builder.getIndexAttr(size0),
                                  builder.getIndexAttr(size1)};
  SmallVector<OpFoldResult> strides{builder.getIndexAttr(1),
                                    builder.getIndexAttr(1)};
  auto subview =
      builder.create<memref::SubViewOp>(loc, base, offsets, sizes, strides);
  if (promoteToLocal)
    subview->setAttr(kPromoteToLocalAttrName, builder.getUnitAttr());
  return subview;
}

static DictionaryAttr buildScheduleAttr(OpBuilder &builder,
                                        MatmulSchedule schedule) {
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr(
      "block_m", builder.getI64IntegerAttr(schedule.blockM)));
  attrs.push_back(builder.getNamedAttr(
      "block_n", builder.getI64IntegerAttr(schedule.blockN)));
  attrs.push_back(builder.getNamedAttr(
      "block_k", builder.getI64IntegerAttr(schedule.blockK)));
  attrs.push_back(builder.getNamedAttr(
      "num_subgroups", builder.getI64IntegerAttr(schedule.numSubgroups)));
  attrs.push_back(builder.getNamedAttr(
      "num_threads", builder.getI64IntegerAttr(schedule.numThreads)));
  attrs.push_back(
      builder.getNamedAttr("promote_lhs", builder.getBoolAttr(true)));
  attrs.push_back(
      builder.getNamedAttr("promote_rhs", builder.getBoolAttr(true)));
  attrs.push_back(builder.getNamedAttr("copy_policy",
                                       builder.getStringAttr("linear_stride")));
  attrs.push_back(builder.getNamedAttr(
      "compute_policy", builder.getStringAttr("linear_tid_2d")));
  return builder.getDictionaryAttr(attrs);
}

static LaneState createLaneState(OpBuilder &builder, Location loc,
                                 Value subgroupId, Value threadId,
                                 Value numSubgroups, Value numThreads) {
  Value subgroupBase =
      builder.create<arith::MulIOp>(loc, subgroupId, numThreads);
  Value linearTid = builder.create<arith::AddIOp>(loc, subgroupBase, threadId);
  Value laneCount = builder.create<arith::MulIOp>(loc, numSubgroups, numThreads);
  return LaneState{linearTid, laneCount};
}

static SmallVector<Value> delinearize2DTileIndex(OpBuilder &builder,
                                                 Location loc, Value linear,
                                                 int64_t cols) {
  Value cCols = createIndexConstant(builder, loc, cols);
  Value row = builder.create<arith::DivUIOp>(loc, linear, cCols);
  Value col = builder.create<arith::RemUIOp>(loc, linear, cCols);
  return SmallVector<Value>{row, col};
}

static void createPerLaneFill(OpBuilder &builder, Location loc,
                              Value fillValue, Value cTile,
                              Value subgroupId, Value threadId,
                              Value numSubgroups, Value numThreads,
                              MatmulSchedule schedule) {
  LaneState lane = createLaneState(builder, loc, subgroupId, threadId,
                                   numSubgroups, numThreads);
  Value tileElements =
      createIndexConstant(builder, loc, schedule.blockM * schedule.blockN);

  auto fillLoop = builder.create<scf::ForOp>(loc, lane.linearTid, tileElements,
                                             lane.laneCount);
  OpBuilder fillBuilder = OpBuilder::atBlockBegin(fillLoop.getBody());
  SmallVector<Value> indices = delinearize2DTileIndex(
      fillBuilder, loc, fillLoop.getInductionVar(), schedule.blockN);
  fillBuilder.create<memref::StoreOp>(loc, fillValue, cTile, indices);
}

static void createPerLaneMatmulCompute(OpBuilder &builder, Location loc,
                                       Value lhs, Value rhs, Value cTile,
                                       Value ii, Value jj, Value kk,
                                       Value subgroupId, Value threadId,
                                       Value numSubgroups, Value numThreads,
                                       MatmulSchedule schedule) {
  auto aTile = createStaticTileSubview(builder, loc, lhs, ii, kk,
                                       schedule.blockM, schedule.blockK,
                                       /*promoteToLocal=*/true);
  auto bTile = createStaticTileSubview(builder, loc, rhs, kk, jj,
                                       schedule.blockK, schedule.blockN,
                                       /*promoteToLocal=*/true);

  LaneState lane = createLaneState(builder, loc, subgroupId, threadId,
                                   numSubgroups, numThreads);
  Value tileElements =
      createIndexConstant(builder, loc, schedule.blockM * schedule.blockN);

  auto elementLoop = builder.create<scf::ForOp>(loc, lane.linearTid,
                                                tileElements, lane.laneCount);
  OpBuilder elementBuilder = OpBuilder::atBlockBegin(elementLoop.getBody());
  SmallVector<Value> cIndices = delinearize2DTileIndex(
      elementBuilder, loc, elementLoop.getInductionVar(), schedule.blockN);
  Value current = elementBuilder.create<memref::LoadOp>(loc, cTile, cIndices);

  Value c0 = createIndexConstant(elementBuilder, loc, 0);
  Value c1 = createIndexConstant(elementBuilder, loc, 1);
  Value cBlockK = createIndexConstant(elementBuilder, loc, schedule.blockK);
  auto kLoop = elementBuilder.create<scf::ForOp>(loc, c0, cBlockK, c1,
                                                 ValueRange{current});
  OpBuilder kBuilder = OpBuilder::atBlockBegin(kLoop.getBody());
  Value kIv = kLoop.getInductionVar();
  Value acc = kLoop.getRegionIterArgs().front();
  Value a = kBuilder.create<memref::LoadOp>(
      loc, aTile.getResult(), ValueRange{cIndices[0], kIv});
  Value b = kBuilder.create<memref::LoadOp>(
      loc, bTile.getResult(), ValueRange{kIv, cIndices[1]});
  Value product = kBuilder.create<arith::MulFOp>(loc, a, b);
  Value next = kBuilder.create<arith::AddFOp>(loc, acc, product);
  kBuilder.create<scf::YieldOp>(loc, next);

  elementBuilder.setInsertionPointAfter(kLoop);
  elementBuilder.create<memref::StoreOp>(loc, kLoop.getResult(0), cTile,
                                         cIndices);
}

template <typename BodyBuilder>
static void createMappedLaneNest(OpBuilder &builder, Location loc, Value c0,
                                 Value c1, Value numSubgroups,
                                 Value numThreads,
                                 BodyBuilder &&bodyBuilder) {
  auto subgroupLoop = builder.create<scf::ForOp>(loc, c0, numSubgroups, c1);
  subgroupLoop->setAttr(kLoopMappingAttrName,
                        builder.getStringAttr("subgroup"));

  OpBuilder subgroupBuilder = OpBuilder::atBlockBegin(subgroupLoop.getBody());
  auto threadLoop = subgroupBuilder.create<scf::ForOp>(loc, c0, numThreads, c1);
  threadLoop->setAttr(kLoopMappingAttrName, builder.getStringAttr("thread"));

  OpBuilder threadBuilder = OpBuilder::atBlockBegin(threadLoop.getBody());
  bodyBuilder(threadBuilder, subgroupLoop.getInductionVar(),
              threadLoop.getInductionVar());
}

static LogicalResult tileFrontendMatmulPattern(FrontendMatmulPattern pattern,
                                               MatmulSchedule schedule) {
  FailureOr<StaticMatmulShape> shape =
      getStaticMatmulShape(pattern.matmul, pattern.outputBuffer);
  if (failed(shape))
    return failure();
  if (failed(verifySchedule(pattern.matmul, *shape, schedule)))
    return failure();

  Location loc = pattern.matmul.getLoc();
  OpBuilder builder(pattern.fill);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value cBlockM = builder.create<arith::ConstantIndexOp>(loc, schedule.blockM);
  Value cBlockN = builder.create<arith::ConstantIndexOp>(loc, schedule.blockN);
  Value cBlockK = builder.create<arith::ConstantIndexOp>(loc, schedule.blockK);
  Value cM = builder.create<arith::ConstantIndexOp>(loc, shape->m);
  Value cN = builder.create<arith::ConstantIndexOp>(loc, shape->n);
  Value cK = builder.create<arith::ConstantIndexOp>(loc, shape->k);
  Value cNumSubgroups =
      builder.create<arith::ConstantIndexOp>(loc, schedule.numSubgroups);
  Value cNumThreads =
      builder.create<arith::ConstantIndexOp>(loc, schedule.numThreads);

  auto outerI = builder.create<scf::ForOp>(loc, c0, cM, cBlockM);
  outerI->setAttr(kMatmulScheduleAttrName,
                  buildScheduleAttr(builder, schedule));
  OpBuilder iBuilder = OpBuilder::atBlockBegin(outerI.getBody());
  Value ii = outerI.getInductionVar();

  auto outerJ = iBuilder.create<scf::ForOp>(loc, c0, cN, cBlockN);
  OpBuilder jBuilder = OpBuilder::atBlockBegin(outerJ.getBody());
  Value jj = outerJ.getInductionVar();

  auto cTileView = createStaticTileSubview(jBuilder, loc, pattern.outputBuffer,
                                           ii, jj, schedule.blockM,
                                           schedule.blockN);

  createMappedLaneNest(
      jBuilder, loc, c0, c1, cNumSubgroups, cNumThreads,
      [&](OpBuilder &laneBuilder, Value subgroupId, Value threadId) {
        createPerLaneFill(laneBuilder, loc, pattern.fill.getInputs().front(),
                          cTileView.getResult(), subgroupId, threadId,
                          cNumSubgroups, cNumThreads, schedule);
      });

  auto innerK = jBuilder.create<scf::ForOp>(loc, c0, cK, cBlockK);
  OpBuilder kBuilder = OpBuilder::atBlockBegin(innerK.getBody());
  Value kk = innerK.getInductionVar();

  createMappedLaneNest(
      kBuilder, loc, c0, c1, cNumSubgroups, cNumThreads,
      [&](OpBuilder &laneBuilder, Value subgroupId, Value threadId) {
        createPerLaneMatmulCompute(
            laneBuilder, loc, pattern.lhs, pattern.rhs, cTileView.getResult(),
            ii, jj, kk, subgroupId, threadId, cNumSubgroups, cNumThreads,
            schedule);
      });

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
    int64_t resolvedBlockM = blockM > 0 ? blockM : tileSize;
    int64_t resolvedBlockN = blockN > 0 ? blockN : tileSize;
    int64_t resolvedBlockK = blockK > 0 ? blockK : tileSize;
    MatmulSchedule schedule{resolvedBlockM, resolvedBlockN, resolvedBlockK,
                            numSubgroups, numThreads};

    if (schedule.blockM <= 0 || schedule.blockN <= 0 ||
        schedule.blockK <= 0) {
      func.emitOpError() << "requires positive tile size or block sizes";
      signalPassFailure();
      return;
    }
    if (schedule.numSubgroups <= 0 || schedule.numThreads <= 0) {
      func.emitOpError() << "requires positive subgroup and thread counts";
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
          failed(tileFrontendMatmulPattern(*pattern, schedule))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::vortex
