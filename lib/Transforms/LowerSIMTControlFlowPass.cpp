#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "vortex/Dialect/Vortex/IR/VortexDialect.h"
#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_LOWERSIMTCONTROLFLOW
#include "vortex/Transforms/Passes.h.inc"

namespace {

static void cloneVortexRegionIntoScf(Region &source, OpBuilder &builder) {
  IRMapping mapping;
  if (source.empty())
    return;

  for (Operation &op : source.front().without_terminator())
    builder.clone(op, mapping);
}

static void lowerPredicated(PredicatedOp predicated, IRRewriter &rewriter) {
  Location loc = predicated.getLoc();
  Value pred = predicated.getPred();

  rewriter.setInsertionPoint(predicated);
  auto split = rewriter.create<SplitOp>(loc, rewriter.getIndexType(), pred);
  auto ifOp = rewriter.create<scf::IfOp>(loc, pred, /*withElseRegion=*/false);
  OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
  cloneVortexRegionIntoScf(predicated.getBody(), thenBuilder);

  rewriter.setInsertionPointAfter(ifOp);
  rewriter.create<JoinOp>(loc, split.getStackPtr());
  rewriter.eraseOp(predicated);
}

static void lowerDivergentIf(DivergentIfOp divergentIf,
                             IRRewriter &rewriter) {
  Location loc = divergentIf.getLoc();
  Value pred = divergentIf.getPred();

  rewriter.setInsertionPoint(divergentIf);
  auto split = rewriter.create<SplitOp>(loc, rewriter.getIndexType(), pred);

  auto ifOp = rewriter.create<scf::IfOp>(loc, pred, /*withElseRegion=*/true);
  {
    OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
    cloneVortexRegionIntoScf(divergentIf.getThenRegion(), thenBuilder);
  }
  {
    OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
    cloneVortexRegionIntoScf(divergentIf.getElseRegion(), elseBuilder);
  }

  rewriter.setInsertionPointAfter(ifOp);
  rewriter.create<JoinOp>(loc, split.getStackPtr());
  rewriter.eraseOp(divergentIf);
}

static LogicalResult lowerStructuredOpsInRegion(Region &region,
                                                IRRewriter &rewriter) {
  for (Block &block : region) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      for (Region &nestedRegion : op.getRegions()) {
        if (failed(lowerStructuredOpsInRegion(nestedRegion, rewriter)))
          return failure();
      }

      if (auto predicated = dyn_cast<PredicatedOp>(op)) {
        lowerPredicated(predicated, rewriter);
        continue;
      }

      if (auto divergentIf = dyn_cast<DivergentIfOp>(op)) {
        lowerDivergentIf(divergentIf, rewriter);
        continue;
      }
    }
  }

  return success();
}

static Value buildThreadMaskFromCount(Location loc, Value threadCount,
                                      IRRewriter &rewriter) {
  Type i64Type = rewriter.getI64Type();
  Value threadCountI64 =
      rewriter.create<arith::IndexCastOp>(loc, i64Type, threadCount);
  Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
  Value maskPlusOne = rewriter.create<arith::ShLIOp>(loc, one, threadCountI64);
  Value maskI64 = rewriter.create<arith::SubIOp>(loc, maskPlusOne, one);
  return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                             maskI64);
}

static void materializeLaunchThreadMask(LaunchOp launch,
                                        IRRewriter &rewriter) {
  if (launch.getBody().empty())
    return;

  Block &body = launch.getBody().front();
  Location loc = launch.getLoc();

  rewriter.setInsertionPointToStart(&body);
  Value restoreMask = rewriter.create<TMaskOp>(loc, rewriter.getIndexType());
  Value activeMask = buildThreadMaskFromCount(loc, launch.getThreads(), rewriter);
  rewriter.create<TmcOp>(loc, activeMask);

  Operation *terminator = body.getTerminator();
  rewriter.setInsertionPoint(terminator);
  rewriter.create<TmcOp>(loc, restoreMask);
}

struct LowerSIMTControlFlow
    : public impl::LowerSIMTControlFlowBase<LowerSIMTControlFlow> {
  using impl::LowerSIMTControlFlowBase<
      LowerSIMTControlFlow>::LowerSIMTControlFlowBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                    VortexDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();

    SmallVector<LaunchOp> launches;
    func.walk([&](LaunchOp launch) { launches.push_back(launch); });

    IRRewriter rewriter(&getContext());
    for (LaunchOp launch : launches) {
      if (!launch || !launch->getParentRegion())
        continue;

      if (failed(lowerStructuredOpsInRegion(launch.getBody(), rewriter))) {
        signalPassFailure();
        return;
      }
      materializeLaunchThreadMask(launch, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::vortex
