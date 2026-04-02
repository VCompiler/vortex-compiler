#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "vortex/Dialect/Vortex/IR/VortexDialect.h"
#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_LOWERVORTEXRUNTIMEBUILTINS
#include "vortex/Transforms/Passes.h.inc"

namespace {

static constexpr StringLiteral kVxBarrier = "vx_barrier";
static constexpr StringLiteral kVxNumWarps = "vx_num_warps";
static constexpr StringLiteral kVxThreadId = "vx_thread_id";
static constexpr StringLiteral kVxWarpId = "vx_warp_id";
static constexpr StringLiteral kVxCoreId = "vx_core_id";
static constexpr StringLiteral kLoweredKernelMarker = "vortex.kernel_entry";

static FailureOr<func::FuncOp>
getOrCreateWrapperDecl(ModuleOp module, StringRef name, FunctionType type,
                       OpBuilder &builder, Location loc) {
  if (auto func = module.lookupSymbol<func::FuncOp>(name)) {
    if (func.getFunctionType() != type) {
      return failure();
    }
    return func;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto func = builder.create<func::FuncOp>(loc, name, type);
  func.setPrivate();
  return func;
}

static LogicalResult ensureKernelContext(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func || !func->hasAttr(VortexDialect::getKernelAttrName())) {
    return op->emitOpError()
           << "requires enclosing func.func marked with vortex.kernel";
  }
  return success();
}

static LogicalResult validateUnsupportedOps(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!isa<LaunchOp, YieldOp, LocalAllocOp, FenceOp, BarrierOp, CoreIdOp,
             SubgroupIdOp, ThreadIdOp>(op)) {
      return WalkResult::advance();
    }

    if (failed(ensureKernelContext(op)))
      return WalkResult::interrupt();

    if (isa<LaunchOp, YieldOp>(op)) {
      op->emitOpError()
          << "requires running vortex-legalize-for-llvm before "
             "vortex-lower-runtime-builtins";
      return WalkResult::interrupt();
    }

    if (isa<LocalAllocOp>(op)) {
      op->emitOpError()
          << "vortex.local_alloc lowering is not implemented yet in "
             "vortex-lower-runtime-builtins";
      return WalkResult::interrupt();
    }

    if (isa<FenceOp>(op)) {
      op->emitOpError()
          << "vortex.fence lowering is not implemented yet in "
             "vortex-lower-runtime-builtins";
      return WalkResult::interrupt();
    }

    auto barrier = dyn_cast<BarrierOp>(op);
    if (barrier && barrier.getScope() != Scope::Core) {
      barrier.emitOpError()
          << "only vortex.barrier <core> is supported in the current MVP";
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}

static LogicalResult
replaceIdQuery(ModuleOp module, Operation *op, StringRef wrapperName,
               IRRewriter &rewriter, func::FuncOp &cachedWrapper) {
  auto callType = rewriter.getFunctionType({}, TypeRange{rewriter.getI32Type()});
  if (!cachedWrapper) {
    FailureOr<func::FuncOp> wrapper = getOrCreateWrapperDecl(
        module, wrapperName, callType, rewriter, op->getLoc());
    if (failed(wrapper)) {
      return op->emitOpError()
             << "wrapper declaration type mismatch for " << wrapperName;
    }
    cachedWrapper = *wrapper;
  }

  rewriter.setInsertionPoint(op);
  Value raw = rewriter.create<func::CallOp>(op->getLoc(), cachedWrapper,
                                            ValueRange{})
                  .getResult(0);
  Value indexValue = rewriter.create<arith::IndexCastOp>(
      op->getLoc(), rewriter.getIndexType(), raw);
  rewriter.replaceOp(op, indexValue);
  return success();
}

static LogicalResult replaceCoreBarrier(ModuleOp module, BarrierOp barrier,
                                        IRRewriter &rewriter,
                                        func::FuncOp &numWarpsWrapper,
                                        func::FuncOp &barrierWrapper) {
  auto i32Type = rewriter.getI32Type();
  auto countType = rewriter.getFunctionType({}, TypeRange{i32Type});
  auto barrierType = rewriter.getFunctionType({i32Type, i32Type}, {});

  if (!numWarpsWrapper) {
    FailureOr<func::FuncOp> wrapper = getOrCreateWrapperDecl(
        module, kVxNumWarps, countType, rewriter, barrier.getLoc());
    if (failed(wrapper)) {
      return barrier.emitOpError()
             << "wrapper declaration type mismatch for " << kVxNumWarps;
    }
    numWarpsWrapper = *wrapper;
  }

  if (!barrierWrapper) {
    FailureOr<func::FuncOp> wrapper = getOrCreateWrapperDecl(
        module, kVxBarrier, barrierType, rewriter, barrier.getLoc());
    if (failed(wrapper)) {
      return barrier.emitOpError()
             << "wrapper declaration type mismatch for " << kVxBarrier;
    }
    barrierWrapper = *wrapper;
  }

  rewriter.setInsertionPoint(barrier);
  Value barrierId =
      rewriter.create<arith::ConstantIntOp>(barrier.getLoc(), 0, 32);
  Value numWarps =
      rewriter.create<func::CallOp>(barrier.getLoc(), numWarpsWrapper,
                                    ValueRange{})
          .getResult(0);
  rewriter.create<func::CallOp>(barrier.getLoc(), barrierWrapper,
                                ValueRange{barrierId, numWarps});
  rewriter.eraseOp(barrier);
  return success();
}

struct LowerVortexRuntimeBuiltins
    : public impl::LowerVortexRuntimeBuiltinsBase<
          LowerVortexRuntimeBuiltins> {
  using impl::LowerVortexRuntimeBuiltinsBase<
      LowerVortexRuntimeBuiltins>::LowerVortexRuntimeBuiltinsBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, VortexDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (failed(validateUnsupportedOps(module))) {
      signalPassFailure();
      return;
    }

    SmallVector<CoreIdOp> coreIds;
    SmallVector<SubgroupIdOp> subgroupIds;
    SmallVector<ThreadIdOp> threadIds;
    SmallVector<BarrierOp> barriers;

    module.walk([&](CoreIdOp op) { coreIds.push_back(op); });
    module.walk([&](SubgroupIdOp op) { subgroupIds.push_back(op); });
    module.walk([&](ThreadIdOp op) { threadIds.push_back(op); });
    module.walk([&](BarrierOp op) { barriers.push_back(op); });

    IRRewriter rewriter(&getContext());
    func::FuncOp coreIdWrapper;
    func::FuncOp warpIdWrapper;
    func::FuncOp threadIdWrapper;
    func::FuncOp numWarpsWrapper;
    func::FuncOp barrierWrapper;

    for (CoreIdOp op : coreIds) {
      if (failed(replaceIdQuery(module, op, kVxCoreId, rewriter,
                                coreIdWrapper))) {
        signalPassFailure();
        return;
      }
    }

    for (SubgroupIdOp op : subgroupIds) {
      if (failed(replaceIdQuery(module, op, kVxWarpId, rewriter,
                                warpIdWrapper))) {
        signalPassFailure();
        return;
      }
    }

    for (ThreadIdOp op : threadIds) {
      if (failed(replaceIdQuery(module, op, kVxThreadId, rewriter,
                                threadIdWrapper))) {
        signalPassFailure();
        return;
      }
    }

    for (BarrierOp op : barriers) {
      if (failed(replaceCoreBarrier(module, op, rewriter, numWarpsWrapper,
                                    barrierWrapper))) {
        signalPassFailure();
        return;
      }
    }

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!func->hasAttr(VortexDialect::getKernelAttrName()))
        continue;
      func->removeAttr(VortexDialect::getKernelAttrName());
      func->setAttr(kLoweredKernelMarker, rewriter.getUnitAttr());
    }
  }
};

} // namespace

} // namespace mlir::vortex
