#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "vortex/Dialect/Vortex/IR/VortexDialect.h"
#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_MATERIALIZESIMTCONTROLFLOW
#include "vortex/Transforms/Passes.h.inc"

namespace {

static constexpr unsigned kMaxPredicatedOps = 4;

enum class Uniformity {
  Uniform,
  MayVarying,
};

class UniformityAnalysis {
public:
  Uniformity classify(Value value) {
    auto it = cache.find(value);
    if (it != cache.end())
      return it->second;

    Uniformity result = classifyUncached(value);
    cache[value] = result;
    return result;
  }

private:
  DenseMap<Value, Uniformity> cache;

  static bool isArithLike(Operation *op) {
    return op->getName().getStringRef().starts_with("arith.");
  }

  Uniformity classifyUncached(Value value) {
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return classifyBlockArgument(blockArg);

    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return Uniformity::MayVarying;

    if (isa<arith::ConstantOp, CoreIdOp, SubgroupIdOp>(defOp))
      return Uniformity::Uniform;

    if (isa<ThreadIdOp, memref::LoadOp, CallOpInterface>(defOp))
      return Uniformity::MayVarying;

    if (isArithLike(defOp))
      return combineOperandUniformity(defOp);

    return Uniformity::MayVarying;
  }

  Uniformity classifyBlockArgument(BlockArgument blockArg) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (isa_and_nonnull<func::FuncOp>(parentOp))
      return Uniformity::Uniform;

    return Uniformity::MayVarying;
  }

  Uniformity combineOperandUniformity(Operation *op) {
    for (Value operand : op->getOperands()) {
      if (classify(operand) == Uniformity::MayVarying)
        return Uniformity::MayVarying;
    }
    return Uniformity::Uniform;
  }
};

static bool hasNonTerminatorOps(Region &region) {
  if (region.empty())
    return false;

  auto ops = region.front().without_terminator();
  return ops.begin() != ops.end();
}

static unsigned countNonTerminatorOps(Region &region) {
  if (region.empty())
    return 0;

  unsigned count = 0;
  for (Operation &op : region.front().without_terminator()) {
    (void)op;
    ++count;
  }
  return count;
}

static bool regionContainsBarrier(Region &region) {
  for (Block &block : region) {
    for (Operation &op : block) {
      WalkResult result = op.walk([&](Operation *nested) -> WalkResult {
        if (isa<BarrierOp>(nested))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return true;
    }
  }
  return false;
}

static bool regionContainsCall(Region &region) {
  for (Block &block : region) {
    for (Operation &op : block) {
      WalkResult result = op.walk([&](Operation *nested) -> WalkResult {
        if (isa<CallOpInterface>(nested))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return true;
    }
  }
  return false;
}

static bool regionContainsReturn(Region &region) {
  for (Block &block : region) {
    for (Operation &op : block) {
      WalkResult result = op.walk([&](Operation *nested) -> WalkResult {
        if (isa<func::ReturnOp>(nested))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return true;
    }
  }
  return false;
}

static LogicalResult validateStraightLineRegion(Region &region,
                                                scf::IfOp ifOp) {
  if (region.empty())
    return success();

  for (Operation &op : region.front().without_terminator()) {
    if (op.getNumRegions() != 0 ||
        isa<BranchOpInterface, RegionBranchOpInterface,
            RegionBranchTerminatorOpInterface>(&op)) {
      return ifOp.emitOpError()
             << "cannot materialize may-varying scf.if with nested complex "
                "control flow";
    }
  }

  return success();
}

static LogicalResult validateMayVaryingIfBody(scf::IfOp ifOp) {
  if (ifOp.getNumResults() != 0)
    return ifOp.emitOpError()
           << "cannot materialize may-varying scf.if with results";

  Region &thenRegion = ifOp.getThenRegion();
  Region &elseRegion = ifOp.getElseRegion();
  if (regionContainsBarrier(thenRegion) || regionContainsBarrier(elseRegion))
    return ifOp.emitOpError()
           << "cannot materialize may-varying scf.if containing "
              "vortex.barrier";

  if (regionContainsCall(thenRegion) || regionContainsCall(elseRegion))
    return ifOp.emitOpError()
           << "cannot materialize may-varying scf.if containing call";

  if (regionContainsReturn(thenRegion) || regionContainsReturn(elseRegion))
    return ifOp.emitOpError()
           << "cannot materialize may-varying scf.if containing return";

  if (failed(validateStraightLineRegion(thenRegion, ifOp)))
    return failure();
  return validateStraightLineRegion(elseRegion, ifOp);
}

static void cloneRegionBodyInto(Region &source, Region &target,
                                OpBuilder &builder, Location yieldLoc) {
  Block *block = new Block();
  target.push_back(block);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);

  IRMapping mapping;
  if (!source.empty()) {
    for (Operation &op : source.front().without_terminator())
      builder.clone(op, mapping);
  }

  if (block->empty() || !isa<YieldOp>(block->back()))
    builder.create<YieldOp>(yieldLoc);
}

static bool shouldUsePredicated(scf::IfOp ifOp) {
  return !hasNonTerminatorOps(ifOp.getElseRegion()) &&
         countNonTerminatorOps(ifOp.getThenRegion()) <= kMaxPredicatedOps;
}

static void rewriteToPredicated(scf::IfOp ifOp) {
  OpBuilder builder(ifOp);
  auto predicated = builder.create<PredicatedOp>(ifOp.getLoc(),
                                                 ifOp.getCondition());
  cloneRegionBodyInto(ifOp.getThenRegion(), predicated.getBody(), builder,
                      ifOp.getLoc());
  ifOp.erase();
}

static void rewriteToDivergentIf(scf::IfOp ifOp) {
  OpBuilder builder(ifOp);
  auto divergentIf = builder.create<DivergentIfOp>(ifOp.getLoc(),
                                                   ifOp.getCondition());
  cloneRegionBodyInto(ifOp.getThenRegion(), divergentIf.getThenRegion(),
                      builder, ifOp.getLoc());
  cloneRegionBodyInto(ifOp.getElseRegion(), divergentIf.getElseRegion(),
                      builder, ifOp.getLoc());
  ifOp.erase();
}

static LogicalResult materializeIf(scf::IfOp ifOp,
                                  UniformityAnalysis &analysis,
                                  bool &rewritten) {
  rewritten = false;
  if (analysis.classify(ifOp.getCondition()) == Uniformity::Uniform)
    return success();

  if (failed(validateMayVaryingIfBody(ifOp)))
    return failure();

  if (shouldUsePredicated(ifOp))
    rewriteToPredicated(ifOp);
  else
    rewriteToDivergentIf(ifOp);
  rewritten = true;
  return success();
}

static LogicalResult processRegion(Region &region, UniformityAnalysis &analysis) {
  for (Block &block : region) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        bool rewritten = false;
        if (failed(materializeIf(ifOp, analysis, rewritten)))
          return failure();
        if (rewritten)
          continue;

        if (failed(processRegion(ifOp.getThenRegion(), analysis)))
          return failure();
        if (failed(processRegion(ifOp.getElseRegion(), analysis)))
          return failure();
        continue;
      }

      for (Region &nestedRegion : op.getRegions()) {
        if (failed(processRegion(nestedRegion, analysis)))
          return failure();
      }
    }
  }
  return success();
}

struct MaterializeSIMTControlFlow
    : public impl::MaterializeSIMTControlFlowBase<
          MaterializeSIMTControlFlow> {
  using impl::MaterializeSIMTControlFlowBase<
      MaterializeSIMTControlFlow>::MaterializeSIMTControlFlowBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, VortexDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();

    SmallVector<LaunchOp> launches;
    func.walk([&](LaunchOp launch) { launches.push_back(launch); });

    for (LaunchOp launch : launches) {
      UniformityAnalysis analysis;
      if (failed(processRegion(launch.getBody(), analysis))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::vortex
