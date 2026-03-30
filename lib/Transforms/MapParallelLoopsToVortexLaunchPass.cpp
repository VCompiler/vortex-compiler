#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "vortex/Dialect/Vortex/IR/VortexDialect.h"
#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_MAPPARALLELLOOPSTOVORTEXLAUNCH
#include "vortex/Transforms/Passes.h.inc"

namespace {

static constexpr llvm::StringLiteral kLoopMappingAttrName = "vortex.mapping";

enum class LoopMappingKind {
  Core = 0,
  Subgroup = 1,
  Thread = 2,
};

struct MappedLoop {
  scf::ForOp loop;
  LoopMappingKind kind;
};

static FailureOr<LoopMappingKind> parseLoopMappingKind(scf::ForOp loop) {
  // 这个 pass 完全由 attribute 驱动，不做执行维度的自动推断。
  auto attr = loop->getAttrOfType<StringAttr>(kLoopMappingAttrName);
  if (!attr)
    return failure();

  StringRef value = attr.getValue();
  if (value == "core")
    return LoopMappingKind::Core;
  if (value == "subgroup")
    return LoopMappingKind::Subgroup;
  if (value == "thread")
    return LoopMappingKind::Thread;

  loop.emitOpError()
      << "expects " << kLoopMappingAttrName
      << " to be one of \"core\", \"subgroup\", or \"thread\"";
  return failure();
}

static bool isConstantIndex(Value value, int64_t expected) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value() == expected;

  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto integerAttr = dyn_cast<IntegerAttr>(constant.getValue());
    return integerAttr && integerAttr.getType().isIndex() &&
           integerAttr.getInt() == expected;
  }

  return false;
}

static bool isUniformLaunchBound(Value value, scf::ForOp root) {
  // launch 维度必须定义在 mapped nest 之外，这样每个逻辑执行实例看到的网格
  // 形状才一致。
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    return !parentOp || (parentOp != root && !root->isAncestor(parentOp));
  }

  Operation *defOp = value.getDefiningOp();
  return !defOp || (defOp != root && !root->isAncestor(defOp));
}

static LogicalResult verifyMappedLoopShape(scf::ForOp loop, scf::ForOp root) {
  if (!loop.getInitArgs().empty() || loop.getNumResults() != 0)
    return loop.emitOpError()
           << "mapped loops must not have iter_args or results";

  if (!isConstantIndex(loop.getLowerBound(), 0))
    return loop.emitOpError()
           << "mapped loops must have lower bound 0";

  if (!isConstantIndex(loop.getStep(), 1))
    return loop.emitOpError() << "mapped loops must have step 1";

  if (!isUniformLaunchBound(loop.getUpperBound(), root))
    return loop.emitOpError()
           << "mapped loop upper bound must be defined outside the mapped loop nest";

  if (loop.getBody()->getNumArguments() != 1)
    return loop.emitOpError()
           << "mapped loops must only carry the induction variable block argument";

  return success();
}

static LogicalResult
collectMappedLoopNest(scf::ForOp root, SmallVectorImpl<MappedLoop> &nest) {
  // 沿着一个已经显式标注好的 perfect nest 往里走，要求执行维度顺序是
  // outer-to-inner。
  llvm::SmallDenseSet<int> seenKinds;
  int lastKindOrdinal = -1;
  scf::ForOp current = root;

  while (true) {
    FailureOr<LoopMappingKind> parsedKind = parseLoopMappingKind(current);
    if (failed(parsedKind))
      return current->hasAttr(kLoopMappingAttrName) ? failure() : success();

    int ordinal = static_cast<int>(*parsedKind);
    if (seenKinds.contains(ordinal))
      return current.emitOpError()
             << "mapped loop nest must not repeat execution dimensions";
    if (ordinal < lastKindOrdinal)
      return current.emitOpError()
             << "mapped loop nest must follow core -> subgroup -> thread order";
    if (failed(verifyMappedLoopShape(current, root)))
      return failure();

    seenKinds.insert(ordinal);
    lastKindOrdinal = ordinal;
    nest.push_back({current, *parsedKind});

    SmallVector<Operation *> nonTerminatorOps;
    for (Operation &op : current.getBody()->without_terminator())
      nonTerminatorOps.push_back(&op);

    if (nonTerminatorOps.size() == 1) {
      if (auto innerLoop = dyn_cast<scf::ForOp>(nonTerminatorOps.front())) {
        if (innerLoop->hasAttr(kLoopMappingAttrName)) {
          current = innerLoop;
          continue;
        }
      }
    }

    for (Operation *op : nonTerminatorOps) {
      if (auto innerLoop = dyn_cast<scf::ForOp>(op)) {
        if (innerLoop->hasAttr(kLoopMappingAttrName))
          return innerLoop.emitOpError()
                 << "mapped loop nests must form a perfect nest";
      }
    }

    return success();
  }
}

static void mapInductionVariable(OpBuilder &builder, Location loc,
                                 LoopMappingKind kind, Value iv,
                                 IRMapping &mapping) {
  // 把结构化循环的 IV 替换成显式的 Vortex 执行 id 查询。
  switch (kind) {
  case LoopMappingKind::Core:
    mapping.map(iv, builder.create<CoreIdOp>(loc, builder.getIndexType()));
    return;
  case LoopMappingKind::Subgroup:
    mapping.map(iv, builder.create<SubgroupIdOp>(loc, builder.getIndexType()));
    return;
  case LoopMappingKind::Thread:
    mapping.map(iv, builder.create<ThreadIdOp>(loc, builder.getIndexType()));
    return;
  }

  llvm_unreachable("unknown loop mapping kind");
}

static LogicalResult rewriteMappedLoopNest(scf::ForOp root) {
  SmallVector<MappedLoop> nest;
  if (failed(collectMappedLoopNest(root, nest)))
    return failure();
  if (nest.empty())
    return success();

  OpBuilder builder(root);
  Location loc = root.getLoc();

  // 没有出现在标注 loop nest 里的执行维度，默认取 1。
  Value oneIndex = root.getStep();
  Value coreCount = oneIndex;
  Value subgroupCount = oneIndex;
  Value threadCount = oneIndex;

  for (MappedLoop &mappedLoop : nest) {
    switch (mappedLoop.kind) {
    case LoopMappingKind::Core:
      coreCount = mappedLoop.loop.getUpperBound();
      break;
    case LoopMappingKind::Subgroup:
      subgroupCount = mappedLoop.loop.getUpperBound();
      break;
    case LoopMappingKind::Thread:
      threadCount = mappedLoop.loop.getUpperBound();
      break;
    }
  }

  auto launch =
      builder.create<LaunchOp>(loc, coreCount, subgroupCount, threadCount);
  Block *launchBlock = new Block();
  launch.getBody().push_back(launchBlock);
  builder.setInsertionPointToStart(launchBlock);

  IRMapping mapping;
  for (MappedLoop &mappedLoop : nest)
    mapInductionVariable(builder, mappedLoop.loop.getLoc(), mappedLoop.kind,
                         mappedLoop.loop.getInductionVar(), mapping);

  // 第一版只在替换完 mapped IV 之后，把最内层循环体克隆进 launch region。
  scf::ForOp innermostLoop = nest.back().loop;
  for (Operation &op : innermostLoop.getBody()->without_terminator())
    builder.clone(op, mapping);

  if (launchBlock->empty() || !isa<YieldOp>(launchBlock->back()))
    builder.create<YieldOp>(loc);

  root.erase();
  return success();
}

struct MapParallelLoopsToVortexLaunch
    : public impl::MapParallelLoopsToVortexLaunchBase<
          MapParallelLoopsToVortexLaunch> {
  using impl::MapParallelLoopsToVortexLaunchBase<
      MapParallelLoopsToVortexLaunch>::MapParallelLoopsToVortexLaunchBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (!func->hasAttr(VortexDialect::getKernelAttrName()))
      return;

    SmallVector<scf::ForOp> rootLoops;
    func.walk([&](scf::ForOp loop) {
      if (!loop->hasAttr(kLoopMappingAttrName))
        return;

      auto parentLoop = loop->getParentOfType<scf::ForOp>();
      if (parentLoop && parentLoop->hasAttr(kLoopMappingAttrName))
        return;

      // 每个 nest 只从最外层 mapped loop 开始重写一次。
      rootLoops.push_back(loop);
    });

    for (scf::ForOp rootLoop : rootLoops) {
      if (!rootLoop || !rootLoop->getParentRegion())
        continue;
      if (failed(rewriteMappedLoopNest(rootLoop))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::vortex
