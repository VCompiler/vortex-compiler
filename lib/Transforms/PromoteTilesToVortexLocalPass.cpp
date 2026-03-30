#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "vortex/Dialect/Vortex/IR/VortexAttributes.h"
#include "vortex/Dialect/Vortex/IR/VortexDialect.h"
#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_PROMOTETILESTOVORTEXLOCAL
#include "vortex/Transforms/Passes.h.inc"

namespace {

static constexpr llvm::StringLiteral kPromoteToLocalAttrName =
    "vortex.promote_to_local";
static constexpr llvm::StringLiteral kWriteBackAttrName = "vortex.write_back";

static Operation *findAncestorInBlock(Operation *op, Block *block) {
  // 当前 promotion 假设所有被改写的 use 都留在同一个 block 内，因此简单的
  // 局部替换就够了。
  Operation *current = op;
  while (current && current->getBlock() != block)
    current = current->getParentOp();
  return current;
}

static bool isExplicitGlobalMemRef(Value value) {
  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return false;

  auto addressSpace =
      dyn_cast_or_null<AddressSpaceAttr>(memrefType.getMemorySpace());
  return addressSpace && addressSpace.getValue() == AddressSpace::Global;
}

static bool dependsOnSubgroupOrThread(Value value,
                                      llvm::SmallDenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return false;

  if (value.getDefiningOp<SubgroupIdOp>() || value.getDefiningOp<ThreadIdOp>())
    return true;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto loop = scf::getForInductionVarOwner(blockArg)) {
      return dependsOnSubgroupOrThread(loop.getLowerBound(), visited) ||
             dependsOnSubgroupOrThread(loop.getUpperBound(), visited) ||
             dependsOnSubgroupOrThread(loop.getStep(), visited);
    }

    if (blockArg.getOwner()->isEntryBlock())
      return false;

    // 对内部 region/block 参数先采用保守策略：一律视为不安全依赖。
    return true;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;

  for (Value operand : defOp->getOperands()) {
    if (dependsOnSubgroupOrThread(operand, visited))
      return true;
  }
  return false;
}

static bool dependsOnSubgroupOrThread(Value value) {
  llvm::SmallDenseSet<Value> visited;
  return dependsOnSubgroupOrThread(value, visited);
}

static bool isEscapingPromotionUse(OpOperand &use) {
  Operation *owner = use.getOwner();

  if (isa<func::ReturnOp>(owner) || isa<BranchOpInterface>(owner) ||
      isa<RegionBranchTerminatorOpInterface>(owner))
    return true;

  // memref 作为 scf.for/scf.while 的输入时，会通过 iter_args / region 边界逃逸。
  if (isa<scf::ForOp, scf::WhileOp>(owner))
    return true;

  return false;
}

static FailureOr<bool> useWritesPromotedTile(OpOperand &use, Value tile) {
  Operation *owner = use.getOwner();

  if (isMemoryEffectFree(owner))
    return false;

  auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(owner);
  if (!memoryEffectOp)
    return failure();

  SmallVector<MemoryEffects::EffectInstance, 4> effects;
  memoryEffectOp.getEffectsOnValue(tile, effects);
  return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
    return isa<MemoryEffects::Write>(effect.getEffect());
  });
}

static LogicalResult verifySubviewPromotionScope(memref::SubViewOp subview) {
  // 第一版只支持在已经建好的执行区域里，把显式标记的 tile 物化成 local buffer。
  if (!subview->getParentOfType<LaunchOp>())
    return subview.emitOpError()
           << "requires enclosing vortex.launch for local promotion";

  auto resultType = dyn_cast<MemRefType>(subview.getResult().getType());
  if (!resultType)
    return subview.emitOpError() << "must produce a ranked memref";

  if (!resultType.hasStaticShape())
    return subview.emitOpError()
           << "currently only static-shaped tiles can be promoted to local memory";

  if (!isExplicitGlobalMemRef(subview.getSource()))
    return subview.emitOpError()
           << "requires source/base memref to use explicit "
              "#vortex.address_space<global>";

  for (Value operand : subview->getOperands()) {
    if (dependsOnSubgroupOrThread(operand))
      return subview.emitOpError()
             << "requires promoted tiles to stay uniform across subgroup/thread";
  }

  bool hasWriteUse = false;
  for (OpOperand &use : subview->getUses()) {
    if (!findAncestorInBlock(use.getOwner(), subview->getBlock()))
      return subview.emitOpError()
             << "requires all uses to stay structurally within the defining block";

    if (isEscapingPromotionUse(use))
      return subview.emitOpError()
             << "cannot escape promotion scope through yield/branch/iter_args";

    FailureOr<bool> writesTile = useWritesPromotedTile(use, subview.getResult());
    if (failed(writesTile))
      return subview.emitOpError()
             << "cannot determine whether user '" << use.getOwner()->getName()
             << "' writes the promoted tile";
    hasWriteUse |= *writesTile;
  }

  if (hasWriteUse && !subview->hasAttr(kWriteBackAttrName))
    return subview.emitOpError()
           << "requires vortex.write_back when promoted tile has write uses";

  return success();
}

static MemRefType buildLocalTileType(memref::SubViewOp subview) {
  // promotion 时有意丢掉原 view 的 layout，转成 Vortex local address space
  // 下的紧凑 tile buffer。
  auto resultType = cast<MemRefType>(subview.getResult().getType());
  Attribute localAddressSpace =
      AddressSpaceAttr::get(subview.getContext(), AddressSpace::Local);
  return MemRefType::get(resultType.getShape(), resultType.getElementType(),
                         MemRefLayoutAttrInterface{}, localAddressSpace);
}

static void stripPromotionAttrs(Operation *op) {
  op->removeAttr(kPromoteToLocalAttrName);
  op->removeAttr(kWriteBackAttrName);
}

static LogicalResult promoteSubviewToLocal(memref::SubViewOp subview) {
  if (failed(verifySubviewPromotionScope(subview)))
    return failure();

  MemRefType localType = buildLocalTileType(subview);

  OpBuilder builder(subview);
  builder.setInsertionPointAfter(subview);
  auto localAlloc =
      builder.create<LocalAllocOp>(subview.getLoc(), localType, ValueRange{});
  // 先用显式 copy 物化 promoted tile，后续 pass 再决定要不要展开成 scalar、
  // cooperative，或者更接近 DMA 的形式。
  auto copyIn = builder.create<memref::CopyOp>(subview.getLoc(), subview,
                                               localAlloc.getBuffer());

  memref::CopyOp copyOut;
  if (subview->hasAttr(kWriteBackAttrName)) {
    builder.setInsertionPoint(subview->getBlock()->getTerminator());
    copyOut = builder.create<memref::CopyOp>(subview.getLoc(),
                                             localAlloc.getBuffer(), subview);
  }

  llvm::SmallPtrSet<Operation *, 2> exceptions;
  exceptions.insert(copyIn.getOperation());
  if (copyOut)
    exceptions.insert(copyOut.getOperation());
  // 普通 tile user 改到 local buffer 上，但保留生成出来的 copy op，让它们继续
  // 引用原始 subview。
  subview.getResult().replaceAllUsesExcept(localAlloc.getBuffer(), exceptions);

  stripPromotionAttrs(subview);
  return success();
}

struct PromoteTilesToVortexLocal
    : public impl::PromoteTilesToVortexLocalBase<PromoteTilesToVortexLocal> {
  using impl::PromoteTilesToVortexLocalBase<
      PromoteTilesToVortexLocal>::PromoteTilesToVortexLocalBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (!func->hasAttr(VortexDialect::getKernelAttrName()))
      return;

    SmallVector<memref::SubViewOp> worklist;
    func.walk([&](memref::SubViewOp subview) {
      if (subview->hasAttr(kPromoteToLocalAttrName))
        worklist.push_back(subview);
    });

    // 先收集稳定 worklist，再做原地改写，避免 walk 过程中被自己的修改打断。
    for (memref::SubViewOp subview : worklist) {
      if (!subview || !subview->getParentRegion())
        continue;
      if (failed(promoteSubviewToLocal(subview))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::vortex
