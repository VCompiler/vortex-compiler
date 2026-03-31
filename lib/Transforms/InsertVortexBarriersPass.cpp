#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include "vortex/Dialect/Vortex/IR/VortexAttributes.h"
#include "vortex/Dialect/Vortex/IR/VortexDialect.h"
#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_INSERTVORTEXBARRIERS
#include "vortex/Transforms/Passes.h.inc"

namespace {

static Operation *findAncestorInBlock(Operation *op, Block *block) {
  // 第一版只在 launch 的顶层 block 上布点；若 use 藏在内部 region 里，
  // 就把插入点提升到它在该 block 内的最外层祖先操作。
  Operation *current = op;
  while (current && current->getBlock() != block)
    current = current->getParentOp();
  return current;
}

static bool hasAddressSpace(Value value, AddressSpace expected) {
  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return false;

  auto addressSpace =
      dyn_cast_or_null<AddressSpaceAttr>(memrefType.getMemorySpace());
  return addressSpace && addressSpace.getValue() == expected;
}

static bool isGlobalToLocalCopy(memref::CopyOp copy) {
  return hasAddressSpace(copy.getSource(), AddressSpace::Global) &&
         hasAddressSpace(copy.getTarget(), AddressSpace::Local);
}

static bool isLocalToGlobalCopy(memref::CopyOp copy) {
  return hasAddressSpace(copy.getSource(), AddressSpace::Local) &&
         hasAddressSpace(copy.getTarget(), AddressSpace::Global);
}

static bool isCoreBarrier(Operation *op) {
  auto barrier = dyn_cast_or_null<BarrierOp>(op);
  return barrier && barrier.getScope() == Scope::Core;
}

static bool hasCoreBarrierBetween(Operation *before, Operation *after) {
  if (!before || !after || before->getBlock() != after->getBlock())
    return false;

  for (Operation *cursor = before->getNextNode(); cursor && cursor != after;
       cursor = cursor->getNextNode()) {
    if (isCoreBarrier(cursor))
      return true;
  }
  return false;
}

static bool analyzeLocalTile(LocalAllocOp localAlloc,
                             SmallPtrSetImpl<Operation *> &beforeFirstUsePoints,
                             SmallPtrSetImpl<Operation *> &beforeCopyOutPoints) {
  LaunchOp launch = localAlloc->getParentOfType<LaunchOp>();
  if (!launch)
    return false;

  Block &launchBlock = launch.getBody().front();
  if (localAlloc->getBlock() != &launchBlock)
    return false;

  memref::CopyOp copyIn;
  memref::CopyOp copyOut;
  llvm::SmallPtrSet<Operation *, 8> useAnchors;

  for (OpOperand &use : localAlloc.getBuffer().getUses()) {
    Operation *owner = use.getOwner();
    Operation *anchor = findAncestorInBlock(owner, &launchBlock);
    if (!anchor)
      return false;

    if (auto copy = dyn_cast<memref::CopyOp>(owner)) {
      if (copy.getTarget() == localAlloc.getBuffer()) {
        if (copyIn || !isGlobalToLocalCopy(copy))
          return false;
        copyIn = copy;
        continue;
      }
      if (copy.getSource() == localAlloc.getBuffer()) {
        if (copyOut || !isLocalToGlobalCopy(copy))
          return false;
        copyOut = copy;
        continue;
      }
    }

    useAnchors.insert(anchor);
  }

  if (!copyIn || useAnchors.empty())
    return false;

  Operation *copyInAnchor = findAncestorInBlock(copyIn, &launchBlock);
  Operation *copyOutAnchor = copyOut ? findAncestorInBlock(copyOut, &launchBlock)
                                     : nullptr;
  if (!copyInAnchor || (copyOut && !copyOutAnchor))
    return false;

  Operation *firstUse = nullptr;
  Operation *lastUse = nullptr;
  for (Operation &op : launchBlock) {
    if (!useAnchors.contains(&op))
      continue;
    if (!firstUse)
      firstUse = &op;
    lastUse = &op;
  }

  if (!firstUse || !lastUse)
    return false;
  if (!copyInAnchor->isBeforeInBlock(firstUse))
    return false;
  if (copyOutAnchor && !lastUse->isBeforeInBlock(copyOutAnchor))
    return false;

  if (!hasCoreBarrierBetween(copyInAnchor, firstUse))
    beforeFirstUsePoints.insert(firstUse);
  if (copyOutAnchor && !hasCoreBarrierBetween(lastUse, copyOutAnchor))
    beforeCopyOutPoints.insert(copyOutAnchor);

  return true;
}

static void insertBarrierBefore(Operation *op) {
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  builder.create<BarrierOp>(op->getLoc(), Scope::Core);
}

static void insertBarriersIntoLaunch(LaunchOp launch) {
  Block &launchBlock = launch.getBody().front();
  llvm::SmallPtrSet<Operation *, 8> beforeFirstUsePoints;
  llvm::SmallPtrSet<Operation *, 8> beforeCopyOutPoints;

  SmallVector<LocalAllocOp> localAllocs;
  for (Operation &op : launchBlock) {
    if (auto localAlloc = dyn_cast<LocalAllocOp>(&op))
      localAllocs.push_back(localAlloc);
  }

  for (LocalAllocOp localAlloc : localAllocs)
    (void)analyzeLocalTile(localAlloc, beforeFirstUsePoints, beforeCopyOutPoints);

  SmallVector<Operation *> insertionOrder;
  insertionOrder.reserve(beforeFirstUsePoints.size() + beforeCopyOutPoints.size());
  for (Operation &op : launchBlock) {
    if (beforeFirstUsePoints.contains(&op) || beforeCopyOutPoints.contains(&op))
      insertionOrder.push_back(&op);
  }

  for (Operation *op : insertionOrder) {
    // copy-out 前的 barrier 允许覆盖一串连续的 write-back copy，因此只要前面的
    // use 到当前 copy-out 之间已经有 core barrier，就不再重复插入。
    if (beforeCopyOutPoints.contains(op)) {
      Operation *previous = op->getPrevNode();
      while (previous && isa<memref::CopyOp>(previous) &&
             isLocalToGlobalCopy(cast<memref::CopyOp>(previous))) {
        previous = previous->getPrevNode();
      }
      if (isCoreBarrier(previous))
        continue;
    } else if (isCoreBarrier(op->getPrevNode())) {
      continue;
    }

    insertBarrierBefore(op);
  }
}

struct InsertVortexBarriers
    : public impl::InsertVortexBarriersBase<InsertVortexBarriers> {
  using impl::InsertVortexBarriersBase<
      InsertVortexBarriers>::InsertVortexBarriersBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (!func->hasAttr(VortexDialect::getKernelAttrName()))
      return;

    SmallVector<LaunchOp> launches;
    func.walk([&](LaunchOp launch) { launches.push_back(launch); });

    for (LaunchOp launch : launches)
      insertBarriersIntoLaunch(launch);
  }
};

} // namespace

} // namespace mlir::vortex
