#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "vortex/Dialect/Vortex/IR/VortexAttributes.h"
#include "vortex/Dialect/Vortex/IR/VortexDialect.h"
#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>

namespace mlir::vortex {

#define GEN_PASS_DEF_PLANVORTEXLOCALMEMORYLAYOUT
#include "vortex/Transforms/Passes.h.inc"

namespace {

static constexpr llvm::StringLiteral kLocalFrameBytesAttrName =
    "vortex.local_frame_bytes";
static constexpr llvm::StringLiteral kLocalByteOffsetAttrName =
    "vortex.local.byte_offset";
static constexpr llvm::StringLiteral kLocalByteSizeAttrName =
    "vortex.local.byte_size";
static constexpr llvm::StringLiteral kLocalAlignmentAttrName =
    "vortex.local.alignment";

struct LocalAllocLayout {
  LocalAllocOp alloc;
  uint64_t byteOffset = 0;
  uint64_t byteSize = 0;
  uint64_t alignment = 0;
};

static bool isExplicitLocalMemRef(MemRefType type) {
  auto addressSpace =
      dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
  return addressSpace && addressSpace.getValue() == AddressSpace::Local;
}

static bool hasCompactIdentityLayout(MemRefType type) {
  MemRefLayoutAttrInterface layout = type.getLayout();
  return !layout || layout.isIdentity();
}

static bool isForwardingAliasOp(Operation *op) {
  return isa<memref::CastOp, memref::SubViewOp, memref::ReinterpretCastOp,
             memref::ExpandShapeOp, memref::CollapseShapeOp,
             memref::TransposeOp, memref::ViewOp>(op);
}

static bool hasMemRefResult(Operation *op) {
  return llvm::any_of(op->getResultTypes(),
                      [](Type type) { return isa<BaseMemRefType>(type); });
}

static LogicalResult verifyLocalValueUses(Value value,
                                          llvm::SmallPtrSetImpl<Value> &visited,
                                          LocalAllocOp rootAlloc) {
  if (!visited.insert(value).second)
    return success();

  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();

    if (isa<CallOpInterface>(owner))
      return rootAlloc.emitOpError()
             << "cannot escape local memory planning scope through call";

    if (isa<func::ReturnOp>(owner) || isa<BranchOpInterface>(owner) ||
        isa<RegionBranchOpInterface>(owner) ||
        isa<RegionBranchTerminatorOpInterface>(owner))
      return rootAlloc.emitOpError()
             << "cannot escape local memory planning scope through "
                "yield/branch/iter_args";

    // 允许简单 memref 别名链继续留在当前函数里，但要把后续 use 一起检查，
    // 避免 local_alloc 通过 cast/subview 等间接逃逸。
    if (isForwardingAliasOp(owner)) {
      for (Value result : owner->getResults()) {
        if (!isa<BaseMemRefType>(result.getType()))
          continue;
        if (failed(verifyLocalValueUses(result, visited, rootAlloc)))
          return failure();
      }
      continue;
    }

    if (hasMemRefResult(owner))
      return rootAlloc.emitOpError()
             << "currently only supports view-like local memref aliases, but "
                "found memref result-producing user '"
             << owner->getName() << "'";
  }

  return success();
}

static LogicalResult verifyLocalAllocScope(LocalAllocOp alloc) {
  llvm::SmallPtrSet<Value, 8> visited;
  return verifyLocalValueUses(alloc.getBuffer(), visited, alloc);
}

static FailureOr<LocalAllocLayout>
planLocalAllocLayout(LocalAllocOp alloc, const DataLayout &dataLayout,
                     uint64_t currentFrameBytes) {
  auto bufferType = dyn_cast<MemRefType>(alloc.getBuffer().getType());
  if (!bufferType)
    return alloc.emitOpError() << "must produce a ranked memref";

  if (!isExplicitLocalMemRef(bufferType))
    return alloc.emitOpError()
           << "requires result memref to use explicit "
              "#vortex.address_space<local>";

  if (!bufferType.hasStaticShape() || !alloc.getDynamicSizes().empty())
    return alloc.emitOpError()
           << "currently requires static-shaped vortex.local_alloc";

  if (!hasCompactIdentityLayout(bufferType))
    return alloc.emitOpError()
           << "currently requires compact identity layout";

  if (failed(verifyLocalAllocScope(alloc)))
    return failure();

  llvm::TypeSize elementSize =
      dataLayout.getTypeSize(bufferType.getElementType());
  if (elementSize.isScalable())
    return alloc.emitOpError()
           << "currently requires fixed-size element types";

  uint64_t elementBytes = elementSize.getFixedValue();
  uint64_t alignment =
      std::max<uint64_t>(elementBytes,
                         dataLayout.getTypeABIAlignment(
                             bufferType.getElementType()));
  if (alignment == 0)
    return alloc.emitOpError()
           << "failed to compute local allocation alignment";

  uint64_t numElements = static_cast<uint64_t>(bufferType.getNumElements());
  if (numElements != 0 &&
      elementBytes > std::numeric_limits<uint64_t>::max() / numElements)
    return alloc.emitOpError() << "local allocation byte size overflows i64";

  uint64_t byteSize = numElements * elementBytes;
  uint64_t byteOffset = llvm::alignTo(currentFrameBytes, alignment);
  if (byteOffset < currentFrameBytes ||
      byteSize > std::numeric_limits<uint64_t>::max() - byteOffset)
    return alloc.emitOpError() << "local frame size overflows i64";

  return LocalAllocLayout{alloc, byteOffset, byteSize, alignment};
}

struct PlanVortexLocalMemoryLayout
    : public impl::PlanVortexLocalMemoryLayoutBase<
          PlanVortexLocalMemoryLayout> {
  using impl::PlanVortexLocalMemoryLayoutBase<
      PlanVortexLocalMemoryLayout>::PlanVortexLocalMemoryLayoutBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (!func->hasAttr(VortexDialect::getKernelAttrName()))
      return;

    SmallVector<LocalAllocOp> allocs;
    func.walk([&](LocalAllocOp alloc) { allocs.push_back(alloc); });

    Builder builder(func.getContext());
    DataLayout dataLayout = DataLayout::closest(func);
    SmallVector<LocalAllocLayout> plannedLayouts;
    plannedLayouts.reserve(allocs.size());

    uint64_t frameBytes = 0;
    for (LocalAllocOp alloc : allocs) {
      FailureOr<LocalAllocLayout> layout =
          planLocalAllocLayout(alloc, dataLayout, frameBytes);
      if (failed(layout)) {
        signalPassFailure();
        return;
      }
      plannedLayouts.push_back(*layout);
      frameBytes = layout->byteOffset + layout->byteSize;
    }

    func->setAttr(kLocalFrameBytesAttrName,
                  builder.getI64IntegerAttr(frameBytes));
    for (const LocalAllocLayout &layout : plannedLayouts) {
      layout.alloc->setAttr(kLocalByteOffsetAttrName,
                            builder.getI64IntegerAttr(layout.byteOffset));
      layout.alloc->setAttr(kLocalByteSizeAttrName,
                            builder.getI64IntegerAttr(layout.byteSize));
      layout.alloc->setAttr(kLocalAlignmentAttrName,
                            builder.getI64IntegerAttr(layout.alignment));
    }
  }
};

} // namespace

} // namespace mlir::vortex
