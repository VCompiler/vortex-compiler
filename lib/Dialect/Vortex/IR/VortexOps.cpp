#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::vortex;

#define GET_OP_CLASSES
#include "vortex/Dialect/Vortex/IR/VortexOps.cpp.inc"

static LogicalResult verifyRegionWithoutBlockArguments(Operation *op,
                                                       Region &region,
                                                       StringRef regionName) {
  if (region.empty())
    return success();

  Block &bodyBlock = region.front();
  if (!bodyBlock.getArguments().empty())
    return op->emitOpError()
           << "expects " << regionName << " without block arguments";

  return success();
}

static LogicalResult verifyNestedInLaunch(Operation *op) {
  if (!op->getParentOfType<LaunchOp>())
    return op->emitOpError() << "must be nested inside vortex.launch";
  return success();
}

static LogicalResult verifyRegionTerminatesWithVortexYield(Operation *op,
                                                          Region &region,
                                                          StringRef regionName) {
  if (region.empty())
    return op->emitOpError()
           << "expects " << regionName << " terminated by vortex.yield";

  Operation *terminator = region.front().getTerminator();
  if (!isa_and_nonnull<YieldOp>(terminator))
    return op->emitOpError()
           << "expects " << regionName << " terminated by vortex.yield";

  return success();
}

LogicalResult LaunchOp::verifyRegions() {
  if (getBody().empty())
    return success();

  Block &bodyBlock = getBody().front();
  if (!bodyBlock.getArguments().empty())
    return emitOpError()
           << "expects launch body without block arguments; use vortex.core_id, "
              "vortex.subgroup_id, and vortex.thread_id inside the body";

  return success();
}

LogicalResult PredicatedOp::verifyRegions() {
  if (failed(verifyNestedInLaunch(getOperation())))
    return failure();
  if (failed(verifyRegionWithoutBlockArguments(getOperation(), getBody(),
                                               "predicated body")))
    return failure();
  return verifyRegionTerminatesWithVortexYield(getOperation(), getBody(),
                                               "predicated body");
}

LogicalResult DivergentIfOp::verifyRegions() {
  if (failed(verifyNestedInLaunch(getOperation())))
    return failure();
  if (failed(verifyRegionWithoutBlockArguments(getOperation(), getThenRegion(),
                                               "divergent_if then region")))
    return failure();
  if (failed(verifyRegionWithoutBlockArguments(getOperation(), getElseRegion(),
                                               "divergent_if else region")))
    return failure();
  if (failed(verifyRegionTerminatesWithVortexYield(
          getOperation(), getThenRegion(), "divergent_if then region")))
    return failure();
  return verifyRegionTerminatesWithVortexYield(
      getOperation(), getElseRegion(), "divergent_if else region");
}

LogicalResult LocalAllocOp::verify() {
  auto memrefType = dyn_cast<MemRefType>(getBuffer().getType());
  if (!memrefType)
    return emitOpError("must return a memref type");

  if (getDynamicSizes().size() !=
      static_cast<size_t>(memrefType.getNumDynamicDims())) {
    return emitOpError() << "expected " << memrefType.getNumDynamicDims()
                         << " dynamic size operands, but got "
                         << getDynamicSizes().size();
  }

  Attribute memorySpace = memrefType.getMemorySpace();
  auto addressSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace);
  if (!addressSpace)
    return emitOpError()
           << "result memref must use #vortex.address_space<local> as its "
              "memory space";

  if (addressSpace.getValue() != AddressSpace::Local)
    return emitOpError()
           << "result memref must use #vortex.address_space<local>, but got "
           << memorySpace;

  return success();
}
