#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::vortex;

#define GET_OP_CLASSES
#include "vortex/Dialect/Vortex/IR/VortexOps.cpp.inc"

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
