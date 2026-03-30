#include "vortex/Dialect/Vortex/IR/VortexDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "vortex/Dialect/Vortex/IR/VortexAttributes.h"
#include "vortex/Dialect/Vortex/IR/VortexOps.h"

using namespace mlir;
using namespace mlir::vortex;

#include "vortex/Dialect/Vortex/IR/VortexOpsDialect.cpp.inc"

void VortexDialect::initialize() {
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "vortex/Dialect/Vortex/IR/VortexOps.cpp.inc"
      >();
}

LogicalResult VortexDialect::verifyOperationAttribute(Operation *op,
                                                      NamedAttribute namedAttr) {
  if (namedAttr.getName() != getKernelAttrName())
    return success();

  if (!isa<func::FuncOp>(op))
    return op->emitOpError()
           << "'" << getKernelAttrName() << "' may only annotate func.func";

  if (!isa<UnitAttr>(namedAttr.getValue()))
    return op->emitOpError()
           << "'" << getKernelAttrName()
           << "' must be a unit attribute in the current dialect version";

  return success();
}
