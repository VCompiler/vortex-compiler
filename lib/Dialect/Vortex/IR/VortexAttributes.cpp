#include "vortex/Dialect/Vortex/IR/VortexAttributes.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "vortex/Dialect/Vortex/IR/VortexDialect.h"

using namespace mlir;
using namespace mlir::vortex;

#include "vortex/Dialect/Vortex/IR/VortexEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "vortex/Dialect/Vortex/IR/VortexAttributes.cpp.inc"

void VortexDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "vortex/Dialect/Vortex/IR/VortexAttributes.cpp.inc"
      >();
}
