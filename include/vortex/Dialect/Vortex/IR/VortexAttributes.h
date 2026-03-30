#ifndef VORTEX_DIALECT_VORTEX_IR_VORTEXATTRIBUTES_H
#define VORTEX_DIALECT_VORTEX_IR_VORTEXATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

#include "vortex/Dialect/Vortex/IR/VortexDialect.h"
#include "vortex/Dialect/Vortex/IR/VortexEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "vortex/Dialect/Vortex/IR/VortexAttributes.h.inc"

#endif // VORTEX_DIALECT_VORTEX_IR_VORTEXATTRIBUTES_H
