#ifndef VORTEX_DIALECT_VORTEX_IR_VORTEXOPS_H
#define VORTEX_DIALECT_VORTEX_IR_VORTEXOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "vortex/Dialect/Vortex/IR/VortexAttributes.h"
#include "vortex/Dialect/Vortex/IR/VortexDialect.h"

#define GET_OP_CLASSES
#include "vortex/Dialect/Vortex/IR/VortexOps.h.inc"

#endif // VORTEX_DIALECT_VORTEX_IR_VORTEXOPS_H
