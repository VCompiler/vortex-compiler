#ifndef VORTEX_INITALLDIALECTS_H
#define VORTEX_INITALLDIALECTS_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir::vortex {

void registerVortexDialects(DialectRegistry &registry);

} // namespace mlir::vortex

#endif // VORTEX_INITALLDIALECTS_H
