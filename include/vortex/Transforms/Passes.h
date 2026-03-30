#ifndef VORTEX_TRANSFORMS_PASSES_H
#define VORTEX_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::vortex {

#define GEN_PASS_DECL
#include "vortex/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "vortex/Transforms/Passes.h.inc"

} // namespace mlir::vortex

#endif // VORTEX_TRANSFORMS_PASSES_H
