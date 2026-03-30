#include "vortex/InitAllPasses.h"

#include "vortex/Pipeline/Pipelines.h"
#include "vortex/Transforms/Passes.h"

namespace mlir::vortex {

void registerVortexPassesAndPipelines() {
  registerVortexPasses();
  registerPreVortexPipeline();
}

} // namespace mlir::vortex
