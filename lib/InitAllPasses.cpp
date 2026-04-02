#include "vortex/InitAllPasses.h"

#include "vortex/Pipeline/Pipelines.h"
#include "vortex/Transforms/Passes.h"

namespace mlir::vortex {

void registerVortexPassesAndPipelines() {
  registerVortexPasses();
  registerONNXMatmulToPreVortexPipeline();
  registerPreVortexPipeline();
  registerMVPBackendPipeline();
}

} // namespace mlir::vortex
