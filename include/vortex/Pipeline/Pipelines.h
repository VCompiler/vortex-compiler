#ifndef VORTEX_PIPELINE_PIPELINES_H
#define VORTEX_PIPELINE_PIPELINES_H

#include "mlir/Pass/PassManager.h"

namespace mlir::vortex {

void buildONNXMatmulToPreVortexPipeline(OpPassManager &pm);
void registerONNXMatmulToPreVortexPipeline();
void buildPreVortexPipeline(OpPassManager &pm);
void registerPreVortexPipeline();
void buildMVPBackendPipeline(OpPassManager &pm);
void registerMVPBackendPipeline();

} // namespace mlir::vortex

#endif // VORTEX_PIPELINE_PIPELINES_H
