#include "vortex/Pipeline/Pipelines.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "vortex/Transforms/Passes.h"

namespace mlir::vortex {

void buildPreVortexPipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createValidatePreVortex());
  pm.addNestedPass<func::FuncOp>(createSummarizePreVortex());
}

void registerPreVortexPipeline() {
  PassPipelineRegistration<> pipeline(
      "vortex-pre-vortex-pipeline",
      "Normalize and summarize the high-level IR that precedes the "
      "target-specific Vortex dialect.",
      buildPreVortexPipeline);
  (void)pipeline;
}

} // namespace mlir::vortex
