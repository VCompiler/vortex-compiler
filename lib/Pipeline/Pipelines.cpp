#include "vortex/Pipeline/Pipelines.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
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

void buildMVPBackendPipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLegalizeVortexForLLVM());
  pm.addPass(createLowerVortexRuntimeBuiltins());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void registerMVPBackendPipeline() {
  PassPipelineRegistration<> pipeline(
      "vortex-mvp-backend-pipeline",
      "Lower post-local-memory Vortex kernel IR through the MVP backend path "
      "into LLVM dialect.",
      buildMVPBackendPipeline);
  (void)pipeline;
}

} // namespace mlir::vortex
