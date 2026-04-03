#include "vortex/Pipeline/Pipelines.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Config/llvm-config.h"

#include "vortex/Transforms/Passes.h"

namespace mlir::vortex {

namespace {

struct ONNXMatmulToPreVortexPipelineOptions
    : public PassPipelineOptions<ONNXMatmulToPreVortexPipelineOptions> {
  Option<int64_t> tileSize{
      *this, "tile-size",
      llvm::cl::desc("Uniform static tile size for the frontend matmul tiling"),
      llvm::cl::init(8)};
};

} // namespace

static void buildONNXMatmulToPreVortexPipeline(
    OpPassManager &pm, const ONNXMatmulToPreVortexPipelineOptions &options) {
#if LLVM_VERSION_MAJOR >= 19
  bufferization::BufferResultsToOutParamsPassOptions outParamOptions;
  outParamOptions.modifyPublicFunctions = true;
#else
  bufferization::BufferResultsToOutParamsOptions outParamOptions;
#endif
  TileMatmulForPreVortexOptions tileOptions;
  tileOptions.tileSize = options.tileSize;

  pm.addPass(bufferization::createBufferResultsToOutParamsPass(
      outParamOptions));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createNormalizeONNXFrontend());
  pm.addNestedPass<func::FuncOp>(createTileMatmulForPreVortex(tileOptions));
  buildPreVortexPipeline(pm);
}

void buildONNXMatmulToPreVortexPipeline(OpPassManager &pm) {
  buildONNXMatmulToPreVortexPipeline(pm,
                                     ONNXMatmulToPreVortexPipelineOptions{});
}

void registerONNXMatmulToPreVortexPipeline() {
  PassPipelineRegistration<ONNXMatmulToPreVortexPipelineOptions> pipeline(
      "vortex-onnx-matmul-to-pre-vortex-pipeline",
      "Bridge the narrow ONNX-MLIR matmul frontend path into tiled "
      "pre-Vortex IR.",
      [](OpPassManager &pm,
         const ONNXMatmulToPreVortexPipelineOptions &options) {
        buildONNXMatmulToPreVortexPipeline(pm, options);
      });
  (void)pipeline;
}

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
#if LLVM_VERSION_MAJOR >= 19
  pm.addPass(createSCFToControlFlowPass());
#else
  pm.addPass(createConvertSCFToCFPass());
#endif
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
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
