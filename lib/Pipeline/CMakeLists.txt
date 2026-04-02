add_mlir_library(MLIRVortexPipeline
  Pipelines.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vortex

  LINK_LIBS PUBLIC
    MLIRArithToLLVM
    MLIRBufferizationTransforms
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRFuncDialect
    MLIRIndexToLLVM
    MLIRMemRefToLLVM
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRTransforms
    MLIRVortexTransforms
)
