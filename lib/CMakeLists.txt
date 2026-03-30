add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(Pipeline)

add_mlir_library(MLIRVortexInit
  InitAllDialects.cpp
  InitAllPasses.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vortex

  LINK_LIBS PUBLIC
    MLIRAffineDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRVectorDialect
    MLIRVortexDialect
    MLIRVortexPipeline
    MLIRVortexTransforms
)
