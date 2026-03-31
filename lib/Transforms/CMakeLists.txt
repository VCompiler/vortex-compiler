add_mlir_library(MLIRVortexTransforms
  InsertVortexBarriersPass.cpp
  LowerLinalgInsideVortexKernelPass.cpp
  MapParallelLoopsToVortexLaunchPass.cpp
  MarkVortexKernelPass.cpp
  MaterializeVortexAddressSpacesPass.cpp
  PromoteTilesToVortexLocalPass.cpp
  SummarizePreVortexPass.cpp
  ValidatePreVortexPass.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vortex

  DEPENDS
    MLIRVortexPassesIncGen

  LINK_LIBS PUBLIC
    MLIRAffineDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRVortexDialect
    MLIRVectorDialect
)
