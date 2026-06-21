add_mlir_library(MLIRVortexTransforms
  ConvertVortexToLLVMPass.cpp
  DistributeVortexLocalCopiesPass.cpp
  FuseLinearWithBiasPass.cpp
  InsertVortexBarriersPass.cpp
  LowerVortexLocalMemoryPass.cpp
  LowerLinalgInsideVortexKernelPass.cpp
  MapParallelLoopsToVortexLaunchPass.cpp
  MaterializeBoardXDMAEntryPass.cpp
  MaterializeSIMTControlFlowPass.cpp
  LowerSIMTControlFlowPass.cpp
  MarkVortexKernelPass.cpp
  MaterializeVortexAddressSpacesPass.cpp
  NormalizeONNXFrontendPass.cpp
  PlanVortexLocalMemoryLayoutPass.cpp
  PrepareVortexToLLVMPass.cpp
  PromoteTilesToVortexLocalPass.cpp
  SummarizePreVortexPass.cpp
  TileMatmulForPreVortexPass.cpp
  ValidatePreVortexPass.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vortex

  DEPENDS
    MLIRVortexPassesIncGen

  LINK_LIBS PUBLIC
    MLIRAffineToStandard
    MLIRAffineDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLLVMDialect
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
