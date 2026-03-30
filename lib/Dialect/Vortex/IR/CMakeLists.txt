add_mlir_dialect_library(MLIRVortexDialect
  VortexAttributes.cpp
  VortexDialect.cpp
  VortexOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vortex

  DEPENDS
    MLIRVortexAttrDefsIncGen
    MLIRVortexEnumsIncGen
    MLIRVortexOpsIncGen

  LINK_LIBS PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRSideEffectInterfaces
)
