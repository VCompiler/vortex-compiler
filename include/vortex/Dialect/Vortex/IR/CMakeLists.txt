set(LLVM_TARGET_DEFINITIONS VortexEnums.td)
mlir_tablegen(VortexEnums.h.inc -gen-enum-decls)
mlir_tablegen(VortexEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRVortexEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS VortexAttributes.td)
mlir_tablegen(VortexAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(VortexAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRVortexAttrDefsIncGen)

add_mlir_dialect(VortexOps vortex)
