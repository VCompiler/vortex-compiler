add_llvm_executable(vx-opt
  vx-opt.cpp
)

llvm_update_compile_flags(vx-opt)
target_link_libraries(vx-opt PRIVATE
  MLIRVortexInit
  MLIROptLib
)

mlir_check_all_link_libraries(vx-opt)
