add_llvm_executable(vx-opt
  vx-opt.cpp
)

llvm_update_compile_flags(vx-opt)
target_link_libraries(vx-opt PRIVATE
  MLIRVortexInit
  MLIROptLib
)

if(TARGET MLIRRegisterAllPasses)
  target_link_libraries(vx-opt PRIVATE MLIRRegisterAllPasses)
endif()

mlir_check_all_link_libraries(vx-opt)
