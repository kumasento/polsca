set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(phism-opt
  phism-opt.cc
)
llvm_update_compile_flags(phism-opt)
target_link_libraries(phism-opt
  PRIVATE
  MLIRStandard
  MLIRLLVMIR
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

  PhismTransforms
  )
