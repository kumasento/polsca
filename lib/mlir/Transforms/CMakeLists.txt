add_mlir_library(PhismTransforms
  ExtractTopFunc.cc
  PhismTransforms.cc

  ADDITIONAL_HEADER_DIRS
  "${PHISM_MAIN_INCLUDE_DIR}/phism/mlir/Transforms"  

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRIR
  MLIRStandard
  MLIRSupport
  MLIRAffineToStandard
)
