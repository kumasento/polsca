add_mlir_library(PhismTransforms
  Mem2PtrCall.cc

  ADDITIONAL_HEADER_DIRS
  "${PHISM_MAIN_INCLUDE_DIR}/phism/Transforms"

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
