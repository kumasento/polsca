add_mlir_library(PhismTransforms
  ExtractTopFunc.cc
  LoopTransforms.cc
  PhismTransforms.cc
  ArrayPartition.cc
  SimpleArrayPartition.cc
  SimplifyPartitionAccess.cc
  FoldIf.cc
  LoopBoundHoisting.cc
  EliminateAffineLoadStore.cc
  SplitNonAffine.cc
  StripExceptTop.cc
  LiftMemRefSubview.cc
  SCoPDecomposition.cc
  Utils.cc

  ADDITIONAL_HEADER_DIRS
  "${PHISM_MAIN_INCLUDE_DIR}/phism/mlir/Transforms"  

  DEPENDS
  PhismMLIRTransformsIncGen

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
