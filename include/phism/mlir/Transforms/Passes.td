#ifndef PHISM_MLIR_TRANSFORMS_PASSES
#define PHISM_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LoopBoundHoisting : FunctionPass<"loop-bound-hoisting"> {
  let summary = "Hoist loop bound calculation to make perfectly nested loops";
  let constructor = "phism::createLoopBoundHoistingPass()";
}

def EliminateAffineLoadStore : FunctionPass<"eliminate-affine-load-store"> {
  let summary = "Eliminate redundant affine load and store operations.";
  let constructor = "phism::createEliminateAffineLoadStorePass()";
  let options = [
    Option<"topOnly", "top-only", "bool", /*default=*/"false",
           "Whether should only apply for phism.top">,
    Option<"loadAfterStore", "load-after-store", "bool", /*default=*/"true",
           "Whether to perform load-after-store elimination">
  ];
}

def SplitNonAffine : FunctionPass<"split-non-affine"> {
  let summary = "Split affine and non-affine loops into functions.";
  let constructor = "phism::createSplitNonAffinePass()";
  let options = [
    Option<"topOnly", "top-only", "bool", /*default=*/"false",
           "Whether should only apply for phism.top">,
    Option<"markOnly", "mark-only", "bool", /*default=*/"false",
           "Whether we just annotate those for loops with non affine accesses.">,
    Option<"maxLoopDepth", "max-loop-depth", "int", /*default=*/"0",
           "Below which loop depth should we put affine loops into functions.">,
    Option<"greedy", "greedy", "bool", /*default=*/"false",
           "Take the contents out once see a non affine loop">,
    ListOption<"inclFuncs", "incl-funcs", "std::string", "Limit to the provided functions",
           "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}

def SimpleArrayPartition : Pass<"simple-array-partition", "::mlir::ModuleOp"> {
  let summary = "Simple array partition, the initial version.";
  let constructor = "phism::createSimpleArrayPartitionPass()";
  let options = [
    Option<"dumpFile", "dump-file", "bool", /*default=*/"false", "Whether to dump file.">,
    Option<"flatten", "flatten", "bool", /*default=*/"false", "Whether to flatten partitioned dims.">,
    Option<"genMain", "gen-main", "bool", /*default=*/"false", "Generate a dummy main function.">
  ];
}

def ArrayPartition : Pass<"array-partition", "::mlir::ModuleOp"> {
  let summary = "Array partition.";
  let constructor = "phism::createArrayPartitionPass()";
  let options = [
  ];
}

def SimplifyPartitionAccess : FunctionPass<"simplify-partition-access"> {
  let summary = "Simplify array partition accesses.";
  let constructor = "phism::createSimplifyPartitionAccessPass()";
  let options = [
  ];
}

def StripExceptTop : Pass<"strip-except-top", "::mlir::ModuleOp"> {
  let summary = "Strip functions except those called by the top.";
  let constructor = "phism::createStripExceptTopPass()";
  let options = [

  ];
}

def AffineLoopUnswitching : FunctionPass<"affine-loop-unswitching"> {
  let summary = "Affine loop unswitching transformation.";
  let constructor = "phism::createAffineLoopUnswitchingPass()";
}

def AnnotatePointLoop : FunctionPass<"anno-point-loop"> {
  let summary = "Annotate point loops from nested affine.for";
  let constructor = "phism::createAnnotatePointLoopPass()";
  let options = [
    Option<"constantIndvar", "constant-indvar", "bool", /*default=*/"false", "Accept constant loop bounds.">
  ];
}

def OutlineProcessElement : Pass<"outline-proc-elem", "::mlir::ModuleOp"> {
  let summary = "Outline process elements";
  let constructor = "phism::createOutlineProcessElementPass()";
  let options = [
    Option<"maxTripcount", "max-tripcount", "int", /*default=*/"0", "Max trip count for a tile loop.">,
    Option<"noIgnored", "no-ignored", "bool", /*default=*/"false", "Ignore scop.ignored.">
  ];
}

def RewritePloopIndvar : FunctionPass<"rewrite-ploop-indvar"> {
  let summary = "Rewrite the indvar of point loops.";
  let constructor = "phism::createRewritePloopIndvarPass()";
  let options = [
  ];
}

def LoadSwitch : FunctionPass<"load-switch"> {
  let summary = "Simplify floordiv related accesses.";
  let constructor = "phism::createLoadSwitchPass()";
  let options = [
  ];
}

def LiftMemRefSubview : Pass<"lift-memref-subview", "::mlir::ModuleOp"> {
  let summary = "Lift the outermost dims into subview.";
  let constructor = "phism::createLiftMemRefSubviewPass()";
  let options= [
    Option<"flatten", "flatten", "bool", /*default=*/"false", "Whether to flatten partitioned dims.">
  ];
}

def SCoPDecomposition : Pass<"scop-decomp", "::mlir::ModuleOp"> {
  let summary = "Decomposite SCoP into smaller, self-dependent regions";
  let constructor = "phism::createSCoPDecompositionPass()";
  let options = [
    Option<"maxLoopDepth", "max-loop-depth", "int", /*default=*/"0",
           "Below which loop depth should we put affine loops into functions.">,
  ];
}

def InlineSCoPAffine : Pass<"inline-scop-affine", "::mlir::ModuleOp"> {
  let summary = "Inline those scop.affine functions back.";
  let constructor = "phism::createInlineSCoPAffinePass()";
  let options = [

  ];
}

#endif
