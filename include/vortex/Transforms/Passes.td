#ifndef VORTEX_PASSES
#define VORTEX_PASSES

include "mlir/Pass/PassBase.td"

def SummarizePreVortex
    : Pass<"vortex-summarize-pre-vortex", "::mlir::func::FuncOp"> {
  let summary = "Attach a summary of the high-level IR seen before the Vortex dialect boundary.";
  let description = [{
    Collects the distinct operation names, dialect namespaces, and memory spaces
    referenced inside each function and stores them in function attributes:

      - vortex.pre_vortex_ops
      - vortex.pre_vortex_dialects
      - vortex.pre_vortex_memory_spaces

    This gives a concrete inventory of which standard MLIR constructs are still
    present before introducing target-specific Vortex operations.
  }];
}

def ValidatePreVortex
    : Pass<"vortex-validate-pre-vortex", "::mlir::func::FuncOp"> {
  let summary = "Reject operations that fall outside the agreed pre-Vortex dialect set.";
  let description = [{
    Ensures the IR remains in the current high-level dialect subset before the
    project starts lowering into a dedicated Vortex dialect. This keeps the
    abstraction boundary explicit and catches accidental low-level dialect usage
    too early in the pipeline.
  }];
}

#endif // VORTEX_PASSES
