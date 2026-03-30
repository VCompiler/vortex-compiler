#ifndef VORTEX_PASSES
#define VORTEX_PASSES

include "mlir/Pass/PassBase.td"

def MarkVortexKernel
    : Pass<"vortex-mark-kernel", "::mlir::func::FuncOp"> {
  let summary = "Mark selected func.func operations as Vortex kernels.";
  let description = [{
    Adds the `vortex.kernel` unit attribute to functions that should become
    Vortex kernel entry points. The first implementation is intentionally
    conservative: it only marks functions selected by exact symbol name or by a
    temporary `vortex.entry` marker attribute.
  }];
  let options = [
    ListOption<"kernelNames", "kernel-name", "std::string",
               "Exact function names to mark as Vortex kernels">,
    Option<"recognizeEntryAttr", "recognize-entry-attr", "bool",
           /*default=*/"true",
           "Treat the temporary `vortex.entry` unit attribute as a kernel marker">,
    Option<"removeEntryAttr", "remove-entry-attr", "bool",
           /*default=*/"false",
           "Remove the temporary `vortex.entry` marker after adding `vortex.kernel`">
  ];
}

def MaterializeVortexAddressSpaces
    : Pass<"vortex-materialize-address-spaces", "::mlir::func::FuncOp"> {
  let summary = "Attach Vortex global address spaces to kernel memref arguments.";
  let description = [{
    Rewrites memref-typed kernel function arguments that still use the default
    memory space so they explicitly carry `#vortex.address_space<global>`.

    The first implementation only touches `func.func` arguments of functions
    already marked with `vortex.kernel`. It does not yet rewrite call sites or
    infer local/private allocations.
  }];
}

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
