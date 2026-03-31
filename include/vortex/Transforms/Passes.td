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

def MapParallelLoopsToVortexLaunch
    : Pass<"vortex-map-parallel-loops-to-launch", "::mlir::func::FuncOp"> {
  let summary = "Convert explicitly mapped scf.for nests into vortex.launch regions.";
  let description = [{
    Materializes Vortex execution regions from conservative, explicitly marked
    loop nests. The pass only rewrites `scf.for` nests inside functions already
    marked with `vortex.kernel`, and only when every mapped loop:

      - carries `vortex.mapping = "core" | "subgroup" | "thread"`
      - has lower bound `0`
      - has step `1`
      - has no iter_args / results
      - forms a perfect nest

    The resulting `vortex.launch` region replaces the mapped induction
    variables with `vortex.core_id`, `vortex.subgroup_id`, and
    `vortex.thread_id`.
  }];
}

def PromoteTilesToVortexLocal
    : Pass<"vortex-promote-tiles-to-local", "::mlir::func::FuncOp"> {
  let summary = "Promote explicitly marked memref.subview tiles into vortex.local_alloc buffers.";
  let description = [{
    Rewrites conservatively marked `memref.subview` operations inside
    `vortex.launch` regions into explicit local-memory buffers:

      - `vortex.promote_to_local` marks the subview for promotion
      - `vortex.write_back` requests a copy back to the original subview

    The first implementation is intentionally narrow. It only handles static
    shaped memref.subview results and does not yet expand the copies into
    cooperative per-lane load/store sequences.
  }];
}

def InsertVortexBarriers
    : Pass<"vortex-insert-barriers", "::mlir::func::FuncOp"> {
  let summary = "Insert conservative vortex.barrier <core> operations around promoted local tiles.";
  let description = [{
    Scans `vortex.launch` regions for the narrow local-tile promotion pattern
    produced by `vortex-promote-tiles-to-local`:

      - `memref.copy global -> local`
      - local tile uses
      - optional `memref.copy local -> global`

    The first implementation stays conservative and only inserts
    `vortex.barrier <core>` at two points:

      - after copy-in, before the first local use
      - after the last local use, before copy-out

    Complex patterns are skipped instead of rewritten.
  }];
}

def LowerLinalgInsideVortexKernel
    : Pass<"vortex-lower-linalg-inside-kernel", "::mlir::func::FuncOp"> {
  let summary = "Lower buffer-semantics linalg operations inside Vortex kernels to scf/memref/arithmetic.";
  let description = [{
    Rewrites remaining `linalg.*` operations inside functions already marked
    with `vortex.kernel` into explicit loop nests and scalar memory accesses.

    The first implementation stays conservative:

      - only buffer-semantics linalg ops are accepted
      - lowering uses standard `scf.for` / `memref.load` / `memref.store`
      - tensor-semantics linalg ops are rejected

    This pass is intended to run after Vortex execution mapping, local tile
    promotion, and barrier insertion, so the Vortex-specific execution and
    memory structure stays explicit while the compute body moves to a form that
    is easier to lower further toward LLVM.
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
