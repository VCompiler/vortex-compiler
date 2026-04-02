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

def PlanVortexLocalMemoryLayout
    : Pass<"vortex-plan-local-memory-layout", "::mlir::func::FuncOp"> {
  let summary = "Assign byte offsets and frame size attributes to vortex.local_alloc buffers.";
  let description = [{
    Computes a conservative per-kernel local-memory frame layout for
    `vortex.local_alloc` operations and materializes the result in attributes:

      - `vortex.local_frame_bytes` on the enclosing `func.func`
      - `vortex.local.byte_offset` on each `vortex.local_alloc`
      - `vortex.local.byte_size` on each `vortex.local_alloc`
      - `vortex.local.alignment` on each `vortex.local_alloc`

    The first implementation stays intentionally narrow:

      - only functions already marked with `vortex.kernel` are processed
      - only static-shaped, compact-layout local buffers are accepted
      - local memrefs must not escape through `call`, `yield`, `branch`, or
        `iter_args`

    This pass only plans layout metadata. It does not yet lower local-memory
    accesses to explicit addresses or LLVM dialect operations.
  }];
}

def LowerVortexLocalMemory
    : Pass<"vortex-lower-local-memory", "::mlir::ModuleOp"> {
  let summary = "Lower the MVP subset of Vortex local memory to explicit LLVM pointer accesses.";
  let description = [{
    Lowers `vortex.local_alloc` and the first supported batch of local-memory
    users into explicit byte-addressed accesses based on a `vx_local_mem_base`
    wrapper:

      - expand `memref.copy global <-> local` into `scf.for` loop nests
      - lower local `memref.load` / `memref.store` into
        `llvm.inttoptr + llvm.load/store`
      - lower local `memref.subview` by composing indices back to the root
        `vortex.local_alloc`
      - erase now-dead `vortex.local_alloc` / `memref.cast` / `memref.subview`

    The first implementation intentionally stays narrow:

      - requires `vortex-plan-local-memory-layout` metadata
      - only accepts local buffers rooted at `vortex.local_alloc`
      - only allows `memref.cast`, `memref.subview`, `memref.copy`,
        `memref.load`, and `memref.store` on local buffers
      - still explicitly rejects more complex local aliasing forms such as
        `memref.transpose` / `reinterpret_cast` / shape-changing views
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

def LegalizeVortexForLLVM
    : Pass<"vortex-legalize-for-llvm", "::mlir::ModuleOp"> {
  let summary = "Legalize Vortex kernel IR at the LLVM lowering boundary.";
  let description = [{
    Performs the first, still-conservative cleanup step before lowering the
    remaining `vortex.*` operations into LLVM dialect.

    The initial implementation intentionally keeps the scope narrow:

      - run standard `lower-affine` on all `func.func`
      - inline `vortex.launch` bodies inside functions marked with
        `vortex.kernel`
      - rewrite `#vortex.address_space<global/private>` into numeric memory
        spaces that the standard LLVM type converter can accept
      - reject residual `vortex.local_alloc` or any remaining
        `#vortex.address_space<local>` memref types, which must have been
        removed by `vortex-lower-local-memory`
      - reject residual `vortex.launch` / `vortex.yield` outside kernels or
        after failed inlining

    This pass runs on `builtin.module`, not `func.func`, so function signatures,
    call sites, and nested memref types can stay consistent when memory-space
    annotations are rewritten.
  }];
}

def LowerVortexRuntimeBuiltins
    : Pass<"vortex-lower-runtime-builtins", "::mlir::ModuleOp"> {
  let summary = "Lower the MVP subset of Vortex runtime-facing builtins into wrapper calls.";
  let description = [{
    Converts the first batch of Vortex execution queries and synchronization
    operations into plain `func.call` wrappers that the downstream LLVM/Vortex
    toolchain already understands.

    The initial implementation intentionally handles only the stable subset:

      - `vortex.core_id      -> call @vx_core_id()`
      - `vortex.subgroup_id  -> call @vx_warp_id()`
      - `vortex.thread_id    -> call @vx_thread_id()`
      - `vortex.barrier<core> -> call @vx_barrier(0, vx_num_warps())`

    The integer wrapper results are cast back to `index` so the surrounding IR
    can keep using normal MLIR index arithmetic before the later
    `convert-index-to-llvm` step.

    After the Vortex runtime-facing builtins are lowered, the pass also
    retags `vortex.kernel` as a plain marker attribute (`vortex.kernel_entry`)
    so later `llvm.func` conversion no longer trips the dialect verifier while
    still preserving the notion of kernel entry points.

    This pass currently rejects:

      - residual `vortex.launch` / `vortex.yield` (run `vortex-legalize-for-llvm` first)
      - `vortex.local_alloc`
      - `vortex.fence`
      - `vortex.barrier<subgroup>`

    The unsupported pieces are deferred to the later local-memory and
    target-specific lowering work.
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
