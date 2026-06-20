# GPGPU Task List

This file is the top-level task list for turning the current Vortex compiler
path into a more complete GPGPU stack. Detailed design notes stay in focused
documents and are linked from each task.

## P0 - Correctness Blockers

- [ ] Fix reduced Guppy local-XDMA semantic output
  - Status: local-XDMA transfer, launch, and readback have reached
    `MANIFEST_RESULT=PASS`, but the reduced run currently reports
    `generated_ids: [4294967295]`.
  - Acceptance: `sequence_length=16, layer_limit=1` produces a next token that
    matches the PyTorch/golden expectation, or a documented narrowed mismatch.
  - Related files: `examples/guppy/chat_driver.py`,
    `examples/guppy/run_board_chat.py`,
    `hw/syn/xilinx/xc7k480t/run_regression_manifest_xdma.py` in
    `vortex-platform`.

- [ ] Connect `vx_local_mem_base()`
  - Status: LLVM intrinsic lowering is wired, `build-vortex-kernel.sh` enables
    the Vortex scheduler route, and scalar, `2 cores x 2 warps x 4 threads`
    no-alias, plus `1 core x 2 warps x 4 threads` cooperative barrier
    local-memory smokes pass on `simx` with `csrr ..., lmem_base` in generated
    assembly. `vx_barrier` lowering now preserves the caller-provided zero-based
    barrier ID. Runtime fallback wrapper and board/local-XDMA validation are
    still pending.
  - Acceptance: local-memory lit tests, LLVM codegen tests, and small
    sim/board kernels prove that `vx_local_mem_base()` reads
    `VX_CSR_LOCAL_MEM_BASE`.
  - Detailed plan: `docs/VX_LOCAL_MEM_BASE_PLAN.md`.

- [ ] Fix one small end-to-end regression route
  - Status: build and runner scripts exist, but the repo still needs a compact
    golden-checked route that is cheap enough to run repeatedly.
  - Acceptance: `MLIR -> LLVM dialect -> LLVM IR -> ELF/bin -> simx or
    local-XDMA -> golden compare` is documented and reproducible.

## P1 - ISA And SIMT Coverage

- [ ] Add LLVM MC/CodeGen tests for existing Vortex custom instructions
  - Scope: `vx_tmc`, `vx_bar`, `vx_pred`, `vx_pred_n`, `vx_split`,
    `vx_split_n`, `vx_join`, `vx_wspawn`, and CSR queries.
  - Acceptance: assembler, disassembler, and intrinsic-to-instruction lowering
    tests cover the currently TableGen-defined instructions.

- [ ] Expose more Vortex execution ops in MLIR
  - Scope: MLIR ops and lowering hooks for `tmc`, `pred`, `split`, `join`, and
    `wspawn`.
  - Acceptance: focused lit tests show each op lowering to the expected wrapper
    or LLVM intrinsic route.

- [ ] Implement SIMT divergence lowering
  - Scope: divergent control flow to `split` / `join` / `pred` / `tmc`.
  - Acceptance: branch-divergent kernels produce correct lane-masked results in
    sim and on board.

## P2 - Runtime And Memory

- [ ] Stabilize resident multi-token execution
  - Acceptance: multiple generated tokens reuse the resident session without
    re-uploading the full kernel image and match the full-rerun path.

- [ ] Complete local/shared memory frame semantics
  - Scope: per-core versus per-group meaning, capacity checks, alignment
    checks, frame metadata handoff, and local alias support beyond the MVP.
  - Acceptance: local memory behavior is documented, checked in compiler
    diagnostics, and validated by at least one cooperative tile kernel.

- [ ] Add `vortex.fence` lowering
  - Acceptance: the compiler, runtime, and hardware semantics agree on the
    fence behavior and tests cover it.

## P3 - Performance And Model Scale

- [ ] Add vote and shuffle support
  - Scope: platform intrinsics such as `vx_vote_*` and `vx_shfl_*`.
  - Acceptance: reduction and warp-exchange tests use these paths directly.

- [ ] Optimize matmul, softmax, and attention kernels
  - Scope: coalesced memory access, local tiling, reductions, scratch reuse, and
    bank-conflict-aware layouts.
  - Acceptance: a documented benchmark shows improvement over the scalar MVP.

- [ ] Improve Guppy full chat execution
  - Scope: resident validation, KV cache, host/runtime weight segmentation, and
    eventually board-side decode control.
  - Acceptance: fixed prompts produce stable multi-token output on board.

## P4 - Later Work

- [ ] Add fp16, bf16, int8, and MMA paths.
- [ ] Add automatic mapping and automatic tiling.
- [ ] Support larger models and dynamic shapes.
- [ ] Add layered CI for lit, simx, rtlsim, and board smoke tests.
