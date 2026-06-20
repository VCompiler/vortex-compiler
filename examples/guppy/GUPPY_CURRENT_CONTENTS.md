# Guppy Current Contents

Date: 2026-05-05

This file indexes the current Guppy-related source, generated artifacts, board
results, diagnostics, and PCIe/XDMA support files on this machine.

## 2026-05-05 Full XDMA Bring-Up Result

Current validated PCIe/XDMA state:

```text
FPGA bitstream: xc7k480t_vortex_pcie_xdma_top.bit
PCIe endpoint: 0000:03:00.0, vendor/device 10ee:7024
XDMA devices: /dev/xdma0_h2c_0, /dev/xdma0_c2h_0, /dev/xdma0_control
DMA-control window: 0x0
DDR window: 0x80000000
Host output dump mode: pre-reset dump, then final host reset
```

Transport/runtime fixes now validated on board:

- [xc7k480t_pcie_dma_ctrl_slave.sv](/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t/xc7k480t_pcie_dma_ctrl_slave.sv): the control block now automatically hands DDR back to the XDMA host after Vortex observes busy and then returns to idle. This fixes post-kernel output dump visibility without resetting before the dump.
- [run_regression_manifest_xdma.py](/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t/run_regression_manifest_xdma.py): local-XDMA runs dump outputs before reset and then perform a final host reset for stable multi-kernel execution.
- [chat_driver.py](/home/xiao/vortex-compiler/examples/guppy/chat_driver.py): local-XDMA split runs now keep `--reset-before-success-dump 0` and restore `--final-host-reset 1`, which is required for continuous multi-token runs.

Latest board validation:

```text
seq16/layer1, max_new_tokens=6: PASS
  generated_ids=[64, 64, 64, 64, 64, 64]
  stop_reason=sequence_full

seq16/layer2, max_new_tokens=4: PASS
  generated_ids=[149, 133, 133, 1071]
  generated_text="the water water tou"

seq128/layer6, max_new_tokens=1: PASS
  generated_ids=[779]
  generated_text="hi"

seq128/layer6, max_new_tokens=2: PASS
  generated_ids=[779, 263]
  generated_text="hi there"
```

The latest full run result is saved at:

- [full_inference_seq128_l6/chat_last_result.json](/home/xiao/vortex-compiler/build/guppy/full_inference_seq128_l6/chat_last_result.json)

Current conclusion:

```text
Full seq128/layer6 Guppy local-XDMA split path is functionally passing for
continuous two-token generation. The remaining blocker is performance, not
PCIe enumeration, XDMA basic transport, post-kernel visibility, or reset
handoff.
```

## 2026-05-05 Active Parallel Bring-Up Notes

Current focus: make `sequence_length=16, layer_limit=2` run through the PCIe
split path with a real core-internal 4-thread stage0 instead of calling the
high-arity MLIR `transformer_block` and relying on `guppy_after_attn_merge()`
for early exit.

Latest source changes:

- [gen_full_inference.py](/home/xiao/vortex-compiler/examples/guppy/gen_full_inference.py): stage0 split for `layer_limit > 1` now dispatches a dedicated C wrapper kernel through `vx_spawn_threads` with one block of four threads. It computes embedding, layernorm, QKV, causal score/softmax, attention heads, and `g_attn_merge` using globals and low-arity calls, then exits at progress `240`.
- [gen_full_inference.py](/home/xiao/vortex-compiler/examples/guppy/gen_full_inference.py): per-layer `guppy_run_block_N()` wrappers no longer use `GUPPY_WRAPPER_O0`; the O0 wrapper path caused the layer2 stage0 run to leave the XDMA path in a bad state after `RUN_PASS=1`.
- [run_regression_manifest_xdma.py](/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t/run_regression_manifest_xdma.py): every XDMA `os.read`/`os.write` now has a syscall timeout, so a bad board state fails with diagnostics instead of hanging forever in `xdma_xfer_submit`.

Latest generated build:

- [full_inference_seq16_l2_split_stage0_cwarp4](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2_split_stage0_cwarp4): prepare-only build passed for the new 4-thread low-arity stage0 split path.
- [full_inference_seq16_l2_split_stage0_cwarp4/out/full_inference.elf](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2_split_stage0_cwarp4/out/full_inference.elf): compiled primary ELF.
- [full_inference_seq16_l2_split_stage0_cwarp4/out_split_post_attn/split_post_attn.elf](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2_split_stage0_cwarp4/out_split_post_attn/split_post_attn.elf): compiled stage1 ELF.

Latest board observations:

```text
layer_limit=1 local-xdma split: PASS, generated_id=43
layer_limit=2 with O0 per-layer wrapper: stage0 timeout at vx_busy
layer_limit=2 after removing O0 wrapper: wait_kernel completed in 152 ms, but
  post-done H2C host/reset write wedged in xdma_xfer_submit and subsequent
  C2H control read returned Unknown error 512
```

Immediate next validation after PCIe/XDMA reset:

```bash
cd /home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t
sudo ./run_vortex_pcie_board_smoke.sh --stress

/home/xiao/.venv-guppy-xdma/bin/python \
  /home/xiao/vortex-compiler/examples/guppy/run_board_chat.py \
  --runner-mode local-xdma \
  --bundle-dir /home/xiao/vortex-compiler/build/guppy/export \
  --stage-dir /home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2_split_stage0_cwarp4 \
  --sequence-length 16 \
  --layer-limit 2 \
  --prompt-text "hi guppy" \
  --max-new-tokens 1 \
  --timeout-sec 240 \
  --force-build
```

## Main Source And Entry Points

- [README.md](/home/xiao/vortex-compiler/examples/guppy/README.md): Guppy flow overview and current `local-xdma` commands.
- [download_assets.py](/home/xiao/vortex-compiler/examples/guppy/download_assets.py): downloads/prepares model assets.
- [dump_reference_logits.py](/home/xiao/vortex-compiler/examples/guppy/dump_reference_logits.py): PyTorch/reference logits export.
- [guppy_to_vortex.py](/home/xiao/vortex-compiler/examples/guppy/guppy_to_vortex.py): converts Guppy assets into Vortex bundle layout.
- [gen_full_inference.py](/home/xiao/vortex-compiler/examples/guppy/gen_full_inference.py): generates MLIR, wrapper C, weights, manifests, split hooks, and host reference helpers.
- [chat_driver.py](/home/xiao/vortex-compiler/examples/guppy/chat_driver.py): main chat orchestration; supports remote/JTAG/local-XDMA and PCIe split workaround.
- [run_board_chat.py](/home/xiao/vortex-compiler/examples/guppy/run_board_chat.py): one-command build/prepare/run wrapper for board chat.
- [run_full_inference.sh](/home/xiao/vortex-compiler/examples/guppy/run_full_inference.sh): older/full inference helper script.
- [fixed_prompt_messages.json](/home/xiao/vortex-compiler/examples/guppy/fixed_prompt_messages.json): fixed prompt input used by reference/export flows.

## Current XDMA/Performance Utilities

- [summarize_xdma_timing.py](/home/xiao/vortex-compiler/examples/guppy/summarize_xdma_timing.py): summarizes `timing.log` across XDMA run stages.
- [benchmark_xdma_progress.py](/home/xiao/vortex-compiler/examples/guppy/benchmark_xdma_progress.py): runs cumulative Guppy progress checkpoint early-exit benchmarks through local XDMA.
- [gen_projection_probe.py](/home/xiao/vortex-compiler/examples/guppy/gen_projection_probe.py): projection microbenchmark/probe generator.
- [write_projection_probe_xdma_manifest.py](/home/xiao/vortex-compiler/examples/guppy/write_projection_probe_xdma_manifest.py): helper for projection probe XDMA manifests.
- [test_chat_driver.py](/home/xiao/vortex-compiler/examples/guppy/test_chat_driver.py): Python tests for chat driver behavior.

## Model Assets And Exported Bundle

- [assets](/home/xiao/vortex-compiler/build/guppy/assets): downloaded/source Guppy assets.
- [export](/home/xiao/vortex-compiler/build/guppy/export): Vortex-ready exported model bundle.
- [export/weights](/home/xiao/vortex-compiler/build/guppy/export/weights): exported weight tensors.

## Current Primary Build: Seq16 Layer2

Main directory:

- [full_inference_seq16_l2](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2)

Important files:

- [full_inference.mlir](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/full_inference.mlir): generated MLIR for `sequence_length=16, layer_limit=2`.
- [full_inference_wrapper.c](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/full_inference_wrapper.c): generated wrapper, runtime controls, progress hooks, and split-exit logic.
- [full_inference_weights.S](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/full_inference_weights.S): generated weights assembly.
- [full_inference_manifest.json](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/full_inference_manifest.json): generated manifest metadata.
- [out/full_inference.elf](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/out/full_inference.elf): primary board ELF.
- [out/full_inference.s](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/out/full_inference.s): generated assembly.
- [out/full_inference.dump](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/out/full_inference.dump): disassembly dump.
- [out_split_post_attn/split_post_attn.elf](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/out_split_post_attn/split_post_attn.elf): tiny stage1 post-split ELF.

Latest validated chat artifacts:

- [chat_step_00_result.json](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/chat_step_00_result.json): first generated token result.
- [chat_step_01_result.json](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/chat_step_01_result.json): second generated token result.
- [chat_last_result.json](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/chat_last_result.json): latest chat result pointer.

Current validated output:

```text
sequence_length=16
layer_limit=2
prompt="hi guppy"
max_new_tokens=2
generated_ids=[149, 133]
generated_text="the water"
```

## XDMA Run Results

### Core-Internal Warp4 Validation

Current single-core, 4-thread experimental stage:

- [full_inference_seq16_l1_warp4_attn_ffn_lm_sanitize](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l1_warp4_attn_ffn_lm_sanitize)
- [full_inference_seq32_l1_warp4_attn_ffn_lm_sanitize](/home/xiao/vortex-compiler/build/guppy/full_inference_seq32_l1_warp4_attn_ffn_lm_sanitize)

Validated on `local-xdma`, monolithic, no PCIe split workaround:

```text
sequence_length=16
layer_limit=1
prompt="hi guppy"
attn_out_thread_mode=warp4
ffn_thread_mode=warp4
max_new_tokens=4
generated_ids=[43, 43, 43, 43]
host_numpy_golden=[43, 43, 43, 43]
MANIFEST_RESULT=PASS for all 4 steps
```

Extended small-scope validation:

```text
sequence_length=16
max_new_tokens=requested 8, stopped after 6 because prompt length 10 fills seq16
generated_ids=[43, 43, 43, 43, 43, 43]
host_numpy_golden_prefix=[43, 43, 43, 43, 43, 43]
MANIFEST_RESULT=PASS for all 6 executed steps

sequence_length=32
max_new_tokens=8
generated_ids=[43, 43, 43, 43, 43, 43, 43, 43]
host_numpy_golden=[43, 43, 43, 43, 43, 43, 43, 43]
MANIFEST_RESULT=PASS for all 8 steps
```

Layer2 monolithic probe after the small-scope layer1 pass:

```text
sequence_length=16
layer_limit=2
prompt="hi guppy"
host_numpy_golden=[149, 133] for the first two generated tokens
```

Two transport/control fixes were made while probing this:

- XDMA manifest staging now exports ELF `PT_LOAD` segments with explicit
  `--pad-bss`, so `memsz > filesz` regions are zero-filled over PCIe instead
  of inheriting stale DDR contents.
- `run_board_chat --no-pcie-split-workaround` now generates
  `guppy_runtime_pcie_split_stage=0`, and `chat_driver` also writes a
  monolithic local-xdma payload for that variable. The generated wrapper also
  guards split fast-exit with `split_stage > 0`.

After those fixes, `seq16/layer2` no longer exits immediately with stale split
state, but the monolithic run still times out after 300 seconds at
`progress_stage=2`, i.e. inside `guppy_run_block_0()`. This is a performance /
observability scaling limit of the full serial transformer block path, not a
basic XDMA transport failure.

Key fixes captured in the generator:

- `attn.out` warp4 broadcasts row/weight/bias through volatile globals before
  `vx_tmc`, because newly activated lanes do not inherit lane0 argument
  registers.
- FFN up warp4 uses branchless integer sign masking for ReLU before bit-store,
  avoiding divergent `sum < 0.0f` behavior in the TMC region.
- Runtime `guppy_lm_head_one` sanitizes impossible logits before argmax. This
  filters observed `1e32`-class token-653 corruption that made the old serial
  multi-token output `[43, 653, 653, 653]` look like a baseline. Host/Numpy
  reference for this reduced setup is `[43, 43, 43, 43]`.

Diagnostic checkpoints used during validation:

- `/tmp/guppy_warp4_step01_checkpoint255/run`: row10 post-attention/FFN
  checkpoint; matched serial exactly for `attn_merge`, `hidden`, and `x_next`.
- `/tmp/guppy_warp4_step01_checkpoint52/run`: row10 LM-head checkpoint; final
  LN matched serial, while raw logits exposed token-653 corruption.
- `/tmp/guppy_serial_step01_checkpoint52/run`: serial LM-head comparison run.

Next validation targets:

- Isolate the underlying LM-head dot-product corruption that the runtime logit
  sanitizer currently contains.
- Before retrying higher layer counts monolithically, add checkpoint/progress
  visibility inside `transformer_block()` or extend the warp4 row schedule into
  the full block path. The current `seq16/layer2` monolithic path times out in
  block0 before reaching the LM head.
- Move from `attn.out + FFN` to the next projection candidate only after
  checkpointed golden comparisons are in place.
- Keep the logit sanitizer as a containment workaround until the underlying
  LM-head dot-product corruption is isolated.

### Primary Layer2 Split Run

Primary run directory:

- [xdma_runs](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_runs)

Step 00:

- [step_00/summary.json](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_runs/step_00/summary.json): split stage0/stage1 summary for token 0.
- [step_00/local_manifest.json](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_runs/step_00/local_manifest.json): split manifest wrapper for token 0.
- [step_00/split_stage0_attn_merge/run.log](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_runs/step_00/split_stage0_attn_merge/run.log): stage0 board run log.
- [step_00/split_stage0_attn_merge/timing.log](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_runs/step_00/split_stage0_attn_merge/timing.log): stage0 timing log.
- [step_00/split_stage1_post_attn/timing.log](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_runs/step_00/split_stage1_post_attn/timing.log): stage1 timing log.

Step 01:

- [step_01/summary.json](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_runs/step_01/summary.json): split stage0/stage1 summary for token 1.
- [step_01/local_manifest.json](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_runs/step_01/local_manifest.json): split manifest wrapper for token 1.

Timing summary from the latest optimized path:

```text
stage0 total_ms ~= 40301
stage0 wait_kernel_ms ~= 40029
stage0 prepare_stage_ms ~= 187
stage0 load_manifest_ms ~= 29
stage0 dump_compare_outputs_ms ~= 50
stage1 total_ms ~= 50-70
```

## Progress Benchmark Results

Benchmark directory:

- [xdma_progress_bench_20260426](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_progress_bench_20260426)
- [xdma_progress_bench_fuse_linear_20260426](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_progress_bench_fuse_linear_20260426)

Key files:

- [BENCHMARK_ANALYSIS.md](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_progress_bench_20260426/BENCHMARK_ANALYSIS.md): detailed benchmark analysis and optimization options.
- [results.tsv](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_progress_bench_20260426/results.tsv): raw checkpoint timing table.
- [BENCHMARK_COMPARISON.md](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_progress_bench_fuse_linear_20260426/BENCHMARK_COMPARISON.md): before/after comparison for `vortex-fuse-linear-with-bias`.
- [fuse-linear results.tsv](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2/xdma_progress_bench_fuse_linear_20260426/results.tsv): raw checkpoint timing table after the pass.

Main benchmark conclusion:

```text
QKV projection ~= 14.7 s
attention output projection ~= 4.9 s
FFN up projection ~= 9.8 s
FFN down projection ~= 9.7 s
full stage0 wait ~= 40.0 s
```

Recommended short-term optimization:

```text
Move PCIe split earlier to attn_merge.
Expected board wait: ~40.0 s -> ~15.6 s.
```

After enabling `vortex-fuse-linear-with-bias`, the same checkpoint benchmark
shows:

```text
full stage0 wait: 40028 ms -> 36317 ms
overall speedup: 9.27%
QKV projection: 14697 ms -> 13265 ms
FFN up: 9788 ms -> 8869 ms
FFN down: 9739 ms -> 8839 ms
attn_merge checkpoint p25: 15594 ms -> 14162 ms
```

The optimized ELF also passed the full reduced split chat correctness run:

```text
sequence_length=16
layer_limit=2
max_new_tokens=2
generated_ids=[149, 133]
generated_text="the water"
```

## Seq128 Layer6 Validation

Larger stage directory:

- [full_inference_seq128_l6](/home/xiao/vortex-compiler/build/guppy/full_inference_seq128_l6)

The stage was rebuilt on 2026-04-26 with the current compiler pipeline,
including `vortex-fuse-linear-with-bias`, and then run with `local-xdma` using
the PCIe split workaround:

```text
sequence_length=128
layer_limit=6
prompt="hi guppy"
max_new_tokens=1
golden next-token argmax=779
generated_ids=[779]
generated_text="hi"
```

Current result artifacts:

- [chat_last_result.json](/home/xiao/vortex-compiler/build/guppy/full_inference_seq128_l6/chat_last_result.json): latest 128-token split run result.
- [xdma_runs/step_00/split_stage0_attn_merge/run.log](/home/xiao/vortex-compiler/build/guppy/full_inference_seq128_l6/xdma_runs/step_00/split_stage0_attn_merge/run.log): board stage0 run log.
- [xdma_runs/step_00/split_stage0_attn_merge/timing.log](/home/xiao/vortex-compiler/build/guppy/full_inference_seq128_l6/xdma_runs/step_00/split_stage0_attn_merge/timing.log): stage0 timing log.
- [xdma_runs/step_00/split_stage1_post_attn/timing.log](/home/xiao/vortex-compiler/build/guppy/full_inference_seq128_l6/xdma_runs/step_00/split_stage1_post_attn/timing.log): stage1 timing log.

Timing:

```text
stage0 total_ms=328685
stage0 wait_kernel_ms=328034
stage0 load_manifest_ms=54
stage0 dump_compare_outputs_ms=406
stage1 total_ms=63
```

Interpretation:

```text
The 128-token, 6-layer path is functionally validated for 1-token decode over
PCIe/XDMA. Runtime is dominated by board-side compute, not XDMA staging.
```

## Other Active Generated Builds

- [full_inference_seq16_l1](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l1): validated reduced layer1 build; known output `[43]`, text `"i"`.
- [full_inference_seq128_l6](/home/xiao/vortex-compiler/build/guppy/full_inference_seq128_l6): validated larger 128-token, 6-layer split `local-xdma` build; known 1-token output `[779]`, text `"hi"`.
- [full_inference_seq16_l2_mono](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2_mono): monolithic layer2 experiment directory.
- [full_inference_seq16_l1_dbg_s2](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l1_dbg_s2): layer1 debug-stage build.
- [full_inference_seq16_l2_dbg_s2](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2_dbg_s2): layer2 progress/debug-stage build.
- [full_inference_seq16_l2_dbg_s101](/home/xiao/vortex-compiler/build/guppy/full_inference_seq16_l2_dbg_s101): layer2 startup/progress debug-stage build.

## Diagnostics

Main diagnostics root:

- [diagnostics](/home/xiao/vortex-compiler/build/guppy/diagnostics)

Important diagnostic groups:

- `diag_abi28*`: high-arity ABI baseline that passes.
- `diag_abi29*`, `diag_abi30*`: high-arity ABI failure/repro and mitigation experiments.
- `diag_embedding*`: embedding-only diagnostics.
- `diag_rodata*`: large rodata access diagnostics.
- `diag_block*`, `diag_gpt2_block*`: transformer/block-level diagnostics.
- `projection_probe_*`: projection kernel and memory-visibility probes.

Current interpretation:

```text
ABI diagnostics remain useful coverage, but the reduced Guppy PCIe failure was
primarily fence/cache visibility. Current performance issue is dense projection
compute, not PCIe transport.
```

## PCIe/XDMA Board Support Files

Board/XDMA scripts live under:

- [xc7k480t board directory](/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t)

Key files:

- [CURRENT_BOARD_STATUS.md](/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t/CURRENT_BOARD_STATUS.md): current board status, latest smoke, Guppy validation, and performance notes.
- [run_regression_manifest_xdma.py](/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t/run_regression_manifest_xdma.py): local XDMA manifest runner used by Guppy.
- [elf_to_ddr_segments.py](/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t/elf_to_ddr_segments.py): ELF PT_LOAD to DDR segment converter, including binary segment sidecars.
- [run_vortex_pcie_board_smoke.sh](/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t/run_vortex_pcie_board_smoke.sh): PCIe/XDMA board smoke and stress validation.
- [run_xdma_stress.sh](/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t/run_xdma_stress.sh): raw XDMA DDR stress tool.
- [PCIE_BASELINES.md](/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t/PCIE_BASELINES.md): PCIe baseline tracking.

Latest known board smoke:

```text
/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t/pcie_board_smoke_logs/20260426T025853Z/run.log
DMA-control PASS at 0x0
DMA smoke PASS at 0x80000000
XDMA stress pass_count=20 fail_count=0
```

## Current Working Mental Model

Functional status:

```text
PCIe/XDMA transport works.
Guppy seq16/layer1 local-xdma works.
Guppy seq16/layer2 split local-xdma works.
Two-token layer2 output is stable: [149, 133] -> "the water".
```

Performance status:

```text
Host staging/load/readback overhead has been reduced to sub-second scale.
The remaining bottleneck is Vortex kernel compute, dominated by dense fp32 projections.
```

Next likely engineering step:

```text
Implement earlier attn_merge split for layer2+ as a PCIe multi-kernel workaround,
then use it to speed up iteration while planning real tiled/fused projection kernels.
```
