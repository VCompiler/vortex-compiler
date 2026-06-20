# Guppy XDMA Benchmarks

Date: 2026-05-05

This file records the latest PCIe/XDMA board benchmark baseline for Guppy.

## Current Passing Baseline

Hardware/software path:

```text
Board: xc7k480t PCIe/XDMA Vortex bitstream
Endpoint: 0000:03:00.0, 10ee:7024
Runner: local-xdma
Split workaround: enabled
Output dump: pre-reset dump, final host reset enabled
Prompt: "hi guppy"
```

Validated runs:

```text
seq16/layer1, max_new_tokens=6: PASS
  generated_text="\n\n\n\n\n\n"

seq16/layer2, max_new_tokens=4: PASS
  generated_ids=[149, 133, 133, 1071]
  generated_text="the water water tou"

seq128/layer6, max_new_tokens=2: PASS
  generated_ids=[779, 263]
  generated_text="hi there"
```

## Full Seq128/Layer6 Timing

Run directory:

- [full_inference_seq128_l6](/home/xiao/vortex-compiler/build/guppy/full_inference_seq128_l6)

Step 0:

```text
stage0 total_ms=329172
stage0 wait_kernel_ms=328025
stage0 load_manifest_ms=70
stage0 dump_compare_outputs_ms=543
stage1 total_ms=267
stage1 wait_kernel_ms=1
next_token_id=779
next_token_text="hi"
```

Step 1:

```text
stage0 total_ms=329037
stage0 wait_kernel_ms=328028
stage0 load_manifest_ms=69
stage0 dump_compare_outputs_ms=518
stage1 total_ms=256
stage1 wait_kernel_ms=1
next_token_id=263
next_token_text=" there"
```

Derived baseline:

```text
Full token latency ~= 329.3 s/token
Dominant component: stage0 kernel compute, ~= 328.0 s/token
Host/PCIe manifest load: < 0.1 s/token
Output dump: ~= 0.5 s/token
Stage1 post-attention/lm-head wrapper: ~= 0.26 s/token
```

## Why Latency Is High

The latency is dominated by `split_stage0_attn_merge`, not PCIe transfer:

```text
stage0 kernel compute ~= 99.65% of token latency
PCIe load/dump overhead ~= 0.2% of token latency
stage1 overhead ~= 0.1% of token latency
```

Current stage0 still performs the main transformer work on one Vortex core with
a limited 4-thread schedule. It computes embedding, layernorm, QKV, causal
attention scores, softmax, attention head output, and attention merge for the
active sequence. For `S=128`, the attention and projection loops scale much
more aggressively than the seq16 bring-up cases, so the single-core schedule is
the bottleneck.

The PCIe path is no longer the primary issue. The load path moves about 43 MB of
segments in roughly 70 ms, and output dump moves the 49,152-word attention merge
buffer in about 0.5 s. These are visible but small compared with the 328 s
kernel time.

## Optimization Directions

Recommended next optimization order:

```text
1. Keep current split/handoff path as the correctness baseline.
2. Add stage0 internal timing/progress counters around embedding, QKV,
   attention score, softmax, attention output, and merge loops.
3. Parallelize the dominant row/column loops inside one core across the four
   hardware threads with deterministic per-thread output ranges.
4. Move projection-heavy loops to a multi-core split once the one-core
   four-thread schedule is stable and golden-checked.
5. Reduce per-token host work only after kernel compute is no longer the
   dominant term.
```

The first performance target should be stage0 compute time. PCIe batching and
resident-kernel work are useful later, but they cannot materially improve the
current 329 s/token baseline while stage0 compute remains above 328 s/token.
