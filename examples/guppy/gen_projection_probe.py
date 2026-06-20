#!/usr/bin/env python3
"""Generate a standalone MLIR probe for the Guppy attention output projection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gen_full_inference import (
    BundleLoader,
    emit_incbin,
    emit_linear_with_bias,
    layernorm_ref,
    linear_ref,
    softmax_ref,
    tensor_symbol,
    write_blob,
)


def float_words(values: np.ndarray) -> list[str]:
    words = values.astype(np.float32).view(np.uint32).ravel()
    return [f"{int(word):08X}" for word in words]


def int_words(values: list[int]) -> list[str]:
    return [f"{value & 0xFFFFFFFF:08X}" for value in values]


def gen_probe_weights_asm(blobs: list[tuple[str, Path]]) -> str:
    parts: list[str] = ["    .text", ""]
    for label, path in blobs:
        parts.append(emit_incbin(label, path))
        parts.append("")
    return "\n".join(parts)


def build_attn_merge(
    bundle: BundleLoader,
    *,
    sequence_length: int,
    layer_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = bundle.model_config["normalized_config"]
    d_model = int(cfg["d_model"])
    n_heads = int(cfg["n_heads"])
    head_dim = d_model // n_heads

    input_ids = list(bundle.prompt["input_ids"])
    if len(input_ids) > sequence_length:
        raise ValueError(
            f"prompt length {len(input_ids)} exceeds sequence length {sequence_length}"
        )
    padded = np.full(sequence_length, int(cfg["pad_id"]), dtype=np.int32)
    padded[: len(input_ids)] = np.array(input_ids, dtype=np.int32)

    tok_emb = bundle.load_tensor("tok_emb.weight").astype(np.float32)
    pos_emb = bundle.load_tensor("pos_emb.weight").astype(np.float32)[:sequence_length]
    x = (tok_emb[padded] + pos_emb).astype(np.float32)

    prefix = f"blocks.{layer_index}"
    ln1 = layernorm_ref(
        x,
        bundle.load_tensor(f"{prefix}.norm1.weight").astype(np.float32),
        bundle.load_tensor(f"{prefix}.norm1.bias").astype(np.float32),
    ).astype(np.float32)
    qkv = linear_ref(
        ln1,
        bundle.load_tensor(f"{prefix}.attn.qkv.weight").astype(np.float32),
        bundle.load_tensor(f"{prefix}.attn.qkv.bias").astype(np.float32),
    ).astype(np.float32)

    qkv = qkv.reshape(sequence_length, 3, n_heads, head_dim).transpose(1, 2, 0, 3)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    score = np.einsum("hsd,htd->hst", q, k).astype(np.float32)
    score *= np.float32(1.0 / math.sqrt(head_dim))
    mask = np.tril(np.ones((sequence_length, sequence_length), dtype=bool))
    score = np.where(mask[None, :, :], score, np.float32(-np.inf)).astype(np.float32)
    prob = softmax_ref(score, axis=-1).astype(np.float32)
    attn_heads = np.einsum("hst,htd->hsd", prob, v).astype(np.float32)
    attn_merge = attn_heads.transpose(1, 0, 2).reshape(sequence_length, d_model)

    out_weight = bundle.load_tensor(f"{prefix}.attn.out.weight").astype(np.float32)
    out_bias = bundle.load_tensor(f"{prefix}.attn.out.bias").astype(np.float32)
    expected = linear_ref(attn_merge, out_weight, out_bias).astype(np.float32)
    return attn_merge.astype(np.float32), out_weight, out_bias, expected


def gen_mlir(sequence_length: int, d_model: int, *, c_only: bool) -> str:
    if c_only:
        return """// Auto-generated C-only projection probe companion module.
module {
  func.func @projection_probe_dummy() attributes {vortex.entry} {
    return
  }
}
"""
    linear = emit_linear_with_bias(
        "%input",
        "%weight",
        "%bias",
        "%output",
        sequence_length,
        d_model,
        d_model,
        tag="probe",
    )
    return f"""// Auto-generated Guppy attention output projection probe.
module {{
  func.func @projection_probe(%input: memref<{sequence_length}x{d_model}xf32>,
                              %weight: memref<{d_model}x{d_model}xf32>,
                              %bias: memref<{d_model}xf32>,
                              %output: memref<{sequence_length}x{d_model}xf32>)
      attributes {{vortex.entry}} {{
    %zero = arith.constant 0.0 : f32
{linear}
    return
  }}
}}
"""


def gen_wrapper(
    sequence_length: int,
    d_model: int,
    tolerance: float,
    *,
    c_only: bool,
    thread_mode: str,
    post_call_delay_iters: int,
    post_call_warmup_reads: int,
) -> str:
    input_sym = tensor_symbol("probe.input")
    weight_sym = tensor_symbol("probe.weight")
    bias_sym = tensor_symbol("probe.bias")
    expected_sym = tensor_symbol("probe.expected")
    call_decl = ""
    call_impl = ""
    call_body = f"""\
  projection_probe_c(
      (float*){input_sym},
      (float*){weight_sym},
      (float*){bias_sym},
      probe_output);"""
    if not c_only:
        call_decl = "extern void projection_probe(float *input, float *weight, float *bias, float *output);"
        call_body = f"""\
  projection_probe(
      (float*){input_sym},
      (float*){weight_sym},
      (float*){bias_sym},
      probe_output);"""
    else:
        if thread_mode == "warp4":
            call_body = f"""\
  projection_probe_c_warp4(
      (float*){input_sym},
      (float*){weight_sym},
      (float*){bias_sym},
      probe_output);"""
        call_impl = """\
static void projection_probe_c(const float *input, const float *weight,
                               const float *bias, float *output) {
  for (int i = 0; i < S; ++i) {
    for (int j = 0; j < D; ++j) {
      float sum = bias[j];
      for (int k = 0; k < D; ++k)
        sum += input[i * D + k] * weight[j * D + k];
      output[i * D + j] = sum;
    }
  }
}

static void __attribute__((noinline))
projection_probe_c_warp4_body(const float *input, const float *weight,
                              const float *bias, float *output) {
  int tid = (int)probe_thread_id();
  int lanes = probe_num_threads_observed;
  if (lanes <= 0)
    lanes = 1;
  for (int task = tid; task < S * D; task += lanes) {
    int i = task / D;
    int j = task - i * D;
    float sum = bias[j];
    for (int k = 0; k < D; ++k)
      sum += input[i * D + k] * weight[j * D + k];
    output[task] = sum;
    probe_thread_task_count[tid] += 1;
  }
}

static void projection_probe_c_warp4(const float *input, const float *weight,
                                     const float *bias, float *output) {
  int lanes = vx_num_threads();
  if (lanes > 4)
    lanes = 4;
  if (lanes < 1)
    lanes = 1;
  probe_num_threads_observed = lanes;
  probe_thread_expected_mask = (1 << lanes) - 1;
  probe_thread_task_count[0] = 0;
  probe_thread_task_count[1] = 0;
  probe_thread_task_count[2] = 0;
  probe_thread_task_count[3] = 0;

  vx_tmc(probe_thread_expected_mask);
  projection_probe_c_warp4_body(input, weight, bias, output);
  vx_tmc_one();

  int nonzero_mask = 0;
  for (int i = 0; i < lanes; ++i) {
    if (probe_thread_task_count[i] != 0)
      nonzero_mask |= 1 << i;
  }
  probe_thread_nonzero_mask = nonzero_mask;
}
"""
    return f"""#include <stdint.h>
#include <vx_intrinsics.h>

{call_decl}

#define S {sequence_length}
#define D {d_model}
#define TOLERANCE {tolerance:.8e}f
#define POST_CALL_DELAY_ITERS {post_call_delay_iters}
#define POST_CALL_WARMUP_READS {post_call_warmup_reads}

extern const float {input_sym}[S * D];
extern const float {weight_sym}[D * D];
extern const float {bias_sym}[D];
extern const float {expected_sym}[S * D];

float probe_output[S * D];
volatile int probe_status = -1;
volatile int probe_nan_count = -1;
volatile int probe_fail_index = -1;
volatile uint32_t probe_first_bits = 0;
volatile uint32_t probe_max_diff_bits = 0;
volatile int probe_done_flag = 0;
volatile float probe_warmup_sink = 0.0f;
volatile int probe_num_threads_observed = 1;
volatile int probe_thread_expected_mask = 1;
volatile int probe_thread_nonzero_mask = 1;
volatile int probe_thread_task_count[4] = {{0, 0, 0, 0}};

static uint32_t f32_bits(float value) {{
  union {{
    float f;
    uint32_t u;
  }} conv;
  conv.f = value;
  return conv.u;
}}

static __attribute__((always_inline)) inline unsigned probe_thread_id(void) {{
  unsigned value;
  __asm__ volatile("csrr %0, %1" : "=r"(value) : "i"(0xCC0));
  return value;
}}

static __attribute__((always_inline)) inline unsigned probe_warp_id(void) {{
  unsigned value;
  __asm__ volatile("csrr %0, %1" : "=r"(value) : "i"(0xCC1));
  return value;
}}

static __attribute__((always_inline)) inline unsigned probe_core_id(void) {{
  unsigned value;
  __asm__ volatile("csrr %0, %1" : "=r"(value) : "i"(0xCC2));
  return value;
}}

static void probe_post_call_delay(void) {{
  for (volatile int i = 0; i < POST_CALL_DELAY_ITERS; ++i) {{
    __asm__ volatile("" ::: "memory");
  }}
}}

static void probe_warmup_output_reads(void) {{
  float sink = 0.0f;
  for (int pass = 0; pass < POST_CALL_WARMUP_READS; ++pass) {{
    for (int i = 0; i < S * D; ++i)
      sink += probe_output[i];
  }}
  probe_warmup_sink = sink;
}}

{call_impl}

int main() {{
  if (probe_thread_id() != 0 || probe_warp_id() != 0 || probe_core_id() != 0) {{
    while (!probe_done_flag) {{
    }}
    return 0;
  }}

  probe_status = 10;
{call_body}
  __asm__ volatile("fence" ::: "memory");
  probe_post_call_delay();
  probe_warmup_output_reads();
  __asm__ volatile("fence" ::: "memory");

  probe_status = 20;
  int failures = 0;
  int nan_count = 0;
  int first_fail = -1;
  float max_diff = 0.0f;
  for (int i = 0; i < S * D; ++i) {{
    float actual = probe_output[i];
    float expected = {expected_sym}[i];
    if (actual != actual) {{
      ++nan_count;
      ++failures;
      if (first_fail < 0)
        first_fail = i;
      continue;
    }}
    float diff = actual - expected;
    if (diff < 0.0f)
      diff = -diff;
    if (diff > max_diff)
      max_diff = diff;
    if (diff > TOLERANCE) {{
      ++failures;
      if (first_fail < 0)
        first_fail = i;
    }}
  }}

  probe_nan_count = nan_count;
  probe_fail_index = first_fail;
  probe_first_bits = f32_bits(probe_output[0]);
  probe_max_diff_bits = f32_bits(max_diff);
  if (S * D >= probe_num_threads_observed &&
      probe_thread_nonzero_mask != probe_thread_expected_mask)
    ++failures;
  probe_status = failures == 0 ? 0 : 1;
  probe_done_flag = 1;
  return 0;
}}
"""


def write_mem_words(path: Path, words: list[str]) -> None:
    path.write_text("".join(f"{word}\n" for word in words), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a standalone Guppy attention output projection probe."
    )
    parser.add_argument("--bundle-dir", default="build/guppy/export")
    parser.add_argument("--out-dir", default="build/guppy/projection_probe_seq16_l1")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Only probe one attention-merge row; default probes all rows.",
    )
    parser.add_argument(
        "--d-limit",
        type=int,
        default=None,
        help="Probe only the leading D columns/rows of the projection.",
    )
    parser.add_argument("--tolerance", type=float, default=5.0e-2)
    parser.add_argument(
        "--c-only",
        action="store_true",
        help="Use a C projection loop instead of the MLIR linalg projection.",
    )
    parser.add_argument(
        "--thread-mode",
        choices=("serial", "warp4"),
        default="serial",
        help=(
            "Projection scheduler used by --c-only: serial uses one lane; "
            "warp4 uses up to four threads in core0/warp0."
        ),
    )
    parser.add_argument(
        "--post-call-delay-iters",
        type=int,
        default=0,
        help="Spin this many iterations after the projection call before comparing.",
    )
    parser.add_argument(
        "--post-call-warmup-reads",
        type=int,
        default=0,
        help="Read the full output this many times after the projection call before comparing.",
    )
    args = parser.parse_args()
    if args.thread_mode != "serial" and not args.c_only:
        raise ValueError("--thread-mode warp4 currently requires --c-only")

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    blobs_dir = out_dir / "blobs"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = BundleLoader(bundle_dir)
    cfg = bundle.model_config["normalized_config"]
    source_d_model = int(cfg["d_model"])
    d_model = source_d_model

    attn_merge, out_weight, out_bias, expected = build_attn_merge(
        bundle,
        sequence_length=args.sequence_length,
        layer_index=args.layer_index,
    )
    probe_rows = args.sequence_length
    if args.row_index is not None:
        if args.row_index < 0 or args.row_index >= args.sequence_length:
            raise ValueError(
                f"row-index must be in 0..{args.sequence_length - 1}, got {args.row_index}"
            )
        attn_merge = attn_merge[args.row_index : args.row_index + 1].copy()
        expected = expected[args.row_index : args.row_index + 1].copy()
        probe_rows = 1
    if args.d_limit is not None:
        if args.d_limit <= 0 or args.d_limit > source_d_model:
            raise ValueError(
                f"d-limit must be in 1..{source_d_model}, got {args.d_limit}"
            )
        d_model = args.d_limit
        attn_merge = attn_merge[:, :d_model].copy()
        out_weight = out_weight[:d_model, :d_model].copy()
        out_bias = out_bias[:d_model].copy()
        expected = linear_ref(attn_merge, out_weight, out_bias).astype(np.float32)

    blobs: list[tuple[str, Path]] = []

    def add_blob(label: str, arr: np.ndarray) -> None:
        blob_path = blobs_dir / f"{label}.bin"
        write_blob(blob_path, np.ascontiguousarray(arr))
        blobs.append((label, blob_path.resolve()))

    add_blob(tensor_symbol("probe.input"), attn_merge)
    add_blob(tensor_symbol("probe.weight"), out_weight)
    add_blob(tensor_symbol("probe.bias"), out_bias)
    add_blob(tensor_symbol("probe.expected"), expected)

    (out_dir / "projection_probe.mlir").write_text(
        gen_mlir(probe_rows, d_model, c_only=args.c_only), encoding="utf-8"
    )
    (out_dir / "projection_probe_wrapper.c").write_text(
        gen_wrapper(
            probe_rows,
            d_model,
            args.tolerance,
            c_only=args.c_only,
            thread_mode=args.thread_mode,
            post_call_delay_iters=args.post_call_delay_iters,
            post_call_warmup_reads=args.post_call_warmup_reads,
        ),
        encoding="utf-8",
    )
    (out_dir / "projection_probe_weights.S").write_text(
        gen_probe_weights_asm(blobs), encoding="utf-8"
    )
    write_mem_words(out_dir / "expected_status.mem", int_words([0]))
    write_mem_words(out_dir / "expected_nan_count.mem", int_words([0]))
    write_mem_words(out_dir / "expected_head.mem", float_words(expected.ravel()[:16]))

    manifest = {
        "schema_version": 1,
        "generator": "examples/guppy/gen_projection_probe.py",
        "bundle_dir": str(bundle_dir),
        "sequence_length": args.sequence_length,
        "probe_rows": probe_rows,
        "row_index": args.row_index,
        "layer_index": args.layer_index,
        "source_d_model": source_d_model,
        "d_model": d_model,
        "d_limit": args.d_limit,
        "tolerance": args.tolerance,
        "c_only": args.c_only,
        "thread_mode": args.thread_mode,
        "post_call_delay_iters": args.post_call_delay_iters,
        "post_call_warmup_reads": args.post_call_warmup_reads,
        "mlir": "projection_probe.mlir",
        "wrapper": "projection_probe_wrapper.c",
        "weights_asm": "projection_probe_weights.S",
        "expected_head_words": "expected_head.mem",
        "expected_status_words": "expected_status.mem",
        "expected_nan_count_words": "expected_nan_count.mem",
    }
    (out_dir / "projection_probe_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"wrote {out_dir / 'projection_probe.mlir'}")
    print(f"wrote {out_dir / 'projection_probe_wrapper.c'}")
    print(f"wrote {out_dir / 'projection_probe_weights.S'}")
    print(f"wrote {out_dir / 'projection_probe_manifest.json'}")
    print(f"expected first output word: {float_words(expected.ravel()[:1])[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
