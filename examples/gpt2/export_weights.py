#!/usr/bin/env python3
"""Export transformer block weights, input, and golden output for bare-metal simx.

Usage:
    python3 export_weights.py --seq S --dim D --ff FF --layers N --out-dir DIR [--seed SEED]

Produces in DIR/:
    weights_layer{i}.bin  -- raw float32 binary per layer
    input.bin             -- random input x: shape [S, D], float32
    golden_output.bin     -- numpy reference output after N transformer blocks
    weights.h             -- C header with static const float arrays
    manifest.json         -- metadata
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Numpy reference implementation
# ---------------------------------------------------------------------------

def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
              eps: float = 1e-5) -> np.ndarray:
    """Layer normalization over last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # numpy uses 1/N (population variance)
    # Match the MLIR kernel which computes var = sum((x-mean)^2)/N
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std * gamma + beta


def gelu(x: np.ndarray) -> np.ndarray:
    """GeLU activation: x * 0.5 * (1 + erf(x / sqrt(2)))."""
    from scipy.special import erf as _erf
    try:
        return x * 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))
    except ImportError:
        pass
    # Fallback without scipy
    return _gelu_fallback(x)


def _gelu_fallback(x: np.ndarray) -> np.ndarray:
    """GeLU without scipy, using numpy's erf approximation via math."""
    result = np.empty_like(x)
    for idx in np.ndindex(x.shape):
        v = float(x[idx])
        result[idx] = v * 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))
    return result


# Try scipy first; if not available, use the fallback
try:
    from scipy.special import erf as scipy_erf

    def gelu(x: np.ndarray) -> np.ndarray:
        return x * 0.5 * (1.0 + scipy_erf(x / math.sqrt(2.0)))
except ImportError:
    def gelu(x: np.ndarray) -> np.ndarray:
        return _gelu_fallback(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    mx = x.max(axis=axis, keepdims=True)
    e = np.exp(x - mx)
    return e / e.sum(axis=axis, keepdims=True)


def attention(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray,
              Wv: np.ndarray, Wo: np.ndarray) -> np.ndarray:
    """Single-head self-attention.

    x:  [S, D]
    Wq, Wk, Wv, Wo: [D, D]
    returns: [S, D]
    """
    D = x.shape[-1]
    Q = x @ Wq   # [S, D]
    K = x @ Wk   # [S, D]
    V = x @ Wv   # [S, D]

    # score = Q @ K^T / sqrt(D)
    score = Q @ K.T / math.sqrt(D)   # [S, S]
    prob = softmax(score, axis=-1)    # [S, S]
    attn = prob @ V                   # [S, D]
    return attn @ Wo                  # [S, D]


def mlp(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Feed-forward MLP block: GELU(x @ W1) @ W2.

    x:  [S, D]
    W1: [D, FF]
    W2: [FF, D]
    returns: [S, D]
    """
    h = x @ W1           # [S, FF]
    h = gelu(h)          # [S, FF]
    return h @ W2         # [S, D]


def transformer_block(x: np.ndarray, weights: dict) -> np.ndarray:
    """Single transformer block: LN1 -> Attention + residual -> LN2 -> MLP + residual.

    weights: dict with keys ln1_gamma, ln1_beta, Wq, Wk, Wv, Wo,
             ln2_gamma, ln2_beta, W1, W2
    returns: [S, D]
    """
    # Pre-norm attention
    x_ln = layernorm(x, weights['ln1_gamma'], weights['ln1_beta'])
    attn_out = attention(x_ln, weights['Wq'], weights['Wk'],
                         weights['Wv'], weights['Wo'])
    x = x + attn_out  # residual

    # Pre-norm MLP
    x_ln2 = layernorm(x, weights['ln2_gamma'], weights['ln2_beta'])
    ff_out = mlp(x_ln2, weights['W1'], weights['W2'])
    x = x + ff_out    # residual

    return x


# ---------------------------------------------------------------------------
# Weight generation
# ---------------------------------------------------------------------------

WEIGHT_ORDER = [
    'ln1_gamma', 'ln1_beta',
    'Wq', 'Wk', 'Wv', 'Wo',
    'ln2_gamma', 'ln2_beta',
    'W1', 'W2',
]


def generate_layer_weights(D: int, FF: int, rng: np.random.RandomState) -> dict:
    """Generate random weights for one transformer layer."""
    return {
        'ln1_gamma': np.ones(D, dtype=np.float32),
        'ln1_beta': np.zeros(D, dtype=np.float32),
        'Wq': (0.02 * rng.randn(D, D)).astype(np.float32),
        'Wk': (0.02 * rng.randn(D, D)).astype(np.float32),
        'Wv': (0.02 * rng.randn(D, D)).astype(np.float32),
        'Wo': (0.02 * rng.randn(D, D)).astype(np.float32),
        'ln2_gamma': np.ones(D, dtype=np.float32),
        'ln2_beta': np.zeros(D, dtype=np.float32),
        'W1': (0.02 * rng.randn(D, FF)).astype(np.float32),
        'W2': (0.02 * rng.randn(FF, D)).astype(np.float32),
    }


def weight_sizes(D: int, FF: int) -> dict:
    """Return the number of float32 elements for each weight tensor."""
    return {
        'ln1_gamma': D,
        'ln1_beta': D,
        'Wq': D * D,
        'Wk': D * D,
        'Wv': D * D,
        'Wo': D * D,
        'ln2_gamma': D,
        'ln2_beta': D,
        'W1': D * FF,
        'W2': FF * D,
    }


# ---------------------------------------------------------------------------
# C header generation
# ---------------------------------------------------------------------------

def float_to_c_hex(f: float) -> str:
    """Convert a float32 to a C hex-float literal for exact representation."""
    # Pack as float32, unpack as uint32, format as hex
    bits = struct.unpack('<I', struct.pack('<f', f))[0]
    # Reconstruct sign, exponent, mantissa for hex float
    # Simpler: just use the decimal repr with enough precision
    return f"{f:.8e}f"


def array_to_c_initializer(arr: np.ndarray, per_line: int = 8) -> str:
    """Convert a flat float32 numpy array to a C initializer string."""
    flat = arr.ravel().astype(np.float32)
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        vals = ", ".join(f"{v:.8e}f" for v in chunk)
        lines.append(f"  {vals},")
    return "\n".join(lines)


def generate_weights_header(layers_weights: list, x_input: np.ndarray,
                            golden_output: np.ndarray,
                            S: int, D: int, FF: int) -> str:
    """Generate weights.h content with static const arrays."""
    parts = []
    parts.append("/* Auto-generated by export_weights.py -- do not edit */")
    parts.append("#ifndef TRANSFORMER_WEIGHTS_H")
    parts.append("#define TRANSFORMER_WEIGHTS_H")
    parts.append("")
    parts.append(f"#define WEIGHT_SEQ   {S}")
    parts.append(f"#define WEIGHT_DIM   {D}")
    parts.append(f"#define WEIGHT_FF    {FF}")
    parts.append(f"#define WEIGHT_LAYERS {len(layers_weights)}")
    parts.append("")

    # Per-layer weight arrays
    for i, weights in enumerate(layers_weights):
        # Concatenate all weights in order into a single array
        all_data = []
        for key in WEIGHT_ORDER:
            all_data.append(weights[key].ravel())
        concat = np.concatenate(all_data)
        total = len(concat)

        parts.append(f"/* Layer {i}: {total} floats */")
        parts.append(f"static const float layer{i}_weights[{total}] = {{")
        parts.append(array_to_c_initializer(concat))
        parts.append("};")
        parts.append("")

    # Input array
    input_flat = x_input.ravel().astype(np.float32)
    parts.append(f"/* Input: [{S}, {D}] = {len(input_flat)} floats */")
    parts.append(f"static const float input_data[{len(input_flat)}] = {{")
    parts.append(array_to_c_initializer(input_flat))
    parts.append("};")
    parts.append("")

    # Golden output array
    golden_flat = golden_output.ravel().astype(np.float32)
    parts.append(f"/* Golden output: [{S}, {D}] = {len(golden_flat)} floats */")
    parts.append(f"static const float golden_output[{len(golden_flat)}] = {{")
    parts.append(array_to_c_initializer(golden_flat))
    parts.append("};")
    parts.append("")

    parts.append("#endif /* TRANSFORMER_WEIGHTS_H */")
    parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export transformer weights, input, and golden output"
    )
    parser.add_argument("--seq", type=int, required=True, help="Sequence length (S)")
    parser.add_argument("--dim", type=int, required=True, help="Model dimension (D)")
    parser.add_argument("--ff", type=int, required=True, help="Feed-forward dimension (FF)")
    parser.add_argument("--layers", type=int, required=True, help="Number of transformer layers")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    S = args.seq
    D = args.dim
    FF = args.ff
    N = args.layers
    out_dir = args.out_dir
    seed = args.seed

    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(seed)

    # Generate weights for all layers
    all_layers_weights = []
    for i in range(N):
        w = generate_layer_weights(D, FF, rng)
        all_layers_weights.append(w)

    # Generate random input
    x_input = (0.02 * rng.randn(S, D)).astype(np.float32)

    # Run numpy reference through all layers
    x = x_input.copy()
    for i in range(N):
        x = transformer_block(x, all_layers_weights[i])
    golden_output = x.astype(np.float32)

    # --- Write binary files ---

    # Per-layer weight binaries
    sizes = weight_sizes(D, FF)
    for i, weights in enumerate(all_layers_weights):
        path = os.path.join(out_dir, f"weights_layer{i}.bin")
        with open(path, "wb") as f:
            for key in WEIGHT_ORDER:
                f.write(weights[key].ravel().astype(np.float32).tobytes())
        print(f"Wrote {path} ({os.path.getsize(path)} bytes)")

    # Input binary
    input_path = os.path.join(out_dir, "input.bin")
    x_input.tofile(input_path)
    print(f"Wrote {input_path} ({os.path.getsize(input_path)} bytes)")

    # Golden output binary
    golden_path = os.path.join(out_dir, "golden_output.bin")
    golden_output.tofile(golden_path)
    print(f"Wrote {golden_path} ({os.path.getsize(golden_path)} bytes)")

    # --- Write C header ---
    header_content = generate_weights_header(
        all_layers_weights, x_input, golden_output, S, D, FF
    )
    header_path = os.path.join(out_dir, "weights.h")
    with open(header_path, "w") as f:
        f.write(header_content)
    print(f"Wrote {header_path}")

    # --- Write manifest ---
    manifest = {
        "seq": S,
        "dim": D,
        "ff": FF,
        "layers": N,
        "seed": seed,
        "weight_order": WEIGHT_ORDER,
        "sizes": sizes,
        "total_weights_per_layer": sum(sizes.values()),
        "files": {
            "weights": [f"weights_layer{i}.bin" for i in range(N)],
            "input": "input.bin",
            "golden_output": "golden_output.bin",
            "header": "weights.h",
        },
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")

    # Print summary
    print(f"\nSummary: seq={S}, dim={D}, ff={FF}, layers={N}, seed={seed}")
    print(f"  Weights per layer: {sum(sizes.values())} floats "
          f"({sum(sizes.values()) * 4} bytes)")
    print(f"  Input: {S * D} floats ({S * D * 4} bytes)")
    print(f"  Golden output: {S * D} floats ({S * D * 4} bytes)")
    print(f"  Golden output sample (first 8): {golden_output.ravel()[:8]}")


if __name__ == "__main__":
    main()
