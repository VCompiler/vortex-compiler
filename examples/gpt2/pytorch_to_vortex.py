#!/usr/bin/env python3
"""Export a simplified GPT-2 model from PyTorch to Vortex MLIR + simx.

Defines a single-head, pre-norm transformer in PyTorch, extracts weights,
runs a forward pass to produce golden logits, then calls gen_full_inference.py
to generate the MLIR module, C wrapper, and weights header.

Usage:
    source .venv/bin/activate
    python3 examples/gpt2/pytorch_to_vortex.py \
        --seq 32 --dim 64 --ff 256 --vocab 256 --layers 4 --seed 42 \
        --out-dir build/gpt2/pytorch_export

Then build and run on simx:
    bash examples/gpt2/run_full_inference.sh build/gpt2/pytorch_export
"""

import argparse
import json
import math
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Optional PyTorch import with helpful error
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print(
        "ERROR: PyTorch is required but not installed.\n"
        "Install it with:\n"
        "  pip install torch  # CPU-only is fine\n"
        "  # or: pip install torch --index-url https://download.pytorch.org/whl/cpu",
        file=sys.stderr,
    )
    sys.exit(1)

import numpy as np


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------
# This model matches EXACTLY the computation in the MLIR kernels generated
# by gen_full_inference.py:
#
#   - Pre-norm architecture (LayerNorm BEFORE attention/MLP, not after)
#   - LayerNorm with eps=1e-5, population variance (divide by N, not N-1)
#   - GeLU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))  (torch default, approximate='none')
#   - Single-head attention: Q=x@Wq, K=x@Wk, V=x@Wv, score=Q@K^T/sqrt(d),
#     softmax, attn=prob@V, out=attn@Wo
#   - No bias in any linear layers
#   - Residual connections: x = x + sublayer_output
# ---------------------------------------------------------------------------


class VortexLayerNorm(nn.Module):
    """LayerNorm using population variance (1/N), matching the MLIR kernel.

    PyTorch's nn.LayerNorm uses 1/N (population variance) by default,
    which matches the MLIR kernel exactly.  eps=1e-5.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, D]
        # Compute mean and variance over the last dimension (population variance)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)  # correction=0 => 1/N
        inv_std = 1.0 / torch.sqrt(var + self.eps)
        return (x - mean) * inv_std * self.gamma + self.beta


class VortexAttention(nn.Module):
    """Single-head self-attention with no bias, matching the MLIR kernel.

    Q = x @ Wq,  K = x @ Wk,  V = x @ Wv
    score = Q @ K^T / sqrt(D)
    prob = softmax(score, dim=-1)
    attn = prob @ V
    output = attn @ Wo
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Linear projections without bias -- weight shape [D, D] stored as (out, in)
        # but we do x @ W (not W @ x), so we use nn.Linear with transposed convention
        # Actually: nn.Linear stores weight as [out_features, in_features] and computes
        # x @ W^T + b.  We want x @ W, so we store W as a raw Parameter of shape [D, D]
        # and do the matmul manually.
        self.Wq = nn.Parameter(torch.empty(dim, dim))
        self.Wk = nn.Parameter(torch.empty(dim, dim))
        self.Wv = nn.Parameter(torch.empty(dim, dim))
        self.Wo = nn.Parameter(torch.empty(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, D]
        Q = x @ self.Wq  # [S, D]
        K = x @ self.Wk  # [S, D]
        V = x @ self.Wv  # [S, D]

        score = Q @ K.T / math.sqrt(self.dim)  # [S, S]
        prob = torch.softmax(score, dim=-1)     # [S, S]
        attn = prob @ V                         # [S, D]
        return attn @ self.Wo                   # [S, D]


class VortexMLP(nn.Module):
    """Feed-forward MLP: GeLU(x @ W1) @ W2, no bias, matching the MLIR kernel.

    W1: [D, FF]   -- up-projection
    W2: [FF, D]   -- down-projection
    GeLU: exact (not approximate)
    """

    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(dim, ff_dim))
        self.W2 = nn.Parameter(torch.empty(ff_dim, dim))
        # GeLU with approximate='none' matches x * 0.5 * (1 + erf(x / sqrt(2)))
        self.gelu = nn.GELU(approximate="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, D]
        h = x @ self.W1       # [S, FF]
        h = self.gelu(h)      # [S, FF]
        return h @ self.W2    # [S, D]


class VortexTransformerBlock(nn.Module):
    """Single pre-norm transformer block, matching the MLIR kernel.

    x_ln  = LayerNorm(x)           -- pre-norm
    attn  = Attention(x_ln)
    x     = x + attn               -- residual
    x_ln2 = LayerNorm(x)           -- pre-norm
    ff    = MLP(x_ln2)
    x     = x + ff                 -- residual
    """

    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.ln1 = VortexLayerNorm(dim)
        self.attn = VortexAttention(dim)
        self.ln2 = VortexLayerNorm(dim)
        self.mlp = VortexMLP(dim, ff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        x_ln = self.ln1(x)
        x = x + self.attn(x_ln)

        # Pre-norm MLP with residual
        x_ln2 = self.ln2(x)
        x = x + self.mlp(x_ln2)

        return x


class VortexGPT2(nn.Module):
    """Simplified GPT-2: embedding -> N transformer blocks -> LM head.

    Matches the full inference pipeline in gen_full_inference.py:
      1. Token embedding + position embedding (lookup + add)
      2. N x transformer blocks (pre-norm, single-head attention, GeLU MLP)
      3. Final LayerNorm + linear projection to vocab (no bias)
    """

    def __init__(self, seq_len: int, dim: int, ff_dim: int, vocab: int,
                 num_layers: int):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.vocab = vocab

        # Embedding tables
        self.tok_embed = nn.Embedding(vocab, dim)
        self.pos_embed = nn.Embedding(seq_len, dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            VortexTransformerBlock(dim, ff_dim) for _ in range(num_layers)
        ])

        # Final LayerNorm + LM head projection
        self.final_ln = VortexLayerNorm(dim)
        self.lm_head = nn.Parameter(torch.empty(dim, vocab))  # [D, V]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [S] int tensor
        returns: logits [S, V]
        """
        S = token_ids.shape[0]

        # 1. Embedding: token lookup + position lookup
        pos_ids = torch.arange(S, device=token_ids.device)
        x = self.tok_embed(token_ids) + self.pos_embed(pos_ids)  # [S, D]

        # 2. Transformer blocks
        for layer in self.layers:
            x = layer(x)

        # 3. LM head: final LayerNorm + vocab projection
        x = self.final_ln(x)       # [S, D]
        logits = x @ self.lm_head   # [S, V]

        return logits


# ---------------------------------------------------------------------------
# Weight initialization -- must match gen_full_inference.py's RNG sequence
# ---------------------------------------------------------------------------

def init_weights(model: VortexGPT2, seed: int):
    """Initialize all weights using torch.manual_seed(seed) and N(0, 0.02).

    LayerNorm gamma = 1, beta = 0 (already set by VortexLayerNorm.__init__).
    All projection matrices: N(0, 0.02).
    Embedding tables: N(0, 0.02).
    """
    torch.manual_seed(seed)

    # Initialize embedding tables
    nn.init.normal_(model.tok_embed.weight, std=0.02)
    nn.init.normal_(model.pos_embed.weight, std=0.02)

    # Initialize transformer block weights
    for layer in model.layers:
        # LayerNorm: gamma=1, beta=0 (already initialized)
        # Re-set explicitly to be safe
        layer.ln1.gamma.data.fill_(1.0)
        layer.ln1.beta.data.fill_(0.0)

        # Attention projection matrices
        nn.init.normal_(layer.attn.Wq, std=0.02)
        nn.init.normal_(layer.attn.Wk, std=0.02)
        nn.init.normal_(layer.attn.Wv, std=0.02)
        nn.init.normal_(layer.attn.Wo, std=0.02)

        # LayerNorm 2
        layer.ln2.gamma.data.fill_(1.0)
        layer.ln2.beta.data.fill_(0.0)

        # MLP projection matrices
        nn.init.normal_(layer.mlp.W1, std=0.02)
        nn.init.normal_(layer.mlp.W2, std=0.02)

    # Final LayerNorm
    model.final_ln.gamma.data.fill_(1.0)
    model.final_ln.beta.data.fill_(0.0)

    # LM head projection
    nn.init.normal_(model.lm_head, std=0.02)


# ---------------------------------------------------------------------------
# Weight extraction and packing
# ---------------------------------------------------------------------------

WEIGHT_ORDER = [
    'ln1_gamma', 'ln1_beta',
    'Wq', 'Wk', 'Wv', 'Wo',
    'ln2_gamma', 'ln2_beta',
    'W1', 'W2',
]


def extract_layer_weights(layer: VortexTransformerBlock) -> dict:
    """Extract weight tensors from a VortexTransformerBlock as numpy arrays."""
    return {
        'ln1_gamma':  layer.ln1.gamma.detach().cpu().numpy().astype(np.float32),
        'ln1_beta':   layer.ln1.beta.detach().cpu().numpy().astype(np.float32),
        'Wq':         layer.attn.Wq.detach().cpu().numpy().astype(np.float32),
        'Wk':         layer.attn.Wk.detach().cpu().numpy().astype(np.float32),
        'Wv':         layer.attn.Wv.detach().cpu().numpy().astype(np.float32),
        'Wo':         layer.attn.Wo.detach().cpu().numpy().astype(np.float32),
        'ln2_gamma':  layer.ln2.gamma.detach().cpu().numpy().astype(np.float32),
        'ln2_beta':   layer.ln2.beta.detach().cpu().numpy().astype(np.float32),
        'W1':         layer.mlp.W1.detach().cpu().numpy().astype(np.float32),
        'W2':         layer.mlp.W2.detach().cpu().numpy().astype(np.float32),
    }


# ---------------------------------------------------------------------------
# C header generation  (matches gen_full_inference.py format exactly)
# ---------------------------------------------------------------------------

def array_to_c_initializer(arr: np.ndarray, per_line: int = 8) -> str:
    """Convert a flat float32 numpy array to a C initializer string."""
    flat = arr.ravel().astype(np.float32)
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        vals = ", ".join(f"{v:.8e}f" for v in chunk)
        lines.append(f"  {vals},")
    return "\n".join(lines)


def int_array_to_c_initializer(arr: np.ndarray, per_line: int = 16) -> str:
    flat = arr.ravel()
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        vals = ", ".join(str(int(v)) for v in chunk)
        lines.append(f"  {vals},")
    return "\n".join(lines)


def gen_weights_header(token_ids: np.ndarray, tok_table: np.ndarray,
                       pos_table: np.ndarray,
                       layers_weights: list,
                       final_ln_gamma: np.ndarray, final_ln_beta: np.ndarray,
                       lm_head_w: np.ndarray,
                       golden_logits_arr: np.ndarray,
                       S: int, D: int, FF: int, V: int) -> str:
    """Generate full_inference_weights.h -- same format as gen_full_inference.py."""
    parts = []
    parts.append("/* Auto-generated by pytorch_to_vortex.py -- do not edit */")
    parts.append("#ifndef FULL_INFERENCE_WEIGHTS_H")
    parts.append("#define FULL_INFERENCE_WEIGHTS_H")
    parts.append("")

    # Input token IDs
    parts.append(f"/* Input token IDs: [{S}] */")
    parts.append(f"static const int input_token_ids[{S}] = {{")
    parts.append(int_array_to_c_initializer(token_ids))
    parts.append("};")
    parts.append("")

    # Token embedding table
    parts.append(f"/* Token embedding table: [{V}, {D}] = {V * D} floats */")
    parts.append(f"static const float tok_embed_table[{V * D}] = {{")
    parts.append(array_to_c_initializer(tok_table))
    parts.append("};")
    parts.append("")

    # Position embedding table
    parts.append(f"/* Position embedding table: [{S}, {D}] = {S * D} floats */")
    parts.append(f"static const float pos_embed_table[{S * D}] = {{")
    parts.append(array_to_c_initializer(pos_table))
    parts.append("};")
    parts.append("")

    # Per-layer weights
    for i, weights in enumerate(layers_weights):
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

    # Final LayerNorm
    parts.append(f"/* Final LayerNorm gamma: [{D}] */")
    parts.append(f"static const float final_ln_gamma[{D}] = {{")
    parts.append(array_to_c_initializer(final_ln_gamma))
    parts.append("};")
    parts.append("")

    parts.append(f"/* Final LayerNorm beta: [{D}] */")
    parts.append(f"static const float final_ln_beta[{D}] = {{")
    parts.append(array_to_c_initializer(final_ln_beta))
    parts.append("};")
    parts.append("")

    # LM head projection
    parts.append(f"/* LM head projection: [{D}, {V}] = {D * V} floats */")
    parts.append(f"static const float lm_head_proj[{D * V}] = {{")
    parts.append(array_to_c_initializer(lm_head_w))
    parts.append("};")
    parts.append("")

    # Golden logits
    golden_flat = golden_logits_arr.ravel().astype(np.float32)
    parts.append(f"/* Golden logits: [{S}, {V}] = {len(golden_flat)} floats */")
    parts.append(f"static const float golden_logits[{len(golden_flat)}] = {{")
    parts.append(array_to_c_initializer(golden_flat))
    parts.append("};")
    parts.append("")

    parts.append("#endif /* FULL_INFERENCE_WEIGHTS_H */")
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export a PyTorch GPT-2 model to Vortex MLIR + simx"
    )
    parser.add_argument("--seq", type=int, required=True,
                        help="Sequence length (S)")
    parser.add_argument("--dim", type=int, required=True,
                        help="Model dimension (D)")
    parser.add_argument("--ff", type=int, required=True,
                        help="Feed-forward dimension (FF)")
    parser.add_argument("--vocab", type=int, required=True,
                        help="Vocabulary size (V)")
    parser.add_argument("--layers", type=int, required=True,
                        help="Number of transformer layers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for generated files")
    parser.add_argument("--external-weights", action="store_true",
                        help="Generate binary weight files + host driver instead of embedded weights.h")
    args = parser.parse_args()

    S = args.seq
    D = args.dim
    FF = args.ff
    V = args.vocab
    N = args.layers
    seed = args.seed
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Build and initialize the PyTorch model
    # ------------------------------------------------------------------

    print(f"Building VortexGPT2 model: seq={S}, dim={D}, ff={FF}, "
          f"vocab={V}, layers={N}, seed={seed}")

    model = VortexGPT2(
        seq_len=S, dim=D, ff_dim=FF, vocab=V, num_layers=N
    )
    init_weights(model, seed)
    model.eval()

    # ------------------------------------------------------------------
    # Step 2: Generate random token IDs as input
    # ------------------------------------------------------------------

    torch.manual_seed(seed + 1000)  # different seed for input tokens
    token_ids = torch.randint(0, V, (S,), dtype=torch.int32)

    print(f"Input token IDs (first 8): {token_ids[:8].tolist()}")

    # ------------------------------------------------------------------
    # Step 3: Run forward pass to get golden logits
    # ------------------------------------------------------------------

    with torch.no_grad():
        logits = model(token_ids.long())

    golden_logits = logits.cpu().numpy().astype(np.float32)
    print(f"Golden logits shape: {golden_logits.shape}")
    print(f"Golden logits sample (first 8): {golden_logits.ravel()[:8]}")

    # ------------------------------------------------------------------
    # Step 4: Extract all weights as numpy arrays
    # ------------------------------------------------------------------

    # Token embedding: [V, D]
    tok_table = model.tok_embed.weight.detach().cpu().numpy().astype(np.float32)

    # Position embedding: [S, D]
    pos_table = model.pos_embed.weight.detach().cpu().numpy().astype(np.float32)

    # Per-layer transformer weights
    all_layers_weights = []
    for i, layer in enumerate(model.layers):
        w = extract_layer_weights(layer)
        all_layers_weights.append(w)

    # Final LayerNorm
    final_ln_gamma = model.final_ln.gamma.detach().cpu().numpy().astype(np.float32)
    final_ln_beta = model.final_ln.beta.detach().cpu().numpy().astype(np.float32)

    # LM head: [D, V]
    lm_head_w = model.lm_head.detach().cpu().numpy().astype(np.float32)

    # Token IDs as numpy
    token_ids_np = token_ids.numpy().astype(np.int32)

    # ------------------------------------------------------------------
    # Step 5: Write weights header (same format as gen_full_inference.py)
    # ------------------------------------------------------------------

    header_content = gen_weights_header(
        token_ids_np, tok_table, pos_table,
        all_layers_weights,
        final_ln_gamma, final_ln_beta,
        lm_head_w, golden_logits,
        S, D, FF, V
    )
    header_path = os.path.join(out_dir, "full_inference_weights.h")
    with open(header_path, "w") as f:
        f.write(header_content)
    print(f"Wrote {header_path}")

    # ------------------------------------------------------------------
    # Step 6: Call gen_full_inference.py to generate MLIR + C wrapper
    #
    # gen_full_inference.py generates its own weights internally using
    # numpy with the same seed.  We only need the MLIR and C wrapper from
    # it, so we call it and then overwrite the weights header with ours.
    #
    # Actually, gen_full_inference.py generates MLIR, wrapper, AND weights
    # as a bundle.  The MLIR and wrapper only depend on dimensions, not
    # weights.  So we call it to get the MLIR + wrapper, then overwrite
    # the weights header with our PyTorch-derived one.
    # ------------------------------------------------------------------

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gen_script = os.path.join(script_dir, "gen_full_inference.py")

    if not os.path.exists(gen_script):
        print(f"ERROR: Cannot find {gen_script}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, gen_script,
        "--seq", str(S),
        "--dim", str(D),
        "--ff", str(FF),
        "--vocab", str(V),
        "--layers", str(N),
        "--seed", str(seed),
        "--out-dir", out_dir,
    ]
    if args.external_weights:
        cmd.append("--external-weights")
    print(f"\nCalling gen_full_inference.py to generate MLIR + C wrapper...")
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: gen_full_inference.py failed:\n{result.stderr}",
              file=sys.stderr)
        sys.exit(1)
    print(result.stdout)

    # Overwrite weights with PyTorch-derived version
    if args.external_weights:
        # Write binary weights blob
        script_dir_for_import = os.path.dirname(os.path.abspath(__file__))
        if script_dir_for_import not in sys.path:
            sys.path.insert(0, script_dir_for_import)
        from gen_full_inference import write_weights_bin
        weights_bin_path = os.path.join(out_dir, "weights.bin")
        write_weights_bin(weights_bin_path, token_ids_np, tok_table, pos_table,
                          all_layers_weights,
                          final_ln_gamma, final_ln_beta, lm_head_w)
        print(f"Overwrote {weights_bin_path} with PyTorch-derived weights")

        # Write golden logits binary
        golden_bin_path = os.path.join(out_dir, "golden.bin")
        with open(golden_bin_path, "wb") as f:
            f.write(golden_logits.ravel().astype(np.float32).tobytes())
        print(f"Overwrote {golden_bin_path} with PyTorch-derived golden")
    else:
        with open(header_path, "w") as f:
            f.write(header_content)
        print(f"Overwrote {header_path} with PyTorch-derived weights")

    # ------------------------------------------------------------------
    # Step 7: Write manifest
    # ------------------------------------------------------------------

    layer_weights_count = (2 * D + 4 * D * D + 2 * D + D * FF + FF * D)
    total_weights = (V * D + S * D + N * layer_weights_count + D + D + D * V)

    manifest = {
        "source": "pytorch_to_vortex.py",
        "seq": S,
        "dim": D,
        "ff": FF,
        "vocab": V,
        "layers": N,
        "seed": seed,
        "weight_order": WEIGHT_ORDER,
        "total_weight_floats": total_weights,
        "golden_logits_shape": list(golden_logits.shape),
        "external_weights": args.external_weights,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print(f"\n{'='*60}")
    print(f"PyTorch -> Vortex export complete!")
    print(f"{'='*60}")
    print(f"  Model:  seq={S}, dim={D}, ff={FF}, vocab={V}, layers={N}")
    print(f"  Seed:   {seed}")
    print(f"  Mode:   {'external weights' if args.external_weights else 'embedded weights'}")
    print(f"  Output: {out_dir}/")
    if args.external_weights:
        print(f"    full_inference.mlir      -- MLIR module")
        print(f"    gpt2_kernel.c            -- kernel code (RISC-V)")
        print(f"    gpt2_host.cpp            -- host driver (x86)")
        print(f"    gpt2_common.h            -- shared header")
        print(f"    weights.bin              -- binary weights")
        print(f"    golden.bin               -- golden logits")
    else:
        print(f"    full_inference.mlir          -- MLIR module")
        print(f"    full_inference_wrapper.c     -- C wrapper")
        print(f"    full_inference_weights.h     -- weights + golden logits")
    print(f"    manifest.json")
    print(f"")
    print(f"  Total weight floats: {total_weights}")
    print(f"  Golden logits: {golden_logits.shape}")
    print(f"  Golden logits sample: {golden_logits.ravel()[:8]}")
    print(f"")
    if args.external_weights:
        print(f"To build and run on simx:")
        print(f"  bash examples/gpt2/run_external_weights.sh {out_dir}")
    else:
        print(f"To build and run on simx:")
        print(f"  bash examples/gpt2/run_full_inference.sh {out_dir}")


if __name__ == "__main__":
    main()
