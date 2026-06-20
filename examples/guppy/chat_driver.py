#!/usr/bin/env python3
"""Host-side driver: 上板执行 Guppy，并回读下一个 token。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def format_prompt(messages: list[dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def safe_name(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    return text.strip("_") or "run"


def run_and_log(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> int:
    log_f = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log_f is not None:
                log_f.write(line)
        return proc.wait()
    finally:
        if log_f is not None:
            log_f.close()


def parse_kernel_manifest(path: Path) -> tuple[str, list[str]]:
    entry_line = ""
    segment_lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        kind = line.split()[0].upper()
        if kind == "ENTRY":
            entry_line = line
        elif kind == "SEGMENT":
            segment_lines.append(line)
        else:
            raise RuntimeError(f"Unsupported line in kernel manifest {path}: {line}")
    if not entry_line:
        raise RuntimeError(f"Missing ENTRY in kernel manifest {path}")
    if not segment_lines:
        raise RuntimeError(f"Missing SEGMENT lines in kernel manifest {path}")
    return entry_line, segment_lines


def parse_run_summary(run_log: Path) -> tuple[str, str]:
    run_pass = "UNKNOWN"
    final_exit = "UNKNOWN"
    for raw in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip().rstrip("\r")
        if line.startswith("RUN_PASS="):
            run_pass = line.split("=", 1)[1]
        elif line.startswith("FINAL_EXIT_WORD="):
            final_exit = line.split("=", 1)[1]
    return run_pass, final_exit


def parse_dump_words(dump_log: Path) -> list[str]:
    words: list[str] = []
    for raw in dump_log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip().rstrip("\r")
        if not line.startswith("READBACK_WORD "):
            continue
        for field in line.split():
            if field.startswith("data=0x"):
                words.append(field.split("0x", 1)[1].upper())
                break
    return words


def write_mem_words(path: Path, words: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{word}\n" for word in words), encoding="utf-8")


def default_platform_root(repo_root: Path) -> Path:
    env = os.environ.get("VORTEX_PLATFORM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    sibling = repo_root.parent / "vortex-platform"
    if sibling.is_dir():
        return sibling.resolve()
    return Path("/home/user/vortex-platform")


def find_default_board_runner(repo_root: Path) -> Path:
    return (
        default_platform_root(repo_root)
        / "hw"
        / "syn"
        / "xilinx"
        / "xc7k480t"
        / "run_regression_manifest_jtag.py"
    )


def find_default_xdma_runner(repo_root: Path) -> Path:
    return (
        default_platform_root(repo_root)
        / "hw"
        / "syn"
        / "xilinx"
        / "xc7k480t"
        / "run_regression_manifest_xdma.py"
    )


def find_default_board_scripts_dir(repo_root: Path) -> Path:
    return (
        default_platform_root(repo_root)
        / "hw"
        / "syn"
        / "xilinx"
        / "xc7k480t"
    )


def build_vivado_cmd(vivado_settings_sh: str | None, tcl_path: Path) -> list[str]:
    if vivado_settings_sh:
        command = (
            f"source {shlex.quote(vivado_settings_sh)} && "
            f"vivado -mode batch -source {shlex.quote(str(tcl_path))} -notrace"
        )
        return ["/bin/bash", "-lc", command]
    return ["vivado", "-mode", "batch", "-source", str(tcl_path), "-notrace"]


def run_local_vivado_tcl(
    *,
    board_scripts_dir: Path,
    tcl_name: str,
    vivado_settings_sh: str | None,
    vivado_env: dict[str, str],
    log_path: Path,
) -> int:
    tcl_path = board_scripts_dir / tcl_name
    if not tcl_path.is_file():
        raise FileNotFoundError(f"Missing board TCL: {tcl_path}")
    env = os.environ.copy()
    env.update(vivado_env)
    return run_and_log(
        build_vivado_cmd(vivado_settings_sh, tcl_path),
        cwd=board_scripts_dir,
        env=env,
        log_path=log_path,
    )


def find_tool(name: str, candidates: list[Path]) -> str:
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"未找到工具: {name}")


def parse_symbols(elf_path: Path, llvm_nm: str) -> dict[str, int]:
    result = subprocess.run(
        [llvm_nm, "--defined-only", str(elf_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    symbols: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            addr = int(parts[0], 16)
        except ValueError:
            continue
        symbols[parts[-1]] = addr
    return symbols


def int_to_word(value: int) -> str:
    return f"{value & 0xFFFFFFFF:08X}"


def ints_to_words(values: list[int]) -> list[str]:
    return [int_to_word(value) for value in values]


def words_to_ints(words: list[str]) -> list[int]:
    return [int(word.strip(), 16) for word in words if word.strip()]


STAGE0_PROFILE_MAGIC = 0x47505330
STAGE0_PROFILE_NAMES = {
    1: "entry",
    20: "embedding_done",
    21: "ln1_done",
    22: "qkv_matmul_done",
    23: "qkv_unpack_done",
    24: "score_softmax_done",
    25: "attn_value_done",
    240: "attn_merge_done",
    241: "spawn_return_done",
    100: "lane_probe_0",
    101: "lane_probe_1",
    102: "lane_probe_2",
    103: "lane_probe_3",
}


def parse_stage0_profile_mem(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path)}

    words = words_to_ints(path.read_text(encoding="utf-8").splitlines())
    if len(words) < 2:
        return {"available": False, "path": str(path), "error": "too_few_words"}

    magic = words[0]
    count = max(0, min(words[1], 15))
    layout_version = words[2] if len(words) > 2 else 0
    if layout_version == 4:
        def profile_slot_base(idx: int) -> int:
            if idx < 8:
                return 96 + idx * 4
            if idx == 8:
                return 84
            return 128 + (idx - 9) * 4
    elif layout_version == 3:
        slot_base_offset = 52
        def profile_slot_base(idx: int) -> int:
            return slot_base_offset + idx * 4
    elif layout_version == 2:
        slot_base_offset = 16
        def profile_slot_base(idx: int) -> int:
            return slot_base_offset + idx * 4
    else:
        slot_base_offset = 2
        def profile_slot_base(idx: int) -> int:
            return slot_base_offset + idx * 4
    events: list[dict[str, Any]] = []
    for idx in range(count):
        base = profile_slot_base(idx)
        if base + 3 >= len(words):
            break
        code = words[base + 0]
        cycle = (words[base + 2] << 32) | words[base + 1]
        info = words[base + 3]
        event: dict[str, Any] = {
            "slot": idx,
            "code": code,
            "name": STAGE0_PROFILE_NAMES.get(code, f"code_{code}"),
            "cycle": cycle,
            "raw_info": f"0x{info:08X}",
            "progress_stage": info & 0xFF,
            "thread_idx_x": (info >> 8) & 0xFF,
            "thread_id": (info >> 16) & 0xFF,
            "warp_id": (info >> 24) & 0x0F,
            "core_id": (info >> 28) & 0x0F,
        }
        if events:
            event["delta_cycles"] = cycle - int(events[-1]["cycle"])
        events.append(event)

    return {
        "available": magic == STAGE0_PROFILE_MAGIC,
        "path": str(path),
        "magic": f"0x{magic:08X}",
        "layout_version": layout_version,
        "event_count": count,
        "events": events,
    }


def append_warp4_attn_out_outputs(outputs: list[dict[str, Any]], symbols: dict[str, int]) -> None:
    optional_outputs = [
        ("warp4_attn_out_status", "guppy_warp4_attn_out_status", 1),
        ("warp4_attn_out_num_threads", "guppy_warp4_attn_out_num_threads", 1),
        ("warp4_attn_out_expected_mask", "guppy_warp4_attn_out_expected_mask", 1),
        ("warp4_attn_out_nonzero_mask", "guppy_warp4_attn_out_nonzero_mask", 1),
        ("warp4_attn_out_task_count", "guppy_warp4_attn_out_task_count", 4),
    ]
    for output_name, symbol_name, words in optional_outputs:
        if symbol_name in symbols:
            outputs.append(
                {
                    "name": output_name,
                    "addr": f"0x{symbols[symbol_name]:08X}",
                    "words": words,
                }
            )


def append_xdma_checkpoint_outputs(
    outputs: list[dict[str, Any]],
    symbols: dict[str, int],
    *,
    request_base: dict[str, Any],
    seq_len: int,
    token_ids: list[int],
) -> None:
    checkpoint_stage = request_base.get("xdma_checkpoint_stage")
    if checkpoint_stage is None:
        return
    d_model = int(request_base.get("xdma_d_model", 0))
    if d_model <= 0:
        return
    row = max(0, min(len(token_ids), seq_len) - 1)
    row_words = d_model
    row_byte_offset = row * d_model * 4
    if int(checkpoint_stage) >= 252:
        for output_name, symbol_name in (
            ("checkpoint_attn_merge_row", "g_attn_merge"),
            ("checkpoint_attn_out_row", "g_attn_out"),
            ("checkpoint_x_next_row", "g_x_next"),
        ):
            if symbol_name in symbols:
                outputs.append(
                    {
                        "name": output_name,
                        "addr": f"0x{symbols[symbol_name] + row_byte_offset:08X}",
                        "words": row_words,
                    }
                )
    if int(checkpoint_stage) >= 253 and "g_x_ln2" in symbols:
        outputs.append(
            {
                "name": "checkpoint_x_ln2_row",
                "addr": f"0x{symbols['g_x_ln2'] + row_byte_offset:08X}",
                "words": row_words,
            }
        )
    if int(checkpoint_stage) >= 254 and "g_hidden" in symbols:
        ffn_hidden = int(request_base.get("xdma_ffn_hidden", 0))
        if ffn_hidden > 0:
            outputs.append(
                {
                    "name": "checkpoint_hidden_row",
                    "addr": f"0x{symbols['g_hidden'] + row * ffn_hidden * 4:08X}",
                    "words": ffn_hidden,
                }
            )
    if 51 <= int(checkpoint_stage) <= 52 and "g_lm_one_ln_out" in symbols:
        outputs.append(
            {
                "name": "checkpoint_lm_ln_out",
                "addr": f"0x{symbols['g_lm_one_ln_out']:08X}",
                "words": row_words,
            }
        )
    if int(checkpoint_stage) == 52 and "guppy_output_last_token_logits" in symbols:
        vocab_size = int(request_base.get("xdma_vocab_size", 0))
        if vocab_size > 0:
            outputs.append(
                {
                    "name": "checkpoint_logits",
                    "addr": f"0x{symbols['guppy_output_last_token_logits']:08X}",
                    "words": vocab_size,
                }
            )


def words_to_f32(words: list[str]) -> list[float]:
    values = []
    for word in words:
        clean = word.strip()
        if not clean:
            continue
        raw = int(clean, 16).to_bytes(4, byteorder="big", signed=False)
        values.append(struct.unpack(">f", raw)[0])
    return values


def f32_to_words(values: Any) -> list[str]:
    words: list[str] = []
    for value in values:
        raw = struct.pack(">f", float(value))
        words.append(f"{int.from_bytes(raw, byteorder='big', signed=False):08X}")
    return words


def compute_split_post_attn_host(
    *,
    request_base: dict[str, Any],
    seq_len: int,
    vocab_size: int,
    token_ids: list[int],
    padded_ids: list[int],
    attn_row_words: list[str],
) -> tuple[int, Any]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PCIe split workaround 需要 numpy 做 host-side post-attn") from exc

    guppy_dir = Path(__file__).resolve().parent
    if str(guppy_dir) not in sys.path:
        sys.path.insert(0, str(guppy_dir))
    from gen_full_inference import BundleLoader, ffn_ref, layernorm_ref, linear_ref

    layer_limit = int(request_base.get("xdma_split_layer_limit", 1))
    if layer_limit != 1:
        raise RuntimeError(
            "PCIe split host-side post-attn workaround 当前只支持 layer_limit=1"
        )

    bundle_dir = Path(str(request_base["bundle_dir"])).expanduser().resolve()
    bundle = BundleLoader(bundle_dir)
    cfg = bundle.model_config["normalized_config"]
    d_model = int(request_base["xdma_split_d_model"])
    if int(cfg["vocab_size"]) != vocab_size:
        raise RuntimeError(
            f"vocab_size mismatch: config={cfg['vocab_size']} runtime={vocab_size}"
        )

    row = max(0, min(len(token_ids), seq_len) - 1)
    token_id = int(padded_ids[row])
    if token_id < 0 or token_id >= vocab_size:
        token_id = 0

    attn_merge_row = np.asarray(words_to_f32(attn_row_words), dtype=np.float32)
    if attn_merge_row.size != d_model:
        raise RuntimeError(
            f"attn_merge_row size mismatch: expected={d_model} actual={attn_merge_row.size}"
        )

    tok_emb = bundle.load_tensor("tok_emb.weight").astype(np.float32)
    pos_emb = bundle.load_tensor("pos_emb.weight").astype(np.float32)[:seq_len]
    prefix = "blocks.0"
    attn_out = linear_ref(
        attn_merge_row.reshape(1, d_model),
        bundle.load_tensor(f"{prefix}.attn.out.weight").astype(np.float32),
        bundle.load_tensor(f"{prefix}.attn.out.bias").astype(np.float32),
    ).astype(np.float32)[0]
    x_next = (tok_emb[token_id] + pos_emb[row] + attn_out).astype(np.float32)

    x_ln2 = layernorm_ref(
        x_next.reshape(1, d_model),
        bundle.load_tensor(f"{prefix}.norm2.weight").astype(np.float32),
        bundle.load_tensor(f"{prefix}.norm2.bias").astype(np.float32),
    ).astype(np.float32)
    ffn_out = ffn_ref(
        x_ln2,
        bundle.load_tensor(f"{prefix}.ffn.up.weight").astype(np.float32),
        bundle.load_tensor(f"{prefix}.ffn.up.bias").astype(np.float32),
        bundle.load_tensor(f"{prefix}.ffn.down.weight").astype(np.float32),
        bundle.load_tensor(f"{prefix}.ffn.down.bias").astype(np.float32),
    ).astype(np.float32)[0]
    x_next = (x_next + ffn_out).astype(np.float32)

    lm_input = layernorm_ref(
        x_next.reshape(1, d_model),
        bundle.load_tensor("norm.weight").astype(np.float32),
        bundle.load_tensor("norm.bias").astype(np.float32),
    ).astype(np.float32)
    logits = linear_ref(lm_input, tok_emb, None).astype(np.float32)[0]
    return int(np.argmax(logits)), logits


def compute_split_layer_tail_from_attn_merge_host(
    *,
    request_base: dict[str, Any],
    seq_len: int,
    vocab_size: int,
    token_ids: list[int],
    padded_ids: list[int],
    attn_merge_words: list[str],
) -> tuple[int, Any]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PCIe split workaround 需要 numpy 做 host-side layer tail") from exc

    guppy_dir = Path(__file__).resolve().parent
    if str(guppy_dir) not in sys.path:
        sys.path.insert(0, str(guppy_dir))
    from gen_full_inference import BundleLoader, block_ref, ffn_ref, layernorm_ref, linear_ref

    layer_limit = int(request_base.get("xdma_split_layer_limit", 1))
    if layer_limit <= 1:
        raise RuntimeError("attn-merge tail split 只用于 layer_limit > 1")

    bundle_dir = Path(str(request_base["bundle_dir"])).expanduser().resolve()
    bundle = BundleLoader(bundle_dir)
    cfg = bundle.model_config["normalized_config"]
    d_model = int(request_base["xdma_split_d_model"])
    if int(cfg["vocab_size"]) != vocab_size:
        raise RuntimeError(
            f"vocab_size mismatch: config={cfg['vocab_size']} runtime={vocab_size}"
        )

    attn_merge = np.asarray(words_to_f32(attn_merge_words), dtype=np.float32)
    expected_words = seq_len * d_model
    if attn_merge.size != expected_words:
        raise RuntimeError(
            f"split attn_merge word count mismatch: "
            f"expected={expected_words} actual={attn_merge.size}"
        )
    attn_merge = attn_merge.reshape(seq_len, d_model).astype(np.float32)

    tok_emb = bundle.load_tensor("tok_emb.weight").astype(np.float32)
    pos_emb = bundle.load_tensor("pos_emb.weight").astype(np.float32)[:seq_len]
    ids = np.asarray(padded_ids[:seq_len], dtype=np.int64)
    if ids.size != seq_len:
        padded = np.zeros(seq_len, dtype=np.int64)
        padded[: ids.size] = ids
        ids = padded
    ids[(ids < 0) | (ids >= vocab_size)] = 0

    x = (tok_emb[ids] + pos_emb).astype(np.float32)
    prefix = "blocks.0"
    attn_out = linear_ref(
        attn_merge,
        bundle.load_tensor(f"{prefix}.attn.out.weight").astype(np.float32),
        bundle.load_tensor(f"{prefix}.attn.out.bias").astype(np.float32),
    ).astype(np.float32)
    x = (x + attn_out).astype(np.float32)

    x_ln2 = layernorm_ref(
        x,
        bundle.load_tensor(f"{prefix}.norm2.weight").astype(np.float32),
        bundle.load_tensor(f"{prefix}.norm2.bias").astype(np.float32),
    ).astype(np.float32)
    ffn_out = ffn_ref(
        x_ln2,
        bundle.load_tensor(f"{prefix}.ffn.up.weight").astype(np.float32),
        bundle.load_tensor(f"{prefix}.ffn.up.bias").astype(np.float32),
        bundle.load_tensor(f"{prefix}.ffn.down.weight").astype(np.float32),
        bundle.load_tensor(f"{prefix}.ffn.down.bias").astype(np.float32),
    ).astype(np.float32)
    x = (x + ffn_out).astype(np.float32)

    for layer_idx in range(1, layer_limit):
        prefix = f"blocks.{layer_idx}"
        layer = {
            "norm1.weight": bundle.load_tensor(f"{prefix}.norm1.weight").astype(np.float32),
            "norm1.bias": bundle.load_tensor(f"{prefix}.norm1.bias").astype(np.float32),
            "attn.qkv.weight": bundle.load_tensor(f"{prefix}.attn.qkv.weight").astype(np.float32),
            "attn.qkv.bias": bundle.load_tensor(f"{prefix}.attn.qkv.bias").astype(np.float32),
            "attn.out.weight": bundle.load_tensor(f"{prefix}.attn.out.weight").astype(np.float32),
            "attn.out.bias": bundle.load_tensor(f"{prefix}.attn.out.bias").astype(np.float32),
            "norm2.weight": bundle.load_tensor(f"{prefix}.norm2.weight").astype(np.float32),
            "norm2.bias": bundle.load_tensor(f"{prefix}.norm2.bias").astype(np.float32),
            "ffn.up.weight": bundle.load_tensor(f"{prefix}.ffn.up.weight").astype(np.float32),
            "ffn.up.bias": bundle.load_tensor(f"{prefix}.ffn.up.bias").astype(np.float32),
            "ffn.down.weight": bundle.load_tensor(f"{prefix}.ffn.down.weight").astype(np.float32),
            "ffn.down.bias": bundle.load_tensor(f"{prefix}.ffn.down.bias").astype(np.float32),
        }
        x = block_ref(x, layer, int(cfg["n_heads"])).astype(np.float32)

    row = max(0, min(len(token_ids), seq_len) - 1)
    lm_input = layernorm_ref(
        x[row : row + 1],
        bundle.load_tensor("norm.weight").astype(np.float32),
        bundle.load_tensor("norm.bias").astype(np.float32),
    ).astype(np.float32)
    logits = linear_ref(lm_input, tok_emb, None).astype(np.float32)[0]
    return int(np.argmax(logits)), logits


def compute_split_layer_tail_host(
    *,
    request_base: dict[str, Any],
    seq_len: int,
    vocab_size: int,
    token_ids: list[int],
    hidden_words: list[str],
) -> tuple[int, Any]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PCIe split workaround 需要 numpy 做 host-side tail") from exc

    guppy_dir = Path(__file__).resolve().parent
    if str(guppy_dir) not in sys.path:
        sys.path.insert(0, str(guppy_dir))
    from gen_full_inference import BundleLoader, block_ref, layernorm_ref, linear_ref

    layer_limit = int(request_base.get("xdma_split_layer_limit", 1))
    if layer_limit <= 1:
        raise RuntimeError("layer-tail split 只用于 layer_limit > 1")

    bundle_dir = Path(str(request_base["bundle_dir"])).expanduser().resolve()
    bundle = BundleLoader(bundle_dir)
    cfg = bundle.model_config["normalized_config"]
    d_model = int(request_base["xdma_split_d_model"])
    if int(cfg["vocab_size"]) != vocab_size:
        raise RuntimeError(
            f"vocab_size mismatch: config={cfg['vocab_size']} runtime={vocab_size}"
        )

    hidden = np.asarray(words_to_f32(hidden_words), dtype=np.float32)
    expected_words = seq_len * d_model
    if hidden.size != expected_words:
        raise RuntimeError(
            f"split hidden_state word count mismatch: "
            f"expected={expected_words} actual={hidden.size}"
        )
    x = hidden.reshape(seq_len, d_model).astype(np.float32)

    for layer_idx in range(1, layer_limit):
        prefix = f"blocks.{layer_idx}"
        layer = {
            "norm1.weight": bundle.load_tensor(f"{prefix}.norm1.weight").astype(np.float32),
            "norm1.bias": bundle.load_tensor(f"{prefix}.norm1.bias").astype(np.float32),
            "attn.qkv.weight": bundle.load_tensor(f"{prefix}.attn.qkv.weight").astype(np.float32),
            "attn.qkv.bias": bundle.load_tensor(f"{prefix}.attn.qkv.bias").astype(np.float32),
            "attn.out.weight": bundle.load_tensor(f"{prefix}.attn.out.weight").astype(np.float32),
            "attn.out.bias": bundle.load_tensor(f"{prefix}.attn.out.bias").astype(np.float32),
            "norm2.weight": bundle.load_tensor(f"{prefix}.norm2.weight").astype(np.float32),
            "norm2.bias": bundle.load_tensor(f"{prefix}.norm2.bias").astype(np.float32),
            "ffn.up.weight": bundle.load_tensor(f"{prefix}.ffn.up.weight").astype(np.float32),
            "ffn.up.bias": bundle.load_tensor(f"{prefix}.ffn.up.bias").astype(np.float32),
            "ffn.down.weight": bundle.load_tensor(f"{prefix}.ffn.down.weight").astype(np.float32),
            "ffn.down.bias": bundle.load_tensor(f"{prefix}.ffn.down.bias").astype(np.float32),
        }
        x = block_ref(x, layer, int(cfg["n_heads"])).astype(np.float32)

    row = max(0, min(len(token_ids), seq_len) - 1)
    tok_emb = bundle.load_tensor("tok_emb.weight").astype(np.float32)
    lm_input = layernorm_ref(
        x[row : row + 1],
        bundle.load_tensor("norm.weight").astype(np.float32),
        bundle.load_tensor("norm.bias").astype(np.float32),
    ).astype(np.float32)
    logits = linear_ref(lm_input, tok_emb, None).astype(np.float32)[0]
    return int(np.argmax(logits)), logits


class JsonHttpClient:
    def __init__(self) -> None:
        self._opener = build_opener(ProxyHandler({}))

    def get_json(self, url: str) -> dict[str, Any]:
        with self._opener.open(url) as response:
            return json.load(response)

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener.open(request) as response:
            return json.load(response)


def fetch_log_content(client: JsonHttpClient, service_url: str, job_id: str, name: str) -> str:
    payload = client.get_json(f"{service_url}/jobs/{job_id}/logs/{name}")
    return payload.get("content", "")


def try_fetch_log_content(
    client: JsonHttpClient, service_url: str, job_id: str, name: str
) -> str | None:
    try:
        return fetch_log_content(client, service_url, job_id, name)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def fetch_log_list(client: JsonHttpClient, service_url: str, job_id: str) -> list[str]:
    payload = client.get_json(f"{service_url}/jobs/{job_id}/logs")
    files = payload.get("files", [])
    return files if isinstance(files, list) else []


def fetch_status_file(client: JsonHttpClient, service_url: str, job_id: str) -> dict[str, Any] | None:
    try:
        content = fetch_log_content(client, service_url, job_id, "status.json")
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if not content:
        return None
    return json.loads(content)


def poll_job(
    client: JsonHttpClient,
    service_url: str,
    job_id: str,
    timeout_sec: int,
    poll_interval_sec: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_status = "unknown"
    while time.time() < deadline:
        status_payload = fetch_status_file(client, service_url, job_id)
        if status_payload is None:
            time.sleep(poll_interval_sec)
            continue
        last_status = status_payload.get("status", last_status)
        if last_status in {"succeeded", "failed", "cancelled"}:
            return status_payload
        time.sleep(poll_interval_sec)
    raise TimeoutError(f"等待 job {job_id} 超时，最后状态: {last_status}")


def extract_stdout_text(stdout_log: str) -> str:
    begin = "STDOUT_TEXT_BEGIN\n"
    end = "\nSTDOUT_TEXT_END"
    start = stdout_log.find(begin)
    if start < 0:
        return ""
    start += len(begin)
    stop = stdout_log.find(end, start)
    if stop < 0:
        return stdout_log[start:].strip()
    return stdout_log[start:stop].strip()


def build_messages(args: argparse.Namespace, bundle_prompt: dict[str, Any]) -> list[dict[str, Any]]:
    if args.messages_json:
        payload = json.loads(Path(args.messages_json).expanduser().resolve().read_text())
        if not isinstance(payload, list) or not payload:
            raise ValueError("--messages-json 必须是非空消息列表")
        return payload
    if args.prompt_text is not None:
        return [{"role": "user", "content": args.prompt_text}]
    messages = bundle_prompt.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("bundle prompt.json 缺少 messages，且未指定 --prompt-text/--messages-json")
    return messages


def pick_next_token_from_logits(
    *,
    logits: list[float],
    tokenizer: Any,
    decode_mode: str,
    temperature: float,
    sample_top_k: int,
    rng: random.Random,
) -> tuple[int, list[dict[str, Any]]]:
    ranked = sorted(enumerate(logits), key=lambda item: item[1], reverse=True)
    preview = [
        {
            "token_id": token_id,
            "logit": float(logit),
            "text": tokenizer.decode([token_id]),
        }
        for token_id, logit in ranked[: max(1, sample_top_k)]
    ]
    if decode_mode == "greedy":
        token_id = preview[0]["token_id"]
        return int(token_id), preview

    top_candidates = ranked[: max(1, sample_top_k)]
    scaled = [candidate[1] / temperature for candidate in top_candidates]
    max_scaled = max(scaled)
    weights = [math.exp(value - max_scaled) for value in scaled]
    total = sum(weights)
    if total <= 0.0:
        token_id = top_candidates[0][0]
        return int(token_id), preview
    threshold = rng.random() * total
    acc = 0.0
    for (token_id, _), weight in zip(top_candidates, weights):
        acc += weight
        if acc >= threshold:
            return int(token_id), preview
    return int(top_candidates[-1][0]), preview


def run_generation_step_service(
    *,
    client: JsonHttpClient,
    tokenizer: Any,
    service_url: str,
    elf_path: Path,
    stage_dir: Path,
    request_base: dict[str, Any],
    symbols: dict[str, int],
    seq_len: int,
    pad_id: int,
    vocab_size: int,
    token_ids: list[int],
    run_name: str,
    step_index: int,
    timeout_sec: int,
    poll_interval_sec: float,
    read_logits: bool,
    top_k: int,
    decode_mode: str,
    temperature: float,
    sample_top_k: int,
    rng: random.Random,
    resident_job_id: str | None,
    reload_all_kernel_segments: bool,
) -> dict[str, Any]:
    padded_ids = token_ids + [pad_id] * (seq_len - len(token_ids))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "name": f"{run_name}_s{step_index:02d}",
        "expect_exit_word": "0x00000000",
        "segments": [
            {
                "name": "input_token_ids",
                "addr": f"0x{symbols['guppy_input_token_ids']:08X}",
                "byte_len": f"0x{seq_len * 4:X}",
                "mem_words": ints_to_words(padded_ids),
            },
            {
                "name": "runtime_prompt_length",
                "addr": f"0x{symbols['guppy_runtime_prompt_length']:08X}",
                "byte_len": "0x4",
                "mem_words": [int_to_word(len(token_ids))],
            },
            {
                "name": "runtime_expect_golden",
                "addr": f"0x{symbols['guppy_runtime_expect_golden']:08X}",
                "byte_len": "0x4",
                "mem_words": [int_to_word(0)],
            },
        ],
        "outputs": [
            {
                "name": "argmax",
                "addr": f"0x{symbols['guppy_output_last_token_argmax']:08X}",
                "words": 1,
            }
        ],
    }
    if read_logits:
        manifest["outputs"].append(
            {
                "name": "last_token_logits",
                "addr": f"0x{symbols['guppy_output_last_token_logits']:08X}",
                "words": vocab_size,
            }
        )

    full_request_payload = {
        **request_base,
        "kernel_elf_name": elf_path.name,
        "kernel_elf_base64": base64.b64encode(elf_path.read_bytes()).decode("ascii"),
        "manifest": manifest,
    }
    resident_request_payload = {
        "resident_job_id": resident_job_id,
        "manifest": manifest,
        "board_scripts_dir": request_base["board_scripts_dir"],
        "ltx_path": request_base["ltx_path"],
        "device_index": request_base["device_index"],
        "debug_jtag_freq_hz": request_base["debug_jtag_freq_hz"],
        "hw_server_url": request_base["hw_server_url"],
        "hw_server_bind": request_base["hw_server_bind"],
        "dump_stdout": request_base["dump_stdout"],
        "persistent_vivado_session": request_base["persistent_vivado_session"],
        "stdout_max_chars": request_base["stdout_max_chars"],
        "timeout_sec": request_base["timeout_sec"],
    }
    if "status_poll_count" in request_base:
        resident_request_payload["status_poll_count"] = request_base["status_poll_count"]
    if "status_poll_ms" in request_base:
        resident_request_payload["status_poll_ms"] = request_base["status_poll_ms"]
    if reload_all_kernel_segments:
        resident_request_payload["reload_all_kernel_segments"] = True

    request_payload = resident_request_payload if resident_job_id else full_request_payload
    endpoint = "run-resident-manifest" if resident_job_id else "run-manifest"

    step_request_path = stage_dir / f"chat_step_{step_index:02d}_request.json"
    step_request_path.write_text(
        json.dumps(request_payload, indent=2, ensure_ascii=False) + "\n"
    )

    used_resident = resident_job_id is not None
    try:
        submission = client.post_json(f"{service_url}/jobs/{endpoint}", request_payload)
    except HTTPError as exc:
        if not used_resident or exc.code not in {400, 404}:
            raise
        endpoint = "run-manifest"
        used_resident = False
        request_payload = full_request_payload
        step_request_path.write_text(
            json.dumps(request_payload, indent=2, ensure_ascii=False) + "\n"
        )
        submission = client.post_json(f"{service_url}/jobs/{endpoint}", request_payload)
    job_id = submission["job_id"]
    (stage_dir / f"chat_step_{step_index:02d}_job_id.txt").write_text(job_id + "\n")
    print(f"step {step_index}: job_id={job_id}", flush=True)

    status_payload = poll_job(
        client,
        service_url,
        job_id,
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
    )

    available_files = fetch_log_list(client, service_url, job_id)
    stdout_log = try_fetch_log_content(client, service_url, job_id, "stdout.log") or ""
    stdout_text = extract_stdout_text(stdout_log)
    summary_text = try_fetch_log_content(client, service_url, job_id, "summary.json")
    summary = json.loads(summary_text) if summary_text else {}

    argmax_mem = try_fetch_log_content(
        client, service_url, job_id, "artifacts/argmax_actual.mem"
    )
    argmax_words = argmax_mem.splitlines() if argmax_mem else []
    if not argmax_words:
        raise RuntimeError(
            "未回读到 argmax_actual.mem; "
            f"step={step_index} status={status_payload.get('status')} files={available_files}"
        )
    board_argmax_id = words_to_ints(argmax_words)[0]
    next_token_id = board_argmax_id
    next_token_text = tokenizer.decode([next_token_id])

    top_k_payload: list[dict[str, Any]] = []
    if read_logits:
        logits_mem = try_fetch_log_content(
            client, service_url, job_id, "artifacts/last_token_logits_actual.mem"
        )
        logits_words = logits_mem.splitlines() if logits_mem else []
        logits = words_to_f32(logits_words)
        next_token_id, top_k_payload = pick_next_token_from_logits(
            logits=logits,
            tokenizer=tokenizer,
            decode_mode=decode_mode,
            temperature=temperature,
            sample_top_k=sample_top_k,
            rng=rng,
        )
        next_token_text = tokenizer.decode([next_token_id])

    step_result = {
        "step_index": step_index,
        "job_id": job_id,
        "status": status_payload.get("status"),
        "exit_code": status_payload.get("exit_code"),
        "input_ids": list(token_ids),
        "padded_input_ids": padded_ids,
        "board_argmax_id": board_argmax_id,
        "next_token_id": next_token_id,
        "next_token_text": next_token_text,
        "stdout_text": stdout_text,
        "metadata": status_payload.get("metadata", {}),
        "summary": summary,
        "top_k": top_k_payload,
        "available_files": available_files,
        "endpoint": endpoint,
        "used_resident": used_resident,
        "resident_job_id": resident_job_id,
    }
    step_result_path = stage_dir / f"chat_step_{step_index:02d}_result.json"
    step_result_path.write_text(
        json.dumps(step_result, indent=2, ensure_ascii=False) + "\n"
    )
    return step_result


def run_generation_step_local_jtag(
    *,
    tokenizer: Any,
    elf_path: Path,
    stage_dir: Path,
    request_base: dict[str, Any],
    symbols: dict[str, int],
    seq_len: int,
    pad_id: int,
    vocab_size: int,
    token_ids: list[int],
    run_name: str,
    step_index: int,
    read_logits: bool,
    top_k: int,
    decode_mode: str,
    temperature: float,
    sample_top_k: int,
    rng: random.Random,
) -> dict[str, Any]:
    board_scripts_dir = Path(str(request_base["board_scripts_dir"])).expanduser().resolve()
    board_runner = Path(str(request_base["board_runner"])).expanduser().resolve()
    vivado_settings_sh = request_base.get("vivado_settings_sh")
    step_dir = stage_dir / "board_runs" / f"step_{step_index:02d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    padded_ids = token_ids + [pad_id] * (seq_len - len(token_ids))
    manifest_name = f"{run_name}_s{step_index:02d}"

    payload_specs = [
        {
            "name": "input_token_ids",
            "mem_file": "input_token_ids.mem",
            "addr": f"0x{symbols['guppy_input_token_ids']:08X}",
            "byte_len": f"0x{seq_len * 4:X}",
            "mem_words": ints_to_words(padded_ids),
        },
        {
            "name": "runtime_prompt_length",
            "mem_file": "runtime_prompt_length.mem",
            "addr": f"0x{symbols['guppy_runtime_prompt_length']:08X}",
            "byte_len": "0x4",
            "mem_words": [int_to_word(len(token_ids))],
        },
        {
            "name": "runtime_expect_golden",
            "mem_file": "runtime_expect_golden.mem",
            "addr": f"0x{symbols['guppy_runtime_expect_golden']:08X}",
            "byte_len": "0x4",
            "mem_words": [int_to_word(0)],
        },
    ]
    checkpoint_stage = request_base.get("xdma_checkpoint_stage")
    if checkpoint_stage is not None:
        payload_specs.append(
            {
                "name": "runtime_pcie_split_stage",
                "mem_file": "runtime_pcie_split_stage.mem",
                "addr": f"0x{symbols['guppy_runtime_pcie_split_stage']:08X}",
                "byte_len": "0x4",
                "mem_words": [int_to_word(int(checkpoint_stage))],
            }
        )
    elif "guppy_runtime_pcie_split_stage" in symbols:
        payload_specs.append(
            {
                "name": "runtime_pcie_split_stage",
                "mem_file": "runtime_pcie_split_stage.mem",
                "addr": f"0x{symbols['guppy_runtime_pcie_split_stage']:08X}",
                "byte_len": "0x4",
                "mem_words": [int_to_word(0)],
            }
        )
    for payload in payload_specs:
        write_mem_words(step_dir / str(payload["mem_file"]), list(payload["mem_words"]))

    outputs = [
        {
            "name": "argmax",
            "addr": f"0x{symbols['guppy_output_last_token_argmax']:08X}",
            "words": 1,
        }
    ]
    if "guppy_progress_stage" in symbols:
        outputs.append(
            {
                "name": "progress_stage",
                "addr": f"0x{symbols['guppy_progress_stage']:08X}",
                "words": 1,
            }
        )
    append_warp4_attn_out_outputs(outputs, symbols)
    append_xdma_checkpoint_outputs(
        outputs,
        symbols,
        request_base=request_base,
        seq_len=seq_len,
        token_ids=token_ids,
    )
    if read_logits:
        outputs.append(
            {
                "name": "last_token_logits",
                "addr": f"0x{symbols['guppy_output_last_token_logits']:08X}",
                "words": vocab_size,
            }
        )

    local_manifest = {
        "schema_version": 1,
        "name": manifest_name,
        "kernel_elf": str(elf_path),
        "startup_addr": "0x80000000",
        "startup_arg": "0x0",
        "expect_exit_word": "0x00000000",
        "require_exit_seen": True,
        "segments": [
            {
                "name": payload["name"],
                "mem_file": str(payload["mem_file"]),
                "addr": payload["addr"],
                "byte_len": payload["byte_len"],
            }
            for payload in payload_specs
        ],
        "outputs": outputs,
    }
    local_manifest_path = step_dir / "local_manifest.json"
    local_manifest_path.write_text(
        json.dumps(local_manifest, indent=2, ensure_ascii=False) + "\n"
    )

    step_request = {
        "runner_mode": "local-jtag",
        "board_runner": str(board_runner),
        "board_scripts_dir": str(board_scripts_dir),
        "vivado_settings_sh": vivado_settings_sh,
        "bit_path": request_base["bit_path"],
        "ltx_path": request_base["ltx_path"],
        "device_index": request_base["device_index"],
        "program_jtag_freq_hz": request_base["program_jtag_freq_hz"],
        "debug_jtag_freq_hz": request_base["debug_jtag_freq_hz"],
        "hw_server_url": request_base["hw_server_url"],
        "hw_server_bind": request_base["hw_server_bind"],
        "stdout_max_chars": request_base["stdout_max_chars"],
        "manifest": local_manifest,
    }
    step_request_path = stage_dir / f"chat_step_{step_index:02d}_request.json"
    step_request_path.write_text(
        json.dumps(step_request, indent=2, ensure_ascii=False) + "\n"
    )

    if not board_runner.is_file():
        raise FileNotFoundError(f"未找到本地 board runner: {board_runner}")

    runner_cmd = [
        sys.executable,
        str(board_runner),
        "--manifest",
        str(local_manifest_path),
        "--runner-mode",
        "local",
        "--local-bit",
        str(request_base["bit_path"]),
        "--local-ltx",
        str(request_base["ltx_path"]),
        "--board-scripts-dir",
        str(board_scripts_dir),
        "--hw-server-url",
        str(request_base["hw_server_url"]),
        "--hw-server-bind",
        str(request_base["hw_server_bind"]),
        "--device-index",
        str(request_base["device_index"]),
        "--program-jtag-freq-hz",
        str(request_base["program_jtag_freq_hz"]),
        "--debug-jtag-freq-hz",
        str(request_base["debug_jtag_freq_hz"]),
        "--verify",
        "0" if not request_base.get("verify", False) else "1",
        "--stdout-max-chars",
        str(request_base["stdout_max_chars"]),
        "--stage-dir",
        str(step_dir),
    ]
    if vivado_settings_sh:
        runner_cmd.extend(["--vivado-settings-sh", str(vivado_settings_sh)])
    if request_base.get("dump_stdout", False):
        runner_cmd.append("--dump-stdout")
    if step_index != 0:
        runner_cmd.append("--skip-program")

    runner_log_path = step_dir / "runner.log"
    runner_rc = run_and_log(runner_cmd, cwd=step_dir, log_path=runner_log_path)
    run_log = step_dir / "run.log"
    run_pass, final_exit = (
        parse_run_summary(run_log) if run_log.is_file() else ("UNKNOWN", "UNKNOWN")
    )

    if runner_rc != 0:
        raise RuntimeError(
            f"本地 JTAG 运行失败: step={step_index} rc={runner_rc} run_pass={run_pass} "
            f"final_exit={final_exit} stage_dir={step_dir}"
        )

    stdout_log = step_dir / "stdout.log"
    stdout_log_content = (
        stdout_log.read_text(encoding="utf-8", errors="replace")
        if stdout_log.is_file()
        else ""
    )
    stdout_text = extract_stdout_text(stdout_log_content)
    summary = {
        "runner_mode": "local-jtag",
        "run_pass": run_pass,
        "final_exit_word": final_exit,
        "stage_dir": str(step_dir),
    }
    summary_path = step_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    argmax_mem_path = step_dir / "argmax_actual.mem"
    if not argmax_mem_path.is_file():
        raise RuntimeError(f"未回读到 argmax_actual.mem: {step_dir}")
    argmax_words = argmax_mem_path.read_text(encoding="utf-8").splitlines()
    if not argmax_words:
        raise RuntimeError(f"argmax_actual.mem 为空: {argmax_mem_path}")

    board_argmax_id = words_to_ints(argmax_words)[0]
    next_token_id = board_argmax_id
    next_token_text = tokenizer.decode([next_token_id])

    top_k_payload: list[dict[str, Any]] = []
    if read_logits:
        logits_mem_path = step_dir / "last_token_logits_actual.mem"
        if not logits_mem_path.is_file():
            raise RuntimeError(f"未回读到 last_token_logits_actual.mem: {step_dir}")
        logits_words = logits_mem_path.read_text(encoding="utf-8").splitlines()
        logits = words_to_f32(logits_words)
        next_token_id, top_k_payload = pick_next_token_from_logits(
            logits=logits,
            tokenizer=tokenizer,
            decode_mode=decode_mode,
            temperature=temperature,
            sample_top_k=sample_top_k,
            rng=rng,
        )
        next_token_text = tokenizer.decode([next_token_id])

    available_files = sorted(
        str(path.relative_to(step_dir))
        for path in step_dir.rglob("*")
        if path.is_file()
    )
    step_result = {
        "step_index": step_index,
        "job_id": f"local-step-{step_index:02d}",
        "status": "succeeded",
        "exit_code": 0,
        "input_ids": list(token_ids),
        "padded_input_ids": padded_ids,
        "board_argmax_id": board_argmax_id,
        "next_token_id": next_token_id,
        "next_token_text": next_token_text,
        "stdout_text": stdout_text,
        "metadata": {
            "board_run_stage_dir": str(step_dir),
        },
        "summary": summary,
        "top_k": top_k_payload,
        "available_files": available_files,
        "endpoint": "local-jtag",
        "used_resident": False,
        "resident_job_id": None,
    }
    step_result_path = stage_dir / f"chat_step_{step_index:02d}_result.json"
    step_result_path.write_text(
        json.dumps(step_result, indent=2, ensure_ascii=False) + "\n"
    )
    return step_result


def run_local_xdma_manifest(
    *,
    xdma_runner: Path,
    manifest_path: Path,
    run_stage_dir: Path,
    request_base: dict[str, Any],
    timeout_sec: int,
) -> tuple[int, str, str]:
    if not xdma_runner.is_file():
        raise FileNotFoundError(f"未找到本地 XDMA runner: {xdma_runner}")

    runner_cmd = [
        sys.executable,
        str(xdma_runner),
        "--manifest",
        str(manifest_path),
        "--stage-dir",
        str(run_stage_dir),
        "--bdf",
        str(request_base.get("xdma_bdf", "0000:03:00.0")),
        "--h2c-dev",
        str(request_base["xdma_h2c_dev"]),
        "--c2h-dev",
        str(request_base["xdma_c2h_dev"]),
        "--ctrl-dma-base",
        str(request_base["xdma_ctrl_dma_base"]),
        "--timeout-sec",
        str(timeout_sec),
        "--require-busy",
        "1" if request_base.get("xdma_require_busy", True) else "0",
        "--verify-payload-load",
        "1",
        "--reset-before-success-dump",
        "0",
        "--final-host-reset",
        "1",
    ]
    runner_log_path = run_stage_dir / "runner.log"
    runner_rc = run_and_log(runner_cmd, cwd=run_stage_dir, log_path=runner_log_path)
    run_log = run_stage_dir / "run.log"
    run_pass, final_exit = (
        parse_run_summary(run_log) if run_log.is_file() else ("UNKNOWN", "UNKNOWN")
    )
    return runner_rc, run_pass, final_exit


def parse_xdma_timing(stage_dir: Path) -> dict[str, Any]:
    timing_path = stage_dir / "timing.log"
    if not timing_path.is_file():
        return {"timing_log": str(timing_path), "events": []}
    events: list[dict[str, Any]] = []
    for raw in timing_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("TIMING "):
            continue
        event: dict[str, Any] = {}
        for part in line.split()[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key.endswith("_ms") or key in {
                "elapsed_ms",
                "segments",
                "segment_words",
                "segment_bytes",
                "payload_verifies",
                "payload_verify_attempts",
                "payload_verify_ms",
                "passed",
                "output_failures",
                "compare_errors",
            }:
                try:
                    event[key] = int(value, 0)
                    continue
                except ValueError:
                    pass
            event[key] = value
        if event:
            events.append(event)
    return {"timing_log": str(timing_path), "events": events}


def run_generation_step_local_xdma(
    *,
    tokenizer: Any,
    elf_path: Path,
    stage_dir: Path,
    request_base: dict[str, Any],
    symbols: dict[str, int],
    seq_len: int,
    pad_id: int,
    vocab_size: int,
    token_ids: list[int],
    run_name: str,
    step_index: int,
    timeout_sec: int,
    read_logits: bool,
    top_k: int,
    decode_mode: str,
    temperature: float,
    sample_top_k: int,
    rng: random.Random,
) -> dict[str, Any]:
    del top_k
    xdma_runner = Path(str(request_base["board_runner"])).expanduser().resolve()
    step_dir = stage_dir / "xdma_runs" / f"step_{step_index:02d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    padded_ids = token_ids + [pad_id] * (seq_len - len(token_ids))
    manifest_name = f"{run_name}_s{step_index:02d}"
    split_enabled = bool(request_base.get("xdma_split_workaround", False))

    payload_specs = [
        {
            "name": "input_token_ids",
            "mem_file": "input_token_ids.mem",
            "addr": f"0x{symbols['guppy_input_token_ids']:08X}",
            "byte_len": f"0x{seq_len * 4:X}",
            "mem_words": ints_to_words(padded_ids),
        },
        {
            "name": "runtime_prompt_length",
            "mem_file": "runtime_prompt_length.mem",
            "addr": f"0x{symbols['guppy_runtime_prompt_length']:08X}",
            "byte_len": "0x4",
            "mem_words": [int_to_word(len(token_ids))],
        },
        {
            "name": "runtime_expect_golden",
            "mem_file": "runtime_expect_golden.mem",
            "addr": f"0x{symbols['guppy_runtime_expect_golden']:08X}",
            "byte_len": "0x4",
            "mem_words": [int_to_word(0)],
        },
    ]
    checkpoint_stage = request_base.get("xdma_checkpoint_stage")
    if checkpoint_stage is not None:
        payload_specs.append(
            {
                "name": "runtime_pcie_split_stage",
                "mem_file": "runtime_pcie_split_stage.mem",
                "addr": f"0x{symbols['guppy_runtime_pcie_split_stage']:08X}",
                "byte_len": "0x4",
                "mem_words": [int_to_word(int(checkpoint_stage))],
            }
        )
    elif not split_enabled and "guppy_runtime_pcie_split_stage" in symbols:
        payload_specs.append(
            {
                "name": "runtime_pcie_split_stage",
                "mem_file": "runtime_pcie_split_stage.mem",
                "addr": f"0x{symbols['guppy_runtime_pcie_split_stage']:08X}",
                "byte_len": "0x4",
                "mem_words": [int_to_word(0)],
            }
        )
    for payload in payload_specs:
        write_mem_words(step_dir / str(payload["mem_file"]), list(payload["mem_words"]))

    split_elf_path = Path(str(request_base.get("xdma_split_elf_path", ""))).expanduser()
    split_symbols = request_base.get("xdma_split_symbols") or {}
    layer_limit = int(request_base.get("xdma_split_layer_limit", 1))
    split_required = ["guppy_runtime_pcie_split_stage"]
    if layer_limit <= 1:
        split_required.append("g_attn_merge")
    else:
        split_required.append("g_attn_merge")
    split_post_required = [
        "guppy_split_host_argmax",
        "guppy_output_last_token_argmax",
    ]
    split_can_run = (
        split_enabled
        and split_elf_path.is_file()
        and all(name in symbols for name in split_required)
        and all(name in split_symbols for name in split_post_required)
    )
    if split_enabled and not split_can_run:
        print(
            "info: local-xdma PCIe split workaround unavailable; falling back to monolithic kernel",
            flush=True,
        )

    if split_can_run:
        d_model = int(request_base["xdma_split_d_model"])
        row = max(0, min(len(token_ids), seq_len) - 1)
        stage0_dir = step_dir / "split_stage0_attn_merge"
        stage1_dir = step_dir / "split_stage1_post_attn"
        stage0_dir.mkdir(parents=True, exist_ok=True)
        stage1_dir.mkdir(parents=True, exist_ok=True)

        if layer_limit <= 1:
            stage0_payload_specs = payload_specs + [
                {
                    "name": "runtime_pcie_split_stage",
                    "mem_file": "runtime_pcie_split_stage.mem",
                    "addr": f"0x{symbols['guppy_runtime_pcie_split_stage']:08X}",
                    "byte_len": "0x4",
                    "mem_words": [int_to_word(1)],
                }
            ]
        else:
            control_words = [
                int_to_word(len(token_ids)),
                int_to_word(0),
                int_to_word(1),
            ] + [int_to_word(0)] * 13
            stage0_payload_specs = [
                payload_specs[0],
                {
                    "name": "runtime_control_block",
                    "mem_file": "runtime_control_block.mem",
                    "addr": f"0x{symbols['guppy_runtime_prompt_length']:08X}",
                    "byte_len": "0x40",
                    "mem_words": control_words,
                },
            ]
        for payload in stage0_payload_specs:
            write_mem_words(stage0_dir / str(payload["mem_file"]), list(payload["mem_words"]))

        if layer_limit <= 1:
            attn_merge_row_addr = symbols["g_attn_merge"] + row * 4 * d_model
            stage0_outputs = [
                {
                    "name": "attn_merge_row",
                    "addr": f"0x{attn_merge_row_addr:08X}",
                    "words": d_model,
                }
            ]
        else:
            stage0_outputs = [
                {
                    "name": "attn_merge",
                    "addr": f"0x{symbols['g_attn_merge']:08X}",
                    "words": seq_len * d_model,
                }
            ]
        if layer_limit <= 1 and "guppy_progress_stage" in symbols:
            stage0_outputs.append(
                {
                    "name": "progress_stage",
                    "addr": f"0x{symbols['guppy_progress_stage']:08X}",
                    "words": 1,
                }
            )
        if "guppy_stage0_profile" in symbols:
            stage0_outputs.append(
                {
                    "name": "stage0_profile",
                    "addr": f"0x{symbols['guppy_stage0_profile']:08X}",
                    "words": 192,
                }
            )

        stage0_manifest = {
            "schema_version": 1,
            "name": f"{manifest_name}_split_stage0_attn_merge",
            "kernel_elf": str(elf_path),
            "startup_addr": "0x80000000",
            "startup_arg": "0x0",
            "expect_exit_word": "0x00000000",
            "require_exit_seen": True,
            "segments": [
                {
                    "name": payload["name"],
                    "mem_file": str(payload["mem_file"]),
                    "addr": payload["addr"],
                    "byte_len": payload["byte_len"],
                }
                for payload in stage0_payload_specs
            ],
            "outputs": stage0_outputs,
        }
        stage0_manifest_path = stage0_dir / "local_manifest.json"
        stage0_manifest_path.write_text(
            json.dumps(stage0_manifest, indent=2, ensure_ascii=False) + "\n"
        )
        runner_rc, stage0_pass, stage0_exit = run_local_xdma_manifest(
            xdma_runner=xdma_runner,
            manifest_path=stage0_manifest_path,
            run_stage_dir=stage0_dir,
            request_base=request_base,
            timeout_sec=timeout_sec,
        )
        stage0_timing = parse_xdma_timing(stage0_dir)
        stage0_profile = parse_stage0_profile_mem(stage0_dir / "stage0_profile_actual.mem")
        if runner_rc != 0:
            raise RuntimeError(
                f"本地 XDMA split stage0 失败: step={step_index} rc={runner_rc} "
                f"run_pass={stage0_pass} final_exit={stage0_exit} stage_dir={stage0_dir}"
            )

        if layer_limit <= 1:
            attn_row_path = stage0_dir / "attn_merge_row_actual.mem"
            if not attn_row_path.is_file():
                raise RuntimeError(f"split stage0 未回读到 attn_merge_row: {attn_row_path}")
            attn_row_words = attn_row_path.read_text(encoding="utf-8").splitlines()
            if len(attn_row_words) != d_model:
                raise RuntimeError(
                    f"split stage0 attn_merge_row word count mismatch: "
                    f"expected={d_model} actual={len(attn_row_words)}"
                )
            if not any(word.strip().upper() not in {"00000000", "80000000"} for word in attn_row_words):
                raise RuntimeError(
                    f"split stage0 attn_merge_row is all zero: {attn_row_path}. "
                    "This means stage0 did not produce a usable attention merge row."
                )
            host_argmax, host_logits = compute_split_post_attn_host(
                request_base=request_base,
                seq_len=seq_len,
                vocab_size=vocab_size,
                token_ids=token_ids,
                padded_ids=padded_ids,
                attn_row_words=attn_row_words,
            )
        else:
            attn_merge_path = stage0_dir / "attn_merge_actual.mem"
            if not attn_merge_path.is_file():
                raise RuntimeError(f"split stage0 未回读到 attn_merge: {attn_merge_path}")
            attn_merge_words = attn_merge_path.read_text(encoding="utf-8").splitlines()
            expected_words = seq_len * d_model
            if len(attn_merge_words) != expected_words:
                raise RuntimeError(
                    f"split stage0 attn_merge word count mismatch: "
                    f"expected={expected_words} actual={len(attn_merge_words)}"
                )
            host_argmax, host_logits = compute_split_layer_tail_from_attn_merge_host(
                request_base=request_base,
                seq_len=seq_len,
                vocab_size=vocab_size,
                token_ids=token_ids,
                padded_ids=padded_ids,
                attn_merge_words=attn_merge_words,
            )

        stage1_payload_specs = [
            {
                "name": "host_argmax",
                "mem_file": "host_argmax.mem",
                "addr": f"0x{split_symbols['guppy_split_host_argmax']:08X}",
                "byte_len": "0x4",
                "mem_words": [int_to_word(host_argmax)],
            }
        ]
        for payload in stage1_payload_specs:
            write_mem_words(stage1_dir / str(payload["mem_file"]), list(payload["mem_words"]))

        stage1_outputs = [
            {
                "name": "argmax",
                "addr": f"0x{split_symbols['guppy_output_last_token_argmax']:08X}",
                "words": 1,
            }
        ]
        if "guppy_progress_stage" in split_symbols:
            stage1_outputs.append(
                {
                    "name": "progress_stage",
                    "addr": f"0x{split_symbols['guppy_progress_stage']:08X}",
                    "words": 1,
                }
            )
        stage1_manifest = {
            "schema_version": 1,
            "name": f"{manifest_name}_split_stage1_post_attn",
            "kernel_elf": str(split_elf_path.resolve()),
            "startup_addr": f"0x{split_symbols.get('STARTUP_ADDR', 0x80000000):08X}",
            "startup_arg": "0x0",
            "expect_exit_word": "0x00000000",
            "require_exit_seen": True,
            "segments": [
                {
                    "name": payload["name"],
                    "mem_file": str(payload["mem_file"]),
                    "addr": payload["addr"],
                    "byte_len": payload["byte_len"],
                }
                for payload in stage1_payload_specs
            ],
            "outputs": stage1_outputs,
        }
        stage1_manifest_path = stage1_dir / "local_manifest.json"
        stage1_manifest_path.write_text(
            json.dumps(stage1_manifest, indent=2, ensure_ascii=False) + "\n"
        )
        runner_rc, run_pass, final_exit = run_local_xdma_manifest(
            xdma_runner=xdma_runner,
            manifest_path=stage1_manifest_path,
            run_stage_dir=stage1_dir,
            request_base=request_base,
            timeout_sec=timeout_sec,
        )
        stage1_timing = parse_xdma_timing(stage1_dir)
        if runner_rc != 0:
            raise RuntimeError(
                f"本地 XDMA split stage1 失败: step={step_index} rc={runner_rc} "
                f"run_pass={run_pass} final_exit={final_exit} stage_dir={stage1_dir}"
            )

        if read_logits:
            write_mem_words(
                stage1_dir / "last_token_logits_actual.mem", f32_to_words(host_logits)
            )

        for name in ("argmax", "progress_stage", "last_token_logits"):
            src = stage1_dir / f"{name}_actual.mem"
            if src.is_file():
                shutil.copyfile(src, step_dir / src.name)

        local_manifest_path = step_dir / "local_manifest.json"
        local_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": f"{manifest_name}_split",
                    "split_stage0": str(stage0_manifest_path),
                    "split_stage1": str(stage1_manifest_path),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        step_request = {
            "runner_mode": "local-xdma",
            "pcie_split_workaround": True,
            "xdma_runner": str(xdma_runner),
            "xdma_h2c_dev": request_base["xdma_h2c_dev"],
            "xdma_c2h_dev": request_base["xdma_c2h_dev"],
            "xdma_ctrl_dma_base": request_base["xdma_ctrl_dma_base"],
            "xdma_bdf": request_base["xdma_bdf"],
            "xdma_require_busy": request_base["xdma_require_busy"],
            "timeout_sec": timeout_sec,
            "host_post_attn_argmax": host_argmax,
            "split_stage0_manifest": stage0_manifest,
            "split_stage1_manifest": stage1_manifest,
        }
        step_request_path = stage_dir / f"chat_step_{step_index:02d}_request.json"
        step_request_path.write_text(
            json.dumps(step_request, indent=2, ensure_ascii=False) + "\n"
        )

        stdout_text = ""
        summary = {
            "runner_mode": "local-xdma",
            "pcie_split_workaround": True,
            "stage0_run_pass": stage0_pass,
            "stage0_final_exit_word": stage0_exit,
            "host_post_attn_argmax": host_argmax,
            "run_pass": run_pass,
            "final_exit_word": final_exit,
            "stage_dir": str(step_dir),
            "split_stage0_dir": str(stage0_dir),
            "split_stage1_dir": str(stage1_dir),
            "split_stage0_timing": stage0_timing,
            "split_stage0_profile": stage0_profile,
            "split_stage1_timing": stage1_timing,
        }
        summary_path = step_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

        argmax_mem_path = step_dir / "argmax_actual.mem"
        if not argmax_mem_path.is_file():
            raise RuntimeError(f"未回读到 argmax_actual.mem: {step_dir}")
        argmax_words = argmax_mem_path.read_text(encoding="utf-8").splitlines()
        if not argmax_words:
            raise RuntimeError(f"argmax_actual.mem 为空: {argmax_mem_path}")

        board_argmax_id = words_to_ints(argmax_words)[0]
        next_token_id = board_argmax_id
        next_token_text = tokenizer.decode([next_token_id])

        top_k_payload: list[dict[str, Any]] = []
        if read_logits:
            logits_mem_path = step_dir / "last_token_logits_actual.mem"
            if not logits_mem_path.is_file():
                raise RuntimeError(f"未回读到 last_token_logits_actual.mem: {step_dir}")
            logits_words = logits_mem_path.read_text(encoding="utf-8").splitlines()
            logits = words_to_f32(logits_words)
            next_token_id, top_k_payload = pick_next_token_from_logits(
                logits=logits,
                tokenizer=tokenizer,
                decode_mode=decode_mode,
                temperature=temperature,
                sample_top_k=sample_top_k,
                rng=rng,
            )
            next_token_text = tokenizer.decode([next_token_id])

        available_files = sorted(
            str(path.relative_to(step_dir))
            for path in step_dir.rglob("*")
            if path.is_file()
        )
        step_result = {
            "step_index": step_index,
            "job_id": f"xdma-split-step-{step_index:02d}",
            "status": "succeeded",
            "exit_code": 0,
            "input_ids": list(token_ids),
            "padded_input_ids": padded_ids,
            "board_argmax_id": board_argmax_id,
            "next_token_id": next_token_id,
            "next_token_text": next_token_text,
            "stdout_text": stdout_text,
            "metadata": {
                "board_run_stage_dir": str(step_dir),
                "pcie_split_workaround": True,
            },
            "summary": summary,
            "top_k": top_k_payload,
            "available_files": available_files,
            "endpoint": "local-xdma",
            "used_resident": False,
            "resident_job_id": None,
        }
        step_result_path = stage_dir / f"chat_step_{step_index:02d}_result.json"
        step_result_path.write_text(
            json.dumps(step_result, indent=2, ensure_ascii=False) + "\n"
        )
        return step_result

    outputs = [
        {
            "name": "argmax",
            "addr": f"0x{symbols['guppy_output_last_token_argmax']:08X}",
            "words": 1,
        }
    ]
    if "guppy_progress_stage" in symbols:
        outputs.append(
            {
                "name": "progress_stage",
                "addr": f"0x{symbols['guppy_progress_stage']:08X}",
                "words": 1,
            }
        )
    append_warp4_attn_out_outputs(outputs, symbols)
    append_xdma_checkpoint_outputs(
        outputs,
        symbols,
        request_base=request_base,
        seq_len=seq_len,
        token_ids=token_ids,
    )
    if read_logits:
        outputs.append(
            {
                "name": "last_token_logits",
                "addr": f"0x{symbols['guppy_output_last_token_logits']:08X}",
                "words": vocab_size,
            }
        )

    local_manifest = {
        "schema_version": 1,
        "name": manifest_name,
        "kernel_elf": str(elf_path),
        "startup_addr": "0x80000000",
        "startup_arg": "0x0",
        "expect_exit_word": "0x00000000",
        "require_exit_seen": True,
        "segments": [
            {
                "name": payload["name"],
                "mem_file": str(payload["mem_file"]),
                "addr": payload["addr"],
                "byte_len": payload["byte_len"],
            }
            for payload in payload_specs
        ],
        "outputs": outputs,
    }
    local_manifest_path = step_dir / "local_manifest.json"
    local_manifest_path.write_text(
        json.dumps(local_manifest, indent=2, ensure_ascii=False) + "\n"
    )

    step_request = {
        "runner_mode": "local-xdma",
        "xdma_runner": str(xdma_runner),
        "xdma_h2c_dev": request_base["xdma_h2c_dev"],
        "xdma_c2h_dev": request_base["xdma_c2h_dev"],
        "xdma_ctrl_dma_base": request_base["xdma_ctrl_dma_base"],
        "xdma_bdf": request_base["xdma_bdf"],
        "xdma_require_busy": request_base["xdma_require_busy"],
        "timeout_sec": timeout_sec,
        "manifest": local_manifest,
    }
    step_request_path = stage_dir / f"chat_step_{step_index:02d}_request.json"
    step_request_path.write_text(
        json.dumps(step_request, indent=2, ensure_ascii=False) + "\n"
    )

    if not xdma_runner.is_file():
        raise FileNotFoundError(f"未找到本地 XDMA runner: {xdma_runner}")

    runner_cmd = [
        sys.executable,
        str(xdma_runner),
        "--manifest",
        str(local_manifest_path),
        "--stage-dir",
        str(step_dir),
        "--bdf",
        str(request_base.get("xdma_bdf", "0000:03:00.0")),
        "--h2c-dev",
        str(request_base["xdma_h2c_dev"]),
        "--c2h-dev",
        str(request_base["xdma_c2h_dev"]),
        "--ctrl-dma-base",
        str(request_base["xdma_ctrl_dma_base"]),
        "--timeout-sec",
        str(timeout_sec),
        "--require-busy",
        "1" if request_base.get("xdma_require_busy", True) else "0",
        "--reset-before-success-dump",
        "0",
        "--final-host-reset",
        "1",
    ]

    runner_log_path = step_dir / "runner.log"
    runner_rc = run_and_log(runner_cmd, cwd=step_dir, log_path=runner_log_path)
    run_log = step_dir / "run.log"
    run_pass, final_exit = (
        parse_run_summary(run_log) if run_log.is_file() else ("UNKNOWN", "UNKNOWN")
    )
    xdma_timing = parse_xdma_timing(step_dir)

    if runner_rc != 0:
        raise RuntimeError(
            f"本地 XDMA 运行失败: step={step_index} rc={runner_rc} run_pass={run_pass} "
            f"final_exit={final_exit} stage_dir={step_dir}"
        )

    stdout_text = ""
    summary = {
        "runner_mode": "local-xdma",
        "run_pass": run_pass,
        "final_exit_word": final_exit,
        "stage_dir": str(step_dir),
        "xdma_timing": xdma_timing,
    }
    summary_path = step_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    argmax_mem_path = step_dir / "argmax_actual.mem"
    if not argmax_mem_path.is_file():
        raise RuntimeError(f"未回读到 argmax_actual.mem: {step_dir}")
    argmax_words = argmax_mem_path.read_text(encoding="utf-8").splitlines()
    if not argmax_words:
        raise RuntimeError(f"argmax_actual.mem 为空: {argmax_mem_path}")

    board_argmax_id = words_to_ints(argmax_words)[0]
    next_token_id = board_argmax_id
    next_token_text = tokenizer.decode([next_token_id])

    top_k_payload: list[dict[str, Any]] = []
    if read_logits:
        logits_mem_path = step_dir / "last_token_logits_actual.mem"
        if not logits_mem_path.is_file():
            raise RuntimeError(f"未回读到 last_token_logits_actual.mem: {step_dir}")
        logits_words = logits_mem_path.read_text(encoding="utf-8").splitlines()
        logits = words_to_f32(logits_words)
        next_token_id, top_k_payload = pick_next_token_from_logits(
            logits=logits,
            tokenizer=tokenizer,
            decode_mode=decode_mode,
            temperature=temperature,
            sample_top_k=sample_top_k,
            rng=rng,
        )
        next_token_text = tokenizer.decode([next_token_id])

    available_files = sorted(
        str(path.relative_to(step_dir))
        for path in step_dir.rglob("*")
        if path.is_file()
    )
    step_result = {
        "step_index": step_index,
        "job_id": f"xdma-step-{step_index:02d}",
        "status": "succeeded",
        "exit_code": 0,
        "input_ids": list(token_ids),
        "padded_input_ids": padded_ids,
        "board_argmax_id": board_argmax_id,
        "next_token_id": next_token_id,
        "next_token_text": next_token_text,
        "stdout_text": stdout_text,
        "metadata": {
            "board_run_stage_dir": str(step_dir),
        },
        "summary": summary,
        "top_k": top_k_payload,
        "available_files": available_files,
        "endpoint": "local-xdma",
        "used_resident": False,
        "resident_job_id": None,
    }
    step_result_path = stage_dir / f"chat_step_{step_index:02d}_result.json"
    step_result_path.write_text(
        json.dumps(step_result, indent=2, ensure_ascii=False) + "\n"
    )
    return step_result


def run_generation_step(
    *,
    runner_mode: str,
    client: JsonHttpClient,
    tokenizer: Any,
    service_url: str,
    elf_path: Path,
    stage_dir: Path,
    request_base: dict[str, Any],
    symbols: dict[str, int],
    seq_len: int,
    pad_id: int,
    vocab_size: int,
    token_ids: list[int],
    run_name: str,
    step_index: int,
    timeout_sec: int,
    poll_interval_sec: float,
    read_logits: bool,
    top_k: int,
    decode_mode: str,
    temperature: float,
    sample_top_k: int,
    rng: random.Random,
    resident_job_id: str | None,
    reload_all_kernel_segments: bool,
) -> dict[str, Any]:
    if runner_mode == "service":
        return run_generation_step_service(
            client=client,
            tokenizer=tokenizer,
            service_url=service_url,
            elf_path=elf_path,
            stage_dir=stage_dir,
            request_base=request_base,
            symbols=symbols,
            seq_len=seq_len,
            pad_id=pad_id,
            vocab_size=vocab_size,
            token_ids=token_ids,
            run_name=run_name,
            step_index=step_index,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            read_logits=read_logits,
            top_k=top_k,
            decode_mode=decode_mode,
            temperature=temperature,
            sample_top_k=sample_top_k,
            rng=rng,
            resident_job_id=resident_job_id,
            reload_all_kernel_segments=reload_all_kernel_segments,
        )
    if runner_mode == "local-jtag":
        return run_generation_step_local_jtag(
            tokenizer=tokenizer,
            elf_path=elf_path,
            stage_dir=stage_dir,
            request_base=request_base,
            symbols=symbols,
            seq_len=seq_len,
            pad_id=pad_id,
            vocab_size=vocab_size,
            token_ids=token_ids,
            run_name=run_name,
            step_index=step_index,
            read_logits=read_logits,
            top_k=top_k,
            decode_mode=decode_mode,
            temperature=temperature,
            sample_top_k=sample_top_k,
            rng=rng,
        )
    if runner_mode == "local-xdma":
        return run_generation_step_local_xdma(
            tokenizer=tokenizer,
            elf_path=elf_path,
            stage_dir=stage_dir,
            request_base=request_base,
            symbols=symbols,
            seq_len=seq_len,
            pad_id=pad_id,
            vocab_size=vocab_size,
            token_ids=token_ids,
            run_name=run_name,
            step_index=step_index,
            timeout_sec=timeout_sec,
            read_logits=read_logits,
            top_k=top_k,
            decode_mode=decode_mode,
            temperature=temperature,
            sample_top_k=sample_top_k,
            rng=rng,
        )
    raise ValueError(f"Unsupported runner mode: {runner_mode}")


def update_resident_job_id(
    current_resident_job_id: str | None,
    step_result: dict[str, Any],
    *,
    reuse_resident: bool,
) -> str | None:
    if not reuse_resident:
        return None
    if step_result.get("used_resident") is True:
        return current_resident_job_id
    if step_result.get("status") != "succeeded":
        return None
    return str(step_result["job_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Guppy next-token inference on the FPGA board.")
    parser.add_argument("--bundle-dir", default="build/guppy/export", help="阶段 B bundle 目录")
    parser.add_argument(
        "--stage-dir",
        default="build/guppy/full_inference_seq10_l1",
        help="阶段 C 产物目录",
    )
    parser.add_argument("--prompt-text", default=None, help="快捷输入：单条 user prompt")
    parser.add_argument("--messages-json", default=None, help="消息列表 JSON，优先级高于 --prompt-text")
    parser.add_argument(
        "--runner-mode",
        choices=["service", "local-jtag", "local-xdma"],
        default="service",
        help="执行后端：remote service、本地 xc7k480t JTAG 或本地 PCIe/XDMA",
    )
    parser.add_argument("--service-url", default="http://100.125.4.76:18001", help="remote_vivado_service URL")
    parser.add_argument("--bit-path", default=None, help="bit 路径")
    parser.add_argument("--ltx-path", default=None, help="ltx 路径")
    parser.add_argument("--board-scripts-dir", default=None, help="板级脚本目录")
    parser.add_argument("--board-runner", default=None, help="manifest runner 脚本路径")
    parser.add_argument("--vivado-settings-sh", default=None, help="本地 Vivado settings64.sh")
    parser.add_argument("--hw-server-url", default=None, help="hw_server URL")
    parser.add_argument("--hw-server-bind", default=None, help="hw_server bind")
    parser.add_argument("--device-index", type=int, default=0, help="板卡索引")
    parser.add_argument("--program-jtag-freq-hz", type=int, default=10_000_000, help="下载 bit 的 JTAG 频率")
    parser.add_argument("--debug-jtag-freq-hz", type=int, default=10_000_000, help="load/run 的 JTAG 频率")
    parser.add_argument("--timeout-sec", type=int, default=3600, help="run-manifest timeout")
    parser.add_argument("--poll-interval-sec", type=float, default=5.0, help="轮询间隔")
    parser.add_argument("--stdout-max-chars", type=int, default=4096, help="stdout drain 上限")
    parser.add_argument("--status-poll-count", type=int, default=None, help="板端 run 轮询次数")
    parser.add_argument("--status-poll-ms", type=int, default=None, help="板端 run 轮询间隔(ms)")
    parser.add_argument("--xdma-h2c-dev", default="/dev/xdma0_h2c_0", help="local-xdma H2C 设备")
    parser.add_argument("--xdma-c2h-dev", default="/dev/xdma0_c2h_0", help="local-xdma C2H 设备")
    parser.add_argument("--xdma-ctrl-dma-base", default="0x0", help="local-xdma DMA-control 基地址")
    parser.add_argument("--xdma-bdf", default="0000:03:00.0", help="local-xdma PCIe BDF，用于预检")
    parser.add_argument(
        "--xdma-require-busy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="local-xdma 要求启动后观察到 Vortex busy/done",
    )
    parser.add_argument(
        "--pcie-split-workaround",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="local-xdma 下把 Guppy attention 后半段拆成第二个 kernel 运行",
    )
    parser.add_argument(
        "--xdma-checkpoint-stage",
        type=int,
        default=None,
        help="local-xdma 单 kernel 调试：写入 guppy_runtime_pcie_split_stage 并回读对应 checkpoint 输出",
    )
    parser.add_argument(
        "--reuse-resident",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="首个 token 用 run-manifest，后续 token 复用 resident session",
    )
    parser.add_argument(
        "--persistent-vivado-session",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否复用常驻 Vivado 控制会话；当前默认关闭以避免已知不稳定问题",
    )
    parser.add_argument(
        "--resident-reload-all-kernel-segments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="resident rerun 时重载全部 kernel 段；仅建议用于小 kernel debug",
    )
    parser.add_argument("--read-logits", action="store_true", help="额外回读完整 last-token logits")
    parser.add_argument("--top-k", type=int, default=8, help="打印 top-k")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="最多生成多少个新 token")
    parser.add_argument(
        "--decode-mode",
        choices=["greedy", "sample"],
        default="greedy",
        help="host 侧 decode 策略",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="sample 模式的温度",
    )
    parser.add_argument(
        "--sample-top-k",
        type=int,
        default=16,
        help="sample 模式的 top-k",
    )
    parser.add_argument("--seed", type=int, default=1234, help="sample 模式随机种子")
    parser.add_argument("--llvm-nm", default=None, help="显式指定 llvm-nm")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    platform_root = default_platform_root(repo_root)
    local_board_scripts_default = find_default_board_scripts_dir(repo_root)
    local_board_runner_default = find_default_board_runner(repo_root)
    local_xdma_runner_default = find_default_xdma_runner(repo_root)
    local_bit_default = (
        platform_root
        / "hw"
        / "syn"
        / "xilinx"
        / "xc7k480t"
        / "xc7k480t_vortex_m1"
        / "xc7k480t_vortex_m1.runs"
        / "impl_1"
        / "xc7k480t_vortex_board_top.bit"
    )
    local_ltx_default = (
        platform_root
        / "hw"
        / "syn"
        / "xilinx"
        / "xc7k480t"
        / "xc7k480t_vortex_m1"
        / "xc7k480t_vortex_m1.runs"
        / "impl_1"
        / "xc7k480t_vortex_board_top.ltx"
    )
    remote_bit_default = (
        "E:/fpga/repo/vx_xc7k480t_jtag_axi_20260405/"
        "jtag_axi_build/jtag_axi_build.runs/impl_1/xc7k480t_vortex_board_top.bit"
    )
    remote_ltx_default = (
        "E:/fpga/repo/vx_xc7k480t_jtag_axi_20260405/"
        "jtag_axi_build/jtag_axi_build.runs/impl_1/xc7k480t_vortex_board_top.ltx"
    )
    remote_board_scripts_default = "E:/fpga/out"
    remote_hw_server_default = "TCP:localhost:3121"
    local_hw_server_default = "TCP:127.0.0.1:53121"

    if args.runner_mode == "local-jtag":
        args.bit_path = str(Path(args.bit_path).expanduser().resolve()) if args.bit_path else str(local_bit_default)
        args.ltx_path = str(Path(args.ltx_path).expanduser().resolve()) if args.ltx_path else str(local_ltx_default)
        args.board_scripts_dir = (
            str(Path(args.board_scripts_dir).expanduser().resolve())
            if args.board_scripts_dir
            else str(local_board_scripts_default)
        )
        args.board_runner = (
            str(Path(args.board_runner).expanduser().resolve())
            if args.board_runner
            else str(local_board_runner_default)
        )
        args.vivado_settings_sh = (
            str(Path(args.vivado_settings_sh).expanduser().resolve())
            if args.vivado_settings_sh
            else "/home/xiao/xilinx/2025.2/Vivado/settings64.sh"
        )
        args.hw_server_url = args.hw_server_url or local_hw_server_default
        args.hw_server_bind = args.hw_server_bind or local_hw_server_default
    elif args.runner_mode == "local-xdma":
        args.bit_path = args.bit_path or ""
        args.ltx_path = args.ltx_path or ""
        args.board_scripts_dir = (
            str(Path(args.board_scripts_dir).expanduser().resolve())
            if args.board_scripts_dir
            else str(local_board_scripts_default)
        )
        args.board_runner = (
            str(Path(args.board_runner).expanduser().resolve())
            if args.board_runner
            else str(local_xdma_runner_default)
        )
        args.vivado_settings_sh = args.vivado_settings_sh or ""
        args.hw_server_url = args.hw_server_url or ""
        args.hw_server_bind = args.hw_server_bind or ""
    else:
        args.bit_path = args.bit_path or remote_bit_default
        args.ltx_path = args.ltx_path or remote_ltx_default
        args.board_scripts_dir = args.board_scripts_dir or remote_board_scripts_default
        args.board_runner = args.board_runner or str(local_board_runner_default)
        args.vivado_settings_sh = args.vivado_settings_sh or "/home/xiao/xilinx/2025.2/Vivado/settings64.sh"
        args.hw_server_url = args.hw_server_url or remote_hw_server_default
        args.hw_server_bind = args.hw_server_bind or remote_hw_server_default

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    stage_dir = Path(args.stage_dir).expanduser().resolve()
    out_dir = stage_dir / "out"
    elf_path = out_dir / "full_inference.elf"
    bundle_prompt = load_json(bundle_dir / "prompt.json")
    bundle_config = load_json(bundle_dir / "model_config.json")
    stage_manifest = load_json(stage_dir / "full_inference_manifest.json")
    if not elf_path.is_file():
        raise FileNotFoundError(f"未找到 ELF: {elf_path}")

    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("需要安装 tokenizers") from exc

    tokenizer = Tokenizer.from_file(str(bundle_dir / "tokenizer.json"))
    messages = build_messages(args, bundle_prompt)
    prompt = format_prompt(messages)
    prompt_ids = tokenizer.encode(prompt).ids

    cfg = bundle_config["normalized_config"]
    seq_len = int(stage_manifest["sequence_length"])
    d_model = int(cfg["d_model"])
    vocab_size = int(cfg["vocab_size"])
    pad_id = int(cfg["pad_id"])
    eos_id = int(cfg["eos_id"])
    if len(prompt_ids) > seq_len:
        raise ValueError(
            f"prompt token 长度 {len(prompt_ids)} 超过当前静态 sequence_length={seq_len}"
        )
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens 必须大于 0")
    if args.temperature <= 0.0:
        raise ValueError("--temperature 必须大于 0")
    if args.sample_top_k <= 0:
        raise ValueError("--sample-top-k 必须大于 0")
    if args.decode_mode == "sample":
        args.read_logits = True

    llvm_nm = args.llvm_nm or find_tool(
        "llvm-nm",
        [
            repo_root / "third_party" / "llvm-vortex-build" / "bin" / "llvm-nm",
            Path("/usr/lib/llvm-18/bin/llvm-nm"),
        ],
    )
    symbols = parse_symbols(elf_path, llvm_nm)
    required = [
        "guppy_input_token_ids",
        "guppy_runtime_prompt_length",
        "guppy_runtime_expect_golden",
        "guppy_output_last_token_argmax",
    ]
    if args.read_logits:
        required.append("guppy_output_last_token_logits")
    if args.runner_mode == "local-xdma" and args.xdma_checkpoint_stage is not None:
        required.extend(
            [
                "guppy_runtime_pcie_split_stage",
                "g_attn_out",
                "g_x_next",
                "g_x_ln2",
                "g_hidden",
                "g_lm_one_ln_out",
                "guppy_output_last_token_logits",
            ]
        )
    missing = [name for name in required if name not in symbols]
    if missing:
        raise KeyError(f"ELF 中缺少符号: {', '.join(missing)}")

    split_elf_path = stage_dir / "out_split_post_attn" / "split_post_attn.elf"
    split_symbols: dict[str, int] = {}
    if args.runner_mode == "local-xdma" and args.pcie_split_workaround:
        if split_elf_path.is_file():
            split_symbols = parse_symbols(split_elf_path, llvm_nm)
        else:
            print(
                f"info: PCIe split workaround ELF 不存在，将回退单 kernel: {split_elf_path}",
                flush=True,
            )

    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    first_msg = messages[0].get("content", "prompt") if messages else "prompt"
    run_name = f"guppy_chat_{safe_name(first_msg)}_{prompt_hash}"
    request_base = {
        "bit_path": args.bit_path,
        "ltx_path": args.ltx_path,
        "board_scripts_dir": args.board_scripts_dir,
        "board_runner": args.board_runner,
        "vivado_settings_sh": args.vivado_settings_sh,
        "device_index": args.device_index,
        "program_jtag_freq_hz": args.program_jtag_freq_hz,
        "debug_jtag_freq_hz": args.debug_jtag_freq_hz,
        "hw_server_url": args.hw_server_url,
        "hw_server_bind": args.hw_server_bind,
        "verify": False,
        "dump_stdout": True,
        "persistent_vivado_session": args.persistent_vivado_session,
        "stdout_max_chars": args.stdout_max_chars,
        "timeout_sec": args.timeout_sec,
        "bundle_dir": str(bundle_dir),
        "xdma_h2c_dev": args.xdma_h2c_dev,
        "xdma_c2h_dev": args.xdma_c2h_dev,
        "xdma_ctrl_dma_base": args.xdma_ctrl_dma_base,
        "xdma_bdf": args.xdma_bdf,
        "xdma_require_busy": args.xdma_require_busy,
        "xdma_split_workaround": args.pcie_split_workaround,
        "xdma_split_elf_path": str(split_elf_path),
        "xdma_split_symbols": split_symbols,
        "xdma_d_model": d_model,
        "xdma_ffn_hidden": int(cfg["ffn_hidden"]),
        "xdma_vocab_size": vocab_size,
        "xdma_checkpoint_stage": args.xdma_checkpoint_stage,
        "xdma_split_d_model": d_model,
        "xdma_split_layer_limit": int(stage_manifest.get("layer_limit", 1)),
    }
    if args.status_poll_count is not None:
        request_base["status_poll_count"] = args.status_poll_count
    if args.status_poll_ms is not None:
        request_base["status_poll_ms"] = args.status_poll_ms

    client = JsonHttpClient()
    if args.runner_mode in {"local-jtag", "local-xdma"} and args.reuse_resident:
        print(
            f"info: {args.runner_mode} 模式当前按 step 全量 rerun，不复用 resident manifest session",
            flush=True,
        )
    rng = random.Random(args.seed)
    current_ids = list(prompt_ids)
    generated_ids: list[int] = []
    step_results: list[dict[str, Any]] = []
    stop_reason = "max_new_tokens"
    resident_job_id: str | None = None

    for step_index in range(args.max_new_tokens):
        if len(current_ids) >= seq_len:
            stop_reason = "sequence_full"
            break
        step_result = run_generation_step(
            runner_mode=args.runner_mode,
            client=client,
            tokenizer=tokenizer,
            service_url=args.service_url,
            elf_path=elf_path,
            stage_dir=stage_dir,
            request_base=request_base,
            symbols=symbols,
            seq_len=seq_len,
            pad_id=pad_id,
            vocab_size=vocab_size,
            token_ids=current_ids,
            run_name=run_name,
            step_index=step_index,
            timeout_sec=args.timeout_sec,
            poll_interval_sec=args.poll_interval_sec,
            read_logits=args.read_logits,
            top_k=args.top_k,
            decode_mode=args.decode_mode,
            temperature=args.temperature,
            sample_top_k=args.sample_top_k,
            rng=rng,
            resident_job_id=resident_job_id if args.reuse_resident else None,
            reload_all_kernel_segments=args.resident_reload_all_kernel_segments,
        )
        step_results.append(step_result)
        resident_job_id = update_resident_job_id(
            resident_job_id,
            step_result,
            reuse_resident=args.reuse_resident,
        )
        next_token_id = int(step_result["next_token_id"])
        next_token_text = str(step_result["next_token_text"])
        generated_ids.append(next_token_id)
        current_ids.append(next_token_id)
        print(
            f"step {step_index}: next_token_id={next_token_id} next_token_text={next_token_text!r}",
            flush=True,
        )
        if next_token_id == eos_id:
            stop_reason = "eos"
            break
        if len(current_ids) >= seq_len:
            stop_reason = "sequence_full"
            break

    generated_text = tokenizer.decode(generated_ids) if generated_ids else ""
    final_job_id = step_results[-1]["job_id"] if step_results else None
    final_status = step_results[-1]["status"] if step_results else "not_run"
    final_exit_code = step_results[-1]["exit_code"] if step_results else None

    result = {
        "job_id": final_job_id,
        "status": final_status,
        "exit_code": final_exit_code,
        "prompt": prompt,
        "input_ids": prompt_ids,
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "full_ids": current_ids,
        "stop_reason": stop_reason,
        "sequence_length_limit": seq_len,
        "decode_mode": args.decode_mode,
        "temperature": args.temperature,
        "sample_top_k": args.sample_top_k,
        "steps": step_results,
    }
    result_path = stage_dir / "chat_last_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    final_request_path = stage_dir / "chat_last_request.json"
    if step_results:
        last_step_request = stage_dir / f"chat_step_{len(step_results) - 1:02d}_request.json"
        if last_step_request.is_file():
            final_request_path.write_text(last_step_request.read_text())
        (stage_dir / "chat_last_job_id.txt").write_text(str(final_job_id) + "\n")

    print(f"job_id: {final_job_id}")
    print(f"status: {final_status}")
    print(f"prompt_token_count: {len(prompt_ids)}")
    print(f"generated_token_count: {len(generated_ids)}")
    print(f"generated_ids: {generated_ids}")
    print(f"generated_text: {generated_text!r}")
    print(f"stop_reason: {stop_reason}")
    if step_results:
        last_stdout = step_results[-1].get("stdout_text") or ""
        if last_stdout:
            print(f"stdout: {last_stdout}")
    print(f"saved request: {final_request_path}")
    print(f"saved result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
