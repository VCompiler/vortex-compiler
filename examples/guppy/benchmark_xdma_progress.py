#!/usr/bin/env python3
"""Benchmark cumulative Guppy progress checkpoints through local XDMA.

The generated Guppy kernel calls ``guppy_set_progress(N)`` at key points.  The
wrapper exits early when ``guppy_runtime_pcie_split_stage == N`` and golden
comparison is disabled.  This script creates temporary manifests that set that
runtime field to each checkpoint, runs them through the repo-native XDMA
runner, and writes a small TSV summary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_RUNNER = (
    "/home/xiao/vortex-platform/hw/syn/xilinx/xc7k480t/"
    "run_regression_manifest_xdma.py"
)
DEFAULT_NM = "/home/xiao/tools/riscv32-gnu-toolchain/bin/riscv32-unknown-elf-nm"


def parse_int(value: str) -> int:
    text = str(value).strip()
    if text.lower().startswith("0x"):
        return int(text, 16)
    if any(ch in text.lower() for ch in "abcdef") or len(text) == 8:
        return int(text, 16)
    return int(text, 10)


def int_to_word(value: int) -> str:
    return f"{value & 0xFFFFFFFF:08X}"


def write_words(path: Path, words: list[int]) -> None:
    path.write_text("".join(f"{int_to_word(word)}\n" for word in words), encoding="utf-8")


def read_first_mem_word(path: Path) -> int | None:
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line:
            return parse_int(line)
    return None


def load_symbols(elf: Path, nm: Path) -> dict[str, int]:
    output = subprocess.check_output([str(nm), "-n", str(elf)], text=True)
    symbols: dict[str, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            symbols[parts[2]] = int(parts[0], 16)
    return symbols


def parse_timing(path: Path) -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.startswith("TIMING "):
            continue
        fields: dict[str, str] = {}
        for part in raw.split()[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
        name = fields.get("name")
        if not name:
            continue
        elapsed = fields.get("elapsed_ms")
        if elapsed is not None:
            try:
                result[f"{name}_ms"] = int(elapsed, 0)
            except ValueError:
                result[f"{name}_ms"] = elapsed
        if name == "wait_kernel":
            result["final_status"] = fields.get("final_status", "")
            result["reason"] = fields.get("reason", "")
    return result


def parse_run_summary(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    if not path.is_file():
        return summary
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        if key in {"FINAL_STATUS", "FINAL_REASON", "FINAL_EXIT_WORD", "RUN_PASS"}:
            summary[key.lower()] = value.strip()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--elf", type=Path)
    parser.add_argument("--runner", type=Path, default=Path(DEFAULT_RUNNER))
    parser.add_argument("--nm", type=Path, default=Path(DEFAULT_NM))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--token-mem", type=Path)
    parser.add_argument("--prompt-len", type=int)
    parser.add_argument("--checkpoints", default="20,21,22,23,24,25,26,27,28,29")
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--bdf", default="0000:03:00.0")
    parser.add_argument("--h2c-dev", default="/dev/xdma0_h2c_0")
    parser.add_argument("--c2h-dev", default="/dev/xdma0_c2h_0")
    parser.add_argument("--ctrl-dma-base", default="0x0")
    args = parser.parse_args()

    stage_dir = args.stage_dir.expanduser().resolve()
    elf = (args.elf or stage_dir / "out" / "full_inference.elf").expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else stage_dir / "xdma_progress_bench"
    )
    token_mem = (
        args.token_mem.expanduser().resolve()
        if args.token_mem
        else stage_dir
        / "xdma_runs"
        / "step_00"
        / "split_stage0_attn_merge"
        / "input_token_ids.mem"
    )
    prompt_len = args.prompt_len
    if prompt_len is None:
        control_mem = (
            stage_dir
            / "xdma_runs"
            / "step_00"
            / "split_stage0_attn_merge"
            / "runtime_control_block.mem"
        )
        prompt_len = read_first_mem_word(control_mem)
    if prompt_len is None:
        raise RuntimeError("prompt length not provided and could not be inferred")
    if not token_mem.is_file():
        raise FileNotFoundError(f"token mem not found: {token_mem}")
    if not elf.is_file():
        raise FileNotFoundError(f"ELF not found: {elf}")
    if not args.runner.is_file():
        raise FileNotFoundError(f"XDMA runner not found: {args.runner}")

    checkpoints = [
        parse_int(part)
        for part in args.checkpoints.replace(",", " ").split()
        if part.strip()
    ]
    symbols = load_symbols(elf, args.nm.expanduser().resolve())
    required = [
        "guppy_input_token_ids",
        "guppy_runtime_prompt_length",
        "guppy_progress_stage",
    ]
    missing = [name for name in required if name not in symbols]
    if missing:
        raise RuntimeError(f"missing ELF symbols: {', '.join(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, str | int]] = []
    for checkpoint in checkpoints:
        run_dir = out_dir / f"p{checkpoint:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(token_mem, run_dir / "input_token_ids.mem")
        write_words(
            run_dir / "runtime_control_block.mem",
            [prompt_len, 0, checkpoint] + [0] * 13,
        )
        manifest = {
            "schema_version": 1,
            "name": f"guppy_progress_{checkpoint:02d}",
            "kernel_elf": str(elf),
            "startup_addr": "0x80000000",
            "startup_arg": "0x0",
            "expect_exit_word": "0x00000000",
            "require_exit_seen": True,
            "segments": [
                {
                    "name": "input_token_ids",
                    "mem_file": "input_token_ids.mem",
                    "addr": f"0x{symbols['guppy_input_token_ids']:08X}",
                    "byte_len": "0x40",
                },
                {
                    "name": "runtime_control_block",
                    "mem_file": "runtime_control_block.mem",
                    "addr": f"0x{symbols['guppy_runtime_prompt_length']:08X}",
                    "byte_len": "0x40",
                },
            ],
            "outputs": [
                {
                    "name": "progress_stage",
                    "addr": f"0x{symbols['guppy_progress_stage']:08X}",
                    "words": 1,
                }
            ],
        }
        manifest_path = run_dir / "local_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        cmd = [
            sys.executable,
            str(args.runner),
            "--manifest",
            str(manifest_path),
            "--stage-dir",
            str(run_dir),
            "--bdf",
            args.bdf,
            "--h2c-dev",
            args.h2c_dev,
            "--c2h-dev",
            args.c2h_dev,
            "--ctrl-dma-base",
            args.ctrl_dma_base,
            "--timeout-sec",
            str(args.timeout_sec),
            "--require-busy",
            "1",
            "--status-interval-sec",
            "5",
        ]
        rc = subprocess.call(cmd, cwd=run_dir)
        timing = parse_timing(run_dir / "timing.log")
        summary = parse_run_summary(run_dir / "run.log")
        progress = read_first_mem_word(run_dir / "progress_stage_actual.mem")
        row: dict[str, str | int] = {
            "checkpoint": checkpoint,
            "rc": rc,
            "progress_actual": "" if progress is None else progress,
            "run_pass": summary.get("run_pass", ""),
            "final_status": summary.get("final_status", timing.get("final_status", "")),
            "final_reason": summary.get("final_reason", timing.get("reason", "")),
            "total_ms": timing.get("total_ms", ""),
            "wait_kernel_ms": timing.get("wait_kernel_ms", ""),
            "load_manifest_ms": timing.get("load_manifest_ms", ""),
        }
        result_rows.append(row)
        print(
            "CHECKPOINT "
            + " ".join(f"{key}={value}" for key, value in row.items()),
            flush=True,
        )
        if rc != 0:
            break

    result_path = out_dir / "results.tsv"
    columns = [
        "checkpoint",
        "rc",
        "progress_actual",
        "run_pass",
        "final_status",
        "final_reason",
        "total_ms",
        "wait_kernel_ms",
        "load_manifest_ms",
    ]
    with result_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in result_rows:
            f.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")
    print(f"results_tsv={result_path}", flush=True)
    return 0 if all(int(row["rc"]) == 0 for row in result_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
