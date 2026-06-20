#!/usr/bin/env python3
"""Write an XDMA manifest for a built Guppy projection probe ELF."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


REQUIRED_SYMBOLS = {
    "probe_status": "probe_status",
    "probe_nan_count": "probe_nan_count",
    "probe_head": "probe_output",
    "probe_fail_index": "probe_fail_index",
    "probe_first_bits": "probe_first_bits",
    "probe_max_diff_bits": "probe_max_diff_bits",
    "probe_num_threads_observed": "probe_num_threads_observed",
    "probe_thread_expected_mask": "probe_thread_expected_mask",
    "probe_thread_nonzero_mask": "probe_thread_nonzero_mask",
    "probe_thread_task_count": "probe_thread_task_count",
}


def find_llvm_nm(explicit: str | None) -> str:
    if explicit:
        return explicit
    for candidate in (
        Path("/home/xiao/tools/llvm-vortex/bin/llvm-nm"),
        Path("/usr/bin/llvm-nm"),
    ):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("llvm-nm")
    if found:
        return found
    raise FileNotFoundError("llvm-nm not found")


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
        if len(parts) >= 3:
            try:
                symbols[parts[-1]] = int(parts[0], 16)
            except ValueError:
                pass
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--kernel-elf", required=True)
    parser.add_argument("--name", default="guppy_projection_probe")
    parser.add_argument("--llvm-nm", default=None)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir).expanduser().resolve()
    kernel_elf = Path(args.kernel_elf).expanduser().resolve()
    symbols = parse_symbols(kernel_elf, find_llvm_nm(args.llvm_nm))
    missing = [symbol for symbol in REQUIRED_SYMBOLS.values() if symbol not in symbols]
    if missing:
        raise RuntimeError(f"ELF missing symbols: {', '.join(missing)}")

    manifest = {
        "schema_version": 1,
        "name": args.name,
        "kernel_elf": str(kernel_elf),
        "startup_addr": "0x80000000",
        "startup_arg": "0x0",
        "expect_exit_word": "0x00000000",
        "require_exit_seen": True,
        "segments": [],
        "outputs": [
            {
                "name": "probe_status",
                "addr": f"0x{symbols['probe_status']:08X}",
                "words": 1,
                "expected_file": "expected_status.mem",
            },
            {
                "name": "probe_nan_count",
                "addr": f"0x{symbols['probe_nan_count']:08X}",
                "words": 1,
                "expected_file": "expected_nan_count.mem",
            },
            {
                "name": "probe_head",
                "addr": f"0x{symbols['probe_output']:08X}",
                "words": 16,
                "expected_file": "expected_head.mem",
                "compare": "float_ulp",
                "ulp": 4096,
            },
            {
                "name": "probe_fail_index",
                "addr": f"0x{symbols['probe_fail_index']:08X}",
                "words": 1,
            },
            {
                "name": "probe_first_bits",
                "addr": f"0x{symbols['probe_first_bits']:08X}",
                "words": 1,
            },
            {
                "name": "probe_max_diff_bits",
                "addr": f"0x{symbols['probe_max_diff_bits']:08X}",
                "words": 1,
            },
            {
                "name": "probe_num_threads_observed",
                "addr": f"0x{symbols['probe_num_threads_observed']:08X}",
                "words": 1,
            },
            {
                "name": "probe_thread_expected_mask",
                "addr": f"0x{symbols['probe_thread_expected_mask']:08X}",
                "words": 1,
            },
            {
                "name": "probe_thread_nonzero_mask",
                "addr": f"0x{symbols['probe_thread_nonzero_mask']:08X}",
                "words": 1,
            },
            {
                "name": "probe_thread_task_count",
                "addr": f"0x{symbols['probe_thread_task_count']:08X}",
                "words": 4,
            },
        ],
    }

    out_path = stage_dir / "local_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
