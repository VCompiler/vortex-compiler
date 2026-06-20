#!/usr/bin/env python3
"""Summarize local-XDMA manifest timing logs under Guppy stage directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TIMING_COLUMNS = [
    "path",
    "total_ms",
    "prepare_stage_ms",
    "xdma_init_ms",
    "load_manifest_ms",
    "start_kernel_ms",
    "wait_kernel_ms",
    "dump_compare_outputs_ms",
    "segment_bytes",
    "payload_verify_ms",
    "result",
    "wait_reason",
    "final_status",
]


def parse_value(value: str) -> int | str:
    try:
        return int(value, 0)
    except ValueError:
        return value


def parse_timing_log(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    by_name: dict[str, list[dict[str, Any]]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("TIMING "):
            continue
        event: dict[str, Any] = {}
        for part in line.split()[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            event[key] = parse_value(value)
        name = str(event.get("name", ""))
        if not name:
            continue
        events.append(event)
        by_name.setdefault(name, []).append(event)

    def sum_elapsed(name: str) -> int:
        return sum(int(event.get("elapsed_ms", 0)) for event in by_name.get(name, []))

    total_events = by_name.get("total", [])
    wait_events = by_name.get("wait_kernel", [])
    load_events = by_name.get("load_manifest", [])
    latest_total = total_events[-1] if total_events else {}
    latest_wait = wait_events[-1] if wait_events else {}
    latest_load = load_events[-1] if load_events else {}
    row = {
        "path": str(path),
        "total_ms": int(latest_total.get("elapsed_ms", 0)),
        "prepare_stage_ms": sum_elapsed("prepare_stage"),
        "xdma_init_ms": sum_elapsed("xdma_init"),
        "load_manifest_ms": sum_elapsed("load_manifest"),
        "start_kernel_ms": sum_elapsed("start_kernel"),
        "wait_kernel_ms": sum_elapsed("wait_kernel"),
        "dump_compare_outputs_ms": sum_elapsed("dump_compare_outputs"),
        "segment_bytes": int(latest_load.get("segment_bytes", 0)),
        "payload_verify_ms": int(latest_load.get("payload_verify_ms", 0)),
        "result": str(latest_total.get("result", "")),
        "wait_reason": str(latest_wait.get("reason", "")),
        "final_status": str(latest_wait.get("final_status", "")),
        "events": events,
    }
    return row


def find_timing_logs(paths: list[Path]) -> list[Path]:
    logs: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_file() and resolved.name == "timing.log":
            logs.append(resolved)
        elif resolved.is_dir():
            logs.extend(sorted(resolved.rglob("timing.log")))
    return logs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Stage dirs or timing.log files")
    parser.add_argument("--json", action="store_true", help="Emit full JSON instead of TSV")
    args = parser.parse_args()

    logs = find_timing_logs([Path(path) for path in args.paths])
    rows = [parse_timing_log(path) for path in logs]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    print("\t".join(TIMING_COLUMNS))
    for row in rows:
        print("\t".join(str(row.get(column, "")) for column in TIMING_COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
