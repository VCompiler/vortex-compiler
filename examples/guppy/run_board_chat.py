#!/usr/bin/env python3
"""一条命令完成 Guppy full board chat：准备资产、生成 stage、编译并上板执行。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PIPELINE = (
    "builtin.module("
    "func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-lower-linalg-inside-kernel),"
    "canonicalize,cse,"
    "vortex-legalize-for-llvm,"
    "vortex-lower-runtime-builtins,"
    "canonicalize,cse,"
    "convert-scf-to-cf,"
    "convert-math-to-llvm,"
    "convert-math-to-libm,"
    "convert-arith-to-llvm,"
    "convert-index-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},"
    "convert-cf-to-llvm,"
    "reconcile-unrealized-casts)"
)


def run_cmd(cmd: list[str], *, cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_repo_relative_path(repo_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def default_platform_root(repo_root: Path) -> Path:
    env = os.environ.get("VORTEX_PLATFORM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    sibling = repo_root.parent / "vortex-platform"
    if sibling.is_dir():
        return sibling.resolve()
    return Path("/home/user/vortex-platform")


def has_assets(assets_dir: Path) -> bool:
    required = [
        assets_dir / "asset_manifest.json",
        assets_dir / "pytorch_model.bin",
        assets_dir / "config.json",
        assets_dir / "tokenizer.json",
    ]
    return all(path.is_file() for path in required)


def has_bundle(bundle_dir: Path) -> bool:
    required = [
        bundle_dir / "manifest.json",
        bundle_dir / "model_config.json",
        bundle_dir / "prompt.json",
        bundle_dir / "tokenizer.json",
        bundle_dir / "weights_index.json",
    ]
    return all(path.is_file() for path in required)


def stage_matches(
    stage_dir: Path,
    *,
    bundle_dir: Path,
    sequence_length: int,
    layer_limit: int,
    tolerance: float,
) -> bool:
    manifest_path = stage_dir / "full_inference_manifest.json"
    required = [
        manifest_path,
        stage_dir / "full_inference.mlir",
        stage_dir / "full_inference_wrapper.c",
        stage_dir / "full_inference_weights.S",
    ]
    if not all(path.is_file() for path in required):
        return False
    manifest = load_json(manifest_path)
    return (
        int(manifest.get("sequence_length", -1)) == sequence_length
        and int(manifest.get("layer_limit", -1)) == layer_limit
        and str(manifest.get("bundle_dir", "")) == str(bundle_dir)
        and abs(float(manifest.get("tolerance", -1.0)) - tolerance) < 1.0e-12
    )


def has_built_elf(stage_dir: Path) -> bool:
    out_dir = stage_dir / "out"
    required = [
        out_dir / "full_inference.elf",
        out_dir / "full_inference.bin",
        out_dir / "full_inference.ll",
        out_dir / "full_inference.s",
    ]
    return all(path.is_file() for path in required)


def ensure_assets(args: argparse.Namespace, repo_root: Path) -> None:
    if has_assets(args.assets_dir) and not args.force_download_assets:
        return
    run_cmd(
        [
            sys.executable,
            str(repo_root / "examples" / "guppy" / "download_assets.py"),
            "--out-dir",
            str(args.assets_dir),
            "--guppylm-root",
            str(args.guppylm_root),
        ],
        cwd=repo_root,
    )


def ensure_bundle(args: argparse.Namespace, repo_root: Path) -> None:
    if has_bundle(args.bundle_dir) and not args.force_export:
        return
    run_cmd(
        [
            sys.executable,
            str(repo_root / "examples" / "guppy" / "guppy_to_vortex.py"),
            "--assets-dir",
            str(args.assets_dir),
            "--guppylm-root",
            str(args.guppylm_root),
            "--messages-json",
            str(args.bundle_messages_json),
            "--out-dir",
            str(args.bundle_dir),
            "--device",
            args.device,
        ],
        cwd=repo_root,
    )


def resolve_stage_shape(args: argparse.Namespace) -> tuple[int, int]:
    cfg = load_json(args.bundle_dir / "model_config.json")["normalized_config"]
    sequence_length = args.sequence_length or int(cfg["max_seq_len"])
    layer_limit = args.layer_limit or int(cfg["n_layers"])
    if sequence_length <= 0 or sequence_length > int(cfg["max_seq_len"]):
        raise ValueError(
            f"sequence-length 必须在 1..{cfg['max_seq_len']} 之间，当前是 {sequence_length}"
        )
    if layer_limit <= 0 or layer_limit > int(cfg["n_layers"]):
        raise ValueError(
            f"layer-limit 必须在 1..{cfg['n_layers']} 之间，当前是 {layer_limit}"
        )
    return sequence_length, layer_limit


def ensure_stage(
    args: argparse.Namespace,
    repo_root: Path,
    *,
    sequence_length: int,
    layer_limit: int,
) -> None:
    if (
        stage_matches(
            args.stage_dir,
            bundle_dir=args.bundle_dir,
            sequence_length=sequence_length,
            layer_limit=layer_limit,
            tolerance=args.tolerance,
        )
        and not args.force_regen
    ):
        return
    run_cmd(
        [
            sys.executable,
            str(repo_root / "examples" / "guppy" / "gen_full_inference.py"),
            "--bundle-dir",
            str(args.bundle_dir),
            "--out-dir",
            str(args.stage_dir),
            "--sequence-length",
            str(sequence_length),
            "--layer-limit",
            str(layer_limit),
            "--tolerance",
            str(args.tolerance),
        ],
        cwd=repo_root,
    )


def ensure_build(args: argparse.Namespace, repo_root: Path) -> None:
    if has_built_elf(args.stage_dir) and not args.force_build:
        return
    vx_opt = repo_root / "build" / "bin" / "vx-opt"
    run_cmd(
        [
            str(repo_root / "scripts" / "build-vortex-kernel.sh"),
            "--input",
            str(args.stage_dir / "full_inference.mlir"),
            "--output-dir",
            str(args.stage_dir / "out"),
            "--platform-root",
            str(args.platform_root),
            "--extra-source",
            str(args.stage_dir / "full_inference_wrapper.c"),
            "--extra-source",
            str(args.stage_dir / "full_inference_weights.S"),
            "--vx-opt",
            str(vx_opt),
            "--pass-pipeline",
            PIPELINE,
        ],
        cwd=repo_root,
    )


def run_board_chat(args: argparse.Namespace, repo_root: Path) -> int:
    cmd = [
        sys.executable,
        str(repo_root / "examples" / "guppy" / "chat_driver.py"),
        "--bundle-dir",
        str(args.bundle_dir),
        "--stage-dir",
        str(args.stage_dir),
        "--service-url",
        args.service_url,
        "--bit-path",
        args.bit_path,
        "--ltx-path",
        args.ltx_path,
        "--board-scripts-dir",
        args.board_scripts_dir,
        "--hw-server-url",
        args.hw_server_url,
        "--hw-server-bind",
        args.hw_server_bind,
        "--device-index",
        str(args.device_index),
        "--program-jtag-freq-hz",
        str(args.program_jtag_freq_hz),
        "--debug-jtag-freq-hz",
        str(args.debug_jtag_freq_hz),
        "--timeout-sec",
        str(args.timeout_sec),
        "--poll-interval-sec",
        str(args.poll_interval_sec),
        "--stdout-max-chars",
        str(args.stdout_max_chars),
        "--top-k",
        str(args.top_k),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--decode-mode",
        args.decode_mode,
        "--temperature",
        str(args.temperature),
        "--sample-top-k",
        str(args.sample_top_k),
        "--seed",
        str(args.seed),
    ]
    if args.status_poll_count is not None:
        cmd.extend(["--status-poll-count", str(args.status_poll_count)])
    if args.status_poll_ms is not None:
        cmd.extend(["--status-poll-ms", str(args.status_poll_ms)])
    if args.persistent_vivado_session:
        cmd.append("--persistent-vivado-session")
    if args.resident_reload_all_kernel_segments:
        cmd.append("--resident-reload-all-kernel-segments")
    if args.prompt_text is not None:
        cmd.extend(["--prompt-text", args.prompt_text])
    if args.messages_json is not None:
        cmd.extend(["--messages-json", str(args.messages_json)])
    if args.read_logits:
        cmd.append("--read-logits")
    if args.llvm_nm:
        cmd.extend(["--llvm-nm", args.llvm_nm])
    run_cmd(cmd, cwd=repo_root)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一条命令完成 Guppy full board chat。"
    )
    parser.add_argument("--assets-dir", default="build/guppy/assets", help="阶段 A 资产目录")
    parser.add_argument("--bundle-dir", default="build/guppy/export", help="阶段 B bundle 目录")
    parser.add_argument("--stage-dir", default=None, help="阶段 C/full stage 目录")
    parser.add_argument(
        "--bundle-messages-json",
        default="examples/guppy/fixed_prompt_messages.json",
        help="缺 bundle 时用于导出 reference 的 messages JSON",
    )
    parser.add_argument(
        "--guppylm-root",
        default=os.path.expanduser("~/guppylm"),
        help="本地 guppylm 仓库目录",
    )
    parser.add_argument("--device", default="cpu", help="导出 reference 时的 torch device")
    parser.add_argument("--sequence-length", type=int, default=None, help="默认取 full max_seq_len")
    parser.add_argument("--layer-limit", type=int, default=None, help="默认取 full n_layers")
    parser.add_argument("--tolerance", type=float, default=5.0e-2, help="wrapper golden 容差")
    parser.add_argument("--platform-root", default=None, help="vortex-platform 根目录")

    parser.add_argument("--prompt-text", default=None, help="快捷输入：单条 user prompt")
    parser.add_argument("--messages-json", default=None, help="运行时消息列表 JSON")
    parser.add_argument("--max-new-tokens", type=int, default=4, help="最多生成多少个新 token")
    parser.add_argument("--decode-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--read-logits", action="store_true", help="额外回读 last-token logits")
    parser.add_argument("--top-k", type=int, default=8, help="打印 top-k")
    parser.add_argument("--llvm-nm", default=None, help="显式指定 llvm-nm")

    parser.add_argument("--service-url", default="http://100.125.4.76:18001")
    parser.add_argument(
        "--bit-path",
        default=(
            "E:/fpga/repo/vx_xc7k480t_jtag_axi_20260405/"
            "jtag_axi_build/jtag_axi_build.runs/impl_1/xc7k480t_vortex_board_top.bit"
        ),
    )
    parser.add_argument(
        "--ltx-path",
        default=(
            "E:/fpga/repo/vx_xc7k480t_jtag_axi_20260405/"
            "jtag_axi_build/jtag_axi_build.runs/impl_1/xc7k480t_vortex_board_top.ltx"
        ),
    )
    parser.add_argument("--board-scripts-dir", default="E:/fpga/out")
    parser.add_argument("--hw-server-url", default="TCP:localhost:3121")
    parser.add_argument("--hw-server-bind", default="TCP:localhost:3121")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--program-jtag-freq-hz", type=int, default=10_000_000)
    parser.add_argument("--debug-jtag-freq-hz", type=int, default=10_000_000)
    parser.add_argument("--timeout-sec", type=int, default=10800)
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    parser.add_argument("--stdout-max-chars", type=int, default=4096)
    parser.add_argument("--status-poll-count", type=int, default=1800)
    parser.add_argument("--status-poll-ms", type=int, default=1000)
    parser.add_argument("--persistent-vivado-session", action="store_true")
    parser.add_argument("--resident-reload-all-kernel-segments", action="store_true")

    parser.add_argument("--force-download-assets", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-regen", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="只准备到 ELF，不上板")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    args.assets_dir = resolve_repo_relative_path(repo_root, args.assets_dir)
    args.bundle_dir = resolve_repo_relative_path(repo_root, args.bundle_dir)
    args.guppylm_root = Path(args.guppylm_root).expanduser().resolve()
    args.bundle_messages_json = resolve_repo_relative_path(
        repo_root, args.bundle_messages_json
    )
    args.platform_root = (
        Path(args.platform_root).expanduser().resolve()
        if args.platform_root
        else default_platform_root(repo_root).resolve()
    )
    if args.messages_json is not None:
        args.messages_json = resolve_repo_relative_path(repo_root, args.messages_json)

    ensure_assets(args, repo_root)
    ensure_bundle(args, repo_root)
    sequence_length, layer_limit = resolve_stage_shape(args)

    if args.stage_dir is None:
        args.stage_dir = (
            repo_root / "build" / "guppy" / f"full_inference_seq{sequence_length}_l{layer_limit}"
        )
    else:
        args.stage_dir = resolve_repo_relative_path(repo_root, args.stage_dir)

    print(f"assets_dir: {args.assets_dir}")
    print(f"bundle_dir: {args.bundle_dir}")
    print(f"stage_dir: {args.stage_dir}")
    print(f"sequence_length: {sequence_length}")
    print(f"layer_limit: {layer_limit}")
    print(f"platform_root: {args.platform_root}")
    print(f"service_url: {args.service_url}")

    ensure_stage(args, repo_root, sequence_length=sequence_length, layer_limit=layer_limit)
    ensure_build(args, repo_root)

    if args.prepare_only:
        print("prepare_only: stage 已准备完成")
        print(f"elf: {args.stage_dir / 'out' / 'full_inference.elf'}")
        return 0

    return run_board_chat(args, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
