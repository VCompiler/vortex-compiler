#!/usr/bin/env python3
"""第一阶段 PyTorch -> ONNX 导出入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models import MODEL_SPECS


def _require_export_deps():
    try:
        import torch  # type: ignore
        import onnx  # noqa: F401  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - 运行时环境相关
        raise SystemExit(
            "缺少导出依赖。请先安装 `torch` 和 `onnx` 后再运行本脚本。"
        ) from exc
    return torch


def export_one(model_name: str, output_dir: Path, opset: int) -> None:
    torch = _require_export_deps()
    spec = MODEL_SPECS[model_name]
    model = spec.build()
    dummy_input = torch.randn(*spec.input_shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"
    meta_path = output_dir / f"{model_name}.json"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    meta = {
        "model": model_name,
        "input_shape": list(spec.input_shape),
        "onnx_file": onnx_path.name,
        "opset": opset,
        "stage": "phase1_pytorch_export",
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"exported: {onnx_path}")
    print(f"metadata: {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出第一阶段 PyTorch toy 模型为 ONNX。"
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *MODEL_SPECS.keys()],
        help="选择要导出的模型。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).resolve().parents[1] / "onnx"
        ),
        help="ONNX 输出目录。",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset 版本。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    model_names = (
        list(MODEL_SPECS.keys()) if args.model == "all" else [args.model]
    )

    for model_name in model_names:
        export_one(model_name, output_dir, args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
