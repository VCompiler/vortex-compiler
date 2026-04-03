#!/usr/bin/env python3
"""生成固定 4x4 MatMul ONNX 模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_onnx():
    try:
        import onnx  # type: ignore
        from onnx import TensorProto, helper  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - 运行时环境相关
        raise SystemExit("缺少 onnx 依赖。请先安装 `onnx` 后再运行本脚本。") from exc
    return onnx, TensorProto, helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成固定 4x4 MatMul ONNX 模型。")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().with_name("matmul4x4.onnx")),
        help="输出 ONNX 文件路径。",
    )
    parser.add_argument(
        "--metadata",
        default=str(Path(__file__).resolve().with_name("matmul4x4.json")),
        help="输出元数据 JSON 路径。",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset 版本。",
    )
    return parser.parse_args()


def main() -> int:
    onnx, tensor_proto, helper = require_onnx()
    args = parse_args()

    output_path = Path(args.output).resolve()
    metadata_path = Path(args.metadata).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    input_a = helper.make_tensor_value_info("A", tensor_proto.FLOAT, [4, 4])
    input_b = helper.make_tensor_value_info("B", tensor_proto.FLOAT, [4, 4])
    output_c = helper.make_tensor_value_info("C", tensor_proto.FLOAT, [4, 4])

    matmul = helper.make_node("MatMul", ["A", "B"], ["C"])
    graph = helper.make_graph([matmul], "matmul4x4_graph", [input_a, input_b], [output_c])
    model = helper.make_model(
        graph,
        producer_name="vortex-compiler",
        opset_imports=[helper.make_operatorsetid("", args.opset)],
    )
    model.ir_version = 10

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    metadata = {
        "model": "matmul4x4",
        "opset": args.opset,
        "inputs": [
            {"name": "A", "shape": [4, 4], "dtype": "float32"},
            {"name": "B", "shape": [4, 4], "dtype": "float32"},
        ],
        "outputs": [
            {"name": "C", "shape": [4, 4], "dtype": "float32"},
        ],
        "stage": "onnx_matmul4x4_generator",
        "onnx_file": output_path.name,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"generated: {output_path}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
