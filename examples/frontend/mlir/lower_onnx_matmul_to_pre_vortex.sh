#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法:
  lower_onnx_matmul_to_pre_vortex.sh \
    --input model.onnx \
    --output model.pre_vortex.mlir \
    [--tile-size 8] \
    [--onnx-mlir /path/to/onnx-mlir] \
    [--onnx-mlir-opt /path/to/onnx-mlir-opt] \
    [--vx-opt /path/to/vx-opt]

说明:
  这条脚本只覆盖当前 MVP 的 matmul 路线:

    ONNX
      -> ONNX Dialect MLIR
      -> linalg + bufferization
      -> tiled pre-vortex

  当前假设:
    1. 输入是静态 shape 的 matmul/Gemm 类 ONNX
    2. onnx-mlir 已经能把核心计算降到 linalg.matmul
    3. tile size 能整除 M/N/K
EOF
}

INPUT=""
OUTPUT=""
TILE_SIZE=8
ONNX_MLIR_BIN="${ONNX_MLIR_BIN:-onnx-mlir}"
ONNX_MLIR_OPT_BIN="${ONNX_MLIR_OPT_BIN:-onnx-mlir-opt}"
VX_OPT_BIN="${VX_OPT_BIN:-vx-opt}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --tile-size)
      TILE_SIZE="$2"
      shift 2
      ;;
    --onnx-mlir)
      ONNX_MLIR_BIN="$2"
      shift 2
      ;;
    --onnx-mlir-opt)
      ONNX_MLIR_OPT_BIN="$2"
      shift 2
      ;;
    --vx-opt)
      VX_OPT_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT}" || -z "${OUTPUT}" ]]; then
  usage >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

ONNX_IR_PREFIX="${TMP_DIR}/frontend"
BUFFERIZED_MLIR="${TMP_DIR}/bufferized.mlir"

"${ONNX_MLIR_BIN}" --EmitONNXIR -o "${ONNX_IR_PREFIX}" "${INPUT}"

"${ONNX_MLIR_OPT_BIN}" "${ONNX_IR_PREFIX}.onnx.mlir" \
  --shape-inference \
  --convert-onnx-to-linalg \
  --canonicalize \
  --empty-tensor-to-alloc-tensor \
  --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
  -o "${BUFFERIZED_MLIR}"

"${VX_OPT_BIN}" "${BUFFERIZED_MLIR}" \
  --allow-unregistered-dialect \
  --pass-pipeline="builtin.module(vortex-onnx-matmul-to-pre-vortex-pipeline{tile-size=${TILE_SIZE}})" \
  -o "${OUTPUT}"

echo "generated: ${OUTPUT}"
