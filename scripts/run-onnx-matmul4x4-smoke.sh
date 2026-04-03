#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

usage() {
  cat <<'EOF'
用法:
  run-onnx-matmul4x4-smoke.sh \
    [--platform-root /path/to/vortex-platform] \
    [--output-dir build/smoke/onnx_matmul4x4_f32] \
    [--python /path/to/python] \
    [--onnx-mlir /path/to/onnx-mlir] \
    [--onnx-mlir-opt /path/to/onnx-mlir-opt] \
    [--driver simx|rtlsim] \
    [--build-sim] \
    [--build-third-party] \
    [--no-proxy] \
    [--make-var 'NAME=VALUE'] ... \
    [--sim-arg ARG] ... \
    [--build-arg ARG] ... \
    [--verbose]

说明:
  1. 先生成固定 4x4 MatMul ONNX 模型
  2. 再走 ONNX -> pre-vortex
  3. 然后继续复用 bare-pointer kernel 构建与 simx/rtlsim 冒烟
EOF
}

log() {
  echo "[onnx-matmul4x4-smoke] $*"
}

die() {
  echo "[onnx-matmul4x4-smoke] error: $*" >&2
  exit 1
}

quote_cmd() {
  local quoted=()
  local arg
  for arg in "$@"; do
    quoted+=("$(printf '%q' "${arg}")")
  done
  printf '%s' "${quoted[*]}"
}

run_cmd() {
  if [[ ${VERBOSE} -eq 1 ]]; then
    echo "+ $(quote_cmd "$@")"
  fi
  "$@"
}

resolve_abs_path() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    (cd "${path}" && pwd -P)
  else
    local dir
    dir="$(cd "$(dirname "${path}")" && pwd -P)"
    printf '%s/%s\n' "${dir}" "$(basename "${path}")"
  fi
}

PYTHON_BIN="${ONNX_FRONT_PYTHON:-python3}"
ONNX_MLIR_BIN="${ONNX_MLIR_BIN:-onnx-mlir}"
ONNX_MLIR_OPT_BIN="${ONNX_MLIR_OPT_BIN:-onnx-mlir-opt}"
PLATFORM_ROOT=""
OUTPUT_DIR="${REPO_ROOT}/build/smoke/onnx_matmul4x4_f32"
DRIVER="simx"
BUILD_SIM=0
BUILD_THIRD_PARTY=0
NO_PROXY=0
VERBOSE=0

PIPELINE="builtin.module(func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-materialize-address-spaces,vortex-lower-linalg-inside-kernel),canonicalize,cse,vortex-legalize-for-llvm,vortex-lower-runtime-builtins,canonicalize,cse,convert-scf-to-cf,convert-arith-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},convert-cf-to-llvm,reconcile-unrealized-casts)"

declare -a BUILD_ARGS=()
declare -a MAKE_VARS=(
  "CONFIGS=-DNUM_CORES=1 -DNUM_WARPS=4 -DNUM_THREADS=4"
)
declare -a SIM_ARGS=(
  -c 1
  -w 4
  -t 4
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform-root)
      PLATFORM_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
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
    --driver)
      DRIVER="$2"
      shift 2
      ;;
    --build-sim)
      BUILD_SIM=1
      shift
      ;;
    --build-third-party)
      BUILD_THIRD_PARTY=1
      shift
      ;;
    --no-proxy)
      NO_PROXY=1
      shift
      ;;
    --make-var)
      MAKE_VARS+=("$2")
      shift 2
      ;;
    --sim-arg)
      SIM_ARGS+=("$2")
      shift 2
      ;;
    --build-arg)
      BUILD_ARGS+=("$2")
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "未知参数: $1"
      ;;
  esac
done

case "${DRIVER}" in
  simx|rtlsim)
    ;;
  *)
    die "--driver 只支持 simx 或 rtlsim"
    ;;
esac

if [[ -n "${PLATFORM_ROOT}" ]]; then
  PLATFORM_ROOT="$(resolve_abs_path "${PLATFORM_ROOT}")"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(resolve_abs_path "${OUTPUT_DIR}")"

ONNX_PATH="${OUTPUT_DIR}/matmul4x4.onnx"
ONNX_META_PATH="${OUTPUT_DIR}/matmul4x4.json"
PRE_VORTEX_PATH="${OUTPUT_DIR}/matmul4x4.pre_vortex.mlir"
WRAPPER_C="${REPO_ROOT}/examples/smoke/matmul4x4_f32_onnx_wrapper.c"
ELF_PATH="${OUTPUT_DIR}/matmul4x4.pre_vortex.elf"

log "生成 ONNX 模型"
run_cmd "${PYTHON_BIN}" \
  "${REPO_ROOT}/examples/frontend/onnx/generate_matmul4x4_model.py" \
  --output "${ONNX_PATH}" \
  --metadata "${ONNX_META_PATH}"

log "降低到 pre-vortex"
run_cmd "${REPO_ROOT}/examples/frontend/mlir/lower_onnx_matmul_to_pre_vortex.sh" \
  --input "${ONNX_PATH}" \
  --output "${PRE_VORTEX_PATH}" \
  --tile-size 4 \
  --onnx-mlir "${ONNX_MLIR_BIN}" \
  --onnx-mlir-opt "${ONNX_MLIR_OPT_BIN}" \
  --vx-opt "${REPO_ROOT}/build/bin/vx-opt"

build_cmd=(
  "${REPO_ROOT}/scripts/build-vortex-kernel.sh"
  --input "${PRE_VORTEX_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --pass-pipeline "${PIPELINE}"
  --extra-source "${WRAPPER_C}"
  --build-runtime
)

if [[ -n "${PLATFORM_ROOT}" ]]; then
  build_cmd+=(--platform-root "${PLATFORM_ROOT}")
fi
if [[ ${VERBOSE} -eq 1 ]]; then
  build_cmd+=(--verbose)
fi
for arg in "${BUILD_ARGS[@]}"; do
  build_cmd+=("${arg}")
done

log "构建 ONNX kernel"
run_cmd "${build_cmd[@]}"

sim_cmd=(
  "${REPO_ROOT}/scripts/run-vortex-sim.sh"
  --driver "${DRIVER}"
  --elf "${ELF_PATH}"
)

if [[ -n "${PLATFORM_ROOT}" ]]; then
  sim_cmd+=(--platform-root "${PLATFORM_ROOT}")
fi
if [[ ${BUILD_SIM} -eq 1 ]]; then
  sim_cmd+=(--build)
fi
if [[ ${BUILD_THIRD_PARTY} -eq 1 ]]; then
  sim_cmd+=(--build-third-party)
fi
if [[ ${NO_PROXY} -eq 1 ]]; then
  sim_cmd+=(--no-proxy)
fi
if [[ ${VERBOSE} -eq 1 ]]; then
  sim_cmd+=(--verbose)
fi
for make_var in "${MAKE_VARS[@]}"; do
  sim_cmd+=(--make-var "${make_var}")
done
for sim_arg in "${SIM_ARGS[@]}"; do
  sim_cmd+=(--sim-arg "${sim_arg}")
done

log "运行 ${DRIVER} 冒烟"
run_cmd "${sim_cmd[@]}"

log "smoke passed: ${ELF_PATH}"
