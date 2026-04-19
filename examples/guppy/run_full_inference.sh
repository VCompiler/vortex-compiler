#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
DIR="${1:?Usage: $0 <out-dir>}"
DIR="$(cd "$DIR" && pwd -P)"

PIPELINE='builtin.module(func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-lower-linalg-inside-kernel),canonicalize,cse,vortex-legalize-for-llvm,vortex-lower-runtime-builtins,canonicalize,cse,convert-scf-to-cf,convert-math-to-llvm,convert-math-to-libm,convert-arith-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},convert-cf-to-llvm,reconcile-unrealized-casts)'

PLATFORM_ROOT="${VORTEX_PLATFORM_ROOT:-/home/user/vortex-platform}"
VX_OPT_BIN="${VX_OPT_BIN:-${REPO_ROOT}/build/bin/vx-opt}"
SIMX_BIN="${VORTEX_SIMX_BIN:-${REPO_ROOT}/build/vortex-sim/simx/simx}"

MLIR="${DIR}/full_inference.mlir"
WRAPPER="${DIR}/full_inference_wrapper.c"
WEIGHTS="${DIR}/full_inference_weights.S"
OUTDIR="${DIR}/out"

[[ -f "${MLIR}" ]] || { echo "error: missing ${MLIR}" >&2; exit 1; }
[[ -f "${WRAPPER}" ]] || { echo "error: missing ${WRAPPER}" >&2; exit 1; }
[[ -f "${WEIGHTS}" ]] || { echo "error: missing ${WEIGHTS}" >&2; exit 1; }
[[ -x "${VX_OPT_BIN}" ]] || { echo "error: missing vx-opt ${VX_OPT_BIN}" >&2; exit 1; }
[[ -x "${SIMX_BIN}" ]] || { echo "error: missing simx ${SIMX_BIN}" >&2; exit 1; }

"${REPO_ROOT}/scripts/build-vortex-kernel.sh" \
  --input "${MLIR}" \
  --output-dir "${OUTDIR}" \
  --platform-root "${PLATFORM_ROOT}" \
  --vx-opt "${VX_OPT_BIN}" \
  --extra-source "${WRAPPER}" \
  --extra-source "${WEIGHTS}" \
  --pass-pipeline "${PIPELINE}"

"${SIMX_BIN}" -c 1 -w 4 -t 4 "${OUTDIR}/full_inference.bin"
