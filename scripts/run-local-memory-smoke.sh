#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

usage() {
  cat <<'EOF'
Usage:
  run-local-memory-smoke.sh \
    [--platform-root /path/to/vortex-platform] \
    [--output-dir build/smoke/local_memory_i32] \
    [--driver simx|rtlsim|board] \
    [--build-sim] \
    [--build-third-party] \
    [--no-proxy] \
    [--make-var 'NAME=VALUE'] ... \
    [--sim-arg ARG] ... \
    [--board-arg ARG] ... \
    [--build-arg ARG] ... \
    [--verbose]

Builds and runs a minimal Vortex local-memory smoke. The generated assembly is
checked to contain a CSR read of lmem_base and no call to vx_local_mem_base.
EOF
}

log() {
  echo "[local-memory-smoke] $*"
}

die() {
  echo "[local-memory-smoke] error: $*" >&2
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

PIPELINE="builtin.module(func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-materialize-address-spaces),func.func(vortex-plan-local-memory-layout),vortex-lower-local-memory,canonicalize,cse,vortex-legalize-for-llvm,vortex-lower-runtime-builtins,canonicalize,cse,convert-scf-to-cf,convert-arith-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},convert-cf-to-llvm,reconcile-unrealized-casts)"

PLATFORM_ROOT=""
OUTPUT_DIR="${REPO_ROOT}/build/smoke/local_memory_i32"
DRIVER="simx"
BUILD_SIM=0
BUILD_THIRD_PARTY=0
NO_PROXY=0
VERBOSE=0

declare -a BUILD_ARGS=()
declare -a MAKE_VARS=(
  "CONFIGS=-DNUM_CORES=1 -DNUM_WARPS=4 -DNUM_THREADS=4"
)
declare -a SIM_ARGS=(
  -c 1
  -w 4
  -t 4
)
declare -a BOARD_ARGS=(
  --hw-server-url
  TCP:localhost:3121
  --hw-server-bind
  ""
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
    --board-arg)
      BOARD_ARGS+=("$2")
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
      die "unknown argument: $1"
      ;;
  esac
done

case "${DRIVER}" in
  simx|rtlsim|board)
    ;;
  *)
    die "--driver only supports simx, rtlsim, or board"
    ;;
esac

if [[ -n "${PLATFORM_ROOT}" ]]; then
  PLATFORM_ROOT="$(resolve_abs_path "${PLATFORM_ROOT}")"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(resolve_abs_path "${OUTPUT_DIR}")"

INPUT_MLIR="${REPO_ROOT}/examples/smoke/local_memory_i32.mlir"
WRAPPER_C="${REPO_ROOT}/examples/smoke/local_memory_i32_wrapper.c"
ELF_PATH="${OUTPUT_DIR}/local_memory_i32.elf"
ASM_PATH="${OUTPUT_DIR}/local_memory_i32.s"

build_cmd=(
  "${REPO_ROOT}/scripts/build-vortex-kernel.sh"
  --input "${INPUT_MLIR}"
  --output-dir "${OUTPUT_DIR}"
  --pass-pipeline "${PIPELINE}"
  --extra-source "${WRAPPER_C}"
  --codegen-backend llc
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

log "building local-memory smoke kernel"
run_cmd "${build_cmd[@]}"

[[ -f "${ASM_PATH}" ]] || die "missing assembly: ${ASM_PATH}"
if ! grep -Eq '(^|[[:space:]])csrr[[:space:]]+[^,]+,[[:space:]]+lmem_base' "${ASM_PATH}"; then
  die "expected csrr ..., lmem_base in ${ASM_PATH}"
fi
if grep -Eq '(^|[[:space:]])call[[:space:]]+vx_local_mem_base' "${ASM_PATH}"; then
  die "unexpected call vx_local_mem_base in ${ASM_PATH}"
fi
log "assembly check passed: csrr ..., lmem_base"

if [[ "${DRIVER}" == "board" ]]; then
  [[ -n "${PLATFORM_ROOT}" ]] || die "--driver board requires --platform-root"
  BOARD_RUNNER="${PLATFORM_ROOT}/hw/syn/xilinx/xc7k480t/board_run_elf.sh"
  [[ -f "${BOARD_RUNNER}" ]] || die "missing board runner: ${BOARD_RUNNER}"

  board_cmd=(
    "${BOARD_RUNNER}"
    "${ELF_PATH}"
  )
  if [[ ${#BOARD_ARGS[@]} -gt 0 ]]; then
    board_cmd+=(-- "${BOARD_ARGS[@]}")
  fi

  log "running board smoke"
  run_cmd "${board_cmd[@]}"
else
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

  log "running ${DRIVER} smoke"
  run_cmd "${sim_cmd[@]}"
fi

log "smoke passed: ${ELF_PATH}"
