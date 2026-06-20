#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

usage() {
  cat <<'EOF'
用法:
  run-vortex-sim.sh \
    [--platform-root /path/to/vortex-platform] \
    [--driver rtlsim|simx] \
    (--bin kernel.bin | --elf kernel.elf) \
    [--objcopy /path/to/llvm-objcopy] \
    [--sim-build-root /path/to/build/vortex-sim] \
    [--build] \
    [--build-third-party] \
    [--no-proxy] \
    [--work-dir /path/to/tmp] \
    [--keep-work-dir] \
    [--make-var 'NAME=VALUE'] ... \
    [--sim-arg ARG] ... \
    [--dry-run] \
    [--verbose]

说明:
  1. 当前仓库负责产出 ELF/bin
  2. 本脚本负责桥接 vortex-platform 的 simx / rtlsim
  3. simulator 默认构建到当前仓库:
       build/vortex-sim/<driver>/

platform-root 解析顺序:
  1. --platform-root
  2. 环境变量 VORTEX_PLATFORM_ROOT
  3. third_party/vortex-platform

示例:
  scripts/run-vortex-sim.sh \
    --platform-root /path/to/vortex-platform \
    --driver rtlsim \
    --elf build/out/kernel.elf \
    --build \
    --make-var 'CONFIGS=-DNUM_CORES=1 -DNUM_WARPS=4 -DNUM_THREADS=4'

  scripts/run-vortex-sim.sh \
    --platform-root /path/to/vortex-platform \
    --driver simx \
    --bin build/out/kernel.bin \
    --build \
    --sim-arg -c --sim-arg 1 \
    --sim-arg -w --sim-arg 4 \
    --sim-arg -t --sim-arg 4
EOF
}

log() {
  echo "[vortex-sim] $*"
}

die() {
  echo "[vortex-sim] error: $*" >&2
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

cmd_prefix() {
  if [[ ${NO_PROXY_MODE} -eq 1 ]]; then
    printf '%s\0' \
      env \
      -u ALL_PROXY \
      -u all_proxy \
      -u HTTP_PROXY \
      -u http_proxy \
      -u HTTPS_PROXY \
      -u https_proxy \
      -u NO_PROXY \
      -u no_proxy
  fi
}

run_cmd() {
  local cmd=()
  if [[ ${NO_PROXY_MODE} -eq 1 ]]; then
    while IFS= read -r -d '' token; do
      cmd+=("${token}")
    done < <(cmd_prefix)
  fi
  cmd+=("$@")

  if [[ ${VERBOSE} -eq 1 || ${DRY_RUN} -eq 1 ]]; then
    echo "+ $(quote_cmd "${cmd[@]}")"
  fi
  if [[ ${DRY_RUN} -eq 0 ]]; then
    "${cmd[@]}"
  fi
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

find_objcopy() {
  if [[ -n "${OBJCOPY_BIN}" ]]; then
    printf '%s\n' "${OBJCOPY_BIN}"
    return
  fi

  local repo_objcopy="${REPO_ROOT}/third_party/llvm-build/bin/llvm-objcopy"
  if [[ -x "${repo_objcopy}" ]]; then
    printf '%s\n' "${repo_objcopy}"
    return
  fi

  if command -v llvm-objcopy >/dev/null 2>&1; then
    command -v llvm-objcopy
    return
  fi

  if command -v objcopy >/dev/null 2>&1; then
    command -v objcopy
    return
  fi

  die "未找到 llvm-objcopy/objcopy，请通过 --objcopy 指定"
}

ensure_platform_root() {
  local candidate=""

  if [[ -n "${PLATFORM_ROOT}" ]]; then
    candidate="${PLATFORM_ROOT}"
  elif [[ -n "${VORTEX_PLATFORM_ROOT:-}" ]]; then
    candidate="${VORTEX_PLATFORM_ROOT}"
  elif [[ -d "${REPO_ROOT}/third_party/vortex-platform" ]]; then
    candidate="${REPO_ROOT}/third_party/vortex-platform"
  else
    die "未找到 vortex-platform，请用 --platform-root 指定，或设置 VORTEX_PLATFORM_ROOT"
  fi

  PLATFORM_ROOT="$(resolve_abs_path "${candidate}")"
  [[ -d "${PLATFORM_ROOT}" ]] || die "vortex-platform 路径不存在: ${PLATFORM_ROOT}"
  [[ -f "${PLATFORM_ROOT}/sim/${DRIVER}/Makefile" ]] || die "缺少 simulator Makefile: ${PLATFORM_ROOT}/sim/${DRIVER}/Makefile"
  [[ -f "${PLATFORM_ROOT}/third_party/Makefile" ]] || die "缺少 third_party/Makefile: ${PLATFORM_ROOT}/third_party/Makefile"
}

ensure_input_image() {
  if [[ -n "${INPUT_BIN}" && -n "${INPUT_ELF}" ]]; then
    die "--bin 和 --elf 只能二选一"
  fi
  if [[ -z "${INPUT_BIN}" && -z "${INPUT_ELF}" ]]; then
    die "必须提供 --bin 或 --elf"
  fi

  if [[ -n "${INPUT_BIN}" ]]; then
    INPUT_BIN="$(resolve_abs_path "${INPUT_BIN}")"
    [[ -f "${INPUT_BIN}" ]] || die "bin 不存在: ${INPUT_BIN}"
    PROGRAM_BIN="${INPUT_BIN}"
    return
  fi

  INPUT_ELF="$(resolve_abs_path "${INPUT_ELF}")"
  [[ -f "${INPUT_ELF}" ]] || die "elf 不存在: ${INPUT_ELF}"

  if [[ -z "${WORK_DIR}" ]]; then
    mkdir -p "${REPO_ROOT}/build"
    WORK_DIR="$(mktemp -d "${REPO_ROOT}/build/vortex-sim-work.XXXXXX")"
    CREATED_TEMP_WORK_DIR=1
  else
    mkdir -p "${WORK_DIR}"
    WORK_DIR="$(resolve_abs_path "${WORK_DIR}")"
  fi

  local objcopy
  objcopy="$(find_objcopy)"
  PROGRAM_BIN="${WORK_DIR}/$(basename "${INPUT_ELF%.*}").bin"
  log "生成 bin: ${PROGRAM_BIN}"
  run_cmd "${objcopy}" -O binary "${INPUT_ELF}" "${PROGRAM_BIN}"
}

cleanup() {
  if [[ ${CREATED_TEMP_WORK_DIR} -eq 1 && ${KEEP_WORK_DIR} -eq 0 ]]; then
    rm -rf "${WORK_DIR}"
  fi
}

ensure_third_party() {
  local softfloat_src="${PLATFORM_ROOT}/third_party/softfloat/build/Linux-x86_64-GCC/Makefile"
  local ramulator_src="${PLATFORM_ROOT}/third_party/ramulator/CMakeLists.txt"
  local softfloat_lib="${PLATFORM_ROOT}/third_party/softfloat/build/Linux-x86_64-GCC/softfloat.a"
  local ramulator_lib="${PLATFORM_ROOT}/third_party/ramulator/libramulator.so"

  if [[ ! -f "${softfloat_src}" || ! -f "${ramulator_src}" ]]; then
    if [[ ${DRY_RUN} -eq 1 ]]; then
      log "dry-run: vortex-platform third_party 未初始化完整，跳过源码完整性校验"
      return
    fi
    die "vortex-platform third_party 未初始化完整，请先在 ${PLATFORM_ROOT} 执行: git submodule update --init --recursive"
  fi

  if [[ ${BUILD_THIRD_PARTY} -eq 1 || ! -f "${softfloat_lib}" || ! -f "${ramulator_lib}" ]]; then
    log "构建 vortex-platform third_party"
    run_cmd env "VORTEX_HOME=${PLATFORM_ROOT}" make -C "${PLATFORM_ROOT}/third_party"
  fi
}

ensure_hw_config() {
  local vx_config="${PLATFORM_ROOT}/hw/VX_config.h"
  if [[ ! -f "${vx_config}" || ${BUILD_SIMULATOR} -eq 1 ]]; then
    log "生成 Vortex 硬件配置头"
    run_cmd env "VORTEX_HOME=${PLATFORM_ROOT}" make -C "${PLATFORM_ROOT}/hw" config
  fi
}

ensure_simulator() {
  mkdir -p "${SIM_BUILD_ROOT}"
  SIM_BUILD_ROOT="$(resolve_abs_path "${SIM_BUILD_ROOT}")"
  SIM_OUT_DIR="${SIM_BUILD_ROOT}/${DRIVER}"
  SIM_BIN="${SIM_OUT_DIR}/${DRIVER}"

  if [[ ${BUILD_SIMULATOR} -eq 0 && -x "${SIM_BIN}" ]]; then
    log "复用已有 simulator: ${SIM_BIN}"
    return
  fi

  if [[ "${DRIVER}" == "rtlsim" ]] && ! command -v verilator >/dev/null 2>&1; then
    if [[ ${DRY_RUN} -eq 1 ]]; then
      log "dry-run: 当前 PATH 中未找到 verilator，跳过依赖校验"
    else
      die "rtlsim 需要 verilator，但当前 PATH 中未找到"
    fi
  fi

  ensure_third_party
  ensure_hw_config

  mkdir -p "${SIM_OUT_DIR}"
  local make_cmd=(
    env
    "VORTEX_HOME=${PLATFORM_ROOT}"
    make
    -C "${PLATFORM_ROOT}/sim/${DRIVER}"
    "DESTDIR=${SIM_OUT_DIR}"
  )
  local var
  for var in "${MAKE_VARS[@]}"; do
    make_cmd+=("${var}")
  done

  log "构建 ${DRIVER}: ${SIM_BIN}"
  run_cmd "${make_cmd[@]}"
}

run_simulator() {
  local cmd=("${SIM_BIN}")
  local arg
  for arg in "${SIM_ARGS[@]}"; do
    cmd+=("${arg}")
  done
  cmd+=("${PROGRAM_BIN}")

  log "运行 ${DRIVER}: ${PROGRAM_BIN}"
  run_cmd "${cmd[@]}"
}

DRIVER="rtlsim"
PLATFORM_ROOT=""
SIM_BUILD_ROOT="${REPO_ROOT}/build/vortex-sim"
INPUT_BIN=""
INPUT_ELF=""
PROGRAM_BIN=""
OBJCOPY_BIN="${LLVM_OBJCOPY:-}"
WORK_DIR=""
KEEP_WORK_DIR=0
CREATED_TEMP_WORK_DIR=0
BUILD_SIMULATOR=0
BUILD_THIRD_PARTY=0
NO_PROXY_MODE=0
DRY_RUN=0
VERBOSE=0
MAKE_VARS=()
SIM_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform-root)
      PLATFORM_ROOT="$2"
      shift 2
      ;;
    --driver)
      DRIVER="$2"
      shift 2
      ;;
    --bin)
      INPUT_BIN="$2"
      shift 2
      ;;
    --elf)
      INPUT_ELF="$2"
      shift 2
      ;;
    --objcopy)
      OBJCOPY_BIN="$2"
      shift 2
      ;;
    --sim-build-root)
      SIM_BUILD_ROOT="$2"
      shift 2
      ;;
    --build|--rebuild)
      BUILD_SIMULATOR=1
      shift
      ;;
    --build-third-party)
      BUILD_THIRD_PARTY=1
      shift
      ;;
    --no-proxy)
      NO_PROXY_MODE=1
      shift
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --keep-work-dir)
      KEEP_WORK_DIR=1
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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        SIM_ARGS+=("$1")
        shift
      done
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

trap cleanup EXIT

ensure_platform_root
ensure_input_image
ensure_simulator
run_simulator
