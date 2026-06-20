#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

usage() {
  cat <<'EOF'
用法:
  run-vortex-board-xdma.sh \
    --manifest local_manifest.json \
    [--platform-root /path/to/vortex-platform] \
    [--stage-dir build/xdma_run] \
    [--timeout-sec 60] \
    [--require-busy 1|0] \
    [--status-interval-sec 2] \
    [--bdf 0000:03:00.0] \
    [--control-dev /dev/xdma0_control] \
    [--h2c-dev /dev/xdma0_h2c_0] \
    [--c2h-dev /dev/xdma0_c2h_0] \
    [--ctrl-dma-base 0x0] \
    [--runner-arg ARG] ... \
    [--verbose]

说明:
  1. 运行已经生成好的 local-XDMA manifest
  2. 运行前检查 PCI COMMAND 的 Mem Space/BusMaster 位和 /dev/xdma* 节点
  3. 调用 vortex-platform 的 run_regression_manifest_xdma.py
  4. 运行后打印 run.log 中的 FINAL_* 状态
EOF
}

log() {
  echo "[vortex-board-xdma] $*"
}

die() {
  echo "[vortex-board-xdma] error: $*" >&2
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

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1"
}

MANIFEST=""
PLATFORM_ROOT=""
STAGE_DIR=""
TIMEOUT_SEC=60
REQUIRE_BUSY=1
STATUS_INTERVAL_SEC=2
BDF="0000:03:00.0"
CONTROL_DEV="/dev/xdma0_control"
H2C_DEV="/dev/xdma0_h2c_0"
C2H_DEV="/dev/xdma0_c2h_0"
CTRL_DMA_BASE="0x0"
VERBOSE=0

declare -a RUNNER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --platform-root)
      PLATFORM_ROOT="$2"
      shift 2
      ;;
    --stage-dir)
      STAGE_DIR="$2"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --require-busy)
      REQUIRE_BUSY="$2"
      shift 2
      ;;
    --status-interval-sec)
      STATUS_INTERVAL_SEC="$2"
      shift 2
      ;;
    --bdf)
      BDF="$2"
      shift 2
      ;;
    --control-dev)
      CONTROL_DEV="$2"
      shift 2
      ;;
    --h2c-dev)
      H2C_DEV="$2"
      shift 2
      ;;
    --c2h-dev)
      C2H_DEV="$2"
      shift 2
      ;;
    --ctrl-dma-base)
      CTRL_DMA_BASE="$2"
      shift 2
      ;;
    --runner-arg)
      RUNNER_ARGS+=("$2")
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

[[ -n "${MANIFEST}" ]] || die "缺少 --manifest"
MANIFEST="$(resolve_abs_path "${MANIFEST}")"
[[ -f "${MANIFEST}" ]] || die "manifest 不存在: ${MANIFEST}"

if [[ -n "${PLATFORM_ROOT}" ]]; then
  PLATFORM_ROOT="$(resolve_abs_path "${PLATFORM_ROOT}")"
elif [[ -n "${VORTEX_PLATFORM_ROOT:-}" ]]; then
  PLATFORM_ROOT="$(resolve_abs_path "${VORTEX_PLATFORM_ROOT}")"
elif [[ -d "${REPO_ROOT}/../vortex-platform" ]]; then
  PLATFORM_ROOT="$(resolve_abs_path "${REPO_ROOT}/../vortex-platform")"
else
  die "未找到 vortex-platform，请通过 --platform-root 或 VORTEX_PLATFORM_ROOT 指定"
fi

[[ -d "${PLATFORM_ROOT}" ]] || die "vortex-platform 路径不存在: ${PLATFORM_ROOT}"

XDMA_RUNNER="${PLATFORM_ROOT}/hw/syn/xilinx/xc7k480t/run_regression_manifest_xdma.py"
[[ -f "${XDMA_RUNNER}" ]] || die "缺少 XDMA runner: ${XDMA_RUNNER}"

if [[ -z "${STAGE_DIR}" ]]; then
  STAGE_DIR="$(dirname "${MANIFEST}")/xdma_run"
else
  STAGE_DIR="$(resolve_abs_path "${STAGE_DIR}")"
fi
mkdir -p "${STAGE_DIR}"

require_cmd python3
require_cmd setpci
require_cmd lspci

if ! lspci -s "${BDF}" >/dev/null 2>&1; then
  die "找不到 PCI endpoint ${BDF}"
fi

COMMAND_HEX="$(setpci -s "${BDF}" COMMAND)"
COMMAND_VAL=$((16#${COMMAND_HEX}))
if (((COMMAND_VAL & 0x6) != 0x6)); then
  die "PCI COMMAND=0x$(printf '%04x' "${COMMAND_VAL}")，缺少 Mem Space/BusMaster 位；先恢复 XDMA endpoint"
fi

for dev in "${CONTROL_DEV}" "${H2C_DEV}" "${C2H_DEV}"; do
  [[ -e "${dev}" ]] || die "缺少 XDMA 设备节点: ${dev}"
done

log "manifest: ${MANIFEST}"
log "stage dir: ${STAGE_DIR}"
log "PCI ${BDF} COMMAND=0x$(printf '%04x' "${COMMAND_VAL}")"

xdma_cmd=(
  python3
  "${XDMA_RUNNER}"
  --manifest "${MANIFEST}"
  --stage-dir "${STAGE_DIR}"
  --timeout-sec "${TIMEOUT_SEC}"
  --require-busy "${REQUIRE_BUSY}"
  --status-interval-sec "${STATUS_INTERVAL_SEC}"
  --bdf "${BDF}"
  --control-dev "${CONTROL_DEV}"
  --h2c-dev "${H2C_DEV}"
  --c2h-dev "${C2H_DEV}"
  --ctrl-dma-base "${CTRL_DMA_BASE}"
)

for arg in "${RUNNER_ARGS[@]}"; do
  xdma_cmd+=("${arg}")
done

run_cmd "${xdma_cmd[@]}"

run_log="${STAGE_DIR}/run.log"
if [[ -f "${run_log}" ]]; then
  grep -E '^(FINAL_|RUN_PASS=|STARTUP_)' "${run_log}" || true
fi

log "local-XDMA run passed: ${STAGE_DIR}"
