#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

usage() {
  cat <<'EOF'
用法:
  build-vortex-kernel.sh \
    --input kernel.mlir \
    --output-dir build/out \
    [--platform-root /path/to/vortex-platform] \
    [--pass-pipeline 'builtin.module(...)'] \
    [--skip-vx-opt] \
    [--allow-unregistered-dialect] \
    [--extra-source wrapper.c] ... \
    [--board-xdma-abi] \
    [--build-runtime] \
    [--xlen 32|64] \
    [--startup-addr 0x80000000] \
    [--vx-opt /path/to/vx-opt] \
    [--mlir-translate /path/to/mlir-translate] \
    [--llc /path/to/llc] \
    [--codegen-backend clang|llc] \
    [--clang /path/to/clang] \
    [--clangxx /path/to/clang++] \
    [--objcopy /path/to/llvm-objcopy] \
    [--objdump /path/to/llvm-objdump] \
    [--clang-arg ARG] ... \
    [--verbose]

说明:
  1. 输入默认是 MLIR 文件
  2. 默认会先跑:
       builtin.module(vortex-mvp-backend-pipeline)
  3. 然后依次产出:
       .llvm.mlir / .ll / .s / .o / .elf / .bin / .dump
  4. 若要自己控制 lowering，可传 --pass-pipeline
  5. 若输入已经是 LLVM dialect MLIR，可传 --skip-vx-opt
  6. 若目标是 local-XDMA board runner，可传 --board-xdma-abi 自动链接标准 entry/exit shim

环境变量:
  VX_OPT_BIN
  MLIR_TRANSLATE_BIN
  LLC_BIN
  CLANG_BIN
  CLANGXX_BIN
  LLVM_OBJCOPY
  LLVM_OBJDUMP
  VORTEX_PLATFORM_ROOT
  LLVM_VORTEX
  RISCV_TOOLCHAIN_PATH
  RISCV_SYSROOT
  LIBC_VORTEX
  LIBCRT_VORTEX
  VORTEX_RUNTIME_CFLAGS
EOF
}

log() {
  echo "[build-vortex-kernel] $*"
}

die() {
  echo "[build-vortex-kernel] error: $*" >&2
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

sanitize_llvm_ir_for_vortex_clang() {
  local path="$1"
  local tmp="${path}.sanitized"
  sed -E \
    -e 's/getelementptr inbounds nuw nusw /getelementptr inbounds /g' \
    -e 's/getelementptr inbounds nusw nuw /getelementptr inbounds /g' \
    -e 's/getelementptr inbounds nuw /getelementptr inbounds /g' \
    -e 's/getelementptr inbounds nusw /getelementptr inbounds /g' \
    -e 's/getelementptr nuw /getelementptr /g' \
    -e 's/getelementptr nusw /getelementptr /g' \
    "${path}" > "${tmp}"
  mv "${tmp}" "${path}"
}

sanitize_llvm_dialect_mlir_for_translate() {
  local path="$1"
  local tmp="${path}.sanitized"
  sed -E \
    -e 's/llvm\.getelementptr inbounds\|nuw /llvm.getelementptr inbounds /g' \
    -e 's/llvm\.getelementptr inbounds\|nusw /llvm.getelementptr inbounds /g' \
    -e 's/llvm\.getelementptr inbounds\|nuw\|nusw /llvm.getelementptr inbounds /g' \
    -e 's/llvm\.getelementptr inbounds\|nusw\|nuw /llvm.getelementptr inbounds /g' \
    "${path}" > "${tmp}"
  mv "${tmp}" "${path}"
}

workaround_large_riscv_sp_offsets() {
  local path="$1"
  local tmp="${path}.sp-offset-workaround"
  python3 - "${path}" "${tmp}" <<'PY'
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
pattern = re.compile(r"^(\s*)(lw|sw|flw|fsw)\s+([^,\s]+),\s*([0-9]+)\(sp\)(.*)$")

with open(src, "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []
for line in lines:
    m = pattern.match(line.rstrip("\n"))
    if not m:
        out.append(line)
        continue

    indent, op, reg, offset_text, suffix = m.groups()
    offset = int(offset_text, 10)
    if offset < 128:
        out.append(line)
        continue

    scratch = "t6"
    if op in ("sw", "fsw") and reg == scratch:
        scratch = "t5"
    out.append(f"{indent}addi\t{scratch}, sp, {offset}\n")
    out.append(f"{indent}{op}\t{reg}, 0({scratch}){suffix}\n")

with open(dst, "w", encoding="utf-8") as f:
    f.writelines(out)
PY
  mv "${tmp}" "${path}"
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

find_tool() {
  local explicit="$1"
  local env_path="$2"
  local bin_dir="$3"
  local name="$4"

  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
    return
  fi

  if [[ -n "${env_path}" ]]; then
    printf '%s\n' "${env_path}"
    return
  fi

  if [[ -n "${bin_dir}" && -x "${bin_dir}/${name}" ]]; then
    printf '%s\n' "${bin_dir}/${name}"
    return
  fi

  if command -v "${name}" >/dev/null 2>&1; then
    command -v "${name}"
    return
  fi

  die "未找到工具 ${name}"
}

find_tool_optional() {
  local explicit="$1"
  local env_path="$2"
  local bin_dir="$3"
  local name="$4"

  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
    return
  fi

  if [[ -n "${env_path}" ]]; then
    printf '%s\n' "${env_path}"
    return
  fi

  if [[ -n "${bin_dir}" && -x "${bin_dir}/${name}" ]]; then
    printf '%s\n' "${bin_dir}/${name}"
    return
  fi

  if command -v "${name}" >/dev/null 2>&1; then
    command -v "${name}"
    return
  fi
}

find_local_tool_optional() {
  local explicit="$1"
  local env_path="$2"
  local bin_dir="$3"
  local name="$4"

  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
    return
  fi

  if [[ -n "${env_path}" ]]; then
    printf '%s\n' "${env_path}"
    return
  fi

  if [[ -n "${bin_dir}" && -x "${bin_dir}/${name}" ]]; then
    printf '%s\n' "${bin_dir}/${name}"
    return
  fi
}

INPUT=""
OUTPUT_DIR=""
PASS_PIPELINE="builtin.module(vortex-mvp-backend-pipeline)"
SKIP_VX_OPT=0
ALLOW_UNREGISTERED=0
BUILD_RUNTIME=0
BOARD_XDMA_ABI=0
VERBOSE=0
XLEN=32
STARTUP_ADDR=0x80000000
PLATFORM_ROOT=""

VX_OPT_EXPLICIT=""
MLIR_TRANSLATE_EXPLICIT=""
LLC_EXPLICIT=""
CLANG_EXPLICIT=""
CLANGXX_EXPLICIT=""
OBJCOPY_EXPLICIT=""
OBJDUMP_EXPLICIT=""
CODEGEN_BACKEND_REQUEST=""

declare -a EXTRA_SOURCES=()
declare -a CLANG_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --pass-pipeline)
      PASS_PIPELINE="$2"
      shift 2
      ;;
    --skip-vx-opt)
      SKIP_VX_OPT=1
      shift
      ;;
    --allow-unregistered-dialect)
      ALLOW_UNREGISTERED=1
      shift
      ;;
    --extra-source)
      EXTRA_SOURCES+=("$2")
      shift 2
      ;;
    --board-xdma-abi)
      BOARD_XDMA_ABI=1
      shift
      ;;
    --build-runtime)
      BUILD_RUNTIME=1
      shift
      ;;
    --platform-root)
      PLATFORM_ROOT="$2"
      shift 2
      ;;
    --xlen)
      XLEN="$2"
      shift 2
      ;;
    --startup-addr)
      STARTUP_ADDR="$2"
      shift 2
      ;;
    --vx-opt)
      VX_OPT_EXPLICIT="$2"
      shift 2
      ;;
    --mlir-translate)
      MLIR_TRANSLATE_EXPLICIT="$2"
      shift 2
      ;;
    --llc)
      LLC_EXPLICIT="$2"
      shift 2
      ;;
    --codegen-backend)
      CODEGEN_BACKEND_REQUEST="$2"
      shift 2
      ;;
    --clang)
      CLANG_EXPLICIT="$2"
      shift 2
      ;;
    --clangxx)
      CLANGXX_EXPLICIT="$2"
      shift 2
      ;;
    --objcopy)
      OBJCOPY_EXPLICIT="$2"
      shift 2
      ;;
    --objdump)
      OBJDUMP_EXPLICIT="$2"
      shift 2
      ;;
    --clang-arg)
      CLANG_EXTRA_ARGS+=("$2")
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

[[ -n "${INPUT}" ]] || die "缺少 --input"
[[ -n "${OUTPUT_DIR}" ]] || die "缺少 --output-dir"

case "${XLEN}" in
  32|64)
    ;;
  *)
    die "--xlen 只支持 32 或 64"
    ;;
esac

INPUT="$(resolve_abs_path "${INPUT}")"
[[ -f "${INPUT}" ]] || die "输入文件不存在: ${INPUT}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(resolve_abs_path "${OUTPUT_DIR}")"

if [[ ${BOARD_XDMA_ABI} -eq 1 ]]; then
  EXTRA_SOURCES+=("${REPO_ROOT}/runtime/board_xdma_abi.c")
fi

for source in "${EXTRA_SOURCES[@]}"; do
  source="$(resolve_abs_path "${source}")"
  [[ -f "${source}" ]] || die "额外源码不存在: ${source}"
done

if [[ -n "${PLATFORM_ROOT}" ]]; then
  PLATFORM_ROOT="$(resolve_abs_path "${PLATFORM_ROOT}")"
elif [[ -n "${VORTEX_PLATFORM_ROOT:-}" ]]; then
  PLATFORM_ROOT="$(resolve_abs_path "${VORTEX_PLATFORM_ROOT}")"
elif [[ -d "${REPO_ROOT}/third_party/vortex-platform" ]]; then
  PLATFORM_ROOT="$(resolve_abs_path "${REPO_ROOT}/third_party/vortex-platform")"
else
  die "未找到 vortex-platform，请通过 --platform-root 或 VORTEX_PLATFORM_ROOT 指定"
fi

[[ -d "${PLATFORM_ROOT}" ]] || die "vortex-platform 路径不存在: ${PLATFORM_ROOT}"

LLVM_VORTEX_ROOT="${LLVM_VORTEX:-}"
if [[ -z "${LLVM_VORTEX_ROOT}" && -d "${REPO_ROOT}/third_party/llvm-vortex-build/bin" ]]; then
  LLVM_VORTEX_ROOT="${REPO_ROOT}/third_party/llvm-vortex-build"
elif [[ -z "${LLVM_VORTEX_ROOT}" && -d "${REPO_ROOT}/third_party/llvm-build/bin" ]]; then
  LLVM_VORTEX_ROOT="${REPO_ROOT}/third_party/llvm-build"
fi

TARGET_BIN_DIR=""
if [[ -n "${LLVM_VORTEX_ROOT}" ]]; then
  if [[ -d "${LLVM_VORTEX_ROOT}/bin" ]]; then
    TARGET_BIN_DIR="${LLVM_VORTEX_ROOT}/bin"
  elif [[ -d "${LLVM_VORTEX_ROOT}" ]]; then
    TARGET_BIN_DIR="${LLVM_VORTEX_ROOT}"
  fi
fi

VX_OPT_BIN_DIR="${REPO_ROOT}/build/bin"
if [[ ! -x "${VX_OPT_BIN_DIR}/vx-opt" && -d "${REPO_ROOT}/build-thirdparty-llvm/bin" ]]; then
  VX_OPT_BIN_DIR="${REPO_ROOT}/build-thirdparty-llvm/bin"
fi

VX_OPT_BIN="$(find_tool "${VX_OPT_EXPLICIT}" "${VX_OPT_BIN:-}" "${VX_OPT_BIN_DIR}" "vx-opt")"
MLIR_TRANSLATE_BIN="$(find_tool "${MLIR_TRANSLATE_EXPLICIT}" "${MLIR_TRANSLATE_BIN:-}" "${TARGET_BIN_DIR}" "mlir-translate")"
LLC_BIN="$(find_tool_optional "${LLC_EXPLICIT}" "${LLC_BIN:-}" "${TARGET_BIN_DIR}" "llc")"
CLANG_BIN="$(find_local_tool_optional "${CLANG_EXPLICIT}" "${CLANG_BIN:-}" "${TARGET_BIN_DIR}" "clang")"
OBJCOPY_BIN="$(find_tool "${OBJCOPY_EXPLICIT}" "${LLVM_OBJCOPY:-}" "${TARGET_BIN_DIR}" "llvm-objcopy")"
OBJDUMP_BIN="$(find_tool "${OBJDUMP_EXPLICIT}" "${LLVM_OBJDUMP:-}" "${TARGET_BIN_DIR}" "llvm-objdump")"

CODEGEN_BACKEND=""
if [[ -n "${CODEGEN_BACKEND_REQUEST}" ]]; then
  CODEGEN_BACKEND="${CODEGEN_BACKEND_REQUEST}"
else
  if [[ -n "${CLANG_BIN}" ]]; then
    CODEGEN_BACKEND="clang"
  elif [[ -n "${LLC_BIN}" ]]; then
    CODEGEN_BACKEND="llc"
  else
    die "未找到可用 codegen backend；请提供本地 clang 或 llc"
  fi
fi

case "${CODEGEN_BACKEND}" in
  clang)
    [[ -n "${CLANG_BIN}" ]] || die "选择了 clang backend，但未找到本地 clang"
    ;;
  llc)
    [[ -n "${LLC_BIN}" ]] || die "选择了 llc backend，但未找到 llc"
    ;;
  *)
    die "--codegen-backend 只支持 clang 或 llc"
    ;;
esac

NEED_CLANGXX=0
for source in "${EXTRA_SOURCES[@]}"; do
  case "${source}" in
    *.cc|*.cpp|*.cxx|*.CPP|*.C)
      NEED_CLANGXX=1
      ;;
  esac
done

CLANGXX_BIN=""
if [[ ${NEED_CLANGXX} -eq 1 && "${CODEGEN_BACKEND}" == "clang" ]]; then
  CLANGXX_BIN="$(find_tool "${CLANGXX_EXPLICIT}" "${CLANGXX_BIN:-}" "${TARGET_BIN_DIR}" "clang++")"
fi

LIBVORTEX_A="${PLATFORM_ROOT}/kernel/libvortex.a"
LINK_SCRIPT="${PLATFORM_ROOT}/kernel/scripts/link${XLEN}.ld"
[[ -f "${LINK_SCRIPT}" ]] || die "缺少链接脚本: ${LINK_SCRIPT}"

if [[ ${BUILD_RUNTIME} -eq 1 || ! -f "${LIBVORTEX_A}" ]]; then
  log "构建 vortex kernel runtime"
  make_args=(
    env
    "VORTEX_HOME=${PLATFORM_ROOT}"
  )
  if [[ -n "${LLVM_VORTEX_ROOT}" ]]; then
    make_args+=("LLVM_VORTEX=${LLVM_VORTEX_ROOT}")
  fi
  if [[ -n "${RISCV_TOOLCHAIN_PATH:-}" ]]; then
    make_args+=("RISCV_TOOLCHAIN_PATH=${RISCV_TOOLCHAIN_PATH}")
  fi
  if [[ -n "${RISCV_SYSROOT:-}" ]]; then
    make_args+=("RISCV_SYSROOT=${RISCV_SYSROOT}")
  fi
  if [[ -n "${LIBC_VORTEX:-}" ]]; then
    make_args+=("LIBC_VORTEX=${LIBC_VORTEX}")
  fi
  if [[ -n "${LIBCRT_VORTEX:-}" ]]; then
    make_args+=("LIBCRT_VORTEX=${LIBCRT_VORTEX}")
  fi
  if [[ -n "${VORTEX_RUNTIME_CFLAGS:-}" ]]; then
    make_args+=("CFLAGS=${VORTEX_RUNTIME_CFLAGS}")
  fi
  make_args+=(make -C "${PLATFORM_ROOT}/kernel")
  run_cmd "${make_args[@]}"
fi

[[ -f "${LIBVORTEX_A}" ]] || die "缺少 runtime 库: ${LIBVORTEX_A}"

RISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_PATH:-}"
RISCV_SYSROOT_PATH="${RISCV_SYSROOT:-}"
LIBC_VORTEX_ROOT="${LIBC_VORTEX:-}"
LIBCRT_VORTEX_ROOT="${LIBCRT_VORTEX:-}"

if [[ -z "${RISCV_TOOLCHAIN_ROOT}" && -n "${PLATFORM_ROOT}" ]]; then
  RISCV_TOOLCHAIN_ROOT="${PLATFORM_ROOT}/../tools/riscv${XLEN}-gnu-toolchain"
fi
if [[ -z "${RISCV_SYSROOT_PATH}" && -n "${RISCV_TOOLCHAIN_ROOT}" ]]; then
  RISCV_SYSROOT_PATH="${RISCV_TOOLCHAIN_ROOT}/riscv${XLEN}-unknown-elf"
fi
if [[ -z "${LIBC_VORTEX_ROOT}" && -d "${REPO_ROOT}/../tools/libc${XLEN}" ]]; then
  LIBC_VORTEX_ROOT="$(resolve_abs_path "${REPO_ROOT}/../tools/libc${XLEN}")"
fi
if [[ -z "${LIBCRT_VORTEX_ROOT}" && -d "${REPO_ROOT}/../tools/libcrt${XLEN}" ]]; then
  LIBCRT_VORTEX_ROOT="$(resolve_abs_path "${REPO_ROOT}/../tools/libcrt${XLEN}")"
fi

declare -a TARGET_FLAGS=()
declare -a GCC_ARCH_FLAGS=()
LLC_FEATURES=""
LLC_CPU=""
if [[ "${XLEN}" == "64" ]]; then
  TARGET_FLAGS+=(--target=riscv64 -march=rv64imafd -mabi=lp64d)
  GCC_ARCH_FLAGS+=(-march=rv64imafd -mabi=lp64d)
  LLC_FEATURES="+m,+a,+f,+d,+vortex"
  LLC_CPU="generic-rv64"
else
  TARGET_FLAGS+=(--target=riscv32 -march=rv32imaf -mabi=ilp32f)
  GCC_ARCH_FLAGS+=(-march=rv32imaf -mabi=ilp32f)
  LLC_FEATURES="+m,+a,+f,+vortex"
  LLC_CPU="generic-rv32"
fi
if [[ -n "${RISCV_SYSROOT_PATH}" ]]; then
  TARGET_FLAGS+=("--sysroot=${RISCV_SYSROOT_PATH}")
fi
if [[ -n "${RISCV_TOOLCHAIN_ROOT}" ]]; then
  TARGET_FLAGS+=("--gcc-toolchain=${RISCV_TOOLCHAIN_ROOT}")
  if [[ -x "${RISCV_TOOLCHAIN_ROOT}/bin/riscv${XLEN}-unknown-elf-ld" ]]; then
    TARGET_FLAGS+=("-fuse-ld=${RISCV_TOOLCHAIN_ROOT}/bin/riscv${XLEN}-unknown-elf-ld")
  fi
fi
TARGET_FLAGS+=(-Xclang -target-feature -Xclang +vortex)
TARGET_FLAGS+=("${CLANG_EXTRA_ARGS[@]}")

declare -a INCLUDE_FLAGS=(
  "-I${REPO_ROOT}/include"
  "-I${PLATFORM_ROOT}/kernel/include"
  "-I${PLATFORM_ROOT}/hw"
)

COMMON_COMPILE_FLAGS=(
  -O3
  -mcmodel=medany
  -fno-exceptions
  -fdata-sections
  -ffunction-sections
)

declare -a LINK_FLAGS=(
  -Wl,-Bstatic,--gc-sections,-T,"${LINK_SCRIPT}",--defsym=STARTUP_ADDR="${STARTUP_ADDR}"
)

declare -a LIBC_SEARCH_FLAGS=()

BUILTINS_LIB=""
if [[ -n "${LIBCRT_VORTEX_ROOT}" ]]; then
  BUILTINS_LIB="${LIBCRT_VORTEX_ROOT}/lib/baremetal/libclang_rt.builtins-riscv${XLEN}.a"
fi

BASE_NAME="$(basename "${INPUT}")"
BASE_NAME="${BASE_NAME%.mlir}"
LOWERED_MLIR="${OUTPUT_DIR}/${BASE_NAME}.llvm.mlir"
PIPELINE_TXT="${OUTPUT_DIR}/${BASE_NAME}.pipeline.txt"
LLVM_IR="${OUTPUT_DIR}/${BASE_NAME}.ll"
ASM="${OUTPUT_DIR}/${BASE_NAME}.s"
MODULE_OBJ="${OUTPUT_DIR}/${BASE_NAME}.mlir.o"
ELF="${OUTPUT_DIR}/${BASE_NAME}.elf"
BIN="${OUTPUT_DIR}/${BASE_NAME}.bin"
DUMP="${OUTPUT_DIR}/${BASE_NAME}.dump"

if [[ ${SKIP_VX_OPT} -eq 1 ]]; then
  run_cmd cp "${INPUT}" "${LOWERED_MLIR}"
else
  printf '%s\n' "${PASS_PIPELINE}" > "${PIPELINE_TXT}"
  vx_opt_cmd=("${VX_OPT_BIN}" "${INPUT}")
  if [[ ${ALLOW_UNREGISTERED} -eq 1 ]]; then
    vx_opt_cmd+=(--allow-unregistered-dialect)
  fi
  vx_opt_cmd+=(--pass-pipeline="${PASS_PIPELINE}" -o "${LOWERED_MLIR}")
  run_cmd "${vx_opt_cmd[@]}"
fi

sanitize_llvm_dialect_mlir_for_translate "${LOWERED_MLIR}"
run_cmd "${MLIR_TRANSLATE_BIN}" -mlir-to-llvmir "${LOWERED_MLIR}" -o "${LLVM_IR}"
sanitize_llvm_ir_for_vortex_clang "${LLVM_IR}"

RISCV_CC_BIN=""
RISCV_CXX_BIN=""
if [[ "${CODEGEN_BACKEND}" == "llc" ]]; then
  if [[ -n "${RISCV_TOOLCHAIN_ROOT}" && -x "${RISCV_TOOLCHAIN_ROOT}/bin/riscv${XLEN}-unknown-elf-gcc" ]]; then
    RISCV_CC_BIN="${RISCV_TOOLCHAIN_ROOT}/bin/riscv${XLEN}-unknown-elf-gcc"
    RISCV_CXX_BIN="${RISCV_TOOLCHAIN_ROOT}/bin/riscv${XLEN}-unknown-elf-g++"
  else
    RISCV_CC_BIN="$(find_tool "" "" "" "riscv${XLEN}-unknown-elf-gcc")"
    RISCV_CXX_BIN="$(find_tool_optional "" "" "" "riscv${XLEN}-unknown-elf-g++")"
  fi
  [[ -n "${RISCV_CC_BIN}" ]] || die "llc backend 需要 riscv${XLEN}-unknown-elf-gcc"
fi

RISCV_CC_FOR_MULTILIB="${RISCV_CC_BIN}"
if [[ -z "${RISCV_CC_FOR_MULTILIB}" && -n "${RISCV_TOOLCHAIN_ROOT}" && -x "${RISCV_TOOLCHAIN_ROOT}/bin/riscv${XLEN}-unknown-elf-gcc" ]]; then
  RISCV_CC_FOR_MULTILIB="${RISCV_TOOLCHAIN_ROOT}/bin/riscv${XLEN}-unknown-elf-gcc"
fi

MULTILIB_DIR=""
if [[ -n "${RISCV_CC_FOR_MULTILIB}" ]]; then
  MULTILIB_DIR="$("${RISCV_CC_FOR_MULTILIB}" "${GCC_ARCH_FLAGS[@]}" -print-multi-directory 2>/dev/null || true)"
fi

if [[ -n "${LIBC_VORTEX_ROOT}" && -d "${LIBC_VORTEX_ROOT}/lib" ]]; then
  if [[ -n "${MULTILIB_DIR}" && -d "${LIBC_VORTEX_ROOT}/lib/${MULTILIB_DIR}" ]]; then
    LIBC_SEARCH_FLAGS+=("-L${LIBC_VORTEX_ROOT}/lib/${MULTILIB_DIR}")
  fi
  LIBC_SEARCH_FLAGS+=("-L${LIBC_VORTEX_ROOT}/lib")
fi

if [[ "${CODEGEN_BACKEND}" == "clang" ]]; then
  run_cmd "${CLANG_BIN}" "${TARGET_FLAGS[@]}" "${COMMON_COMPILE_FLAGS[@]}" \
    -mllvm -vortex-kernel-scheduler=1 -S -x ir "${LLVM_IR}" -o "${ASM}"
  run_cmd "${CLANG_BIN}" "${TARGET_FLAGS[@]}" "${COMMON_COMPILE_FLAGS[@]}" \
    -mllvm -vortex-kernel-scheduler=1 -c -x ir "${LLVM_IR}" -o "${MODULE_OBJ}"
else
  llc_common=(
    "${LLC_BIN}"
    "-mtriple=riscv${XLEN}-unknown-elf"
    "-mcpu=${LLC_CPU}"
    "-mattr=${LLC_FEATURES}"
    -code-model=medium
    -relocation-model=static
    --vortex-kernel-scheduler=1
  )
  run_cmd "${llc_common[@]}" -filetype=asm "${LLVM_IR}" -o "${ASM}"
  if [[ "${VORTEX_RISCV_SP_OFFSET_WORKAROUND:-0}" == "1" ]]; then
    workaround_large_riscv_sp_offsets "${ASM}"
    LLVM_MC_BIN="$(find_tool "" "${LLVM_MC_BIN:-}" "${TARGET_BIN_DIR}" "llvm-mc")"
    run_cmd "${LLVM_MC_BIN}" \
      "-triple=riscv${XLEN}-unknown-elf" \
      "-mcpu=${LLC_CPU}" \
      "-mattr=${LLC_FEATURES}" \
      -filetype=obj \
      "${ASM}" \
      -o "${MODULE_OBJ}"
  else
    run_cmd "${llc_common[@]}" -filetype=obj "${LLVM_IR}" -o "${MODULE_OBJ}"
  fi
fi

declare -a EXTRA_OBJECTS=()
source_index=0
for source in "${EXTRA_SOURCES[@]}"; do
  source="$(resolve_abs_path "${source}")"
  extension="${source##*.}"
  object_path="${OUTPUT_DIR}/${BASE_NAME}.extra${source_index}.o"
  compiler="${CLANG_BIN}"
  compile_cmd=()

  if [[ "${CODEGEN_BACKEND}" == "clang" ]]; then
    case "${source}" in
      *.cc|*.cpp|*.cxx|*.CPP|*.C)
        compiler="${CLANGXX_BIN}"
        ;;
    esac

    compile_cmd=(
      "${compiler}"
      "${TARGET_FLAGS[@]}"
      "${COMMON_COMPILE_FLAGS[@]}"
      "${INCLUDE_FLAGS[@]}"
      -c
      "${source}"
      -o
      "${object_path}"
    )
  else
    case "${source}" in
      *.cc|*.cpp|*.cxx|*.CPP|*.C)
        [[ -n "${RISCV_CXX_BIN}" ]] || die "llc backend 遇到 C++ 额外源码，但未找到 riscv${XLEN}-unknown-elf-g++"
        compiler="${RISCV_CXX_BIN}"
        ;;
      *)
        compiler="${RISCV_CC_BIN}"
        ;;
    esac

    compile_cmd=(
      "${compiler}"
      "${GCC_ARCH_FLAGS[@]}"
      "${COMMON_COMPILE_FLAGS[@]}"
      "${INCLUDE_FLAGS[@]}"
      -c
      "${source}"
      -o
      "${object_path}"
    )
  fi

  run_cmd "${compile_cmd[@]}"
  EXTRA_OBJECTS+=("${object_path}")
  source_index=$((source_index + 1))
done

link_cmd=()
if [[ "${CODEGEN_BACKEND}" == "clang" ]]; then
  link_cmd=(
    "${CLANG_BIN}"
    "${TARGET_FLAGS[@]}"
    -O3
    -mcmodel=medany
    -fno-exceptions
    -nostartfiles
    -nostdlib
    "${MODULE_OBJ}"
    "${EXTRA_OBJECTS[@]}"
    "${LINK_FLAGS[@]}"
    "${LIBC_SEARCH_FLAGS[@]}"
    "${LIBVORTEX_A}"
    -lm
    -lc
  )
else
  link_cmd=(
    "${RISCV_CC_BIN}"
    "${GCC_ARCH_FLAGS[@]}"
    -O3
    -mcmodel=medany
    -fno-exceptions
    -nostartfiles
    -nostdlib
    "${MODULE_OBJ}"
    "${EXTRA_OBJECTS[@]}"
    "${LINK_FLAGS[@]}"
    "${LIBC_SEARCH_FLAGS[@]}"
    "${LIBVORTEX_A}"
    -lm
    -lc
  )
fi

# Find and add libgcc for compiler runtime builtins (soft-float, etc.)
LIBGCC_PATH=""
if [[ -n "${RISCV_CC_FOR_MULTILIB}" ]]; then
  LIBGCC_PATH="$("${RISCV_CC_FOR_MULTILIB}" "${GCC_ARCH_FLAGS[@]}" -print-libgcc-file-name 2>/dev/null || true)"
fi
if [[ ( -z "${LIBGCC_PATH}" || ! -f "${LIBGCC_PATH}" ) && -n "${RISCV_TOOLCHAIN_ROOT}" ]]; then
  LIBGCC_PATH="$(find -L "${RISCV_TOOLCHAIN_ROOT}" -name "libgcc.a" -path "*/ilp32f/*" 2>/dev/null | head -1)"
  if [[ -z "${LIBGCC_PATH}" ]]; then
    LIBGCC_PATH="$(find -L "${RISCV_TOOLCHAIN_ROOT}" -name "libgcc.a" 2>/dev/null | head -1)"
  fi
fi
if [[ -n "${LIBGCC_PATH}" ]]; then
  link_cmd+=("${LIBGCC_PATH}")
elif [[ -n "${LIBGCC:-}" ]]; then
  link_cmd+=("${LIBGCC}")
fi

if [[ -n "${BUILTINS_LIB}" && -f "${BUILTINS_LIB}" ]]; then
  link_cmd+=("${BUILTINS_LIB}")
fi

link_cmd+=(-o "${ELF}")
run_cmd "${link_cmd[@]}"

run_cmd "${OBJCOPY_BIN}" -O binary "${ELF}" "${BIN}"
run_cmd "${OBJDUMP_BIN}" -D "${ELF}" > "${DUMP}"

log "codegen backend: ${CODEGEN_BACKEND}"
log "generated: ${LOWERED_MLIR}"
log "generated: ${LLVM_IR}"
log "generated: ${ASM}"
log "generated: ${MODULE_OBJ}"
log "generated: ${ELF}"
log "generated: ${BIN}"
log "generated: ${DUMP}"
