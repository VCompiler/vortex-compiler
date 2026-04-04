#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_full_inference.sh <out-dir>
# Builds and runs the full GPT-2 inference pipeline on simx.
#
# Expects in <out-dir>:
#   full_inference.mlir
#   full_inference_wrapper.c
#   full_inference_weights.h

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
DIR="${1:?Usage: $0 <out-dir>}"

DIR="$(cd "$DIR" && pwd -P)"
NAME="full_inference"
MLIR="$DIR/${NAME}.mlir"
WRAPPER="$DIR/${NAME}_wrapper.c"
OUTDIR="$DIR/out"

LLVM_BUILD="${REPO_ROOT}/third_party/llvm-build"
VX_OPT="${REPO_ROOT}/build/bin/vx-opt"
PLATFORM="${VORTEX_PLATFORM_ROOT:-/home/leoric/code/vortex-platform}"
TOOLCHAIN="${RISCV_TOOLCHAIN_PATH:-/home/leoric/tools/riscv32-gnu-toolchain}"
SIMX="${PLATFORM}/sim/simx/simx"

PIPELINE='builtin.module(func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-materialize-address-spaces,vortex-lower-linalg-inside-kernel),canonicalize,cse,vortex-legalize-for-llvm,vortex-lower-runtime-builtins,canonicalize,cse,convert-scf-to-cf,convert-math-to-llvm,convert-math-to-libm,convert-arith-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},convert-cf-to-llvm,reconcile-unrealized-casts)'

[[ -f "$MLIR" ]]    || { echo "error: $MLIR not found"; exit 1; }
[[ -f "$WRAPPER" ]] || { echo "error: $WRAPPER not found"; exit 1; }

mkdir -p "$OUTDIR"

echo "[1/5] vx-opt (${NAME})"
"$VX_OPT" "$MLIR" --pass-pipeline="$PIPELINE" -o "$OUTDIR/${NAME}.llvm.mlir"

echo "[2/5] mlir-translate"
"$LLVM_BUILD/bin/mlir-translate" -mlir-to-llvmir "$OUTDIR/${NAME}.llvm.mlir" -o "$OUTDIR/${NAME}.ll"
sed -i -E \
  -e 's/getelementptr inbounds nuw nusw /getelementptr inbounds /g' \
  -e 's/getelementptr inbounds nusw nuw /getelementptr inbounds /g' \
  -e 's/getelementptr inbounds nuw /getelementptr inbounds /g' \
  -e 's/getelementptr inbounds nusw /getelementptr inbounds /g' \
  -e 's/getelementptr nuw /getelementptr /g' \
  -e 's/getelementptr nusw /getelementptr /g' \
  "$OUTDIR/${NAME}.ll"

echo "[3/5] compile"
clang-18 --target=riscv32 -march=rv32imaf -mabi=ilp32f \
  --sysroot="$TOOLCHAIN/riscv32-unknown-elf" --gcc-toolchain="$TOOLCHAIN" \
  -O3 -mcmodel=medany -fno-exceptions -fdata-sections -ffunction-sections \
  -c -x ir "$OUTDIR/${NAME}.ll" -o "$OUTDIR/${NAME}.o" 2>&1

clang-18 --target=riscv32 -march=rv32imaf -mabi=ilp32f \
  --sysroot="$TOOLCHAIN/riscv32-unknown-elf" --gcc-toolchain="$TOOLCHAIN" \
  -O3 -mcmodel=medany -fno-exceptions -fdata-sections -ffunction-sections \
  -I"$PLATFORM/kernel/include" -I"$PLATFORM/hw" -I"$REPO_ROOT" -I"$DIR" \
  -c "$WRAPPER" -o "$OUTDIR/${NAME}_wrapper.o" 2>&1

echo "[4/5] link"
"$TOOLCHAIN/bin/riscv32-unknown-elf-gcc" -march=rv32imaf -mabi=ilp32f \
  -O3 -mcmodel=medany -fno-exceptions -nostartfiles \
  "$OUTDIR/${NAME}.o" "$OUTDIR/${NAME}_wrapper.o" \
  -Wl,-Bstatic,--gc-sections,-T,"$PLATFORM/kernel/scripts/link32.ld",--defsym=STARTUP_ADDR=0x80000000 \
  "$PLATFORM/kernel/libvortex.a" -lm -o "$OUTDIR/${NAME}.elf"

"$TOOLCHAIN/bin/riscv32-unknown-elf-objcopy" -O binary "$OUTDIR/${NAME}.elf" "$OUTDIR/${NAME}.bin"

echo "[5/5] simx"
"$SIMX" -c 1 -w 4 -t 4 "$OUTDIR/${NAME}.bin"
