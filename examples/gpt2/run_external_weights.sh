#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_external_weights.sh <out-dir>
#
# Builds and runs GPT-2 inference with external weight loading.
# Uses vortex runtime API: host driver loads weights into DDR separately.
#
# Expects in <out-dir>:
#   full_inference.mlir   -- MLIR module
#   gpt2_kernel.c         -- kernel code (RISC-V)
#   gpt2_common.h         -- shared header
#   gpt2_host.cpp         -- host driver (x86)
#   weights.bin           -- binary weights blob
#   golden.bin            -- golden logits
#
# Generate with:
#   python3 gen_full_inference.py --external-weights --seq 32 --dim 64 \
#       --ff 256 --vocab 256 --layers 4 --seed 42 --out-dir DIR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
DIR="${1:?Usage: $0 <out-dir>}"

DIR="$(cd "$DIR" && pwd -P)"
NAME="full_inference"
MLIR="$DIR/${NAME}.mlir"
OUTDIR="$DIR/out"

LLVM_BUILD="${REPO_ROOT}/third_party/llvm-build"
VX_OPT="${REPO_ROOT}/build/bin/vx-opt"
PLATFORM="${VORTEX_PLATFORM_ROOT:-/home/leoric/code/vortex-platform}"
TOOLCHAIN="${RISCV_TOOLCHAIN_PATH:-/home/leoric/tools/riscv32-gnu-toolchain}"
RUNTIME_LIB_DIR="${PLATFORM}/runtime"

PIPELINE='builtin.module(func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-materialize-address-spaces,vortex-lower-linalg-inside-kernel),canonicalize,cse,vortex-legalize-for-llvm,vortex-lower-runtime-builtins,canonicalize,cse,convert-scf-to-cf,convert-math-to-llvm,convert-math-to-libm,convert-arith-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},convert-cf-to-llvm,reconcile-unrealized-casts)'

[[ -f "$MLIR" ]] || { echo "error: $MLIR not found (did you run gen_full_inference.py --external-weights?)"; exit 1; }
[[ -f "$DIR/gpt2_kernel.c" ]] || { echo "error: gpt2_kernel.c not found"; exit 1; }
[[ -f "$DIR/gpt2_host.cpp" ]] || { echo "error: gpt2_host.cpp not found"; exit 1; }
[[ -f "$DIR/weights.bin" ]] || { echo "error: weights.bin not found"; exit 1; }

mkdir -p "$OUTDIR"

echo "===== Building GPT-2 with external weight loading ====="

# Step 1: Compile MLIR to LLVM IR
echo "[1/6] vx-opt"
"$VX_OPT" "$MLIR" --pass-pipeline="$PIPELINE" -o "$OUTDIR/${NAME}.llvm.mlir"

echo "[2/6] mlir-translate"
"$LLVM_BUILD/bin/mlir-translate" -mlir-to-llvmir "$OUTDIR/${NAME}.llvm.mlir" -o "$OUTDIR/${NAME}.ll"
sed -i -E \
  -e 's/getelementptr inbounds nuw nusw /getelementptr inbounds /g' \
  -e 's/getelementptr inbounds nusw nuw /getelementptr inbounds /g' \
  -e 's/getelementptr inbounds nuw /getelementptr inbounds /g' \
  -e 's/getelementptr inbounds nusw /getelementptr inbounds /g' \
  -e 's/getelementptr nuw /getelementptr /g' \
  -e 's/getelementptr nusw /getelementptr /g' \
  "$OUTDIR/${NAME}.ll"

# Step 2: Compile MLIR kernels to RISC-V object
echo "[3/6] compile MLIR kernels"
clang-18 --target=riscv32 -march=rv32imaf -mabi=ilp32f \
  --sysroot="$TOOLCHAIN/riscv32-unknown-elf" --gcc-toolchain="$TOOLCHAIN" \
  -O3 -mcmodel=medany -fno-exceptions -fdata-sections -ffunction-sections \
  -c -x ir "$OUTDIR/${NAME}.ll" -o "$OUTDIR/${NAME}.o" 2>&1

# Step 3: Compile kernel wrapper (RISC-V)
echo "[4/6] compile kernel wrapper"
clang-18 --target=riscv32 -march=rv32imaf -mabi=ilp32f \
  --sysroot="$TOOLCHAIN/riscv32-unknown-elf" --gcc-toolchain="$TOOLCHAIN" \
  -O3 -mcmodel=medany -fno-exceptions -fdata-sections -ffunction-sections \
  -I"$PLATFORM/kernel/include" -I"$PLATFORM/hw" -I"$DIR" \
  -c "$DIR/gpt2_kernel.c" -o "$OUTDIR/gpt2_kernel.o" 2>&1

# Step 4: Link kernel (small ELF -- code only, no weights!)
echo "[5/6] link kernel"
"$TOOLCHAIN/bin/riscv32-unknown-elf-gcc" -march=rv32imaf -mabi=ilp32f \
  -O3 -mcmodel=medany -fno-exceptions -nostartfiles \
  "$OUTDIR/${NAME}.o" "$OUTDIR/gpt2_kernel.o" \
  -Wl,-Bstatic,--gc-sections,-T,"$PLATFORM/kernel/scripts/link32.ld",--defsym=STARTUP_ADDR=0x80000000 \
  "$PLATFORM/kernel/libvortex.a" -lm -o "$OUTDIR/kernel.elf"

OBJCOPY="$TOOLCHAIN/bin/riscv32-unknown-elf-objcopy" \
  python3 "$PLATFORM/kernel/scripts/vxbin.py" "$OUTDIR/kernel.elf" "$OUTDIR/kernel.vxbin"

KERNEL_SIZE=$(stat -c%s "$OUTDIR/kernel.vxbin")
WEIGHTS_SIZE=$(stat -c%s "$DIR/weights.bin")
echo "  Kernel:  $KERNEL_SIZE bytes ($(echo "scale=1; $KERNEL_SIZE / 1024" | bc) KB)"
echo "  Weights: $WEIGHTS_SIZE bytes ($(echo "scale=2; $WEIGHTS_SIZE / 1024 / 1024" | bc) MB)"

# Step 5: Compile host driver (x86)
echo "[6/6] compile host driver"
g++ -std=c++17 -O2 \
  -I"$PLATFORM/runtime/include" -I"$PLATFORM/hw" -I"$DIR" \
  "$DIR/gpt2_host.cpp" \
  -L"$RUNTIME_LIB_DIR" -lvortex \
  -Wl,-rpath,"$RUNTIME_LIB_DIR" \
  -o "$OUTDIR/gpt2_host"

echo ""
echo "===== Build complete ====="
echo "  Kernel binary: $OUTDIR/kernel.vxbin ($KERNEL_SIZE bytes)"
echo "  Host driver:   $OUTDIR/gpt2_host"
echo ""
echo "===== Running ====="
LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${PLATFORM}/third_party/ramulator:${LD_LIBRARY_PATH:-}" \
  VORTEX_DRIVER=simx \
  "$OUTDIR/gpt2_host" \
    -k "$OUTDIR/kernel.vxbin" \
    -w "$DIR/weights.bin" \
    -g "$DIR/golden.bin"
