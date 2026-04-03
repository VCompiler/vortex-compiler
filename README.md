# vortex-compiler

Out-of-tree MLIR project for:

1. `Vortex dialect`
2. `pre-vortex -> vortex` passes
3. `ONNX -> pre-vortex` minimal frontend bridge
4. `vortex -> LLVM dialect` MVP backend pipeline
5. `vx-opt`

## Repository

- `include/`
- `lib/`
- `tools/`
- `scripts/`
- `test/`
- `examples/`
- `docs/`
- `third_party/llvm/`

## Requirements

### Core

- `git`
- `cmake >= 3.20`
- `ninja`
- `python3`
- C++17 compiler

Ubuntu:

```bash
sudo apt update
sudo apt install -y \
  git \
  cmake \
  ninja-build \
  python3 \
  python3-pip \
  build-essential
```

### Optional frontend

- `torch`
- `onnx`
- `onnx-mlir`
- `onnx-mlir-opt`

Install Python packages:

```bash
python3 -m pip install torch onnx
```

Current frontend script is verified with:

- `onnx-mlir @ 504fcfe`

### Optional board validation

- `vortex-platform`
- Vivado / JTAG environment
- matching `.bit` / `.ltx`

### Optional simulation backend

- `vortex-platform`
- `make`
- `verilator` for `rtlsim`

## Clone

```bash
git clone --recurse-submodules git@github.com:VCompiler/vortex-compiler.git
cd vortex-compiler
git submodule update --init --recursive
```

## Build LLVM/MLIR

```bash
cmake -S third_party/llvm/llvm -B third_party/llvm-build \
  -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_BUILD_TYPE=Release

cmake --build third_party/llvm-build -j$(nproc)
```

## Build vortex-compiler

```bash
cmake -S . -B build \
  -G Ninja \
  -DMLIR_DIR=$PWD/third_party/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/third_party/llvm-build/lib/cmake/llvm \
  -DVORTEX_ENABLE_TESTS=ON

cmake --build build -j$(nproc)
```

If you already have an external LLVM/MLIR build:

```bash
cmake -S . -B build \
  -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm \
  -DVORTEX_ENABLE_TESTS=ON

cmake --build build -j$(nproc)
```

## Test

```bash
cmake --build build --target check-vortex -j$(nproc)
```

## vx-opt

```bash
./build/bin/vx-opt --help
```

## Build Kernel

```bash
MLIR_TRANSLATE_BIN=/path/to/mlir-translate \
LLVM_VORTEX=/path/to/llvm-vortex \
RISCV_TOOLCHAIN_PATH=/path/to/riscv32-gnu-toolchain \
RISCV_SYSROOT=$RISCV_TOOLCHAIN_PATH/riscv32-unknown-elf \
LIBC_VORTEX=/path/to/libc32 \
LIBCRT_VORTEX=/path/to/libcrt32 \
scripts/build-vortex-kernel.sh \
  --input examples/smoke/matmul4x4_f32.mlir \
  --output-dir build/smoke/matmul4x4_f32 \
  --platform-root /path/to/vortex-platform \
  --pass-pipeline 'builtin.module(func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-materialize-address-spaces,vortex-lower-linalg-inside-kernel),canonicalize,cse,vortex-legalize-for-llvm,vortex-lower-runtime-builtins,canonicalize,cse,convert-scf-to-cf,convert-arith-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},convert-cf-to-llvm,reconcile-unrealized-casts)' \
  --extra-source examples/smoke/matmul4x4_f32_wrapper.c \
  --build-runtime
```

## Smoke

```bash
MLIR_TRANSLATE_BIN=/path/to/mlir-translate \
LLVM_VORTEX=/path/to/llvm-vortex \
RISCV_TOOLCHAIN_PATH=/path/to/riscv32-gnu-toolchain \
RISCV_SYSROOT=$RISCV_TOOLCHAIN_PATH/riscv32-unknown-elf \
LIBC_VORTEX=/path/to/libc32 \
LIBCRT_VORTEX=/path/to/libcrt32 \
scripts/run-matmul4x4-smoke.sh \
  --platform-root /path/to/vortex-platform \
  --build-sim
```

## ONNX Smoke

```bash
ONNX_FRONT_PYTHON=/path/to/python-with-onnx \
ONNX_MLIR_BIN=/path/to/onnx-mlir \
ONNX_MLIR_OPT_BIN=/path/to/onnx-mlir-opt \
MLIR_TRANSLATE_BIN=/path/to/mlir-translate \
LLVM_VORTEX=/path/to/llvm-vortex \
RISCV_TOOLCHAIN_PATH=/path/to/riscv32-gnu-toolchain \
RISCV_SYSROOT=$RISCV_TOOLCHAIN_PATH/riscv32-unknown-elf \
LIBC_VORTEX=/path/to/libc32 \
LIBCRT_VORTEX=/path/to/libcrt32 \
scripts/run-onnx-matmul4x4-smoke.sh \
  --platform-root /path/to/vortex-platform \
  --build-sim
```

## Frontend Example

Export ONNX:

```bash
python3 examples/frontend/pytorch/export_models.py --model all
```

Lower a static matmul ONNX model to `pre-vortex`:

```bash
examples/frontend/mlir/lower_onnx_matmul_to_pre_vortex.sh \
  --input examples/frontend/onnx/matmul_mlp.onnx \
  --output examples/frontend/mlir/matmul_mlp.pre_vortex.mlir \
  --onnx-mlir /path/to/onnx-mlir \
  --onnx-mlir-opt /path/to/onnx-mlir-opt \
  --vx-opt $PWD/build/bin/vx-opt
```

## Simulation

Use external `vortex-platform`, or place it at `third_party/vortex-platform`.

Prepare `vortex-platform`:

```bash
git -C /path/to/vortex-platform submodule update --init --recursive
```

Run `rtlsim`:

```bash
scripts/run-vortex-sim.sh \
  --platform-root /path/to/vortex-platform \
  --driver rtlsim \
  --elf /path/to/kernel.elf \
  --build \
  --make-var 'CONFIGS=-DNUM_CORES=1 -DNUM_WARPS=4 -DNUM_THREADS=4'
```

Run `simx`:

```bash
scripts/run-vortex-sim.sh \
  --platform-root /path/to/vortex-platform \
  --driver simx \
  --bin /path/to/kernel.bin \
  --build \
  --sim-arg -c --sim-arg 1 \
  --sim-arg -w --sim-arg 4 \
  --sim-arg -t --sim-arg 4
```

If the shell exports unusable proxy variables, add `--no-proxy`.

## Registered Pipelines

- `vortex-pre-vortex-pipeline`
- `vortex-onnx-matmul-to-pre-vortex-pipeline`
- `vortex-mvp-backend-pipeline`

## Implemented Passes

- `vortex-validate-pre-vortex`
- `vortex-summarize-pre-vortex`
- `vortex-normalize-onnx-frontend`
- `vortex-tile-matmul-for-pre-vortex`
- `vortex-mark-kernel`
- `vortex-materialize-address-spaces`
- `vortex-map-parallel-loops-to-launch`
- `vortex-promote-tiles-to-local`
- `vortex-insert-barriers`
- `vortex-plan-local-memory-layout`
- `vortex-lower-local-memory`
- `vortex-lower-linalg-inside-kernel`
- `vortex-legalize-for-llvm`
- `vortex-lower-runtime-builtins`

## Docs

- `docs/MLIR_IR_DESIGN_AND_PLAN.md`
- `docs/PRE_VORTEX_TO_VORTEX_PASS_PLAN.md`
- `docs/PRE_VORTEX_TO_VORTEX_GENERAL_LOWERING.md`
- `docs/MVP_BACKEND_PASS_CHECKLIST.md`
- `docs/VORTEX_LOCAL_ALLOC_LOWERING_DESIGN.md`
- `docs/PYTORCH_ONNX_ONNX_MLIR_FRONTEND_PLAN.md`
- `docs/PHASE1_PYTORCH_FRONTEND_TASKS.md`
- `docs/SIMULATION_BACKEND_INTEGRATION.md`
