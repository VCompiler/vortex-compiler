# mlir-vortex

Out-of-tree MLIR project for:

1. `Vortex dialect`
2. `pre-vortex -> vortex` transformation passes
3. `vx-opt`
4. tests, examples, and design docs

## 仓库结构

- `include/`：头文件、TableGen 定义
- `lib/`：dialect、pass、pipeline 实现
- `tools/vx-opt/`：命令行驱动
- `test/`：lit 测试
- `examples/`：示例 IR
- `examples/frontend/`：PyTorch / ONNX / ONNX-MLIR 前端样例与中间产物
- `docs/`：设计和开发计划
- `third_party/llvm/`：固定版本的 LLVM/MLIR submodule

## LLVM 依赖

- 路径：`third_party/llvm`
- 远程：`https://github.com/vortexgpgpu/llvm.git`
- 分支：`vortex_2.x`
- 当前固定 commit：`d78d4a25ebfa0a9145e2c5b2590daccdb56da93a`

要求：

1. 先单独构建 LLVM/MLIR
2. `MLIR_DIR` / `LLVM_DIR` 指向 LLVM/MLIR build 目录中的 CMake package
3. `third_party/llvm` 只是源码树，不是 build 目录

## 构建

### 1. 拉取仓库和 submodule

```bash
git clone --recurse-submodules git@github.com:VCompiler/vortex-compiler.git
cd vortex-compiler
```

### 2. 构建 LLVM/MLIR

```bash
cmake -S third_party/llvm/llvm -B third_party/llvm-build \
  -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_BUILD_TYPE=Release

cmake --build third_party/llvm-build -j$(nproc)
```

### 3. 构建 `mlir-vortex`

```bash
cmake -S . -B build \
  -G Ninja \
  -DMLIR_DIR=$PWD/third_party/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/third_party/llvm-build/lib/cmake/llvm \
  -DVORTEX_ENABLE_TESTS=ON

cmake --build build -j$(nproc)
```

### 4. 运行测试

```bash
cmake --build build --target check-vortex -j$(nproc)
```

### 5. 使用 `vx-opt`

```bash
./build/bin/vx-opt --help
```

## 使用外部 LLVM build

如果已有独立的 LLVM/MLIR build：

```bash
cmake -S . -B build \
  -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm \
  -DVORTEX_ENABLE_TESTS=ON

cmake --build build -j$(nproc)
cmake --build build --target check-vortex -j$(nproc)
```

## 当前内容

1. `Vortex dialect` 基础骨架
2. `vortex-pre-vortex-pipeline`
3. `vortex-mark-kernel`
4. `vortex-materialize-address-spaces`
5. `vortex-map-parallel-loops-to-launch`
6. `vortex-promote-tiles-to-local`

## 相关文档

- `docs/MLIR_IR_DESIGN_AND_PLAN.md`
- `docs/PRE_VORTEX_TO_VORTEX_PASS_PLAN.md`
- `docs/PRE_VORTEX_TO_VORTEX_GENERAL_LOWERING.md`
- `docs/MVP_BACKEND_PASS_CHECKLIST.md`
- `docs/PYTORCH_ONNX_ONNX_MLIR_FRONTEND_PLAN.md`
- `docs/PHASE1_PYTORCH_FRONTEND_TASKS.md`
