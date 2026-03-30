# mlir-vortex

`mlir-vortex` 是一个 out-of-tree MLIR 工程，用来承载 Vortex 相关的：

1. `Vortex dialect`
2. `pre-vortex -> vortex` 转换 pass
3. `vx-opt` 驱动
4. 测试、样例和设计文档

当前仓库仍处于原型阶段，目标是先打通：

```text
pre-vortex 标准 MLIR
  -> 混合 Vortex IR
  -> LLVM dialect
  -> LLVM IR
  -> Vortex 后端
```

## 仓库结构

- `include/`：头文件、TableGen 定义
- `lib/`：dialect、pass、pipeline 实现
- `tools/vx-opt/`：命令行驱动
- `test/`：lit 测试
- `examples/`：示例 IR
- `docs/`：设计和开发计划
- `third_party/llvm/`：固定版本的 LLVM/MLIR submodule

## LLVM 依赖

仓库内已经带了 LLVM submodule：

- 路径：`third_party/llvm`
- 远程：`https://github.com/vortexgpgpu/llvm.git`
- 分支：`vortex_2.x`
- 当前固定 commit：`d78d4a25ebfa0a9145e2c5b2590daccdb56da93a`

注意：

1. `mlir-vortex` 当前依赖一个**已经构建好的** LLVM/MLIR
2. 仓库里的 `third_party/llvm` 只是源码，不会自动替你完成 LLVM 构建
3. `CMakeLists.txt` 当前通过 `find_package(MLIR REQUIRED CONFIG)` 查找 MLIR

## 推荐构建方式

下面给出一套不依赖任何本机私有路径、其他人 clone 后也能直接照着跑的方式。

### 1. clone 仓库并拉 submodule

```bash
git clone --recurse-submodules git@github.com:VCompiler/vortex-compiler.git
cd vortex-compiler
```

如果已经 clone 了主仓库：

```bash
git submodule update --init --recursive
```

### 2. 先单独构建 LLVM/MLIR

可以直接使用 `third_party/llvm` 作为源码树，并在仓库内放一个单独的 LLVM build 目录，例如：

```bash
cmake -S third_party/llvm/llvm -B third_party/llvm-build \
  -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_BUILD_TYPE=Release

cmake --build third_party/llvm-build -j$(nproc)
```

如果你不想把 LLVM build 放在仓库里，也可以放到仓库外任意位置。关键点只有一个：

```text
MLIR_DIR 和 LLVM_DIR 必须指向“LLVM/MLIR build 目录”里的 cmake package，
而不是指向 third_party/llvm 这个源码目录本身。
```

### 3. 再构建 `mlir-vortex`

```bash
cmake -S . -B build \
  -G Ninja \
  -DMLIR_DIR=$PWD/third_party/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/third_party/llvm-build/lib/cmake/llvm \
  -DVORTEX_ENABLE_TESTS=ON

cmake --build build -j$(nproc)
```

### 4. 跑测试

```bash
cmake --build build --target check-vortex -j$(nproc)
```

### 5. 使用 `vx-opt`

```bash
./build/bin/vx-opt --help
```

## 使用外部 LLVM build

如果你已经有一份单独构建好的 LLVM/MLIR，也可以直接复用，只要把 `MLIR_DIR` / `LLVM_DIR` 指到那份 build：

```bash
cmake -S . -B build \
  -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm \
  -DVORTEX_ENABLE_TESTS=ON

cmake --build build -j$(nproc)
cmake --build build --target check-vortex -j$(nproc)
```

## 当前已有内容

当前仓库已经包含：

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
