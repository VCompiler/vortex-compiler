# Vortex 仿真后端接入方案

## 1. 目标

把 `vortex-platform` 的本地仿真接到当前 `vortex-compiler` 仓库里，作为上板前的标准替代后端。

当前推荐边界：

```text
MLIR / LLVM lowering
  -> asm / obj / elf / bin
  -> 本仓库 wrapper
  -> simx 或 rtlsim(verilator)
```

这里不把整套 `vortex-platform` runtime、测试工程、板级脚本直接揉进当前 CMake。
第一阶段只做一层稳定桥接：

1. `vortex-compiler` 负责生成 `ELF/bin`
2. `vortex-platform` 负责提供 `simx/rtlsim`
3. 当前仓库提供统一脚本，把产物送进仿真

---

## 2. 为什么这样接

## 2.1 不直接把 `vortex-platform` 编进当前仓库

第一阶段不建议这么做，原因很直接：

1. `vortex-platform` 体量大，依赖和构建逻辑比当前编译器仓库重很多
2. `simx/rtlsim` 的职责是执行后端，不是 MLIR pass 本身
3. 把它们直接塞进当前 CMake，会让编译器开发和平台构建耦合过深

所以当前更合理的是：

1. 接口在当前仓库统一
2. simulator 代码仍然来自 `vortex-platform`
3. simulator 构建产物默认落在当前仓库 `build/` 下

## 2.2 为什么同时保留 `simx` 和 `rtlsim`

两者职责不同：

1. `simx`
   - 编译快
   - 适合 pass 快速回归
   - 适合做前端/IR/代码生成链路冒烟
2. `rtlsim`
   - 基于 Verilator
   - 更接近 RTL 和板级行为
   - 更适合替代“先上板再看”的流程

建议顺序：

1. 日常开发默认先跑 `simx`
2. 在准备替代上板验证时再跑 `rtlsim`

---

## 3. 当前仓库里的落地方式

当前已经新增统一入口：

```text
scripts/run-vortex-sim.sh
```

这个脚本负责三件事：

1. 如果输入是 `ELF`，先转成 simulator 需要的 `bin`
2. 按需构建 `simx` 或 `rtlsim`
3. 调用对应 simulator 运行 kernel

默认输出位置：

```text
build/vortex-sim/<driver>/
```

这样做的目的，是尽量不把 simulator 可执行文件和中间目录堆回 `vortex-platform` 源码树。

---

## 4. `vortex-platform` 的查找方式

脚本按下面顺序找 `vortex-platform`：

1. `--platform-root /path/to/vortex-platform`
2. 环境变量 `VORTEX_PLATFORM_ROOT`
3. 当前仓库内的 `third_party/vortex-platform`

因此当前有两种推荐接法：

### 4.1 外部 checkout

适合当前阶段，最轻量：

```bash
scripts/run-vortex-sim.sh \
  --platform-root /path/to/vortex-platform \
  --driver simx \
  --bin /path/to/kernel.bin
```

### 4.2 后续再收成 submodule

如果后面要把仿真环境也固定进当前工程，可以再把：

```text
third_party/vortex-platform
```

做成 git submodule。

到那时脚本接口不需要改，只是路径自动命中。

---

## 5. 运行方式

先准备 `vortex-platform` 依赖：

```bash
git -C /path/to/vortex-platform submodule update --init --recursive
```

## 5.1 运行 `rtlsim` 作为上板替代

```bash
scripts/run-vortex-sim.sh \
  --platform-root /path/to/vortex-platform \
  --driver rtlsim \
  --elf /path/to/kernel.elf \
  --build \
  --make-var 'CONFIGS=-DNUM_CORES=1 -DNUM_WARPS=4 -DNUM_THREADS=4'
```

说明：

1. `rtlsim` 需要 `verilator`
2. 硬件配置主要通过 `--make-var 'CONFIGS=...'` 传给 simulator 构建
3. 这条链更适合做“替代上板”的本地验证

## 5.2 运行 `simx` 做快速回归

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

说明：

1. `simx` 除了编译时配置，还支持一部分运行时参数
2. 适合先验证编译链路和返回值/输出 buffer

如果当前 shell 里残留了不可用的代理变量，可以额外加：

```bash
--no-proxy
```

## 5.3 只看命令，不实际执行

```bash
scripts/run-vortex-sim.sh \
  --platform-root /path/to/vortex-platform \
  --driver rtlsim \
  --elf /path/to/kernel.elf \
  --dry-run
```

---

## 6. 当前边界下还没有做的事

当前脚手架只解决“把 kernel image 喂给 simulator”。

还没有做的是：

1. 从当前仓库一键把 `MLIR -> ELF/bin` 全串起来
2. 自动对接 host 侧输入输出封装
3. 自动把 simulator 结果和 CPU golden 做 compare
4. 把仿真跑进 `lit` / CI

---

## 7. 下一步建议

按 MVP 顺序，建议继续补这几项：

1. `build-vortex-kernel.sh`
   - 输入 `mlir/ll/s`
   - 输出 `elf/bin`
2. `run-vortex-sim.sh` 冒烟 case
   - 至少固定一个 `matmul 4x4`
   - 输出结果与 CPU golden 对比
3. `run-vortex-board.sh`
   - 保持和 `run-vortex-sim.sh` 尽量一致的参数形状
4. CI / lit 条件测试
   - 当环境里有 `VORTEX_PLATFORM_ROOT` 时自动启用

这样后面仓库里的执行后端就会稳定成三类：

1. `simx`
2. `rtlsim`
3. `board`
