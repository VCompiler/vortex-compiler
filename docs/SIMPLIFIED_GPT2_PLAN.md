# 简化版 GPT-2 on Vortex：差距分析与实施清单

## 0. 当前进度

> 最后更新：2026-04-04

| Step | 内容 | 状态 |
|------|------|------|
| Step 1 | math-to-llvm/libm lowering 通路 | ✅ 已合入 main |
| Step 2 | reduce_sum / reduce_max 端到端 simx 验证 | ✅ passed |
| Step 3 | GeLU / Softmax / LayerNorm 端到端 simx 验证 | ✅ passed |
| Step 4 | MLP block (matmul → GeLU → matmul) | ✅ passed |
| Step 5 | Single-head attention (QKV + K^T + softmax + output) | ✅ passed |
| Step 6 | 完整 Transformer block (LN + Attn + Res + LN + MLP + Res) | ✅ passed |
| Step 7 | 扩展到真实尺寸 (seq=32, d=64) | 未开始 |
| Step 8 | 多 block / embedding / lm_head | 未开始 |
| Step 9 | ONNX 前端自动化 | 未开始 |

**核心结论：单个 Transformer block 已在 simx 上端到端跑通（seq=4, d=8, 1-head, f32），
8 个 kernel 全部通过数值验证。**

已验证的 kernel（全部在 `examples/gpt2/` 下）：

```
reduce_sum.mlir          → simx passed (sum=136)
reduce_max.mlir          → simx passed (max=16)
gelu.mlir                → simx passed (vs erff reference)
softmax.mlir             → simx passed (row sums=1.0)
layernorm.mlir           → simx passed (mean≈0, var≈1)
mlp_block.mlir           → simx passed
attention.mlir           → simx passed
transformer_block.mlir   → simx passed
```

---

## 1. 目标

### 1.1 最终目标

在 Vortex simx 仿真器上跑通一个简化版 GPT-2 Transformer block 的 forward pass，
结果与 CPU (Python/numpy) 数值对齐。

### 1.2 约束

- batch = 1
- seq_len = 32
- d_model = 64
- num_heads = 1（single-head attention，避开 reshape/multi-head split）
- f32 精度
- inference only，静态 shape
- 不做 KV cache、decode loop、多层 block、embedding

### 1.3 计算图

一个 Transformer block 的完整计算图：

```
x: [32, 64]

--- Pre-Attention LayerNorm ---
mean = reduce_mean(x, axis=-1)              # [32, 1]
var  = reduce_mean((x - mean)^2, axis=-1)   # [32, 1]
x_ln = (x - mean) / sqrt(var + eps) * gamma + beta

--- Self-Attention (single head) ---
Q = x_ln @ Wq                               # [32, 64] @ [64, 64] -> [32, 64]
K = x_ln @ Wk                               # [32, 64] @ [64, 64] -> [32, 64]
V = x_ln @ Wv                               # [32, 64] @ [64, 64] -> [32, 64]
score = Q @ K^T / sqrt(64)                   # [32, 64] @ [64, 32] -> [32, 32]
prob  = softmax(score, axis=-1)              # [32, 32]
attn  = prob @ V                             # [32, 32] @ [32, 64] -> [32, 64]
out   = attn @ Wo                            # [32, 64] @ [64, 64] -> [32, 64]
x = x + out                                 # residual

--- Pre-MLP LayerNorm ---
x_ln2 = layernorm(x)

--- MLP ---
h = x_ln2 @ W1                              # [32, 64] @ [64, 256] -> [32, 256]
h = gelu(h)                                  # [32, 256]
h = h @ W2                                   # [32, 256] @ [256, 64] -> [32, 64]
x = x + h                                   # residual
```

涉及的原子操作：

| 操作 | 出现位置 | MLIR 表达 |
|------|----------|-----------|
| matmul | QKV projection, score, attn, output, MLP | `linalg.matmul` |
| reduce_mean | LayerNorm | `linalg.generic` (reduction iterator) |
| reduce_max | softmax 数值稳定 | `linalg.generic` (reduction iterator) |
| reduce_sum | softmax 归一化 | `linalg.generic` (reduction iterator) |
| exp | softmax | `math.exp` in `linalg.generic` body |
| sqrt | LayerNorm, attention scale | `math.sqrt` in `linalg.generic` body |
| erf | GeLU | `math.erf` in `linalg.generic` body |
| elementwise add/sub/mul/div | 各处 | `arith.*` in `linalg.generic` body |
| transpose (K^T) | attention score | `linalg.generic` indexing_maps 表达 |

## 2. 当前状态

### 2.1 已验证可工作

| 能力 | 依据 |
|------|------|
| 静态 rank-2 `linalg.matmul` (buffer semantics) → LLVM → RISC-V → simx | `matmul4x4` smoke test 端到端通过 |
| `linalg.fill` + `linalg.matmul` pattern tiling | `TileMatmulForPreVortexPass` |
| `linalg.generic` (buffer semantics) → `scf.for` loops | `LowerLinalgInsideVortexKernelPass` 用 `linalgOpToLoops` |
| `scf.for` / `arith` / `memref` → LLVM dialect | MVP backend pipeline |
| f32 加减乘除、fma、sqrt | rv32imaf 硬件指令 |
| MLIR → `mlir-translate` → LLVM IR → clang → ELF → simx | 完整链路已验证 |
| `-lm` (newlib libm) 链接成功 | smoke test 链接了 libm |

### 2.2 确认不可工作

| 能力 | 原因 | 影响 |
|------|------|------|
| `math.exp` / `math.erf` / `math.tanh` → LLVM | `buildMVPBackendPipeline` 缺 `convert-math-to-llvm` 和 `convert-math-to-libm` | softmax / GeLU / LayerNorm 全部无法编译 |
| reduction `linalg.generic` 端到端 | 从未验证过 | softmax / LayerNorm 无法确认正确性 |
| 复合 elementwise `linalg.generic` 端到端 | 从未验证过 | GeLU / softmax / LayerNorm 无法确认正确性 |

### 2.3 瓶颈判断

**核心瓶颈极小：** `Pipelines.cpp:82` 的 `buildMVPBackendPipeline` 缺少两个标准 MLIR pass：

1. `createConvertMathToLLVMPass()` — 把 `math.exp`/`math.sqrt` 等映射到 LLVM intrinsics
2. `createConvertMathToLibmPass()` — 把 `math.erf`/`math.tanh` 等映射到 libm 函数调用

这两个 pass 的库（`libMLIRMathToLLVM.a`、`libMLIRMathToLibm.a`）已经在 LLVM build 里了，
头文件也已经被 `mlir/Conversion/Passes.h` include 了。**加两行代码 + CMake 链接就能解锁。**

后端不需要改。LLVM fork 的 RISC-V backend 会把 LLVM intrinsic `@llvm.exp.f32` 等
展开成 libm 调用（`expf`），链接时 `-lm` 提供实现。

## 3. 解决方案

### 3.1 策略

**手写 MLIR，逐层验证，自底向上组装。**

不走 ONNX 前端（当前只支持 matmul，扩展成本高且与核心目标无关）。
直接用 `linalg.matmul` + `linalg.generic` 手写每个子图的 pre-vortex IR，
配合 C wrapper 做 simx 数值验证。

### 3.2 分层拆解

```
Layer 0: 基础设施                    ← 补 math lowering pass
Layer 1: 原子 kernel 验证            ← reduce_sum, reduce_max, exp, erf
Layer 2: 复合 kernel 验证            ← GeLU, softmax, LayerNorm
Layer 3: 子图组装                    ← MLP block, attention block
Layer 4: 完整 Transformer block      ← 组装 + 端到端数值对齐
```

每一层都必须 simx 通过后才进入下一层。

## 4. 方案拆解

### Step 1: 补 math lowering 通路 ✅

**目标：** 让 `math.exp`、`math.sqrt`、`math.erf`、`math.tanh` 能走完 pipeline 到 LLVM IR。

**修改清单：**

1. `lib/Pipeline/Pipelines.cpp` — `buildMVPBackendPipeline` 中，在 `createArithToLLVMConversionPass()` 之前加入：
   ```cpp
   pm.addPass(createConvertMathToLLVMPass());  // exp, sqrt, log → LLVM intrinsics
   pm.addPass(createConvertMathToLibmPass());  // erf, tanh → libm calls
   ```

2. `lib/Pipeline/CMakeLists.txt` — LINK_LIBS 加入：
   ```
   MLIRMathToLLVM
   MLIRMathToLibm
   ```

3. `test/Pipeline/math-to-llvm.mlir` — lit 测试：
   - 输入包含 `math.exp`、`math.sqrt`、`math.erf` 的 kernel
   - 验证输出为纯 LLVM dialect（无残留 math op）

**验收标准：** `check-vortex` 全部通过，新 lit 测试确认 math ops 被消除。

---

### Step 2: 验证 reduction kernel 端到端 ✅

**目标：** 确认 `linalg.generic` 带 reduction iterator 能走完 pipeline 到 simx 并数值正确。

**交付物：**

1. `examples/gpt2/reduce_sum.mlir` — f32 向量 reduce_sum：
   ```mlir
   // input: memref<32xf32>, output: memref<1xf32>
   // linalg.generic with reduction iterator, body: arith.addf
   ```

2. `examples/gpt2/reduce_sum_wrapper.c` — C wrapper：
   - 填入固定输入数据
   - 调用 kernel
   - 与预期结果比较（`fabs(actual - expected) < 1e-4`）
   - 返回 0 成功 / 非 0 失败

3. `examples/gpt2/reduce_max.mlir` + wrapper — 同上，body 改为 `arith.maximumf`

**验收标准：** `simx -c 1 -w 4 -t 4 reduce_sum.bin` 输出 pass，数值正确。

---

### Step 3: 验证 math kernel 端到端 ✅

**目标：** 确认 `math.exp`、`math.sqrt`、`math.erf` 在 simx 上数值正确。

**交付物：**

1. `examples/gpt2/exp_f32.mlir` — elementwise exp：
   ```mlir
   // linalg.generic, body: math.exp
   ```

2. `examples/gpt2/exp_f32_wrapper.c` — 与 C `expf()` 结果对比

3. `examples/gpt2/sqrt_f32.mlir` + wrapper — 同理

4. `examples/gpt2/erf_f32.mlir` + wrapper — 同理

**验收标准：** 每个 kernel simx 通过，与 C 标准库函数结果误差 < 1e-4。

---

### Step 4: GeLU kernel ✅

**目标：** GeLU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

**交付物：**

1. `examples/gpt2/gelu.mlir` — 输入 `memref<32x256xf32>`，输出同 shape：
   ```mlir
   // linalg.generic, parallel x parallel, body:
   //   %c = arith.constant 0.5
   //   %sqrt2 = arith.constant 1.41421356...
   //   %x_scaled = arith.divf %x, %sqrt2
   //   %erf_val = math.erf %x_scaled
   //   %one = arith.constant 1.0
   //   %sum = arith.addf %one, %erf_val
   //   %half = arith.mulf %x, %c
   //   %result = arith.mulf %half, %sum
   ```

2. `examples/gpt2/gelu_wrapper.c` — 与 C 实现对比

**验收标准：** simx 数值对齐。

---

### Step 5: Softmax kernel ✅

**目标：** softmax(x, axis=-1) 数值稳定版

实现拆成三步：
1. `reduce_max` 沿 axis=-1 → `max_val: memref<32x1xf32>`
2. `exp(x - max_val)` → `exp_val: memref<32x32xf32>`
3. `reduce_sum(exp_val)` → `sum_val: memref<32x1xf32>`
4. `exp_val / sum_val` → `output: memref<32x32xf32>`

**交付物：**

1. `examples/gpt2/softmax.mlir` — 输入 `memref<32x32xf32>`
2. `examples/gpt2/softmax_wrapper.c` — 与 C softmax 对比

**验收标准：** simx 数值对齐。

---

### Step 6: LayerNorm kernel ✅

**目标：** LayerNorm(x, gamma, beta, eps=1e-5)

实现拆成：
1. `reduce_mean(x)` 沿 axis=-1
2. `x - mean`
3. `reduce_mean((x-mean)^2)` 沿 axis=-1
4. `rsqrt(var + eps)`
5. `normalize * gamma + beta`

**交付物：**

1. `examples/gpt2/layernorm.mlir` — 输入 `memref<32x64xf32>` + gamma/beta `memref<64xf32>`
2. `examples/gpt2/layernorm_wrapper.c` — 与 C 实现对比

**验收标准：** simx 数值对齐。

---

### Step 7: MLP subgraph ✅

**目标：** matmul → GeLU → matmul

**交付物：**

1. `examples/gpt2/mlp_block.mlir`:
   ```
   input [32, 64] @ W1 [64, 256] → [32, 256]
   → GeLU
   → @ W2 [256, 64] → [32, 64]
   ```
2. `examples/gpt2/mlp_block_wrapper.c` — 用小尺寸权重验证

**验收标准：** simx 数值与 numpy 对齐。

---

### Step 8: Single-head attention subgraph ✅

**目标：** Q/K/V projection → score → softmax → value aggregation → output projection

**交付物：**

1. `examples/gpt2/attention.mlir`:
   ```
   Q = x @ Wq, K = x @ Wk, V = x @ Wv
   score = Q @ K^T / sqrt(64)            // K^T 通过 indexing_maps 转置
   prob = softmax(score)
   attn = prob @ V
   out = attn @ Wo
   ```
   K 的转置通过 `linalg.generic` 的 `indexing_maps` 中交换维度实现，
   不需要显式 `memref.transpose`。

2. `examples/gpt2/attention_wrapper.c`

**验收标准：** simx 数值与 numpy 对齐。

---

### Step 9: 完整 Transformer block ✅

**目标：** LayerNorm + Attention + Residual + LayerNorm + MLP + Residual

**交付物：**

1. `examples/gpt2/transformer_block.mlir` — 组装 Step 5-8 的所有操作
2. `examples/gpt2/transformer_block_wrapper.c` — 完整数值验证
3. `examples/gpt2/reference.py` — 用 numpy 或 PyTorch 生成 golden 数据

**验收标准：** simx 输出与 Python reference 逐元素误差 < 1e-3（f32 累积误差允许稍大）。

## 5. 目录结构

```
examples/gpt2/
  ├── reduce_sum.mlir
  ├── reduce_sum_wrapper.c
  ├── reduce_max.mlir
  ├── reduce_max_wrapper.c
  ├── exp_f32.mlir
  ├── exp_f32_wrapper.c
  ├── sqrt_f32.mlir
  ├── sqrt_f32_wrapper.c
  ├── erf_f32.mlir
  ├── erf_f32_wrapper.c
  ├── gelu.mlir
  ├── gelu_wrapper.c
  ├── softmax.mlir
  ├── softmax_wrapper.c
  ├── layernorm.mlir
  ├── layernorm_wrapper.c
  ├── mlp_block.mlir
  ├── mlp_block_wrapper.c
  ├── attention.mlir
  ├── attention_wrapper.c
  ├── transformer_block.mlir
  ├── transformer_block_wrapper.c
  └── reference.py
```

## 6. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| newlib libm 缺 `erff` | GeLU 链接失败 | 确认 newlib 版本；如不行改用 tanh 近似 GeLU |
| f32 累积误差 | 完整 block 数值偏差大 | 用数值稳定算法（softmax 减 max, LayerNorm Welford）；放宽 block 级阈值到 1e-3 |
| `linalg.generic` 复杂 body codegen 有 bug | 某步生成错误 LLVM IR | 逐原子操作验证，每步独立测试 |
| kernel 太大寄存器溢出 | 完整 block 编译失败或性能极差 | 拆成多个 kernel 分步调用 |
| simx exit code 约定 | 脚本误判失败 | wrapper 统一用 `vx_putchar` 输出结果标记 |

## 7. 下一阶段工作

第一阶段目标（单个 Transformer block, simx 数值验证）已完成。

下一阶段可选方向：

1. **扩展尺寸** — seq=32, d=64, d_ff=256，验证更大 kernel 的 codegen 稳定性和寄存器压力
2. **多 block** — 串联 N 个 Transformer block
3. **加载真实权重** — 从文件读取 GPT-2 权重而非硬编码
4. **embedding / lm_head** — 补齐推理链路的首尾
5. **ONNX 前端自动化** — PyTorch 导出 → ONNX → 自动编译
6. **多核并行** — 利用 vortex.launch 做 core/warp/thread 映射
7. **性能优化** — tiling、local memory promotion for attention

## 8. 不在当前范围内

- KV cache / decode loop
- bf16 / fp16
- 训练
- 动态 shape
