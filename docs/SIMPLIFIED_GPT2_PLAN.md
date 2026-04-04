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
| **第二阶段** | | |
| M2.1 | 放大到 seq=32, d=64 | 未开始 |
| M2.2 | 权重从文件加载 | 未开始 |
| M2.3 | 4 层 block 串联 | 未开始 |
| **第三阶段** | | |
| M3.1 | Token embedding kernel | 未开始 |
| M3.2 | Position embedding | 未开始 |
| M3.3 | LM head (output projection) | 未开始 |
| M3.4 | 端到端 forward (token_ids → logits) | 未开始 |
| **第四阶段** | | |
| M4.1 | matmul 多核并行 | 未开始 |
| M4.2 | attention 并行 | 未开始 |
| M4.3 | 性能基线建立 | 未开始 |
| **第五阶段** | | |
| M5.1-M5.2 | ONNX bridge 基础算子 | 未开始 |
| M5.3-M5.4 | ONNX bridge 复合算子 | 未开始 |
| M5.5 | 全自动 PyTorch → simx | 未开始 |

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

## 7. 第二阶段：扩展到真实尺寸 + 多 block

### 7.1 目标

```
4 层 Transformer block, seq_len=32, d_model=64, d_ff=256, 1 head, f32
权重从二进制文件加载，simx 数值对齐
```

内存预估：4 层约 900 KB 权重 + 120 KB 激活，simx 虚拟内存按需分配，无硬限制。

### 7.2 里程碑

| 编号 | 里程碑 | 验收标准 |
|------|--------|----------|
| M2.1 | 单 block 放大到 seq=32, d=64 | simx passed，与 Python reference 对齐 |
| M2.2 | 权重从文件加载 | 不再硬编码权重，C wrapper 从 `.bin` 文件读取 |
| M2.3 | 4 层 block 串联 | simx passed，逐层数值与 PyTorch 对齐 |

### 7.3 任务拆解

**M2.1 — 放大尺寸**

现状：当前 transformer_block.mlir 硬编码 seq=4, d=8。
风险：更大 kernel 可能触发寄存器溢出、栈溢出、或 codegen bug。

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T2.1.1 | 参数化 MLIR 生成 | Python 脚本 `gen_transformer_mlir.py`，输入 seq/d/d_ff，输出 `.mlir` 文件 |
| T2.1.2 | 参数化 C wrapper 生成 | Python 脚本 `gen_wrapper.py`，输入 seq/d/d_ff，输出 `_wrapper.c` 文件 |
| T2.1.3 | 渐进测试 | seq=8/d=16 → seq=16/d=32 → seq=32/d=64，每步 simx 验证 |
| T2.1.4 | 如果寄存器溢出 | 拆成多个 kernel（attention 单独、MLP 单独），wrapper 依次调用 |

**M2.2 — 权重加载**

现状：权重硬编码在 C wrapper 里。真实推理需要从文件读取。

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T2.2.1 | Python 权重导出 | `export_weights.py`：PyTorch 随机初始化或加载预训练权重 → 逐层导出为 raw float32 `.bin` |
| T2.2.2 | Python reference 生成 | `gen_golden.py`：用 PyTorch 跑 forward，导出每层输入/输出为 `.bin`，供逐层对比 |
| T2.2.3 | C 权重加载器 | `weight_loader.c`：在 simx 上用 `vx_dev_read()` 或直接内存映射从 `.bin` 加载权重到 buffer |
| T2.2.4 | 端到端验证 | wrapper 加载权重 → 跑 kernel → 比对 golden → simx passed |

**M2.3 — 多 block 串联**

现状：只有单个 block。GPT-2-small 有 12 层，我们先做 4 层。

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T2.3.1 | 多 block MLIR | 生成脚本输出 4 层 block，每层独立 kernel 函数（`@block_0`, `@block_1`, ...） |
| T2.3.2 | 多 block wrapper | C wrapper 依次调用 4 个 block kernel，中间传递激活 |
| T2.3.3 | 逐层数值对比 | 每层输出与 PyTorch golden 比对，定位精度累积 |
| T2.3.4 | 放宽误差阈值 | 如果 4 层累积误差超出 1e-2，评估是否可接受或需要改进数值稳定性 |

---

## 8. 第三阶段：完整推理链路

### 8.1 目标

```
完整 GPT-2 推理：token_ids → embedding → N x Transformer block → lm_head → logits
在 simx 上跑通，输出 logits 与 PyTorch 对齐
```

### 8.2 里程碑

| 编号 | 里程碑 | 验收标准 |
|------|--------|----------|
| M3.1 | Token embedding kernel | 给定 token_id 序列，查表输出 embedding 矩阵，simx 数值对齐 |
| M3.2 | Position embedding | 位置编码加到 token embedding 上 |
| M3.3 | LM head (output projection) | 最后一层输出 → vocab logits |
| M3.4 | 端到端 forward | token_ids → logits，单次推理完整链路 |

### 8.3 任务拆解

**M3.1 — Token embedding**

embedding 本质是查表：`output[i] = embedding_table[token_id[i]]`

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T3.1.1 | embedding kernel MLIR | 输入 `memref<32xi32>`(token_ids) + `memref<256x64xf32>`(table)，输出 `memref<32x64xf32>` |
| T3.1.2 | 实现方式 | 用 `scf.for` + `memref.load`(index) + `memref.copy`(行) 或 linalg.generic |
| T3.1.3 | 验证 | simx 输出与直接查表结果一致 |

**M3.2 — Position embedding**

GPT-2 的位置编码是可学习参数，不是 sinusoidal。

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T3.2.1 | position embedding kernel | 和 token embedding 同样的查表，然后 elementwise add |
| T3.2.2 | 合并到 embedding 阶段 | `output = token_emb + pos_emb`，一个 kernel 完成 |

**M3.3 — LM head**

LM head = 最终 LayerNorm + 线性投影到 vocab 维度。

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T3.3.1 | final LayerNorm | 复用已有 LayerNorm pattern |
| T3.3.2 | vocab projection | `memref<32x64xf32>` @ `memref<64x256xf32>` → `memref<32x256xf32>` (logits) |
| T3.3.3 | argmax (可选) | 取 logits 最大值的 index 作为预测 token |

**M3.4 — 端到端**

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T3.4.1 | 完整 wrapper | C main: 加载权重 → embedding → N blocks → lm_head → 输出 logits |
| T3.4.2 | PyTorch golden | 用相同权重和输入跑 PyTorch forward，导出 logits |
| T3.4.3 | 数值对比 | simx logits 与 PyTorch logits 对比，验证 top-k 一致性 |

---

## 9. 第四阶段：多核并行

### 9.1 目标

```
利用 Vortex 多核/多 warp/多 thread 硬件并行，加速 Transformer block 执行
```

### 9.2 里程碑

| 编号 | 里程碑 | 验收标准 |
|------|--------|----------|
| M4.1 | matmul 行并行 | matmul 外层循环映射到 core/warp，simx 多核配置下数值正确 |
| M4.2 | attention 并行 | seq 维度映射到并行层级 |
| M4.3 | 性能基线 | 建立 simx cycle count 基线，与单核对比 |

### 9.3 任务拆解

**M4.1 — matmul 并行**

当前所有 kernel 都在单核单线程上跑。Vortex 的编程模型是
给 `scf.for` 加 `vortex.mapping` 属性，让 pass 自动映射到 `vortex.launch`。

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T4.1.1 | 标注 matmul 外层循环 | 给 M 维度的 `scf.for` 加 `vortex.mapping = "core"` |
| T4.1.2 | 跑通 vortex.launch pipeline | 确认 mark-kernel → map-loops-to-launch → lower → simx 正确 |
| T4.1.3 | 多核 simx 验证 | `simx -c 4 -w 4 -t 4`，数值与单核一致 |

**M4.2 — attention 并行**

attention 里的多个 matmul 可以在 seq 维度并行。softmax 的 reduction 需要在并行后做 barrier。

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T4.2.1 | attention seq 维度并行 | Q/K/V projection 的 seq 维度映射到 core |
| T4.2.2 | softmax 并行化 | reduce_max/reduce_sum 需要跨线程协作，用 local memory + barrier |
| T4.2.3 | 完整 attention 并行验证 | 多核 simx 数值对齐 |

**M4.3 — 性能基线**

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T4.3.1 | 统计 simx cycle count | 用 `simx -s` 输出性能统计 |
| T4.3.2 | 建立基线表 | 单核 vs 多核 cycle count 对比 |
| T4.3.3 | 识别瓶颈 | 分析是 compute bound 还是 memory bound |

---

## 10. 第五阶段：前端自动化

### 10.1 目标

```
PyTorch GPT-2 模型 → ONNX → ONNX-MLIR → vortex-compiler → simx
全自动，不需要手写 MLIR
```

### 10.2 里程碑

| 编号 | 里程碑 | 验收标准 |
|------|--------|----------|
| M5.1 | ONNX bridge 支持 elementwise ops | Add/Sub/Mul/Div/Sqrt 等走通 |
| M5.2 | ONNX bridge 支持 reduction | ReduceMean/ReduceSum/ReduceMax 走通 |
| M5.3 | ONNX bridge 支持 Softmax/LayerNorm | 作为复合子图走通 |
| M5.4 | ONNX bridge 支持 attention pattern | MatMul + Transpose + Softmax 组合走通 |
| M5.5 | 自动编译简化版 GPT-2 | PyTorch → ONNX → vx-opt → simx 全自动 |

### 10.3 任务拆解

**M5.1-M5.2 — 基础算子 bridge**

当前 `NormalizeONNXFrontendPass` 只做 metadata 清理，`TileMatmulForPreVortexPass` 只处理 matmul。
ONNX-MLIR 本身会把大部分 ONNX ops lower 到 `linalg`/`arith`/`math`，
所以关键是确认 ONNX-MLIR 的输出能被现有 vortex pipeline 接受。

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T5.1.1 | 调研 ONNX-MLIR 输出 | 用 ONNX-MLIR 编译简化版 GPT-2 block 的 ONNX，检查输出 IR 的 dialect 集合 |
| T5.1.2 | 识别不兼容 op | 列出 ONNX-MLIR 输出中 vortex pre-vortex 白名单不接受的 op |
| T5.1.3 | 补齐 bridge pass | 对不兼容 op 写转换规则或在 ONNX-MLIR 侧调整 lowering 选项 |
| T5.1.4 | elementwise 验证 | 用 ONNX Add/Mul/Div 模型走通 ONNX → vx-opt → simx |

**M5.3-M5.4 — 复合算子 bridge**

ONNX 的 Softmax/LayerNormalization 可能被 ONNX-MLIR 分解成多个 linalg 操作。
需要确认分解结果能走完 vortex pipeline。

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T5.3.1 | Softmax 子图验证 | ONNX Softmax → ONNX-MLIR → vx-opt → simx |
| T5.3.2 | LayerNorm 子图验证 | 同上 |
| T5.3.3 | Attention pattern | ONNX MatMul + Softmax 组合 → vx-opt → simx |
| T5.3.4 | bufferization 调优 | 确认 ONNX-MLIR 的 tensor → buffer 转换适配 vortex pipeline |

**M5.5 — 端到端自动化**

| 任务 | 内容 | 交付物 |
|------|------|--------|
| T5.5.1 | 导出脚本 | `export_gpt2_block.py`：PyTorch → ONNX |
| T5.5.2 | 一键编译脚本 | `compile_gpt2.sh`：ONNX → ONNX-MLIR → vx-opt → ELF |
| T5.5.3 | 端到端验证 | 从 PyTorch 到 simx 全自动，数值对齐 |

---

## 11. 阶段间依赖关系

```
第一阶段 (已完成)
  单 block, 小尺寸, 手写 MLIR, simx 验证
       │
       ▼
第二阶段 ─────────────────────────────────┐
  放大尺寸, 多 block, 权重加载             │
       │                                   │
       ▼                                   ▼
第三阶段                             第五阶段
  完整推理链路                         前端自动化
  (embedding → blocks → lm_head)     (ONNX bridge 扩展)
       │                                   │
       ▼                                   │
第四阶段                                   │
  多核并行                                 │
       │                                   │
       ▼                                   ▼
   ┌────────────────────────────────────────┐
   │  最终目标：自动编译 + 并行执行 GPT-2   │
   └────────────────────────────────────────┘
```

第二阶段是后续所有阶段的前提。第三阶段和第五阶段可以并行推进。
第四阶段依赖第三阶段（需要完整推理链路才能有意义地做并行优化）。

## 12. 不在规划范围内

- KV cache / 自回归 decode loop
- bf16 / fp16 混合精度
- 训练 / 反向传播
- 动态 shape / 动态 seq_len
- 真实 GPT-2-small (d=768, 12 层) — 当前规划用缩减版 (d=64, 4 层)
- FPGA 板级性能调优
