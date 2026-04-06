# torch-mlir 集成方案

## 1. 目标

用编译器 pass 替代手写代码生成器，实现：

```
PyTorch model
  → torch.export (PyTorch 自带)
  → torch-mlir (torch dialect → linalg-on-tensors)
  → bufferize (tensor → memref, MLIR 自带)
  → vortex pass pipeline (已有)
  → LLVM → RISC-V ELF
```

替换当前的：

```
PyTorch model
  → pytorch_to_vortex.py (手写代码生成器)
  → gen_full_inference.py (模板 MLIR + C wrapper)
```

## 2. 现状分析

### 2.1 已有的能力

| 组件 | 状态 |
|------|------|
| vortex pass pipeline（linalg-on-memref → LLVM） | ✅ 已验证 |
| ValidatePreVortexPass（检查输入 IR 合法性） | ✅ 已有 |
| bufferization dialect（CMake 已链接） | ✅ 可用 |
| NormalizeONNXFrontendPass（ONNX bridge） | ✅ 已有，可参考 |
| LLVM 18.1.7 fork | ✅ torch-mlir 有 LLVM 18 分支 |

### 2.2 缺失的部分

| 缺什么 | 说明 |
|--------|------|
| torch-mlir 本身 | 需要作为外部依赖构建或预编译使用 |
| tensor → memref bufferize 接入 | MLIR 自带 pass，需要接入 pipeline |
| func 签名转换 | torch-mlir 输出的 func 用 tensor 返回值，需要转成 memref 入参（out-param 风格） |
| vortex.entry 标记 | torch-mlir 不知道哪些 func 是 kernel entry |
| 静态 shape 保证 | torch-mlir 可能输出动态 shape，需要在接入时验证/拒绝 |

## 3. 两条路线对比

### 路线 A：torch-mlir 直连（本方案）

```
PyTorch → torch-mlir → linalg-on-tensors → bufferize → vortex pipeline
```

优点：
- 链路最短，中间表示损失最少
- torch-mlir 社区活跃，PyTorch 新特性跟进快
- 可以处理 Transformer / Attention 等 ONNX 支持不好的 op

缺点：
- torch-mlir 构建复杂（依赖 LLVM + PyTorch）
- 对 LLVM 版本敏感（需要和我们的 fork 对齐）

### 路线 B：ONNX-MLIR 桥接（已有文档 `PYTORCH_ONNX_ONNX_MLIR_FRONTEND_PLAN.md`）

```
PyTorch → ONNX → ONNX-MLIR → onnx-to-linalg → bufferize → vortex pipeline
```

优点：
- ONNX 是通用交换格式，不只服务 PyTorch
- ONNX-MLIR 有成熟的 shape inference

缺点：
- 多一层转换（PyTorch → ONNX 可能丢信息）
- ONNX 对 Transformer ops 支持有限（Attention、LayerNorm 版本碎片化）
- ONNX-MLIR 本身也需要构建和维护

### 推荐

**先走路线 A（torch-mlir），因为我们的目标模型是 Transformer**。ONNX 对 Transformer 的支持不如 torch-mlir 直接。路线 B 更适合 CNN 场景，可以后续再补。

## 4. 具体实施方案

### Phase 1：torch-mlir 外部调用（不改编译器，1-2 周）

不把 torch-mlir 编译进 vortex-compiler，而是作为外部工具使用：

```bash
# Step 1: torch-mlir 导出 linalg-on-tensors MLIR
python3 export_via_torch_mlir.py --model gpt2 --out model.linalg.mlir

# Step 2: vortex-compiler 的新前端 pipeline 处理
vx-opt model.linalg.mlir \
  --pass-pipeline="builtin.module(
    vortex-normalize-torch-frontend,     # 新增：清理 torch-mlir 特有的属性
    one-shot-bufferize,                  # MLIR 自带：tensor → memref
    buffer-results-to-out-params,        # MLIR 自带：返回值 → 出参
    func.func(
      vortex-mark-kernel,               # 已有
      vortex-materialize-address-spaces, # 已有
      vortex-lower-linalg-inside-kernel  # 已有
    ),
    ...后续已有 pipeline...
  )"
```

#### 需要新增的 pass

**1. `vortex-normalize-torch-frontend`**（约 100-200 行）

职责：
- 移除 torch-mlir 特有的属性和 metadata
- 验证所有 shape 是静态的（拒绝动态 shape）
- 给入口 func 添加 `vortex.entry` 属性
- 验证只使用允许的 dialect（func, arith, linalg, tensor, math, scf）

**2. 在 Pipeline 中接入标准 bufferize pass**

MLIR 自带这些 pass，只需在 pipeline 中添加：
```cpp
// 在 vortex-mark-kernel 之前
pm.addPass(bufferization::createOneShotBufferizePass());
pm.addPass(bufferization::createBufferResultsToOutParamsPass());
pm.addPass(createCanonicalizerPass());
```

**3. `export_via_torch_mlir.py`**（Python 脚本，约 50 行）

```python
import torch
from torch_mlir import fx
# 导出模型为 linalg-on-tensors MLIR
module = fx.export_and_import(model, *example_inputs,
                              output_type="linalg-on-tensors")
with open(output_path, "w") as f:
    f.write(module.operation.get_asm())
```

### Phase 2：端到端验证（1 周）

1. 用 torch-mlir 导出一个简单 matmul 模型
2. 通过新 pipeline → LLVM → simx 验证数值一致性
3. 逐步增加 op 覆盖：LayerNorm, Softmax, GeLU, Attention

### Phase 3：GPT-2 全模型替换（1-2 周）

1. 用 torch-mlir 导出完整 VortexGPT2 模型
2. 验证与当前手写 MLIR 输出一致
3. 替换 `gen_full_inference.py` 的代码生成逻辑

### Phase 4（可选）：torch-mlir 编译集成

如果外部调用模式验证通过，可以考虑把 torch-mlir 作为 submodule 编译进 vortex-compiler，实现单命令端到端编译。

## 5. 关键技术细节

### 5.1 torch-mlir 输出 vs vortex pipeline 输入

```
torch-mlir 输出:                    vortex pipeline 需要:
─────────────────                    ────────────────────
func.func @forward(                  func.func @forward(
    %arg0: tensor<64x256xf32>)           %arg0: memref<64x256xf32>,
    -> tensor<64x256xf32> {               %arg1: memref<64x256xf32>)
  %0 = linalg.matmul                     attributes {vortex.entry} {
    ins(... : tensor, tensor)           linalg.matmul
    outs(... : tensor)                    ins(... : memref, memref)
    -> tensor<64x256xf32>                 outs(... : memref)
  return %0 : tensor                    return
}                                    }
```

需要的转换：
1. `tensor` → `memref`（one-shot-bufferize）
2. 返回值 → 出参（buffer-results-to-out-params）
3. 添加 `vortex.entry`（normalize pass）

### 5.2 torch-mlir 安装

```bash
# 推荐 pip 安装预编译版（不需要编译 LLVM）
pip install torch-mlir -f https://github.com/llvm/torch-mlir/releases
```

或从源码构建（需要对齐 LLVM 18）：
```bash
git clone https://github.com/llvm/torch-mlir.git
cd torch-mlir && git checkout llvm-18
cmake -B build -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir
```

### 5.3 与现有 ONNX-MLIR 路线的关系

两条路线不冲突。它们共享同一个接入点：

```
torch-mlir ──→ linalg-on-tensors ──┐
                                    ├─→ bufferize → vortex pipeline
ONNX-MLIR  ──→ linalg-on-tensors ──┘
```

`vortex-normalize-torch-frontend` 和 `NormalizeONNXFrontendPass` 是并列的前端适配 pass。

## 6. 里程碑

| 里程碑 | 完成标准 | 预估 |
|--------|---------|------|
| M1: torch-mlir 环境 | pip install 成功，能导出 linalg MLIR | 1 天 |
| M2: matmul 端到端 | torch matmul → vx-opt → simx PASSED | 3 天 |
| M3: Transformer block | LayerNorm + Attention + MLP 通过 | 1 周 |
| M4: GPT-2 替换 | 替代 gen_full_inference.py，结果一致 | 1 周 |
