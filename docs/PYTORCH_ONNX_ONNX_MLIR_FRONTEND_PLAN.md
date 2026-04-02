# PyTorch / ONNX / ONNX-MLIR 到 Pre-Vortex / Vortex 的前端接入计划

## 1. 文档目标

这份文档只回答一个问题：

```text
如果要让 Vortex 支持 CNN / GEMM 类模型，
前端应该如何规划，
接下来优先做哪些任务，
并且尽可能复用已有工具链。
```

这里的推荐路线是：

```text
PyTorch
  -> ONNX
  -> ONNX-MLIR
  -> pre-vortex 标准 MLIR
  -> 现有 vortex pass pipeline
  -> LLVM Dialect / LLVM IR
  -> 现有 Vortex LLVM 后端
```

这条路线的核心目标不是立刻做一个“完美前端”，而是：

1. 先打通一个对 CNN / GEMM 友好的模型输入入口
2. 先尽量复用 PyTorch、ONNX、ONNX-MLIR 和现有 `mlir-vortex` pass
3. 先避免重新发明一套模型导出、shape 推理、基础算子语义系统

---

## 2. 为什么当前优先选这条路线

## 2.1 为什么不是直接从普通 C / C++ / CIR 开始

普通 C / C++ 前端更自然地产生：

1. 循环
2. 指针
3. load / store
4. if / branch
5. kernel 级执行语义

但它不会天然给出：

1. `linalg.matmul`
2. `linalg.conv_*`
3. `tensor.extract_slice`
4. `memref.subview`
5. 高层张量算子图

这意味着如果从普通 C / C++ / CIR 起步，要额外做：

1. loop 模式识别
2. 高层算子恢复
3. 张量 / buffer 语义重建

对于当前目标来说，这条路成本太高。

## 2.2 为什么优先选 ONNX-MLIR，而不是先接 torch-mlir

当前更推荐：

1. 先把 `ONNX-MLIR` 当作主前端桥
2. 后续如果需要更深的 PyTorch 绑定，再考虑 `torch-mlir`

原因是：

1. `ONNX` 更适合作为模型交换和部署入口
2. `ONNX-MLIR` 已经提供了 ONNX Dialect、shape 推理、导入和后续 lowering 基础
3. 你当前的 Vortex 目标更像“后端/部署编译器”，不是“PyTorch 原生训练前端”
4. 先接 `ONNX-MLIR`，前端边界更稳定

一句话概括：

```text
先用 ONNX-MLIR 把“模型图 -> MLIR”这段复用起来，
再把精力集中在“如何接入 pre-vortex / vortex”。
```

---

## 3. 总体接入原则

## 3.1 第一阶段不要 fork ONNX-MLIR

第一阶段最合理的方式不是：

1. 立刻把 `ONNX-MLIR` 大量代码搬进 `mlir-vortex`
2. 立刻修改 ONNX-MLIR 主线 pass

而是：

1. 先把 `ONNX-MLIR` 当作外部前端工具
2. 让它输出中间 MLIR
3. 在 `mlir-vortex` 里新增桥接 pass，把这份 MLIR 接到 `pre-vortex`

这样做的好处是：

1. 复用现有导出和 shape 推理能力
2. 降低维护成本
3. 不会一开始就把两个工程深度耦合

## 3.2 接入点要尽量高，不要太低

不建议在以下位置接入：

1. LLVM Dialect
2. LLVM IR
3. ONNX-MLIR 已经很低层、很 target-like 的末端 IR

更合理的接入位置应该是：

1. 仍然保留张量/结构化语义的阶段
2. 仍然便于转成 `linalg` / `tensor` / `scf` / `memref` 的阶段

目标是得到：

```text
func + arith + scf + tensor/memref + linalg + 少量附加属性
```

也就是当前 `pre-vortex` 能够自然接住的形态。

## 3.3 先做静态 shape、推理态、算子子集

MVP 阶段建议收窄到：

1. inference only
2. 静态 shape
3. NCHW / NHWC 中先固定一种主布局
4. CNN / GEMM 常见算子子集

MVP 不先做：

1. 训练
2. 动态 shape 全覆盖
3. control-flow ONNX 全覆盖
4. 稀疏 / 量化 / 混合精度全覆盖

---

## 4. 当前应尽量复用的已有工具链

## 4.1 PyTorch 侧复用

直接复用：

1. `torch.onnx.export`
2. 或更新的 PyTorch 导出接口

职责：

1. 把模型导成 ONNX
2. 固化输入 shape
3. 固化 inference 图

不要在 `mlir-vortex` 里自己实现：

1. PyTorch 模型解析
2. PyTorch 到 IR 的直接导出

## 4.2 ONNX 侧复用

直接复用：

1. ONNX 模型格式
2. ONNX checker
3. ONNX shape inference

职责：

1. 做模型合法性检查
2. 让图在进入 ONNX-MLIR 前尽量 shape 明确

## 4.3 ONNX-MLIR 侧复用

直接复用：

1. ONNX Dialect
2. ONNX importer
3. 现有 canonicalization / inference / decomposition 体系
4. 现有“accelerator”扩展接口思想

第一阶段不建议自己重写：

1. ONNX importer
2. ONNX op 定义
3. ONNX 基础 shape helper

## 4.4 mlir-vortex 侧复用

当前已经有的后半段能力应尽量复用：

1. `vortex-mark-kernel`
2. `vortex-materialize-address-spaces`
3. `vortex-map-parallel-loops-to-launch`
4. `vortex-promote-tiles-to-local`
5. `vortex-insert-barriers`
6. `vortex-plan-local-memory-layout`
7. `vortex-lower-linalg-inside-kernel`
8. `vortex-lower-local-memory`
9. `vortex-legalize-for-llvm`
10. `vortex-lower-runtime-builtins`

也就是说：

```text
真正新增的工作重点，
应该放在 “ONNX-MLIR 输出 -> pre-vortex 输入” 这一小段。
```

---

## 5. 推荐的接入边界

## 5.1 建议的工程边界

推荐把前端工程边界定义为：

```text
PyTorch / ONNX / ONNX-MLIR 负责：
  1. 模型导出
  2. 模型校验
  3. ONNX 算子图表示
  4. 高层 shape / 类型推理

mlir-vortex 负责：
  1. 把 ONNX-MLIR 输出变成 pre-vortex
  2. 把 pre-vortex 映射成 vortex
  3. 把 vortex 接到 LLVM / Vortex backend
```

## 5.2 pre-vortex 新输入契约

为了对接 ONNX-MLIR，建议把 `pre-vortex` 输入契约扩成：

1. `func`
2. `arith`
3. `tensor`
4. `linalg`
5. `scf`
6. `memref`
7. 按需允许少量：
   `shape`、`affine`、`cf`

第一阶段目标不是让 ONNX-MLIR 直接吐出纯 `memref` 风格 IR，
而是允许它先经过：

1. tensor 语义阶段
2. 再 bufferize 到 memref
3. 再进入现有 Vortex pipeline

---

## 6. 第一批建议支持的模型与算子

## 6.1 模型优先级

建议按下面顺序推进：

### 第一批

1. 单层 `MatMul`
2. 两层 MLP
3. 小型 Conv + Relu + Pool

### 第二批

1. LeNet
2. 小型 ResNet block
3. 多层 CNN inference 子图

### 暂缓

1. Transformer 全模型
2. 动态 shape 模型
3. 训练图

## 6.2 算子优先级

MVP 最值得先支持的算子：

1. `MatMul`
2. `Gemm`
3. `Conv`
4. `Relu`
5. `Add`
6. `Mul`
7. `MaxPool`
8. `AveragePool`
9. `Reshape`
10. `Transpose`
11. `Flatten`

建议延期的算子：

1. `BatchNormalization`
   优先尝试在前面 fold 掉
2. `Softmax`
3. `LayerNormalization`
4. `Attention`
5. `DynamicSlice` / `NonZero` / `Where` 等动态图味道较重的算子

---

## 7. 建议的 lowering 方向

## 7.1 算子到 pre-vortex 的建议映射

建议先建立如下映射：

| ONNX 侧语义 | pre-vortex 目标形态 |
| --- | --- |
| `MatMul` / `Gemm` | `linalg.matmul` + 可选 bias add |
| `Conv` | `linalg.conv_*` |
| `Relu` | `linalg.map` 或 `linalg.generic` |
| `Add` / `Mul` | `linalg.generic` 或张量 elementwise |
| `MaxPool` / `AveragePool` | `linalg.pooling_*` |
| `Reshape` | `tensor.expand_shape` / `tensor.collapse_shape` / cast 类重解释 |
| `Transpose` | `linalg.transpose` 或通用 tensor/linalg 变换 |
| `Flatten` | reshape / collapse shape |

## 7.2 为什么这样映射

这样做的原因是：

1. 它能最大化复用 MLIR 标准生态
2. 它最容易接你当前已有的 `pre-vortex -> vortex` 路线
3. `linalg` / `tensor` 仍然保留了高层结构化优化空间

这比直接把 ONNX 算子硬塞成 `vortex.*` 更合理。

---

## 8. 接下来的前端开发任务清单

下面是推荐的开发顺序。

## 8.1 任务 1：固定输入输出样例

目标：

1. 固定 2 到 3 个 PyTorch 小模型
2. 导出 ONNX
3. 保存 ONNX-MLIR 各阶段 IR 快照

建议样例：

1. `matmul_mlp`
2. `conv_relu_pool`
3. `conv_conv_relu`

产出物：

1. `examples/frontend/*.py`
2. `examples/frontend/*.onnx`
3. `examples/frontend/*.mlir`

## 8.2 任务 2：确定 ONNX-MLIR 的最佳接入点

目标：

1. 看 ONNX-MLIR 哪个阶段最接近 `pre-vortex`
2. 决定我们到底是接：
   1. ONNX Dialect
   2. ONNX lowering 后的较高层 tensor/linalg 形式

建议结论方向：

1. 优先接“仍保留高层张量语义”的阶段
2. 不接最终 LLVM 前的低层阶段

## 8.3 任务 3：定义 `onnx-to-pre-vortex` 输入契约

目标：

1. 明确第一版允许哪些 dialect
2. 明确第一版禁止哪些动态特性
3. 明确 tensor 到 memref 的边界在哪里

建议第一版约束：

1. static shape only
2. inference only
3. 无 ONNX control-flow
4. 无动态 rank
5. 无稀疏和量化特例

## 8.4 任务 4：实现 `ONNX / Tensor -> PreVortex` 桥接 pass

建议拆成小 pass：

1. `vortex-normalize-onnx-frontend`
2. `vortex-lower-frontend-matmul-to-linalg`
3. `vortex-lower-frontend-conv-to-linalg`
4. `vortex-lower-frontend-elementwise-to-linalg`
5. `vortex-lower-frontend-pool-to-linalg`

第一阶段目标不是做全 ONNX lowering，而是吃一个受限子集。

## 8.5 任务 5：接入 bufferization

目标：

1. 把 tensor 语义变成 memref 语义
2. 让 IR 进入你当前已有的 `pre-vortex` 管线

这一步应优先复用标准 MLIR pass，而不是自己造 bufferization。

## 8.6 任务 6：复用现有 Vortex 后半段 pipeline

目标：

1. 把前端桥接后的 IR 直接接到现有 pass：
   1. `vortex-mark-kernel`
   2. `vortex-materialize-address-spaces`
   3. `vortex-map-parallel-loops-to-launch`
   4. `vortex-promote-tiles-to-local`
   5. 后续 LLVM lowering

这里要求：

1. 不要因为有了 ONNX 前端，就重写后半段 pass
2. 只修 pre-vortex 输入契约不匹配的地方

## 8.7 任务 7：建立端到端正确性回归

目标：

1. ONNX reference 输出
2. CPU 参考输出
3. Vortex 仿真输出
4. 板上输出

至少要建立：

1. `matmul` 数值一致性
2. `conv` 数值一致性
3. 小 CNN 子图一致性

---

## 9. 推荐的里程碑

## 里程碑 A：前端样例打通

完成标准：

1. `PyTorch -> ONNX` 稳定
2. `ONNX -> ONNX-MLIR` 能稳定拿到中间 MLIR
3. 有固定小样例和测试脚本

## 里程碑 B：首个高层算子接入 pre-vortex

完成标准：

1. `MatMul` 或 `Gemm` 能转成 `linalg.matmul`
2. 经过 bufferization 后能进入现有 `pre-vortex` pipeline

## 里程碑 C：首个 CNN 子图接入 Vortex

完成标准：

1. `Conv + Relu + Pool` 能进入 `pre-vortex`
2. 能继续 lower 到 vortex / LLVM / backend

## 里程碑 D：端到端正确性

完成标准：

1. 仿真结果和 CPU 参考一致
2. 至少一个小模型可以走完整链路

---

## 10. 当前最推荐的实际执行顺序

如果按“尽可能复用已有工具链”的原则，接下来最合理的顺序是：

1. 先准备 2 到 3 个 PyTorch 小模型并导出 ONNX
2. 跑 ONNX-MLIR，收集各阶段 IR 快照
3. 明确 `onnx-to-pre-vortex` 的最小输入契约
4. 先实现 `MatMul/Gemm -> linalg.matmul`
5. 再实现 `Conv -> linalg.conv_*`
6. 接上标准 bufferization
7. 再复用现有 `pre-vortex -> vortex -> LLVM` pipeline

一句话总结：

```text
前端阶段的重点不是重做模型编译器，
而是把 ONNX-MLIR 输出的高层语义
稳妥地接进你已经写好的 pre-vortex / vortex 后半段。
```
