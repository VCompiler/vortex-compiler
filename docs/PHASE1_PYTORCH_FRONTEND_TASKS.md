# PyTorch 前端接入第一阶段任务

## 1. 阶段目标

第一阶段只做一件事：

```text
把 PyTorch 侧的输入、导出、样例和产物边界固定下来，
为后续 ONNX / ONNX-MLIR / pre-vortex 接入建立稳定入口。
```

这一阶段**不追求**：

1. 直接从 PyTorch 一步 lower 到 `vortex`
2. 一次性支持完整 CNN 模型族
3. 一次性支持动态 shape
4. 一次性实现 ONNX 全算子覆盖

这一阶段的成功标准是：

1. 仓库里有固定的 PyTorch toy 模型
2. 仓库里有固定的 ONNX 导出入口
3. 仓库里有固定的 ONNX 与中间 MLIR 产物目录
4. 后续 `ONNX-MLIR -> pre-vortex` 桥接有明确输入契约

---

## 2. 阶段边界

第一阶段推荐边界如下：

```text
PyTorch
  -> ONNX
  -> 保存 ONNX 模型
  -> 预留 ONNX-MLIR 中间产物目录
```

也就是说：

1. 先把 `PyTorch -> ONNX` 做稳定
2. 先把样例模型和固定输入 shape 固定下来
3. 先不在这一阶段实现 `ONNX -> pre-vortex` 桥接 pass

---

## 3. 第一阶段要完成的任务

## 3.1 任务 A：固定 toy 模型集合

第一阶段只保留三个样例：

1. `matmul_mlp`
2. `conv_relu_pool`
3. `conv_conv_relu`

选择原因：

1. `matmul_mlp` 对应 GEMM 主链
2. `conv_relu_pool` 对应最小 CNN 子图
3. `conv_conv_relu` 对应多层卷积组合

第一阶段不要扩到：

1. ResNet
2. Transformer
3. 动态 shape 网络

## 3.2 任务 B：固定导出脚本入口

仓库内固定一个入口脚本：

```text
examples/frontend/pytorch/export_models.py
```

这个脚本负责：

1. 生成 toy 模型
2. 固定输入 shape
3. 导出 ONNX
4. 写出简单元数据

第一阶段不要在仓库里实现：

1. PyTorch 训练逻辑
2. 数据集加载
3. 大模型下载

## 3.3 任务 C：固定产物目录

第一阶段统一使用：

1. `examples/frontend/pytorch/`
2. `examples/frontend/onnx/`
3. `examples/frontend/mlir/`

各目录职责：

1. `pytorch/`：模型定义、导出脚本、使用说明
2. `onnx/`：导出的 `.onnx` 与元数据
3. `mlir/`：后续 ONNX-MLIR 各阶段快照

## 3.4 任务 D：固定输入契约

第一阶段固定如下约束：

1. inference only
2. static shape only
3. batch size 先固定
4. 不支持训练态算子
5. 不支持 ONNX control-flow

## 3.5 任务 E：为 ONNX-MLIR 留出接入位置

第一阶段虽然不实现桥接 pass，但要提前固定后续工作边界：

1. `ONNX-MLIR` 输出中间 MLIR 快照
2. 后续把这些快照转成 `pre-vortex`
3. 再接现有 Vortex pass pipeline

---

## 4. 第一阶段交付物

第一阶段结束时，仓库内应至少具备：

1. 一个阶段任务文档
2. 一个 PyTorch toy 模型定义文件
3. 一个 ONNX 导出脚本
4. 一个前端目录 README
5. 预留的 ONNX / MLIR 产物目录

---

## 5. 第一阶段后的下一步

第一阶段完成后，再进入第二阶段：

```text
ONNX
  -> ONNX-MLIR
  -> 选定中间 IR 接入点
  -> onnx/tensor -> pre-vortex 桥接
```

第二阶段最值得优先做的是：

1. `MatMul/Gemm -> linalg.matmul`
2. `Conv -> linalg.conv_*`
3. `Relu/Add/Mul -> linalg.generic`

---

## 6. 当前推荐执行顺序

建议严格按下面顺序做：

1. 先补 PyTorch toy 模型与导出脚本
2. 再实际导出 ONNX 样例
3. 再跑 ONNX checker / shape inference
4. 再引入 ONNX-MLIR 做中间 IR 快照
5. 最后再开始 `onnx-to-pre-vortex` pass 设计

第一阶段的重点不是 lowering，而是：

```text
把前端输入稳定下来。
```
