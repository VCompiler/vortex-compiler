# Frontend Examples

这个目录用于放置前端接入相关样例和中间产物。

当前第一阶段只固定三类内容：

1. `pytorch/`
   PyTorch toy 模型与 ONNX 导出脚本
2. `onnx/`
   导出的 ONNX 模型与元数据
3. `mlir/`
   后续 ONNX-MLIR 中间 IR 快照

当前目标不是把前端完整做完，而是先固定：

1. 输入模型
2. 导出方式
3. 产物位置
4. 后续桥接边界

当前已经补上的第二阶段最小能力是：

1. `onnx-mlir` 产出 ONNX Dialect MLIR
2. `onnx-mlir-opt` 产出 bufferized `linalg.matmul`
3. `vx-opt` 通过前端桥接 pass 把它整理成 tiled `pre-vortex`

对应脚本见：

```text
examples/frontend/mlir/lower_onnx_matmul_to_pre_vortex.sh
```
