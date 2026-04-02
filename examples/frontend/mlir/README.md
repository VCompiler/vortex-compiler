# MLIR Snapshots

这个目录现在不再只是占位，已经开始承接 `ONNX -> pre-vortex` 的中间产物。

当前已落地的最小链路是：

1. `ONNX`
2. `onnx-mlir --EmitONNXIR`
3. `onnx-mlir-opt`
   产出 `linalg + memref` 的 bufferized matmul
4. `vx-opt`
   通过 `vortex-normalize-onnx-frontend`
   和 `vortex-tile-matmul-for-pre-vortex`
   进入当前仓库已有的 tiled pre-vortex 形态

## 当前脚本

仓库内提供了一个最小脚本：

```bash
examples/frontend/mlir/lower_onnx_matmul_to_pre_vortex.sh
```

它负责把一个静态 shape 的 matmul/Gemm 类 ONNX 模型，降到 pre-vortex：

```bash
examples/frontend/mlir/lower_onnx_matmul_to_pre_vortex.sh \
  --input examples/frontend/onnx/matmul_mlp.onnx \
  --output examples/frontend/mlir/matmul_mlp.pre_vortex.mlir \
  --onnx-mlir /path/to/onnx-mlir \
  --onnx-mlir-opt /path/to/onnx-mlir-opt \
  --vx-opt /path/to/vx-opt \
  --tile-size 8
```

## 当前保存建议

建议至少保留下面几类快照：

1. ONNX Dialect 快照
2. `linalg` 快照
3. bufferization 后快照
4. 最终 `pre-vortex` 快照

当前这条脚本默认只保留最终 `pre-vortex` 产物；
如果要调试前端问题，建议把中间文件也单独落盘保存。
