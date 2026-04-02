# PyTorch Frontend Scaffold

这个目录保存第一阶段的 PyTorch 侧骨架：

1. toy 模型定义
2. ONNX 导出脚本
3. 导出约束说明

第一阶段只支持：

1. `matmul_mlp`
2. `conv_relu_pool`
3. `conv_conv_relu`

统一要求：

1. inference only
2. static shape only
3. 固定输入 shape

示例导出命令：

```bash
python3 examples/frontend/pytorch/export_models.py --model all
```

如果本机缺少 `torch` 或 `onnx`，脚本会直接报错并提示先安装依赖。
