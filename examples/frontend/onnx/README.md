# ONNX Artifacts

这个目录保存第一阶段导出的 ONNX 产物。

当前约定：

1. 每个模型导出一个 `.onnx`
2. 每个模型导出一个同名 `.json`
3. `.json` 里记录输入 shape、opset 和阶段信息

这个目录中的文件应被视为：

1. 前端固定输入样例
2. 后续 ONNX checker / shape inference 输入
3. 后续 ONNX-MLIR 导入输入
