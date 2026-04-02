"""第一阶段 PyTorch toy 模型定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_shape: tuple[int, ...]
    build: Callable[[], "object"]


def _require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - 运行时环境相关
        raise SystemExit(
            "缺少 PyTorch 依赖。请先安装 `torch` 后再运行导出脚本。"
        ) from exc
    return torch, nn


def build_matmul_mlp():
    torch, nn = _require_torch()

    class MatmulMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 8)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.fc2(self.relu(self.fc1(x)))

    return MatmulMLP().eval()


def build_conv_relu_pool():
    torch, nn = _require_torch()

    class ConvReluPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.pool(self.relu(self.conv(x)))

    return ConvReluPool().eval()


def build_conv_conv_relu():
    torch, nn = _require_torch()

    class ConvConvRelu(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.relu(self.conv2(self.conv1(x)))

    return ConvConvRelu().eval()


MODEL_SPECS = {
    "matmul_mlp": ModelSpec(
        name="matmul_mlp",
        input_shape=(1, 16),
        build=build_matmul_mlp,
    ),
    "conv_relu_pool": ModelSpec(
        name="conv_relu_pool",
        input_shape=(1, 3, 8, 8),
        build=build_conv_relu_pool,
    ),
    "conv_conv_relu": ModelSpec(
        name="conv_conv_relu",
        input_shape=(1, 3, 8, 8),
        build=build_conv_conv_relu,
    ),
}
