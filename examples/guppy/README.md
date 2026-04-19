# Guppy

## 依赖

- `python3`
- `torch`
- `tokenizers`
- 本地 `guppylm` 仓库
- 已编译好的 `vx-opt`
- 可用的 `vortex-platform`
- 可访问的 `remote_vivado_service`

## 目录

- `download_assets.py`
- `dump_reference_logits.py`
- `guppy_to_vortex.py`
- `gen_full_inference.py`
- `chat_driver.py`
- `run_full_inference.sh`
- `run_board_chat.py`
- `fixed_prompt_messages.json`

## 阶段 A

```bash
python3 examples/guppy/download_assets.py \
  --out-dir build/guppy/assets \
  --guppylm-root ~/guppylm
```

```bash
python3 examples/guppy/dump_reference_logits.py \
  --assets-dir build/guppy/assets \
  --guppylm-root ~/guppylm \
  --messages-json examples/guppy/fixed_prompt_messages.json \
  --out-dir build/guppy/reference
```

## 阶段 B

```bash
python3 examples/guppy/guppy_to_vortex.py \
  --assets-dir build/guppy/assets \
  --guppylm-root ~/guppylm \
  --messages-json examples/guppy/fixed_prompt_messages.json \
  --out-dir build/guppy/export
```

## 阶段 C

```bash
python3 examples/guppy/gen_full_inference.py \
  --bundle-dir build/guppy/export \
  --out-dir build/guppy/full_inference_seq128_l6 \
  --sequence-length 128 \
  --layer-limit 6
```

## 本地编译

```bash
scripts/build-vortex-kernel.sh \
  --input build/guppy/full_inference_seq128_l6/full_inference.mlir \
  --output-dir build/guppy/full_inference_seq128_l6/out \
  --platform-root ../vortex-platform \
  --vx-opt build/bin/vx-opt \
  --extra-source build/guppy/full_inference_seq128_l6/full_inference_wrapper.c \
  --extra-source build/guppy/full_inference_seq128_l6/full_inference_weights.S \
  --pass-pipeline 'builtin.module(func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-lower-linalg-inside-kernel),canonicalize,cse,vortex-legalize-for-llvm,vortex-lower-runtime-builtins,canonicalize,cse,convert-scf-to-cf,convert-math-to-llvm,convert-math-to-libm,convert-arith-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},convert-cf-to-llvm,reconcile-unrealized-casts)'
```

## 一次性上板 chat

```bash
python3 examples/guppy/run_board_chat.py \
  --prompt-text "hi" \
  --max-new-tokens 4
```

默认行为是：

1. 第 1 个 token 走一次完整 `run-manifest`
2. 后续 token 默认复用 resident session，改走 `run-resident-manifest`

这样不会在每个 token 都重复上传整份 Guppy ELF。

如果你要对 resident 状态复用做小 kernel 调试，可以额外加：

```bash
--resident-reload-all-kernel-segments
```

这个开关会让 resident rerun 重新装载全部 kernel 段。它适合定位 resident 复用错误，不适合默认用于完整 Guppy 模型。

只准备 full stage 和 ELF，不上板：

```bash
python3 examples/guppy/run_board_chat.py \
  --prepare-only
```
