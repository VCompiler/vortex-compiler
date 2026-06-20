# Guppy

## 依赖

- `python3`
- `torch`
- `tokenizers`
- 本地 `guppylm` 仓库
- 已编译好的 `vx-opt`
- 可用的 `vortex-platform`
- 可访问的 `remote_vivado_service`，或本地 `xc7k480t` JTAG/XDMA 板卡路径

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

## 本地 PCIe/XDMA 上板

如果已经在 `xc7k480t` 上加载 repo-native Vortex PCIe/XDMA bitstream，并且
`xdma` 驱动已绑定到 `10ee:7024`，可以直接走本地 XDMA runner：

```bash
python3 examples/guppy/run_board_chat.py \
  --runner-mode local-xdma \
  --bundle-dir build/guppy/export \
  --stage-dir build/guppy/full_inference_seq16_l2 \
  --sequence-length 16 \
  --layer-limit 2 \
  --prompt-text "hi guppy" \
  --max-new-tokens 1 \
  --timeout-sec 300
```

当前 `local-xdma` 行为：

1. 每个 token 走一次完整 manifest rerun，不复用 resident session。
2. 默认设备节点是 `/dev/xdma0_h2c_0`、`/dev/xdma0_c2h_0` 和 DMA-control 基地址 `0x0`。
3. 在 2026-04-25 的 Vortex LSU fence-drain RTL 修复后，reduced `sequence_length=16, layer_limit=1` monolithic 路径已验证 `MANIFEST_RESULT=PASS`，1-token 输出 `generated_ids: [43]` / `generated_text: 'i'`，2-token 输出 `generated_ids: [43, 653]` / `generated_text: 'i better'`。
4. 在 2026-04-26 的 Vortex cache flush bank0 wait 修复后，reduced `sequence_length=16, layer_limit=2` 默认 split 路径已验证 `MANIFEST_RESULT=PASS`，1-token 输出 `generated_ids: [149]` / `generated_text: 'the'`，2-token 输出 `generated_ids: [149, 133]` / `generated_text: 'the water'`。
5. 默认 `--pcie-split-workaround` 会把 reduced chat 拆成 attention-merge 和 post-attn 两个 kernel，用于绕过长 kernel 的 PCIe/cache 可见性调试风险；layer2 已在这个路径上验证通过。
6. 更长 sequence、更大 layer count、resident 复用、以及完整 generated MLIR correctness 仍需要单独扩展验证。

如果要复现 2-token layer2 PCIe split 结果：

```bash
python3 examples/guppy/run_board_chat.py \
  --runner-mode local-xdma \
  --bundle-dir build/guppy/export \
  --stage-dir build/guppy/full_inference_seq16_l2 \
  --sequence-length 16 \
  --layer-limit 2 \
  --prompt-text "hi guppy" \
  --max-new-tokens 2 \
  --timeout-sec 300
```

如果要显式验证 monolithic PCIe 路径，不使用 split workaround：

```bash
python3 examples/guppy/run_board_chat.py \
  --runner-mode local-xdma \
  --bundle-dir build/guppy/export \
  --stage-dir build/guppy/full_inference_seq16_l1 \
  --sequence-length 16 \
  --layer-limit 1 \
  --prompt-text "hi guppy" \
  --max-new-tokens 1 \
  --timeout-sec 300 \
  --no-pcie-split-workaround
```

可以用 timing 汇总脚本查看 XDMA runner 各阶段耗时：

```bash
python3 examples/guppy/summarize_xdma_timing.py \
  build/guppy/full_inference_seq16_l2/xdma_runs/step_00/split_stage0_attn_merge \
  build/guppy/full_inference_seq16_l2/xdma_runs/step_00/split_stage1_post_attn
```

当前 layer2 split stage0 优化后大约是：

```text
total_ms ~= 40301
prepare_stage_ms ~= 187
load_manifest_ms ~= 29
dump_compare_outputs_ms ~= 50
wait_kernel_ms ~= 40029
```

也就是说，host 侧 staging/load/readback 已经不是主瓶颈；下一阶段性能优化应优先针对 Vortex kernel 执行时间。

只准备 full stage 和 ELF，不上板：

```bash
python3 examples/guppy/run_board_chat.py \
  --prepare-only
```
