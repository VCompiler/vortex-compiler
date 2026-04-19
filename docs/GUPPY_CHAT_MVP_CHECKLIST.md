# Guppy 完整 Chat MVP 清单

## 0. 目标

这份清单的目标不是“先跑一个简化 Transformer forward”，而是：

```text
让用户输入一句话，
最终在 Vortex 路线上拿到 Guppy 的完整回复。
```

这里的“完整 chat MVP”定义为：

1. 用户输入文本
2. tokenizer 编码
3. 按 Guppy 的 chat prompt 格式拼接输入
4. Vortex 负责每一步 forward，输出 logits
5. 主机侧做 temperature / top-k / 采样
6. 遇到 EOS 或长度上限停止
7. tokenizer 解码，输出最终文本

第一阶段**不要求**：

1. 板上独立完成 decode loop
2. 板上独立完成随机采样
3. 支持 KV cache
4. 支持动态 shape
5. 支持 batch > 1

---

## 1. 当前结论

## 1.1 已有能力

当前 `vortex-compiler` 已经具备：

1. `linalg` / `scf` / `arith` / `memref` -> LLVM 的 MVP backend 通路
2. `math.exp` / `math.erf` / `math.sqrt` 的 lowering 通路
3. embedding / layernorm / softmax / attention / lm_head / full forward 的 GPT 风格样例
4. `examples/gpt2/pytorch_to_vortex.py` 这条“PyTorch 权重 -> MLIR -> simx”路线

也就是说：

```text
后端底座已经基本够用，
当前真正的缺口主要在 Guppy 专用前端导出和模型结构适配。
```

## 1.2 当前不能直接跑 Guppy 的原因

当前不能直接把 `~/guppylm` 跑起来，原因是：

1. 现有 `examples/gpt2` 导出器是单头 attention，不是 Guppy 的 6 头 attention
2. 现有 full inference 路线没有 causal mask
3. 现有简化 GPT 路线默认无 bias，而 Guppy 使用带 bias 的 `nn.Linear`
4. Guppy 使用 tied embedding / lm_head 权重复用，当前路线默认把输出投影单独放一份
5. 本地 `guppylm` 目录当前只有量化 ONNX 和 tokenizer，没有 PyTorch checkpoint
6. 当前 chat 只具备“单次 full forward”，还没有主机侧 decode loop
7. 当前大权重仍偏向“静态数组 + wrapper 拷贝”风格，不适合 Guppy 的 9M 级模型

---

## 2. 总体策略

推荐采用：

```text
PyTorch checkpoint
  -> Guppy 专用导出器
  -> MLIR full forward kernel
  -> Vortex 编译
  -> 主机侧 decode loop
  -> 完整 chat MVP
```

明确不建议第一步采用：

1. 量化 ONNX -> ONNX-MLIR -> pre-vortex
2. 板上独立采样和 decode loop
3. 一开始就做 KV cache

原因：

1. 当前 ONNX 前端只覆盖窄 matmul 路线，不适合直接承接 Guppy 全图
2. 当前最短路径是复用已经跑通的 `examples/gpt2` PyTorch 直出 MLIR 思路
3. 主机侧 decode loop 可以先把“完整 chat”打通，再逐步把控制逻辑下沉

---

## 3. 分阶段清单

## 3.1 阶段 A：补齐 Guppy 真实模型资产

目标：

```text
拿到 Guppy 的真实 PyTorch 权重和配置，
不再依赖本地量化 ONNX。
```

清单：

- [x] 下载 `pytorch_model.bin`
- [x] 下载 `config.json`
- [x] 下载 `tokenizer.json`
- [x] 记录下载方式和固定路径
- [x] 明确 `~/guppylm` 本地运行所依赖的最小资产集合

建议落点：

1. `~/guppylm/checkpoints/pytorch_model.bin`
2. `~/guppylm/checkpoints/config.json`
3. `~/guppylm/data/tokenizer.json` 或统一复制到导出目录

验收标准：

1. 本地 Python 可以成功加载 Guppy 模型
2. 能输出固定 prompt 的 reference logits

---

## 3.2 阶段 B：建立 Guppy 专用导出入口

目标：

```text
在 vortex-compiler 中新增一条 Guppy 专用导出链，
不要继续把 Guppy 强塞进当前简化 GPT2 生成器。
```

当前文件：

1. `examples/guppy/guppy_to_vortex.py`
2. `examples/guppy/README.md`

清单：

- [x] 新建 `examples/guppy/`
- [x] 实现 `guppy_to_vortex.py`
- [x] 支持从 `pytorch_model.bin + config.json` 读取真实参数
- [x] 导出 `manifest.json`
- [x] 导出 `model_config.json`
- [x] 导出 `prompt.json`
- [x] 导出 `reference_logits.json`
- [x] 导出 `reference_logits.npy`
- [x] 导出 `reference_last_token_logits.npy`
- [x] 导出 `weights_index.json`
- [x] 导出 `weights/**/*.npy`
- [x] 输出 `seq / dim / heads / ff / vocab / layers` metadata
- [x] 固定 `tok_emb.weight <-> lm_head.weight` 复用关系
- [x] 固定阶段 C 需要补齐的语义清单
- [ ] 生成 `full_inference.mlir`
- [ ] 生成 `full_inference_wrapper.c`
- [ ] 生成 Guppy 专用权重载入产物

验收标准：

1. 不依赖随机初始化
2. 导出的权重与 PyTorch 原模型一致
3. 生成器能复现同一输入下的 golden logits
4. 当前明确停在 bundle 导出，不假装已经生成 MLIR / wrapper

---

## 3.3 阶段 C：适配 Guppy 的真实模型结构

目标：

```text
让导出器表达的 forward 与 Guppy 原始 PyTorch 模型语义一致。
```

Guppy 当前需要的结构点：

1. multi-head attention
2. fused qkv projection
3. causal mask
4. linear bias
5. ReLU FFN
6. tied embedding / lm_head

清单：

- [ ] 支持 `qkv = linear(x)` 的 fused projection
- [ ] 支持 `reshape + transpose` 形成多头 `Q/K/V`
- [ ] 支持 `Q @ K^T / sqrt(head_dim)`
- [ ] 支持下三角 causal mask
- [ ] 支持 masked softmax
- [ ] 支持 `prob @ V`
- [ ] 支持 head merge
- [ ] 支持 output projection 的 bias
- [ ] 支持 FFN 的 `Linear + ReLU + Linear`
- [ ] 支持 final `LayerNorm + lm_head`
- [ ] 支持 `lm_head.weight` 与 `tok_emb.weight` 复用

明确说明：

1. ReLU 比当前 GeLU 路线更简单，不是难点
2. 真正新增的核心复杂度是“多头 + causal mask + bias + tied weights”

验收标准：

1. 单层 block 对齐 PyTorch
2. 全模型 forward 对齐 PyTorch

---

## 3.4 阶段 D：确定 Vortex 侧 MVP 运行边界

目标：

```text
在工程上固定第一版 Guppy Chat 的约束，
避免范围失控。
```

建议固定如下：

1. `batch = 1`
2. `seq <= 128`
3. 静态 shape
4. inference only
5. dropout 关闭
6. 无 KV cache
7. 每轮 decode 都重跑完整前缀 forward

清单：

- [ ] 在文档中固定 MVP 参数边界
- [ ] 在导出器中强制检查 shape / config
- [ ] 在 host chat 入口中限制最大 prompt 长度
- [ ] 在 decode loop 中限制最大生成 token 数

验收标准：

1. 第一版不需要讨论动态 shape
2. 第一版不需要讨论训练态算子

---

## 3.5 阶段 E：建立主机侧完整 Chat 驱动

目标：

```text
先用主机驱动 decode loop，
实现用户可见的完整 chat。
```

主机侧负责：

1. tokenizer 编码
2. prompt 模板拼接
3. 调用 Vortex forward
4. 从最后一个位置取 logits
5. `temperature / top-k / 采样`
6. EOS 检查
7. tokenizer 解码

清单：

- [ ] 复用 Guppy tokenizer
- [ ] 复用 Guppy prompt 格式
- [ ] 写 host 侧 `chat_driver.py` 或等价脚本
- [ ] 支持单轮 prompt
- [ ] 支持多轮 messages -> prompt 拼接
- [ ] 支持 `temperature`
- [ ] 支持 `top_k`
- [ ] 支持 EOS 停止
- [ ] 支持 `max_new_tokens`

建议不要第一版做：

1. `multinomial` 下沉到板上
2. tokenizer 下沉到板上
3. 多轮状态缓存到板上

验收标准：

1. 用户输入一句话
2. 最终返回完整回复文本
3. 输出与本地 PyTorch chat 在可接受误差范围内一致

---

## 3.6 阶段 F：权重与内存组织改造

目标：

```text
避免 Guppy 权重沿用“小样例时代”的堆栈/静态头文件组织方式。
```

当前要重点解决：

1. 权重体积 30MB+ 级别
2. wrapper 中不应再有大规模局部数组拷贝
3. tied weights 应避免重复存储

清单：

- [ ] 盘点当前权重、activation、scratch 的内存占用
- [ ] 去掉 wrapper 中不必要的 stack allocation
- [ ] 让大权重直接驻留在只读段或独立权重段
- [ ] 让 `tok_emb` / `lm_head` 共享同一份存储
- [ ] 设计 activation buffer 复用策略
- [ ] 设计 scratch buffer 复用策略

第二阶段再考虑：

- [ ] manifest 外部权重加载
- [ ] 权重与代码 ELF 分离

验收标准：

1. 能生成并编译 Guppy 规模的产物
2. 不因为 wrapper 栈爆或重复拷贝导致失败

---

## 3.7 阶段 G：数值验证

目标：

```text
逐层对齐，避免直接调 chat 时才发现误差来源。
```

建议验证顺序：

1. embedding
2. single-head reference 拆分验证
3. multi-head attention
4. causal mask
5. block
6. final lm_head
7. full forward
8. decode 逐步对齐

清单：

- [ ] 对齐 embedding 输出
- [ ] 对齐 attention score
- [ ] 对齐 masked softmax 输出
- [ ] 对齐 attention 输出
- [ ] 对齐 block 输出
- [ ] 对齐 full logits
- [ ] 对齐每一步 next-token logits
- [ ] 对齐 top-k/argmax 结果

验收标准：

1. full forward 与 PyTorch 的最大误差在可控范围内
2. decode 前若干步的 token 选择一致

---

## 3.8 阶段 H：simx / 上板完整 chat 冒烟

目标：

```text
让 Guppy 的完整 chat 在 simx 先跑通，再推进到板上。
```

清单：

- [ ] simx 固定 prompt 冒烟
- [ ] simx 多个 prompt 冒烟
- [ ] 板上固定 prompt 冒烟
- [ ] 板上记录运行时间
- [ ] 板上记录加载时间
- [ ] 板上记录内存占用

验收标准：

1. simx 能输出完整一轮回复
2. 板上能输出完整一轮回复
3. 输出文本基本符合 PyTorch Guppy 参考行为

---

## 3.9 当前状态与下一步

当前已经确认的状态：

- [x] Guppy 真实 PyTorch 资产已接入
- [x] Guppy bundle 导出已打通
- [x] multi-head attention / causal mask / bias / tied embedding 已表达进 full forward 生成器
- [x] `chat_driver.py` 已支持 host-side decode loop
- [x] 6 层完整模型已在板上回出 next token
- [x] 已新增 `examples/guppy/run_board_chat.py`，可一条命令自动准备 full stage、编译并上板
- [x] 已在本地实现 resident 执行模式：
  第 1 步走 `run-manifest`，后续步走 `run-resident-manifest`

当前默认 full 运行方式：

```bash
python3 examples/guppy/run_board_chat.py \
  --prompt-text "hi" \
  --max-new-tokens 4
```

当前最需要继续改进的点：

- [ ] resident 路线还需要在 Windows 侧 service 同步并完成板测验证
- [ ] 需要确认 resident 模式下多 token 输出是否稳定
- [ ] 当前 decode 仍是 host-side loop，板上尚未做自循环
- [ ] 当前没有 KV cache，生成长度上来后每步都是完整前缀重算

---

## 4. 当前最关键的 blocker

当前最关键的 blocker 已经不是前端和算子覆盖，而是执行模式：

- [x] 已补 `run-resident-manifest` 接口，允许复用首个 full run 的 resident session
- [x] `chat_driver.py` 已默认切到“首步 full run，后续 resident run”
- [ ] 仍需在真实 Windows service + 板卡环境上完成 resident 路线回归

---

## 5. 第一阶段推荐执行顺序

建议严格按下面顺序推进：

1. 先补真实 Guppy 模型资产
2. 再新建 `examples/guppy/guppy_to_vortex.py`
3. 先只打通 Guppy full forward
4. 再补多头 attention
5. 再补 causal mask
6. 再补 bias 和 tied weights
7. 再写主机侧 chat driver
8. 最后做 simx / 板上完整 chat 冒烟

---

## 6. 第一阶段交付物

第一阶段结束时，仓库里至少应有：

- [x] `docs/GUPPY_CHAT_MVP_CHECKLIST.md`
- [x] `examples/guppy/guppy_to_vortex.py`
- [x] `examples/guppy/run_board_chat.py`
- [x] `examples/guppy/README.md`
- [x] 一组可复现的 Guppy full forward 产物
- [x] 一条 host-side decode loop 路线
- [ ] 一组固定 prompt 的 simx / 板上完整 chat 冒烟结果

---

## 7. 第二阶段再做的内容

下面这些不属于第一阶段 MVP：

- [x] resident session / `run-resident-manifest` 执行模式
- [ ] resident 路线板测验证与耗时记录
- [ ] Windows 侧 service 驻留启动稳定化
- [ ] KV cache
- [ ] 板上 decode loop
- [ ] 板上随机采样
- [ ] 多核并行
- [ ] 权重 manifest 分段加载
- [ ] 更大模型配置
- [ ] 动态 shape

---

## 8. 一句话版本

```text
先把 Guppy 当成“真实 PyTorch 模型 + 主机侧 decode loop + Vortex 负责 forward”的项目来做，
不要第一步就追求全板上生成。
```
