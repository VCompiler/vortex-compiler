# Pre-Vortex / Vortex 分层边界与 Pass 输入契约

## 1. 文档目标

这份文档专门回答下面几个问题：

1. `vortex` dialect 是否应该直接接 `linalg.fill`、`linalg.matmul` 这类高层语义
2. `pre-vortex` 与 `vortex` 的职责边界应该怎么划
3. 从哪个 pass 开始，不再允许把高层 `linalg/tensor` 继续往后带
4. 当前 `fill + matmul` 路线为什么“部分跑通”，但还没有自动接上完整的
   `launch/local/barrier` 执行模型

本文档给出的结论是当前仓库后续实现的推荐边界，不要求一步到位纯化成
“只剩一种 dialect”的形态，但要求每一层语义职责清楚。

## 2. 结论先行

### 2.1 核心结论

`vortex` dialect 不应直接承接 `linalg.fill`、`linalg.matmul`、`linalg.generic`
这类高层计算语义。

更合理的分层是：

```text
前端 / 高层张量语义
  -> pre-vortex
     以标准 MLIR 为主，保留 linalg/tensor/scf/memref
  -> vortex
     只承接硬件执行、地址空间、local memory、同步语义
  -> LLVM dialect
  -> LLVM IR
  -> Vortex LLVM backend
```

### 2.2 一句话概括每层职责

1. `pre-vortex` 负责“看懂算法和调度”
2. `vortex` 负责“表达硬件执行和存储语义”
3. LLVM 负责“表达后端真正可消费的低层形式”

### 2.3 对 `fill` / `matmul` 的直接结论

1. `linalg.fill` 不值得直接进入 `vortex`
2. `linalg.matmul` 可以在 `pre-vortex` 被专门识别、tile、promotion、mapping
   规划，但不建议作为 `vortex.matmul` 长期保留
3. 真正应该进入 `vortex` 的，不是 `fill/matmul` 本身，而是它们调度后形成的：
   `launch`、执行 id、`local_alloc`、`barrier`、带地址空间的访存边界

## 3. 为什么不让 Vortex 直接接 Linalg

### 3.1 `linalg` 描述的是算法结构，不是硬件结构

`linalg.matmul` 表达的是矩阵乘这一类结构化计算。

但后端真正需要稳定承接的是：

1. 哪个循环维映射到 `core`
2. 哪个循环维映射到 `subgroup`
3. 哪个循环维映射到 `thread`
4. 哪块 buffer 被提升到 `local`
5. 哪里需要 `barrier`
6. 哪些 buffer 落在哪个地址空间

这些都不是 `linalg` 本身的语义，而是调度和硬件绑定后的语义。

### 3.2 直接把高层算子塞进目标 dialect，会把前后端绑死

如果 `vortex` 直接以 `linalg.matmul` 为核心输入：

1. 你会把后端和某一种前端表达强耦合
2. 以后来自 ONNX、PyTorch、Polygeist、手写 MLIR 的输入都要尽量长成同一种
   `linalg` 形态
3. 一旦上层不再是 `linalg.matmul`，而是 `linalg.generic`、`scf` loops、或更低层
   的 buffer 语义，后端边界就会变得不稳定

### 3.3 复用标准 MLIR 的收益更大

`linalg/tensor/vector` 这几层已经有大量现成能力：

1. tile
2. fusion
3. promotion planning
4. bufferization
5. canonicalize / CSE
6. loop lowering

因此更合理的方式是：

1. 在 `pre-vortex` 尽量复用这些能力
2. 等到必须引入目标相关语义时，再进入 `vortex`

### 3.4 否则会产生“名字像目标相关，实质仍是高层语义”的伪目标 op

例如：

1. `vortex.matmul`
2. `vortex.fill`
3. `vortex.subview`

如果这些 op 只是把原来标准 op 改个名字，并没有额外硬件含义，
那只会让后续 lowering 更混乱，不会更清晰。

## 4. 推荐分层

### 4.1 高层输入层

典型 dialect：

1. `onnx`
2. `torch`
3. `tensor`
4. `linalg`
5. `vector`

职责：

1. 表达模型和算子语义
2. 保留高层结构化优化机会

### 4.2 `pre-vortex` 层

典型 dialect：

1. `func`
2. `arith`
3. `scf`
4. `memref`
5. `linalg`
6. `tensor`
7. `vector`
8. 按需保留 `affine`

职责：

1. 把前端输入规范化到标准 MLIR 形态
2. 做 tile / loop normalization / bufferization
3. 建立进入 Vortex 前所需的显式结构：
   `scf.for`、`memref.subview`、地址空间标记、mapping 标记
4. 尽量不提前注入硬件细节

### 4.3 `vortex` 层

这一层应只承接硬件语义：

1. `vortex.kernel`
2. `vortex.launch`
3. `vortex.core_id`
4. `vortex.subgroup_id`
5. `vortex.thread_id`
6. `vortex.local_alloc`
7. `vortex.barrier`
8. `vortex.fence`
9. `#vortex.address_space<global/local/private>`
10. `#vortex.scope<subgroup/core>`

这一层允许和标准 dialect 混合存在，但主体应已经是：

```text
func + arith + scf + memref + vortex
```

`linalg` 在这一层最多只是过渡性残留，不应该成为长期核心输入契约。

### 4.4 LLVM 层

职责：

1. 地址空间整数化
2. local memory 的显式地址化
3. runtime builtin / intrinsic 对接
4. `scf/cf/arith/memref` 到 LLVM dialect 的标准转换

## 5. 哪些语义应该停在 Pre-Vortex

下面这些内容，原则上应在进入 `vortex` 前就基本处理完，或者至少不再作为
`vortex` 的核心输入契约：

| 语义 | 典型 op | 处理位置 |
| --- | --- | --- |
| 结构化计算 | `linalg.fill`, `linalg.matmul`, `linalg.generic` | `pre-vortex` |
| 张量值语义 | `tensor.*` | `pre-vortex` |
| 向量高层语义 | `vector.contract`, `vector.transfer_*` | `pre-vortex` 或后续专门路线 |
| 纯算法级循环变换 | `tile`, `interchange`, `fusion` | `pre-vortex` |
| 普通切片视图 | `memref.subview` | `pre-vortex` 建模阶段 |

说明：

1. `memref.subview` 可以在 `pre-vortex` 作为 tile 视图继续存在
2. 但一旦它被解释为“要提升到 local 的协作 tile”，就不再只是普通视图，
   后续应进入 `vortex.local_alloc + copy + barrier` 路线

## 6. 哪些语义必须进入 Vortex

下面这些语义如果仍停留在标准 dialect 里，就无法稳定表达 Vortex 硬件语义：

| 语义 | 推荐表示 |
| --- | --- |
| kernel 入口及 ABI 边界 | `vortex.kernel` |
| 显式执行区域 | `vortex.launch` |
| core 级执行 id | `vortex.core_id` |
| subgroup 级执行 id | `vortex.subgroup_id` |
| thread 级执行 id | `vortex.thread_id` |
| per-core local memory | `vortex.local_alloc` |
| local / global / private 地址空间 | `#vortex.address_space<...>` |
| 协作式同步边界 | `vortex.barrier` |
| 显式顺序 / 可见性边界 | `vortex.fence` |

判断标准很简单：

1. 如果只是计算语义，尽量留在标准 dialect
2. 如果已经是硬件执行、存储、同步边界，就进入 `vortex`

## 7. `fill` / `matmul` 的推荐路线

### 7.1 `linalg.fill`

推荐路线：

```text
linalg.fill
  -> 视情况保留在 pre-vortex
  -> 或 lower 为 scf + memref.store
  -> 不直接变成 vortex.fill
```

原因：

1. `fill` 没有 Vortex 特有硬件语义
2. 它通常只是初始化行为
3. 更适合被标准 lowering、融合、或普通 loop/store 吃掉

### 7.2 `linalg.matmul`

推荐路线：

```text
linalg.matmul
  -> 在 pre-vortex 做 tile / subview / loop materialization
  -> 给并行 loop 加 mapping 标记
  -> 给可 promotion 的 tile 加 local promotion 标记
  -> 进入 vortex.launch / vortex.local_alloc / vortex.barrier
  -> 最终再把剩余计算体 lower 成 scf + memref + arith
```

关键点：

1. `matmul` 在 `pre-vortex` 可以被当作“重要模式”专门识别
2. 但它不是 `vortex` 的稳定核心算子
3. 真正进入 `vortex` 的，是围绕这个模式形成的执行与内存结构

## 8. Pass 输入输出契约

这一节给出当前推荐的 pass 边界。

### 8.1 允许消费高层 `linalg/tensor` 的 pass

下面这些 pass 可以继续看到高层语义：

1. `vortex-normalize-onnx-frontend`
2. `vortex-tile-matmul-for-pre-vortex`
3. `vortex-validate-pre-vortex`
4. `vortex-summarize-pre-vortex`
5. 各类标准 `linalg` / `tensor` / `bufferization` pass

它们的职责是：

1. 看懂高层算子
2. 把它们转成更适合 Vortex 的标准 MLIR 结构

### 8.2 进入 `vortex` 前必须显式化的结构

在进入 `vortex-map-parallel-loops-to-launch` 之前，建议至少显式化出：

1. tiled `scf.for` nest
2. 显式 `memref.subview`
3. kernel 边界标记
4. 地址空间标记
5. 显式 loop mapping 标记
6. local promotion 标记

也就是说，进入这一步之前，重点已经不再是“识别 `linalg.matmul`”，
而是“给出可以硬件映射的 loops / views / buffers 结构”。

### 8.3 可以仍然容忍残留 `linalg` 的 Vortex 前半段 pass

下面这些 pass 可以在输入中容忍残留 `linalg`，但它们自己不应该再依赖
“高层算子名字”来工作：

1. `vortex-mark-kernel`
2. `vortex-materialize-address-spaces`
3. `vortex-map-parallel-loops-to-launch`
4. `vortex-promote-tiles-to-local`
5. `vortex-insert-barriers`
6. `vortex-plan-local-memory-layout`

这些 pass 消费的应是：

1. `func.func`
2. `scf.for`
3. `memref.subview`
4. `memref.load/store/copy`
5. 各类属性和地址空间标记

而不是直接消费：

1. `linalg.fill`
2. `linalg.matmul`
3. `linalg.generic`

### 8.4 最后一个允许消费 `linalg` 的 pass

当前推荐边界是：

`vortex-lower-linalg-inside-kernel`

它是最后一个允许消费 residual `linalg` 的 pass。

这一步之后，kernel 内主体应当收敛为：

```text
func + arith + scf + memref + vortex
```

MVP 阶段建议直接要求：

1. `vortex-legalize-for-llvm` 不再接受 residual `linalg`
2. `vortex-lower-runtime-builtins` 不再接受 residual `linalg`
3. `convert-scf-to-cf`、`finalize-memref-to-llvm` 前，也不应再看到 `linalg`

### 8.5 当前推荐的边界线

最推荐的规则可以直接写成一句话：

```text
进入 vortex 前，允许有 linalg；
离开 vortex 的高层阶段后，不再允许把 linalg 当作后端输入契约。
```

如果一定要给出一个明确切分点，那么切分点就是：

```text
vortex-lower-linalg-inside-kernel
```

这之后，后面的 pipeline 应面向：

1. `scf`
2. `memref`
3. `arith`
4. `cf`
5. `llvm`
6. `vortex` runtime builtin 边界

## 9. 当前 `fill + matmul` 路线为什么只是“部分跑通”

当前仓库里，`fill/matmul` 的顺序 MVP 路线已经可以跑通到：

```text
pre-vortex
  -> LLVM dialect
  -> LLVM IR
  -> ELF
  -> simx
```

这说明：

1. `linalg.fill`
2. `linalg.matmul`

本身并不是“完全没有 lowering 方案”。

但当前还没有自动接上完整的：

```text
pre-vortex
  -> vortex.launch
  -> vortex.local_alloc
  -> vortex.barrier
  -> LLVM
```

原因不是 `fill/matmul` 本身不能 lower，而是当前输入通常还缺：

1. tiled `scf.for`
2. `vortex.mapping = core/subgroup/thread`
3. 可 promotion 的 `memref.subview`
4. `vortex.promote_to_local`
5. 明确的 write-back 边界

也就是说，缺的是“进入 Vortex 执行模型所需的结构”，
不是“高层算子本身没有解”。

## 10. 当前实现建议

按当前仓库状态，后续实现应遵循下面的顺序：

1. 继续把高层前端收敛到标准 `pre-vortex`
2. 补强 `matmul` 的 tiling + mapping annotation + local promotion planning
3. 让 `vortex-map-parallel-loops-to-launch` 和
   `vortex-promote-tiles-to-local` 接住这些显式结构
4. 用 `vortex-lower-linalg-inside-kernel` 把 residual `linalg` 收口
5. 后续 LLVM 侧只面向 `scf/memref/arith/vortex`

换句话说，后续工作的重点不应是“继续为 `vortex` 发明更多高层计算 op”，
而应是：

1. 把 `pre-vortex` 调度结构显式化
2. 把 `vortex` 的硬件语义边界收紧
3. 把进入 LLVM 的后半段输入契约固定下来

## 11. 最终推荐规则

最后把整件事压缩成 4 条规则：

1. `vortex` 不直接承接 `linalg` 这类高层计算语义
2. `pre-vortex` 负责把高层算子转成可映射的 `scf/memref` 结构
3. `vortex` 只承接执行映射、地址空间、local memory、同步语义
4. 从 `vortex-lower-linalg-inside-kernel` 之后，后端输入契约收敛为
   `func + arith + scf + memref + vortex`
