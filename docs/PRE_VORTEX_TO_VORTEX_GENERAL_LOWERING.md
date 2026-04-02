# Pre-Vortex 到 Vortex 的通用 Lowering 设计

## 1. 文档目标

这份文档不是围绕某一个 `matmul` 样例来“逐条翻译 IR”，而是要回答一个更通用的问题：

```text
上层 pre-vortex 标准 MLIR
  在什么条件下
  以什么规则
  分几步
  lower 到当前的 Vortex dialect
```

这里的目标不是把所有标准方言都机械替换成 `vortex.*`，而是建立一套可扩展、可复用、可落地到更多 kernel 的 lowering 规则。

当前讨论对象包括但不限于：

1. tiled matmul
2. cooperative tiled copy
3. elementwise / map
4. reduction
5. stencil / 卷积类 tile kernel
6. 后续可能出现的 vectorized kernel

---

## 2. 核心设计原则

### 2.1 只把“硬件语义”下沉到 Vortex

`vortex` 方言应该承载的是：

1. 执行层级
2. 地址空间
3. local memory
4. barrier / fence
5. 显式谓词 / mask
6. 真正存在的硬件专用原语

而不是承载所有算法和调度细节。

### 2.2 只在“语义闭合”时替换标准 op

如果一个标准 op 仍然只是表达：

1. 结构化循环
2. 普通 tile 视图
3. 标量算术
4. 通用张量/向量计算

那么它应继续保留标准形式。

### 2.3 Lowering 结果应是“混合 IR”，而不是“纯 Vortex IR”

在较长一段时间内，合理的目标形态应该是：

```text
func + arith + scf + memref + vector + vortex
```

或者：

```text
func + arith + scf + memref + linalg + vortex
```

也就是说：

- `vortex` 用来接住目标语义
- 标准 MLIR 继续承载通用控制流与计算语义

---

## 3. 当前 Vortex dialect 能表达什么

当前已经实现的 Vortex 方言最小集合包括：

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
11. `vortex.yield`

这意味着当前 Vortex dialect 可以稳定表达：

1. 一个已经映射好的执行区域
2. core/subgroup/thread 三层执行 id
3. per-core local memory
4. 带作用域的同步语义
5. Vortex-visible memory space

但它**还不能完整表达**：

1. 多维 grid/block 级 launch 语义
2. kernel 外层 tile 网格的全局实例 id
3. 专用矩阵原语
4. 显式 predication / mask
5. local memory 的 bank/layout 细节

因此当前 lowering 的通用目标不应是假设“所有并行层级都已经可以完全用 `vortex.*` 表达”。

---

## 4. 通用 lowering 的目标分层

建议把 pre-vortex 到 Vortex 的 lowering 看成 4 个抽象层：

### 层 A：规范化 pre-vortex

输入仍然主要是标准方言：

- `func`
- `arith`
- `scf`
- `memref`
- `tensor`
- `linalg`
- `vector`
- `affine`

这一层做：

1. canonicalize
2. CSE
3. bufferization 前准备
4. loop normalization
5. tile / fusion / shape simplification

### 层 B：映射分析层

这层先**决定**而不是直接改写：

1. 哪些循环维映射到 `core`
2. 哪些循环维映射到 `subgroup`
3. 哪些循环维映射到 `thread`
4. 哪些 tile 要 promotion 到 `local`
5. 哪些地方需要 `barrier` / `fence`
6. 哪些值应视为 `uniform`

这一层的输出可以是：

1. analysis 结果
2. 临时 annotation
3. mapping attribute

### 层 C：混合 Vortex IR

这一层开始把**真正的目标相关语义** materialize 出来：

1. `vortex.kernel`
2. `vortex.launch`
3. `vortex.core_id / subgroup_id / thread_id`
4. `vortex.local_alloc`
5. `vortex.barrier / fence`
6. `#vortex.address_space<...>`

同时继续保留：

1. `arith`
2. `scf`
3. `memref`
4. `vector`
5. 必要时保留 `linalg`

### 层 D：Vortex-aware low-level IR

这时再继续做：

1. 把剩余 `linalg` lower 掉
2. 把 `vector` 映射成更低层表达
3. 把访存与控制流压到更接近 LLVM 的形式
4. 为 `VortexToLLVM` 做准备

---

## 5. 通用 lowering 决策框架

面对一个 pre-vortex op，应该按下面的问题来判断：

### 问题 1：它是在表达硬件语义吗？

- 如果是：考虑 lower 到 `vortex.*`
- 如果不是：继续保留标准 MLIR

### 问题 2：它是否已经完成了硬件映射分析？

- 如果没有：先不要替换
- 如果已经确定 execution/memory/sync 语义：可以 materialize 成 `vortex.*`

### 问题 3：当前 Vortex dialect 是否真的有更稳定的对应抽象？

- 如果没有：不要为了“目标相关”而造不稳定的新 op
- 应优先保持标准 op + 附加属性/地址空间/外围 launch 结构

### 问题 4：替换后是否更接近 LLVM backend，而不是更远？

- 如果更接近：可以下沉
- 如果只是换个名字、并不能帮助后续 lowering：不要替换

---

## 6. 各类 op 的通用 lowering 策略

## 6.1 `func` 方言

| pre-vortex op | 通用 lowering 策略 | 是否进入 vortex |
| --- | --- | --- |
| `func.func` | 如果是 kernel 入口，则保留 `func.func`，并加 `vortex.kernel` attr；参数/结果补地址空间 | 部分进入 |
| `func.return` | 保留 | 否 |
| `func.call` | 当前阶段通常保留；只有 runtime ABI 稳定后才特殊处理 | 否 |

### 推荐规则

1. 不造 `vortex.func`
2. kernel 入口继续使用 `func.func`
3. `vortex.kernel` 只作为目标相关补充信息

---

## 6.2 `arith` 方言

`arith` 原则上应尽量保留。

| pre-vortex op | 通用 lowering 策略 |
| --- | --- |
| `arith.constant` | 保留 |
| `arith.addi/muli/addf/mulf` | 保留 |
| `arith.cmpi/cmpf/select` | 保留；后续若要做 predication，再转 mask/predicate |
| `arith.index_cast` | 保留 |

### 原因

这些 op 表达的是通用 SSA 算术，不是 Vortex 特有语义。

---

## 6.3 `scf` 方言

`scf` 是 pre-vortex 到 Vortex 之间最需要“选择性下沉”的部分。

| pre-vortex op | 通用 lowering 策略 | 是否直接替换 |
| --- | --- | --- |
| `scf.for` | 若仍是算法循环/遍历循环，保留；若已经明确表示一个映射到执行层级的区域，则重组为 `vortex.launch + id query` | 选择性 |
| `scf.if` | 保留；若未来需要显式谓词控制，再改造成 predication 形式 | 通常不替换 |
| `scf.while` | 保留 | 否 |
| `scf.parallel` | 可先 lower 成 `scf.for` 或作为 mapping 候选输入 | 选择性 |
| `scf.reduce` | 通常先保留或 lower 到普通循环 | 通常不替换 |

### 关键原则

#### 不是所有 `scf.for` 都该变成 `vortex.launch`

只有当一个循环已经不再代表“算法流程”，而是代表“某个 tile/region 交给某个 Vortex 执行层级来并行完成”时，才应该被 materialize 成 `vortex.launch`。

#### 当前 dialect 下，外层 grid 级循环可以继续保留

因为当前 dialect 只有：

1. `core_id`
2. `subgroup_id`
3. `thread_id`

还没有更强的 grid/block id 语义，所以很多 kernel 外层 tile 网格在第一版里仍应保留 `scf.for`。

---

## 6.4 `memref` 方言

`memref` 是另一个不能机械替换的核心区域。

| pre-vortex op | 通用 lowering 策略 | 是否进入 vortex |
| --- | --- | --- |
| `memref.subview` | 大多数情况下保留；promotion 时再重写 | 选择性 |
| `memref.load/store` | 通常保留；只通过 base memref 的 memory space 获得目标语义 | 通常不替换 |
| `memref.alloc` / `alloca` | 如果确定分配的是 Vortex per-core local memory，则改成 `vortex.local_alloc` | 选择性 |
| `memref.copy` | 若是 tile promotion 或 cooperative copy，可展开为显式 load/store + barrier | 选择性 |
| `memref.cast` / `reinterpret_cast` | 先尽量消解；必要时保留 | 否 |

### `memref.subview` 的通用判断规则

#### 保留为标准 subview 的场景

1. 普通切片
2. 仅用于索引推导
3. 仍处于 global buffer 视图层
4. 还未决定是否 promotion

#### 重写的场景

1. tile 已确定进入 local memory
2. 需要 cooperative load/store
3. 需要把 buffer 放入 `#vortex.address_space<local>`
4. 需要显式同步

### 典型转换形式

```text
memref.subview(global tile)
  -> vortex.local_alloc(local tile)
  -> cooperative copy global -> local
  -> vortex.barrier <core>
```

---

## 6.5 `tensor` 与 bufferization

### 原则

`tensor` 不应直接 lower 到 `vortex.*`。

更合理流程是：

```text
tensor/linalg on tensors
  -> bufferization
  -> memref/linalg on buffers
  -> 再考虑 Vortex lowering
```

也就是说：

1. `tensor.empty`
2. `tensor.extract_slice`
3. `tensor.insert_slice`
4. `tensor.extract`
5. `tensor.insert`

这些都应在进入 Vortex-specific lowering 前尽量完成 bufferization。

---

## 6.6 `linalg` 方言

`linalg` 通常不是直接替换对象，而是分析与逐步 lower 的中间层。

| pre-vortex op | 通用 lowering 策略 | 备注 |
| --- | --- | --- |
| `linalg.fill` | 可保留；也可 lower 为循环填充 local/global buffer | 通常不直接变 `vortex.fill` |
| `linalg.generic` | 保留作结构化计算入口；后续 lower 为 loops/vector | 最重要的通用入口 |
| `linalg.matmul` | 先 tile / bufferize / vectorize；未来如有硬件原语，再匹配成 `vortex.mma` | 当前不直接替换 |
| `linalg.reduce` | 先保留，后 lower 为 loops/vector/reduction tree | 当前不直接替换 |
| `linalg.map` | 保留到 vector/loop lowering | 当前不直接替换 |

### 关键原则

#### 不应一开始就设计 `vortex.matmul`

`linalg.matmul` 更合理的通用路径是：

```text
linalg.matmul
  -> tile / fuse / bufferize
  -> vector.contract 或 scf loops
  -> 如果硬件确有矩阵原语，再识别成 vortex.mma
```

#### `linalg.generic` 是通用 lowering 的重点输入

因为很多上层算子最终都会落到 `linalg.generic`，所以真正重要的是：

1. 如何识别并行维 / reduction 维
2. 如何把迭代空间映射到 core/subgroup/thread
3. 如何把输入/输出 tile promotion 到 local/private

而不是去为每个高层 op 单独造一个 `vortex.*` 对应物。

---

## 6.7 `vector` 方言

`vector` 是一个非常重要但应谨慎下沉的层。

| pre-vortex op | 通用 lowering 策略 |
| --- | --- |
| `vector.transfer_read/write` | 保留；或 lower 为局部 load/store 展开 |
| `vector.contract` | 若未来有硬件矩阵原语可识别，则作为 `vortex.mma` 候选；否则继续 lower |
| `vector.broadcast/extract/insert` | 通常保留或 lower 为标量/pack-unpack |
| `vector.multi_reduction` | 继续 lower 为 reduction tree / loops |

### 原则

如果 Vortex backend 暂时没有稳定的向量专用原语，那么 `vector` 应作为：

1. 中间优化层
2. 通往更低层 `arith + memref + scf` 的中转层

而不应被强行映射到新的 `vortex.vector_*` 系列算子。

---

## 6.8 `affine` / `cf` / `math`

### `affine`

- 优先用于分析和规范化
- 在 Vortex-specific lowering 前，多数情况可先 lower 到 `scf`

### `cf`

- 通常在更低层控制流阶段出现
- 不是 pre-vortex 到 Vortex 的主要设计重心

### `math`

- 若只是普通数学函数，先保留
- 如果未来确实要映射到 Vortex/LLVM intrinsic，再单独处理

---

## 7. 通用 lowering 模式

## 7.1 Kernel 边界 materialization

### 输入

```mlir
func.func @kernel(%a: memref<...>, %b: memref<...>)
```

### 输出

```mlir
func.func @kernel(
  %a: memref<..., #vortex.address_space<global>>,
  %b: memref<..., #vortex.address_space<global>>)
  attributes {vortex.kernel}
```

### 作用

1. 标记 kernel 入口
2. 把参数 buffer 明确落到 global address space

---

## 7.2 执行映射 materialization

### 输入

一个已经通过 analysis 确定“由某个 Vortex 执行组协作完成”的 tile 计算区域。

### 输出

```mlir
vortex.launch %cores, %subgroups, %threads {
  %core = vortex.core_id : index
  %sg = vortex.subgroup_id : index
  %th = vortex.thread_id : index
  ...
}
```

### 作用

1. 显式标记 execution region
2. 在 SSA 中获得执行 id
3. 为 cooperative memory movement 与同步提供边界

---

## 7.3 Local memory promotion

### 输入

```text
global tile subview + repeated reuse
```

### 输出

```mlir
%tile = vortex.local_alloc() : memref<..., #vortex.address_space<local>>
... cooperative copy ...
vortex.barrier <core>
```

### 作用

1. 显式 local scratchpad 分配
2. 把“隐式缓存复用”转成“显式软件可控近端存储”
3. 为后续 LLVM address space lowering 建立稳定语义

---

## 7.4 Cooperative copy

当前 dialect 还没有 `vortex.copy`。

因此推荐的通用表达是：

1. `vortex.launch`
2. `vortex.subgroup_id / thread_id`
3. `scf.for` 或算术索引计算
4. `memref.load/store`
5. `vortex.barrier <core>`

也就是说，copy 本身仍然由标准 MLIR 访存 op 表达；Vortex 只负责：

- 谁在拷
- 拷到哪里
- 什么时候同步

---

## 7.5 计算体 lowering

### 元素级 / map 类 kernel

```text
linalg.generic (pure elementwise)
  -> 保留在 launch 内
  -> 或 lower 为 scf + arith
```

### reduction 类 kernel

```text
linalg.reduce / scf.reduce
  -> 先保留 reduction 结构
  -> 决定是否要在 subgroup/core 内树形规约
  -> 当前阶段可先 lower 为 loops + arith
```

### matmul / contraction 类 kernel

```text
linalg.matmul / vector.contract
  -> tile + promotion + launch mapping
  -> 保留 structured compute
  -> 或进一步 lower 为 loops + load/store + add/mul
  -> 未来再评估是否匹配 vortex.mma
```

---

## 7.6 控制流与 predication

当前 dialect 暂时没有 `vortex.mask` / `vortex.pred`。

所以当前通用策略是：

1. `scf.if` 先保留
2. 尾块处理先保留标准 control flow
3. 谓词化仅作为未来扩展方向

未来若要支持：

- subgroup 内部分 lane 生效
- masked load/store
- 尾块 predication

再考虑加入：

1. `vortex.pred`
2. `vortex.mask`
3. `vortex.select_active`
4. `vortex.masked_load/store`

---

## 8. 不同 kernel 类型的通用 lowering 路线

## 8.1 Tiled matmul

推荐路线：

```text
linalg.matmul
  -> tile/fuse/bufferize
  -> outer tile loops 保留 scf
  -> inner cooperative tile region -> vortex.launch
  -> A/B/C tile promotion -> vortex.local_alloc
  -> cooperative copy + vortex.barrier
  -> compute 保留 linalg.matmul 或 lower 为 loops
```

## 8.2 Cooperative tiled copy

推荐路线：

```text
memref.subview + memref.copy
  -> decide launch mapping
  -> vortex.launch
  -> vortex.local_alloc / global subview
  -> memref.load/store cooperative copy
  -> vortex.barrier <core>
```

## 8.3 Elementwise map

推荐路线：

```text
linalg.map / linalg.generic
  -> decide parallel dimensions
  -> vortex.launch
  -> per-lane indexing by subgroup_id/thread_id
  -> arith + memref load/store
```

## 8.4 Reduction

推荐路线：

```text
linalg.reduce / scf.reduce
  -> 保留 reduction structure
  -> decide subgroup/core reduction strategy
  -> 当前阶段 lower 为 loops + arith
  -> 未来再引入 shuffle/reduction intrinsic
```

---

## 9. 建议的 pass pipeline

## 阶段 0：Pre-Vortex 规范化

1. canonicalize
2. CSE
3. affine simplification
4. shape simplification
5. tensor -> memref bufferization（如需要）
6. linalg tiling/fusion

## 阶段 1：Mapping Analysis

1. 识别 kernel 入口
2. 识别 tile 结构
3. 识别并行维 / reduction 维
4. 选择 execution mapping
5. 选择 local promotion 候选
6. 推导 barrier/fence 插入点

## 阶段 2：Materialize Vortex Boundary

1. 给 kernel 打 `vortex.kernel`
2. 参数/结果 memref 加 `#vortex.address_space<global>`
3. 在选定区域插入 `vortex.launch`
4. 把 execution ids materialize 成 `core_id/subgroup_id/thread_id`

## 阶段 3：Materialize Memory & Sync

1. 将选中的 tile promotion 成 `vortex.local_alloc`
2. 展开 cooperative copy
3. 插入 `vortex.barrier <core>` / `<subgroup>`
4. 必要时插入 `vortex.fence`

## 阶段 4：Lower Structured Compute

1. 保留 `linalg` 一段时间，或
2. lower `linalg` -> `scf + memref + arith`
3. 处理 reduction / vector contract

## 阶段 5：Vortex To LLVM

1. `vortex.kernel` -> LLVM ABI / function metadata
2. `vortex.launch` -> lower 为控制结构 / builtin 环境
3. `core_id/subgroup_id/thread_id` -> LLVM builtin / intrinsic
4. `#vortex.address_space<...>` -> LLVM address space
5. `vortex.local_alloc` -> local address space object
6. `vortex.barrier/fence` -> LLVM call / intrinsic

---

## 10. 当前阶段推荐的“稳定中间目标 IR”

对于大多数 kernel，当前阶段最现实的目标形态是：

```text
func + arith + scf + memref + vortex
```

对于还未拆开的结构化计算，也可以暂时接受：

```text
func + arith + scf + memref + linalg + vortex
```

这比“全变成 vortex.*”更合理，因为：

1. 不会过早冻结高层语义
2. 更便于复用 MLIR 现有 pass
3. 更利于逐步验证 lowering 正确性
4. 更贴合当前 Vortex dialect 的实际能力边界

---

## 11. 当前 dialect 下的现实约束

### 11.1 外层 grid 映射不要过早绑定

当前没有完整 grid/block id 抽象，因此很多 kernel 的外层 tile 网格：

- 仍然应保留为 `scf.for`
- 或先以 annotation 形式表示 mapping 决策

### 11.2 先不要设计一大批 `vortex.<algo-op>`

在没有确定硬件原语前，不建议先造：

1. `vortex.matmul`
2. `vortex.subview`
3. `vortex.copy`
4. `vortex.fill`
5. `vortex.reduce`

因为它们很容易只是“标准 MLIR 的重命名版本”。

### 11.3 当前最值得先实现的是“语义型 pass”

优先级建议是：

1. kernel/address-space materialization
2. launch mapping pass
3. local promotion pass
4. barrier/fence insertion pass
5. linalg-in-launch lowering pass

---

## 12. 建议的近期实现顺序

### 第一阶段

1. `MarkVortexKernelPass`
2. `MaterializeVortexAddressSpacesPass`
3. `MapParallelLoopsToVortexLaunchPass`

### 第二阶段

1. `PromoteTilesToVortexLocalPass`
2. `InsertVortexBarriersPass`
3. `ExpandCooperativeCopyPass`

### 第三阶段

1. `LowerLinalgInsideVortexPass`
2. `LegalizeVortexForLLVMPass`
3. `VortexToLLVMConversionPass`

---

## 13. 一句话总结

**pre-vortex 到 Vortex 的 lowering，不应理解成“把某个样例里的 op 一个个替换成 vortex.*”，而应理解成：把标准 MLIR 中已经确定的执行、内存、同步硬件语义显式 materialize 出来，同时尽量保留标准控制流与计算 op。**

这才是更通用、也更适合后续接 LLVM backend 的路线。
