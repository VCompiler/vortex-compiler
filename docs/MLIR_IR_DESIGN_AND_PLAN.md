# Vortex MLIR IR 设计与开发计划

## 1. 文档目标

本文档用于明确当前 Vortex MLIR 路线里的 IR 分层边界：

```text
pre-vortex 标准 MLIR
  -> 按需引入 Vortex 方言
  -> LLVM Dialect
  -> LLVM IR
  -> 现有 Vortex LLVM 后端
```

核心原则只有三条：

1. 不能把所有 `linalg` / `scf` / `memref` 算子机械地一一替换成 `vortex.*`
2. 只要标准 MLIR 还能清楚表达算法和调度，就继续保留标准算子
3. 只有当 IR 需要承载 Vortex 硬件语义时，才引入 `vortex.*`

本文档当前基于以下内容整理：

1. `/home/cx/VORTEX_MLIR_GPGPU后端开发计划.md`
2. `/home/cx/mlir-vortex/examples/pre_vortex_tiled_matmul.mlir`
3. `/home/cx/mlir-vortex/examples/pre_vortex_tiled_matmul.annotated.mlir`

## 2. IR 分层边界

### 2.1 pre-vortex 层

`pre-vortex` 是高层、弱目标绑定的 IR 层。它主要使用：

1. `func`
2. `arith`
3. `scf`
4. `memref`
5. `tensor`
6. `linalg`
7. `vector`
8. 按需补充：
   `affine`、`cf`、`math`、`bufferization`

这一层的职责是：

1. 表达 kernel 结构、循环嵌套、tiling 和数据流
2. 在真正绑定到 Vortex 硬件前，尽量保留优化空间
3. 让 IR 更容易调试、检查和测试

### 2.2 Vortex 方言层

`vortex` 是目标相关层。它应当承载：

1. 执行层级语义
2. kernel launch 语义
3. 同步语义
4. 近端 local memory 与地址空间语义
5. mask / predicate 语义
6. 硬件真正具备的专用计算原语

这一层的职责是：

1. 把高层调度绑定到 Vortex 执行模型
2. 显式表达硬件可见的内存和同步行为
3. 最终可靠地 lower 到 LLVM Dialect，再接现有 LLVM 后端

## 3. pre-vortex 阶段会遇到的标准算子

下面这些算子是 `pre-vortex` 阶段预期会出现的主体。大部分都不应该直接变成 `vortex.*` 的一一对应替代。

| Dialect | 代表算子 | 含义 | 默认处理策略 |
| --- | --- | --- | --- |
| `func` | `func.func`, `func.call`, `func.return` | 函数与 kernel 边界 | 保留标准形式 |
| `arith` | `arith.constant`, `arith.addi`, `arith.muli`, `arith.addf`, `arith.mulf`, `arith.cmpi`, `arith.cmpf`, `arith.select`, `arith.index_cast` | 标量与索引运算 | 保留标准形式 |
| `scf` | `scf.for`, `scf.if`, `scf.while`, `scf.parallel`, `scf.reduce`, `scf.yield` | 结构化控制流 | 在映射到硬件前保持标准形式 |
| `memref` | `memref.load`, `memref.store`, `memref.subview`, `memref.alloc`, `memref.alloca`, `memref.copy`, `memref.cast`, `memref.reinterpret_cast` | buffer 视图和访存 | 只有涉及特殊地址空间时才转目标相关形式 |
| `tensor` | `tensor.extract`, `tensor.insert`, `tensor.extract_slice`, `tensor.insert_slice`, `tensor.empty` | 张量值语义 | bufferization 前保持标准形式 |
| `linalg` | `linalg.generic`, `linalg.fill`, `linalg.matmul`, `linalg.batch_matmul`, `linalg.map`, `linalg.reduce` | 结构化计算 | 先保留并走标准 lowering |
| `vector` | `vector.transfer_read`, `vector.transfer_write`, `vector.contract`, `vector.broadcast`, `vector.multi_reduction`, `vector.extract`, `vector.insert` | 向量化计算和数据搬运 | 除非存在真实硬件原语，否则继续保留标准形式 |
| `affine` | `affine.for`, `affine.if`, `affine.apply` | 可分析的循环和索引 | 保留或 lower 到 `scf` |
| `cf` | `cf.br`, `cf.cond_br` | 更低层控制流 | 作为标准 lowering 中间阶段 |
| `math` | `math.exp`, `math.rsqrt`, `math.fma` | 数学库风格算子 | 除非需要目标 intrinsic，否则保持标准形式 |

## 4. 当前样例里已经出现的 pre-vortex 算子

从当前 tiled matmul 样例统计结果来看，已经出现的核心方言和算子是：

1. 方言：
   `arith`、`linalg`、`memref`、`scf`
2. 算子：
   `arith.addf`、`arith.mulf`、`linalg.fill`、`linalg.matmul`、
   `memref.subview`、`scf.for`

这说明当前边界是合理的：

1. IR 还主要是在表达算法与分块调度
2. 还没有提前混入 Vortex 硬件执行语义
3. 这是决定哪些语义必须进入 `vortex` 的正确位置

## 5. Vortex 方言里应该具备的算子与语义

只有当标准方言已经不足以表达 Vortex 硬件行为时，才引入 `vortex.*`

### 5.1 第一批建议纳入的执行模型算子

这一层的设计应优先绑定“稳定的架构语义”，而不是当前板级实现细节。

当前来看，`core` 与 `VX_local_mem`、`local` 地址空间、`core` 级同步边界天然相关，
因此 `vortex.core_id` 保留是合理的。

当前硬件真实层级是：

1. `core`
2. `warp/subgroup`
3. `thread`

因此第一版如果只有 `core_id` 和 `thread_id`，语义并不闭合：

1. 无法完整表达当前 `core -> subgroup -> thread` 的并行层级
2. 会让 `thread_id` 混淆成“subgroup 内 id”还是“core 内 flatten 后 id”

因此这里采用更清晰的三层执行 id 设计：

1. `vortex.core_id`
2. `vortex.subgroup_id`
3. `vortex.thread_id`

其中：

1. `vortex.subgroup_id` 用来表达当前 `warp-like` 执行组
2. `vortex.thread_id` 明确表示 `subgroup` 内的 thread id
3. 第一版仍然不直接使用 `vortex.warp_id`，优先使用更中性的 `subgroup_id`

| 建议算子/属性 | 作用 | 典型来源 | 后续 lowering 方向 |
| --- | --- | --- | --- |
| `vortex.kernel` 属性 | 标记 kernel 入口及 ABI 信息 | 顶层 kernel 函数 | LLVM 函数属性 / ABI 物化 |
| `vortex.launch` | 显式表示 launch / 网格映射区域 | 外层并行 tile 循环 | lower 为控制结构加 builtin id |
| `vortex.core_id` | 当前 core id | 外层循环维度映射 | lower 为 intrinsic / builtin |
| `vortex.subgroup_id` | 当前 core 内的 subgroup id | 中层并行区域映射 | lower 为 intrinsic / builtin |
| `vortex.thread_id` | 当前 subgroup 内的 thread id | 线程级工作映射 | lower 为 intrinsic / builtin |
| `vortex.uniform` | 表示 execution group 内统一值 | 分析结果 | lower 为普通 SSA 或 metadata |

说明：

1. 第一版加入 `vortex.subgroup_id`
2. 第一版仍然不把 `vortex.warp_id` 放入核心算子集合
3. 如果后续确实需要 CUDA 风格的 warp-specific 命名，可以再评估是否将
   `subgroup` 语义进一步细化或别名化为 `warp`

### 5.2 第一批建议纳入的内存与同步算子

当前 Vortex 硬件里，软件可控的近端存储主要对应 `VX_local_mem`。
它是 `per-core local memory`，不是更大范围共享的 scratchpad，也不应与
`icache` / `dcache` / `L2` 这类硬件自动管理缓存混淆。

因此第一批 `vortex.*` 内存与同步语义应优先表达：

1. `global / local / private` 地址空间区分
2. `local` 对应的显式分配
3. 带作用域的同步语义

| 建议算子/属性 | 作用 | 典型来源 | 后续 lowering 方向 |
| --- | --- | --- | --- |
| `vortex.local_alloc` | 分配 per-core 近端 local memory / scratchpad | tile promotion、显式局部缓冲 | lower 为目标 `local` address space 中的对象 |
| `vortex.barrier` | 执行组同步 | 协作式 load/store、tile consume 前后 | intrinsic / builtin |
| `vortex.barrier` 的 `scope` 属性 | 区分 `subgroup` / `core` 级同步 | mapping 结果 | lower 为不同同步原语或约束 |
| `#vortex.address_space<global>` | 全局可见存储 | 原始输入/输出 buffer | 转换为 LLVM global address space |
| `#vortex.address_space<local>` | per-core local memory | local tile / scratchpad | 转换为 LLVM local address space |
| `#vortex.address_space<private>` | 私有寄存器/线程私有存储 | 临时值、private buffer | 转换为 LLVM private address space |
| `vortex.fence` | 显式可见性/顺序边界 | memory ordering 边界 | intrinsic / fence |

第一版不建议直接使用 `vortex.workgroup_alloc` 这个名字，避免它暗示一个
超过 `per-core` 语义范围的共享层级。除非后续硬件抽象明确把
`workgroup == core` 固定下来，否则优先使用 `vortex.local_alloc`。

### 5.3 可选加入的硬件专用计算原语

这些算子只有在硬件和 LLVM 后端都确实有直接承载方式时才应该加入。

| 建议算子 | 加入条件 | 典型来源 |
| --- | --- | --- |
| `vortex.mma` | 硬件有矩阵乘原语 | `linalg.matmul` 或 `vector.contract` 模式识别后 |
| `vortex.dot` | 硬件有 dot-product 原语 | 向量 reduction |
| `vortex.shuffle` | 硬件支持 lane 交换 | lane 级数据重排 |
| `vortex.tmc` | 硬件暴露显式线程掩码控制 | control-to-mask lowering |
| `vortex.wspawn` | 硬件暴露 subgroup / warp spawn 机制 | 多 subgroup / warp 启动 |
| `vortex.mask` / `vortex.pred` | 硬件需要显式谓词控制 | 尾块处理 / masked memory op |

## 6. 哪些算子不应该一开始就一一替换

下面这些点需要特别强调。

### 6.1 `memref.subview`

不要因为出现 tile 视图就立刻设计 `vortex.subview`

`memref.subview` 在以下场景中应继续保留：

1. 普通切片
2. 形状/视图推导
3. local memory promotion 前的 tile 地址表示

只有当它已经变成目标可见的内存语义时，才需要重写，例如：

1. tile 被提升到 local memory
2. 需要显式 banked/local layout
3. 需要落到特定地址空间

这时更合理的形式通常是：

```text
memref.subview + copy
  -> vortex.local_alloc + 显式 copy + vortex.barrier {scope = "core"}
```

### 6.2 `linalg.matmul`

不要第一步就造 `vortex.matmul`

更合理的路径是：

```text
linalg.matmul
  -> linalg tiling / fusion
  -> vectorization
  -> vector.contract
  -> 如果硬件支持，再模式匹配成 vortex.mma
```

只有在下面两个条件都满足时，才值得引入 `vortex.mma`：

1. Vortex 硬件确实存在对应矩阵计算原语
2. LLVM 后端能用 intrinsic 或特定 LLVM IR 形式承接它

### 6.3 `scf.for`

不要把所有循环都替换成 `vortex.loop`

以下场景中的 `scf.for` 应继续保留：

1. 算法级循环嵌套
2. reduction 结构
3. tile 遍历

只有当循环已经明确表示硬件执行映射时，才需要替换，例如：

1. grid / core 映射
2. warp / thread 映射
3. 显式 launch 区域

## 7. 推荐的 lowering 路线

建议采用分阶段 lowering。

### 阶段 A：pre-vortex 规范化

输入主要是：

1. `func`
2. `arith`
3. `scf`
4. `memref`
5. `tensor`
6. `linalg`
7. `vector`

这一阶段做：

1. canonicalize
2. CSE
3. tiling
4. fusion
5. bufferization
6. vectorization

### 阶段 B：执行映射与内存提升

这一阶段做：

1. 决定哪些循环维映射到 `core/subgroup/thread`
2. 决定哪些 tile 要进入 local memory
3. 插入显式同步点及其作用域
4. 给 IR 附加 mapping 属性，或把相关区域改写成 `vortex.*`

### 阶段 C：按需引入 Vortex 方言

只把真正目标相关的语义改写为 `vortex.*`：

1. launch 与执行 id
2. local allocation 与地址空间
3. barrier（带 scope）与 fence
4. predication / mask
5. 硬件原生计算原语

### 阶段 D：Vortex 到 LLVM Dialect

典型 lowering 形式包括：

1. `vortex.thread_id` -> LLVM Dialect 中的 builtin / intrinsic
2. `vortex.subgroup_id` -> LLVM Dialect 中的 builtin / intrinsic
3. `vortex.barrier` -> LLVM Dialect 中带作用域语义的 call / intrinsic
4. `vortex.local_alloc` -> 带 `local` address space 的 LLVM pointer / 对象
5. `#vortex.address_space<...>` -> LLVM address space
6. `vortex.kernel` 属性 -> LLVM 函数级 ABI 信息

### 阶段 E：其余标准方言继续走常规 lowering

仍然尽量复用标准 lowering：

1. `linalg` -> `scf` / `vector`
2. `scf` -> `cf`
3. `arith` / `memref` / `func` / `vector` -> `llvm`

### 阶段 F：接现有 LLVM 后端

最终链路是：

1. LLVM Dialect -> LLVM IR
2. LLVM IR -> 现有 Vortex LLVM backend
3. backend -> ELF

## 8. 当前工程状态

当前 `/home/cx/mlir-vortex` 已完成的内容：

1. `vx-opt` 已可构建和运行
2. `pre-vortex` 方言合法性检查已实现
3. `pre-vortex` IR 汇总 pass 已实现
4. `vortex-pre-vortex-pipeline` 已实现
5. lit 测试已通过

当前已经新增的内容：

1. 第一版 `vortex` 自定义方言骨架已实现
2. 已接入 `vx-opt` 的 dialect 注册与构建流程
3. 已补充第一批 dialect round-trip / verifier lit 测试
4. 已新增 `examples/vortex_kernel_skeleton.mlir` 示例

当前还没有做的内容：

1. `PreVortexToVortex` lowering 还没有实现
2. `VortexToLLVM` conversion 还没有实现
3. 真正的 `kernel ABI` 细化信息还没有落到专门 attribute 结构里，当前先用 `vortex.kernel` unit attr 占位

所以当前进度可以概括为：

1. `pre-vortex` 边界定义：已可用
2. `vortex` 方言第一版骨架：已可用
3. `PreVortexToVortex` lowering：下一步
4. `vortex -> llvm` lowering：尚未开始

## 9. 当前开发计划

### 阶段 0：冻结硬件边界

目标：

1. 明确执行层级
2. 明确 local/global/private 内存模型
3. 明确 barrier / fence 语义
4. 明确 kernel ABI

输出：

1. 硬件语义说明
2. ABI 说明
3. 地址空间映射说明

状态：

1. 待完成

### 阶段 1：继续强化 pre-vortex 层

目标：

1. 增加更多代表性 kernel
2. 统计真实出现的算子组合
3. 确定稳定的 pre-vortex allowlist

输出：

1. 更多样例 IR
2. 更多汇总结果
3. 稳定的 pre-vortex 方言集合

状态：

1. 进行中

### 阶段 2：定义第一版 Vortex 方言

第一版已经落地的最小集合：

1. `vortex.kernel`（当前以 `func.func` 上的 unit attr 形式存在）
2. `vortex.launch`
3. `vortex.core_id`
4. `vortex.subgroup_id`
5. `vortex.thread_id`
6. `vortex.local_alloc`
7. `vortex.barrier`
8. `vortex.fence`
9. `#vortex.address_space<...>`
10. `#vortex.scope<...>`
11. `vortex.yield`（结构性 terminator）

输出：

1. dialect 骨架
2. parser / printer
3. verifier
4. lit 测试
5. 样例 IR

状态：

1. 已完成第一版 scaffold

### 阶段 3：实现 PreVortexToVortex passes

目标：

1. 把外层循环映射成执行层级
2. 把需要的 tile 提升到 local memory
3. 插入带作用域的 barrier
4. 其他部分继续保留标准 MLIR

输出：

1. loop mapping pass
2. local memory promotion pass
3. 对应测试

状态：

1. 待完成

### 阶段 4：实现 VortexToLLVM conversion

目标：

1. 把 execution id lower 到 LLVM 可识别形式
2. 把 `local` memory 与 `#vortex.address_space<...>` lower 到目标地址空间
3. 把带 scope 的 `barrier` 与 `fence` lower 到 builtin 或 intrinsic
4. 把 kernel ABI lower 到 LLVM 函数契约

输出：

1. conversion patterns
2. type converter
3. conversion target
4. lit / FileCheck 测试

状态：

1. 待完成

### 阶段 5：后端联调与板级验证

目标：

1. 把 LLVM IR 喂给现有 Vortex LLVM backend
2. 生成 ELF
3. 复用 unified manifest 和板级验证链路

输出：

1. 端到端样例
2. baseline 对比结果
3. 板级验证记录

状态：

1. 待完成

## 10. 下一步的直接动作

最合理的下一步是：

1. 再增加 2 到 3 个 `pre-vortex` 样例：
   卷积、reduction、cooperative tiled copy
2. 基于这些样例固定第一版 `vortex` 算子清单
3. 基于当前已落地的 `Dialect/Vortex/IR` 骨架，开始实现 `PreVortexToVortex`
   passes
4. 先实现第一版 loop mapping pass，把部分 `scf.for` 映射到
   `vortex.launch/core_id/subgroup_id/thread_id`
