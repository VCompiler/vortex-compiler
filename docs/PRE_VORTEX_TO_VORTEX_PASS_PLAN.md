# Pre-Vortex 到 Vortex 的 MLIR Pass 设计与实现顺序

## 1. 文档目标

这份文档专门回答一个问题：

```text
Pre-Vortex 到 Vortex 的 lowering，
在 MLIR 工程里应该拆成哪些 pass，
这些 pass 的职责边界是什么，
又应该按什么顺序落地实现。
```

这份文档不再讨论总体 IR 分层原则，也不围绕某一个具体 kernel 样例逐行分析，
而是把重点放在：

1. pass 如何拆分
2. pass 之间的前后依赖
3. 每个 pass 该做什么
4. 每个 pass 不该做什么
5. 当前第一步应该先实现哪两个 pass

补充说明：

如果需要一份更聚焦的“从 pre-vortex 一直到 LLVM / backend 的 MVP 清单”，
见同目录下的：

```text
docs/MVP_BACKEND_PASS_CHECKLIST.md
```

---

## 2. 总体原则

### 2.1 先做“稳定语义”，再做“复杂重写”

优先实现：

1. kernel 边界识别
2. 地址空间 materialization
3. execution mapping 的显式化

而不是一开始就去做：

1. 大规模循环改写
2. 智能 tile promotion
3. `linalg` 到低层循环的完全展开
4. 复杂 predication

### 2.2 pass 要尽量单一职责

不要把下面这些事情塞进一个 pass：

1. 既识别 kernel
2. 又改函数类型
3. 又映射循环
4. 又 promotion local tile
5. 又插 barrier

更合理的方式是拆开，让每个 pass 只负责一类稳定语义。

### 2.3 先建立“混合 IR”，不要追求一步到位

当前阶段合理目标不是：

```text
pre-vortex -> 纯 vortex IR
```

而是：

```text
pre-vortex -> func + arith + scf + memref + linalg/vector + vortex
```

也就是：

1. Vortex 只接住目标相关语义
2. 标准 MLIR 继续承载通用计算和控制流

---

## 3. 推荐的 pass 总顺序

建议按下面顺序落地：

### 第一阶段：边界建立

1. `MarkVortexKernelPass`
2. `MaterializeVortexAddressSpacesPass`

### 第二阶段：执行映射

3. `MapParallelLoopsToVortexLaunchPass`

### 第三阶段：内存与同步

4. `PromoteTilesToVortexLocalPass`
5. `InsertVortexBarriersPass`
6. `InsertVortexFencesPass`

### 第四阶段：计算体下沉

7. `LowerLinalgInsideVortexPass`
8. `LowerVectorInsideVortexPass`

### 第五阶段：LLVM 接口准备

9. `LegalizeVortexForLLVMPass`
10. `LowerVortexRuntimeBuiltinsPass`

---

## 4. 为什么先做前两个 pass

当前最值得先落地的是：

1. `MarkVortexKernelPass`
2. `MaterializeVortexAddressSpacesPass`

原因很简单：

### 4.1 它们最稳定

这两个 pass 不依赖复杂的调度分析，也不需要先决定：

1. 哪个循环映射到 core
2. 哪个循环映射到 subgroup
3. tile 是否要 promotion 到 local

它们只是在建立两个稳定边界：

1. 哪些函数是 Vortex kernel
2. 哪些 memref 已经是 Vortex 可见地址空间

### 4.2 它们风险最低

这两个 pass：

1. 不改计算语义
2. 不改循环结构
3. 不改访存顺序
4. 不引入同步

因此最适合先实现并打通测试。

### 4.3 它们是后续 pass 的公共前提

后面的 `launch` mapping、local promotion、barrier insertion 都会依赖：

1. kernel 入口已经明确
2. global/local/private 地址空间约定已经一致

所以先做这两个 pass，后面的 pass 才更容易写干净。

---

## 5. Pass 1：`MarkVortexKernelPass`

## 5.1 目标

识别 kernel 入口函数，并给它加上：

```mlir
attributes {vortex.kernel}
```

### 5.2 输入假设

输入仍然是 pre-vortex 标准 MLIR，例如：

```mlir
func.func @tiled_matmul(%a: memref<128x128xf32>, %b: memref<128x128xf32>, %c: memref<128x128xf32>) {
  return
}
```

### 5.3 输出目标

```mlir
func.func @tiled_matmul(%a: memref<128x128xf32>, %b: memref<128x128xf32>, %c: memref<128x128xf32>)
    attributes {vortex.kernel} {
  return
}
```

### 5.4 第一版建议规则

第一版不要做复杂推断，建议只支持非常保守的识别方式：

1. 函数名匹配约定
2. 或已有显式标记 attr
3. 或 pass 选项里指定函数名列表

第一版不要做：

1. 基于调用图自动推断
2. 基于 side-effect 自动推断
3. 基于参数类型自动猜测

### 5.5 这个 pass 该做什么

1. 遍历 `func.func`
2. 判定是否是 kernel
3. 给目标函数设置 `vortex.kernel`
4. 可选地清理旧的临时标记

### 5.6 这个 pass 不该做什么

1. 不改函数签名
2. 不改参数类型
3. 不插 `vortex.launch`
4. 不做 local promotion
5. 不做 barrier/fence

### 5.7 需要的测试

至少应覆盖：

1. 正常标记 kernel 函数
2. 非 kernel 函数不应被标记
3. 已有 `vortex.kernel` 时保持幂等

---

## 6. Pass 2：`MaterializeVortexAddressSpacesPass`

## 6.1 目标

把 kernel 参数中的 memref 类型显式改写成：

```text
#vortex.address_space<global>
```

为后续 local promotion 和 LLVM lowering 建立地址空间边界。

### 6.2 输入假设

该 pass 依赖：

1. kernel 函数已经被 `MarkVortexKernelPass` 标记
2. 输入仍处于 pre-vortex / 混合 IR 的较高层

### 6.3 输出目标

输入：

```mlir
func.func @kernel(%a: memref<128x128xf32>, %b: memref<128x128xf32>) attributes {vortex.kernel}
```

输出：

```mlir
func.func @kernel(
    %a: memref<128x128xf32, #vortex.address_space<global>>,
    %b: memref<128x128xf32, #vortex.address_space<global>>)
    attributes {vortex.kernel}
```

### 6.4 第一版建议范围

第一版先只处理：

1. kernel 函数参数

第一版先不要处理：

1. 函数返回值
2. 中间 `alloc` 出来的 buffer
3. `memref.subview` 的结果类型
4. `call` 边界上的跨函数 ABI

这样范围最容易控制。

### 6.5 这个 pass 该做什么

1. 找到带 `vortex.kernel` 的 `func.func`
2. 扫描函数参数中的 `memref`
3. 如果 memory space 为空，则补成 `#vortex.address_space<global>`
4. 如果已经是 `#vortex.address_space<...>`，则保持不变
5. 必要时更新函数类型

### 6.6 这个 pass 不该做什么

1. 不决定 local/private
2. 不自动 promotion buffer
3. 不改 load/store 本体
4. 不改循环
5. 不插入 Vortex op

### 6.7 需要的测试

至少应覆盖：

1. kernel 参数补 global address space
2. 非 kernel 函数不应被改写
3. 已有 Vortex memory space 时保持不变
4. 非 memref 参数保持不变

---

## 7. Pass 3：`MapParallelLoopsToVortexLaunchPass`

这是第一个真正开始引入执行语义的 pass。

### 7.1 目标

把已经确定映射到 Vortex 执行层级的区域显式改写为：

1. `vortex.launch`
2. `vortex.core_id`
3. `vortex.subgroup_id`
4. `vortex.thread_id`

### 7.2 第一版建议策略

第一版必须保守：

1. 不吞掉所有 `scf.for`
2. 不自动推断所有并行层级
3. 只转换已经明确识别的“内层协作区域”

### 7.3 当前阶段不建议做的事

1. 不把所有外层 tile grid 直接转成 Vortex launch
2. 不假设当前 dialect 已经有完整 grid/block 语义
3. 不在这个 pass 里顺便做 local promotion

---

## 8. Pass 4：`PromoteTilesToVortexLocalPass`

### 8.1 目标

把已识别的 tile promotion 候选改成：

1. `vortex.local_alloc`
2. global-to-local cooperative copy
3. 必要的 local buffer 使用

### 8.2 第一版建议只做静态、明确模式

优先支持：

1. 静态 tile 形状
2. 清晰的 `memref.subview`
3. 明确存在 reuse 的 tile

暂时不做：

1. 动态复杂切片
2. 自动 bank layout 选择
3. 复杂 alias 情况

---

## 9. Pass 5：`InsertVortexBarriersPass`

### 9.1 目标

在 cooperative copy 与 local tile 使用之间插入：

1. `vortex.barrier <core>`
2. 必要时 `vortex.barrier <subgroup>`

### 9.2 为什么不应更早做

因为 barrier 的位置依赖：

1. launch region 已经存在
2. local promotion 已经发生
3. copy 边界已经明确

所以 barrier insertion 应排在 launch mapping 和 local promotion 之后。

---

## 10. Pass 6：`LowerLinalgInsideVortexPass`

### 10.1 目标

把 Vortex launch 内部还残留的：

1. `linalg.fill`
2. `linalg.generic`
3. `linalg.matmul`

继续压低到：

1. `scf`
2. `memref`
3. `arith`

### 10.2 当前不建议直接做的事

第一版不要直接引入：

1. `vortex.matmul`
2. `vortex.fill`
3. `vortex.reduce`
4. `vortex.generic`

因为这些很容易只是标准 op 的改名版本。

---

## 11. 当前阶段建议的 pass pipeline

当前最近期最建议实现的是一个最小 pipeline：

```text
vortex-mark-kernel
  -> vortex-materialize-address-spaces
```

然后下一步再接：

```text
vortex-map-loops-to-launch
```

再往后才是：

```text
vortex-promote-tiles-to-local
  -> vortex-insert-barriers
```

---

## 12. 当前阶段的实现优先级

### 优先级 P0

1. `MarkVortexKernelPass`
2. `MaterializeVortexAddressSpacesPass`

### 优先级 P1

3. `MapParallelLoopsToVortexLaunchPass`

### 优先级 P2

4. `PromoteTilesToVortexLocalPass`
5. `InsertVortexBarriersPass`

### 优先级 P3

6. `LowerLinalgInsideVortexPass`
7. `LegalizeVortexForLLVMPass`

---

## 13. 建议的测试组织

建议在 `test/Transforms/` 下按 pass 分开：

1. `mark-vortex-kernel.mlir`
2. `materialize-vortex-address-spaces.mlir`
3. `map-loops-to-launch.mlir`
4. `promote-tiles-to-local.mlir`
5. `insert-vortex-barriers.mlir`

这样可以保证：

1. 每个 pass 单测独立
2. 出错时容易定位
3. pipeline 测试和单 pass 测试可以同时保留

---

## 14. 一句话结论

当前最合理的落地顺序不是先去碰复杂的 loop mapping 或 matmul lowering，
而是先把最稳定的两层边界建起来：

1. `MarkVortexKernelPass`
2. `MaterializeVortexAddressSpacesPass`

这两个 pass 做完之后，整个工程才算真正开始从“pre-vortex 分析阶段”
进入“Vortex 目标语义 materialization 阶段”。
