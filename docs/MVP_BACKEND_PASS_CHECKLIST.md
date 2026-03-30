# Pre-Vortex 到后端的 MVP Pass 清单

## 1. 文档目标

这份文档只回答一个问题：

```text
如果按 MVP 原则推进，
从 pre-vortex 标准 MLIR
一直打通到 LLVM / Vortex 后端，
最少需要哪些 pass，
它们的先后顺序是什么，
哪些可以明确延期。
```

这里强调的是：

1. 先打通一条最小可运行链路
2. 先建立稳定语义边界
3. 先让 Vortex dialect 真正接住执行、地址空间、local memory、同步
4. 不在 MVP 阶段追求自动分析、复杂 vector lowering、predication、fence 全覆盖

---

## 2. MVP 的边界

当前建议把 MVP 范围收窄为：

1. 只支持 `memref` 形式的 kernel
2. 只支持静态 shape 或至少 local promotion 相关 tile 为静态 shape
3. execution mapping 先靠显式 attr，不做自动推断
4. tile promotion 先靠显式 attr，不做自动推断
5. `local` 只先建模为 Vortex 的 per-core local memory
6. 先打通 `linalg -> loops`，不要求先做复杂 `vector` 专项 lowering
7. 先做到：

```text
pre-vortex
  -> 混合 vortex IR
  -> LLVM dialect
  -> LLVM IR
  -> Vortex LLVM backend
```

MVP 不要求：

1. 自动化 mapping 分析
2. 自动 tile 识别
3. `vortex.fence` 的完整内存模型
4. predication / mask
5. async / DMA
6. bank conflict / local layout 调优
7. 专用矩阵原语

---

## 3. 当前已经有的 pass

下面这些 pass 已经实现：

1. `vortex-validate-pre-vortex`
2. `vortex-summarize-pre-vortex`
3. `vortex-mark-kernel`
4. `vortex-materialize-address-spaces`
5. `vortex-map-parallel-loops-to-launch`
6. `vortex-promote-tiles-to-local`

也就是说，当前已经具备：

1. pre-vortex 边界校验
2. kernel 标记
3. global address space materialization
4. 显式执行区域 `vortex.launch`
5. 显式 local tile `vortex.local_alloc`

当前还没有的是：

1. barrier 插入
2. kernel 内 `linalg` 下沉
3. Vortex 到 LLVM 的专用转换
4. 端到端 pipeline 注册

---

## 4. MVP 必须完成的 pass 清单

## 4.1 Pre-Vortex 规范化阶段

### 1. `canonicalize`

作用：

1. 清理冗余 IR
2. 简化 constant / subview / loop 结构

状态：

1. 直接使用 MLIR 标准 pass

### 2. `cse`

作用：

1. 做公共子表达式消除
2. 收紧高层 IR

状态：

1. 直接使用 MLIR 标准 pass

### 3. `vortex-validate-pre-vortex`

作用：

1. 限定当前允许的 pre-vortex dialect 集合
2. 防止过早混入低层 dialect

状态：

1. 已实现

### 4. `vortex-summarize-pre-vortex`

作用：

1. 收集函数中出现的 op / dialect / memory space
2. 给后续 pass 设计提供输入画像

状态：

1. 已实现

---

## 4.2 Kernel 与地址空间边界阶段

### 5. `vortex-mark-kernel`

作用：

1. 标记 kernel 入口
2. 建立 target-specific 边界

状态：

1. 已实现

### 6. `vortex-materialize-address-spaces`

作用：

1. 把 kernel 参数上的 `memref` 显式改成 `#vortex.address_space<global>`
2. 为后续 local promotion / LLVM lowering 做准备

状态：

1. 已实现

---

## 4.3 执行映射阶段

### 7. `vortex-map-parallel-loops-to-launch`

作用：

1. 把显式映射的 `scf.for` nest 变成 `vortex.launch`
2. 用 `vortex.core_id / subgroup_id / thread_id` 替换 mapped IV

状态：

1. 已实现

MVP 说明：

1. 这一步先继续依赖显式 `vortex.mapping` attr
2. 不在 MVP 阶段引入自动 mapping analysis

---

## 4.4 Local Memory 阶段

### 8. `vortex-promote-tiles-to-local`

作用：

1. 把显式标记的 `memref.subview` promotion 成 `vortex.local_alloc`
2. 插入 `copy-in`
3. 在需要时插入 `copy-out`

状态：

1. 已实现

MVP 说明：

1. 当前仍是显式 attr 驱动
2. 当前已经收紧了 global source、uniform tile、no-escape、write-back 约束

### 9. `vortex-insert-barriers`

作用：

1. 在 local tile 的协作使用边界插入 `vortex.barrier`
2. 至少先覆盖：
   1. copy-in 之后、compute 之前
   2. compute 之后、copy-out 之前

状态：

1. MVP 必须实现
2. 这是当前最缺的一步

MVP 边界：

1. 先只处理 `vortex.launch` 内、由 `vortex-promote-tiles-to-local` 产生的 local tile
2. 先只插 `#vortex.scope<core>`
3. 不做复杂自动同步分析

---

## 4.5 计算体下沉阶段

### 10. `vortex-lower-linalg-inside-kernel`

作用：

1. 把 `vortex.launch` 内剩余的 `linalg.*` 下沉成更低层的 `scf + memref + arith`
2. 让计算体进入 LLVM conversion 友好的形态

状态：

1. MVP 必须实现

MVP 边界：

1. 先只处理 kernel 内的 `linalg.fill`、`linalg.matmul`、常见 elementwise
2. 优先 lower 成 loops
3. 不要求单独保留一层复杂的 `vector` 专项 pass

### 11. `lower-affine-to-scf`

作用：

1. 如果前端仍带 `affine.*`，则先统一落回 `scf`

状态：

1. 视输入而定
2. 如果 pre-vortex 已经不使用 `affine`，这步可以省

### 12. `canonicalize`

作用：

1. 清理前面几步引入的中间冗余

状态：

1. 标准 pass

### 13. `cse`

作用：

1. 再收紧一遍 SSA

状态：

1. 标准 pass

---

## 4.6 Vortex 到 LLVM 的接口准备阶段

### 14. `vortex-prepare-to-llvm`

作用：

1. 为 `vortex.* -> llvm.*` 转换统一接口
2. 明确 kernel ABI
3. 明确地址空间编号映射
4. 明确 barrier / id query 的 lowering 目标

状态：

1. MVP 必须实现

这一 pass 至少应处理：

1. `#vortex.address_space<global/local/private>` 到 LLVM address space 的固定映射
2. `vortex.local_alloc` 的后端对象表示约定
3. `vortex.core_id / subgroup_id / thread_id` 的 lowering 入口约定
4. `vortex.barrier` 的 intrinsic / builtin / runtime stub 约定
5. `vortex.launch` 的结构性消解方式

### 15. `vortex-convert-to-llvm`

作用：

1. 把 `vortex.*` 真正转成 LLVM dialect

状态：

1. MVP 必须实现

最少需要覆盖：

1. `vortex.core_id`
2. `vortex.subgroup_id`
3. `vortex.thread_id`
4. `vortex.local_alloc`
5. `vortex.barrier`
6. `vortex.launch`
7. `#vortex.address_space<...>`

---

## 4.7 标准 MLIR 到 LLVM 的收尾阶段

下面这些 pass 原则上直接复用标准 MLIR conversion：

### 16. `convert-scf-to-cf`

作用：

1. 把结构化控制流变成 CFG

### 17. `convert-arith-to-llvm`

作用：

1. 把通用标量算术转成 LLVM dialect

### 18. `finalize-memref-to-llvm`

作用：

1. 完成 memref descriptor 到 LLVM dialect 的 lowering

### 19. `convert-func-to-llvm`

作用：

1. 把 `func.func/call/return` 变成 LLVM dialect

### 20. `convert-cf-to-llvm`

作用：

1. 把 `cf.*` 转成 LLVM dialect

### 21. `reconcile-unrealized-casts`

作用：

1. 清理 conversion 过程遗留的 cast 桥接

---

## 4.8 LLVM IR 与后端阶段

### 22. `mlir-to-llvmir`

作用：

1. 从 LLVM dialect 导出 LLVM IR

### 23. LLVM backend codegen

作用：

1. 让 LLVM IR 进入真正的 Vortex target backend
2. 生成目标机器码 / 汇编 / 目标文件

说明：

1. 这一步已经不是 MLIR pass
2. 但它必须纳入端到端 MVP 路线里

---

## 5. MVP 明确不做的 pass

下面这些可以明确延期：

1. `AnalyzeExecutionMappingPass`
2. `AnalyzeTilePromotionPass`
3. `InsertVortexFencesPass`
4. `LowerVectorInsideVortexPass`
5. predication / mask lowering
6. async / DMA lowering
7. 自动 barrier 推断
8. 专用矩阵 / tensor core 类原语

延期理由：

1. 它们不是打通第一条后端链路的必要条件
2. 很多都依赖更成熟的执行模型和 memory model

---

## 6. 推荐的 MVP 实现顺序

建议按下面顺序推进：

1. `canonicalize`
2. `cse`
3. `vortex-validate-pre-vortex`
4. `vortex-summarize-pre-vortex`
5. `vortex-mark-kernel`
6. `vortex-materialize-address-spaces`
7. `vortex-map-parallel-loops-to-launch`
8. `vortex-promote-tiles-to-local`
9. `vortex-insert-barriers`
10. `vortex-lower-linalg-inside-kernel`
11. `canonicalize`
12. `cse`
13. `vortex-prepare-to-llvm`
14. `vortex-convert-to-llvm`
15. `convert-scf-to-cf`
16. `convert-arith-to-llvm`
17. `finalize-memref-to-llvm`
18. `convert-func-to-llvm`
19. `convert-cf-to-llvm`
20. `reconcile-unrealized-casts`
21. `mlir-to-llvmir`
22. LLVM backend codegen

---

## 7. 对当前代码基的直接结论

如果只看当前仓库状态，下一批最值得实现的 pass 是：

1. `vortex-insert-barriers`
2. `vortex-lower-linalg-inside-kernel`
3. `vortex-prepare-to-llvm`
4. `vortex-convert-to-llvm`

原因是：

1. `launch` 和 `local_alloc` 已经有了
2. 但同步语义还没闭合
3. kernel 内高层计算还没压低
4. Vortex dialect 还没有真正接到 LLVM/backend

---

## 8. 配套 Pipeline 建议

最终建议至少注册三条 pipeline：

### 1. `vortex-pre-vortex-pipeline`

作用：

1. 只做 pre-vortex 规范化与校验

### 2. `vortex-pre-to-vortex-mvp-pipeline`

作用：

1. 从高层 IR 走到混合 Vortex IR

### 3. `vortex-to-llvm-mvp-pipeline`

作用：

1. 从混合 Vortex IR 走到 LLVM dialect / LLVM IR

如果只想先跑通端到端，也可以额外注册：

### 4. `vortex-end-to-end-mvp-pipeline`

作用：

1. 一次性串起上面所有必要阶段

