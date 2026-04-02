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
7. `vortex-insert-barriers`
8. `vortex-plan-local-memory-layout`
9. `vortex-lower-linalg-inside-kernel`
10. `vortex-lower-local-memory`

也就是说，当前已经具备：

1. pre-vortex 边界校验
2. kernel 标记
3. global address space materialization
4. 显式执行区域 `vortex.launch`
5. 显式 local tile `vortex.local_alloc`
6. local tile 协作使用边界上的 `vortex.barrier <core>`
7. 每个 kernel 的 local frame 大小与 `local_alloc` 字节偏移规划
8. kernel 内 buffer-semantics `linalg.* -> scf + memref + arith`
9. local `memref.load/store/copy` 到显式地址化 LLVM 访问的 MVP lowering

当前还没有的是：

1. 更完整的端到端 pipeline 注册
2. 从 LLVM dialect 到最终 LLVM IR / backend 的一键集成流程

当前已经补上的 pipeline：

1. `vortex-mvp-backend-pipeline`
2. 可把 post-local-memory 的混合 Vortex IR 继续降到 LLVM dialect

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

1. 已实现

MVP 边界：

1. 先只处理 `vortex.launch` 内、由 `vortex-promote-tiles-to-local` 产生的 local tile
2. 先只插 `#vortex.scope<core>`
3. 不做复杂自动同步分析

---

### 10. `vortex-plan-local-memory-layout`

作用：

1. 为 kernel 内所有 `vortex.local_alloc` 规划 local frame
2. 计算每个 alloc 的 `byte_offset / byte_size / alignment`
3. 给函数附加 `vortex.local_frame_bytes`

状态：

1. MVP 基础版已实现
2. 当前已覆盖：
   静态 shape、紧凑布局、no-escape 的 `vortex.local_alloc`
3. 当前仍未覆盖：
   动态 shape、复杂 local alias、真正的 local address lowering

MVP 边界：

1. 先只接受 `#vortex.address_space<local>` 的静态 shaped memref
2. 禁止 local memref 通过 `call / yield / branch / iter_args` 逃逸
3. 这一 pass 只做布局属性规划，不做真实 lowering

---

## 4.5 计算体下沉阶段

### 11. `vortex-lower-linalg-inside-kernel`

作用：

1. 把 `vortex.launch` 内剩余的 `linalg.*` 下沉成更低层的 `scf + memref + arith`
2. 让计算体进入 LLVM conversion 友好的形态

状态：

1. 已实现

MVP 边界：

1. 当前按保守策略处理 kernel 内的 buffer-semantics `linalg.*`
2. 当前 lower 成 `scf.for + memref.load/store + arith`
3. tensor-semantics `linalg` 仍直接拒绝

### 12. `vortex-lower-local-memory`

作用：

1. 把 local `memref.copy` 展开成显式 loop nest
2. 把 local `memref.load/store` lower 成 `llvm.inttoptr + llvm.load/store`
3. 擦除已无用的 `vortex.local_alloc / memref.cast`
4. 对 local `memref.subview` 做索引回推并映射回 root `vortex.local_alloc`

状态：

1. MVP 基础版已实现
2. 当前已覆盖：
   `memref.cast`、`memref.subview`、`memref.copy`、`memref.load`、`memref.store`
3. 当前仍未覆盖：
   `memref.reinterpret_cast / expand_shape / collapse_shape / transpose / view`
   等更复杂 local alias、`vx_local_mem_base` 的真实 runtime 实现

MVP 边界：

1. 要求先跑 `vortex-plan-local-memory-layout`
2. 要求 local use 最终收敛到 `cast/subview/copy/load/store`
3. 当前 `memref.subview` 仅按“地址变换”处理，不生成新的 local 对象
4. 更复杂 local alias 仍显式拒绝

### 13. `lower-affine-to-scf`

作用：

1. 如果前端仍带 `affine.*`，则先统一落回 `scf`

状态：

1. 视输入而定
2. 如果 pre-vortex 已经不使用 `affine`，这步可以省

### 14. `canonicalize`

作用：

1. 清理前面几步引入的中间冗余

状态：

1. 标准 pass

### 15. `cse`

作用：

1. 再收紧一遍 SSA

状态：

1. 标准 pass

---

## 4.6 Vortex 到 LLVM 的接口准备阶段

### 16. `vortex-legalize-for-llvm`

作用：

1. 为 `vortex.* -> llvm.*` 转换统一接口
2. 明确 kernel ABI
3. 明确地址空间编号映射
4. 明确 barrier / id query 的 lowering 目标

状态：

1. MVP 基础版已实现
2. 当前已覆盖：
   `lower-affine`、`vortex.launch` 内联、`global/private` 地址空间整数化、
   residual local / residual launch 合法性检查
3. 当前仍未覆盖：
   `vortex.fence`、Vortex intrinsic 真正转 LLVM、剩余 local wrapper ABI 收口

这一 pass 至少应处理：

1. 先跑标准 `lower-affine`
2. 把 `vortex.launch` 的 region inline 到外层 block，并擦掉 launch 外壳
3. 把 `#vortex.address_space<global/private>` 改成整数 memory space
4. 若还残留 `vortex.local_alloc` 或 `#vortex.address_space<local>`，直接报错，
   要求先跑 `vortex-lower-local-memory`
5. 给后续 `vortex-lower-runtime-builtins` 留下更稳定的 Vortex op 边界

### 17. `vortex-lower-runtime-builtins`

作用：

1. 把 `vortex.*` 真正转成 LLVM dialect

状态：

1. MVP 第一阶段已实现
2. 当前已覆盖：
   `core_id / subgroup_id / thread_id / barrier<core> -> vx_* wrapper call`
3. 当前还会做：
   把 `vortex.kernel` 改写成普通 marker `vortex.kernel_entry`，
   避免后续 `llvm.func` 触发 Vortex dialect verifier
4. 当前仍未覆盖：
   `fence`、`barrier<subgroup>`、可能残留的 target-specific local ABI 收口

最少需要覆盖：

1. `vortex.core_id`
2. `vortex.subgroup_id`
3. `vortex.thread_id`
4. `vortex.barrier`
5. `vortex.launch`
6. `#vortex.address_space<...>`
7. 后续保留下来的 Vortex target wrapper / intrinsic

---

## 4.7 标准 MLIR 到 LLVM 的收尾阶段

下面这些 pass 原则上直接复用标准 MLIR conversion：

### 18. `convert-scf-to-cf`

作用：

1. 把结构化控制流变成 CFG

### 19. `convert-arith-to-llvm`

作用：

1. 把通用标量算术转成 LLVM dialect

### 20. `finalize-memref-to-llvm`

作用：

1. 完成 memref descriptor 到 LLVM dialect 的 lowering

### 21. `convert-func-to-llvm`

作用：

1. 把 `func.func/call/return` 变成 LLVM dialect

### 22. `convert-cf-to-llvm`

作用：

1. 把 `cf.*` 转成 LLVM dialect

### 23. `reconcile-unrealized-casts`

作用：

1. 清理 conversion 过程遗留的 cast 桥接

---

## 4.8 LLVM IR 与后端阶段

### 24. `mlir-to-llvmir`

作用：

1. 从 LLVM dialect 导出 LLVM IR

### 25. LLVM backend codegen

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
10. `vortex-plan-local-memory-layout`
11. `vortex-lower-linalg-inside-kernel`
12. `vortex-lower-local-memory`
13. `canonicalize`
14. `cse`
15. `vortex-legalize-for-llvm`
16. `vortex-lower-runtime-builtins`
17. `convert-scf-to-cf`
18. `convert-arith-to-llvm`
19. `finalize-memref-to-llvm`
20. `convert-func-to-llvm`
21. `convert-cf-to-llvm`
22. `reconcile-unrealized-casts`
23. `mlir-to-llvmir`
24. LLVM backend codegen

---

## 7. 对当前代码基的直接结论

如果只看当前仓库状态，下一批最值得实现的 pass 是：

1. `vortex-legalize-for-llvm` 的 local memory / LLVM 边界收口
2. `vortex-lower-runtime-builtins` 的 `fence` / `barrier<subgroup>` 补完
3. 更复杂 local alias 与 local memory runtime 约定继续补完

原因是：

1. `launch`、`local_alloc`、barrier、local frame layout、local direct access 已经有了
2. 但 `fence` 与 subgroup barrier 仍未闭合
3. local `subview` 已接入新的地址模型，但更复杂 alias 仍未覆盖
4. Vortex dialect 到 LLVM/backend 的边界还需要继续收口

---

## 8. 配套 Pipeline 建议

最终建议至少注册三条 pipeline：

### 1. `vortex-pre-vortex-pipeline`

作用：

1. 只做 pre-vortex 规范化与校验

### 2. `vortex-pre-to-vortex-mvp-pipeline`

作用：

1. 从高层 IR 走到混合 Vortex IR

### 3. `vortex-mvp-backend-pipeline`

作用：

1. 从 post-local-memory 的混合 Vortex IR 走到 LLVM dialect

状态：

1. 已实现
2. 当前串起：
   `canonicalize/cse`
   -> `vortex-legalize-for-llvm`
   -> `vortex-lower-runtime-builtins`
   -> `convert-scf-to-cf`
   -> `convert-arith-to-llvm`
   -> `convert-index-to-llvm`
   -> `finalize-memref-to-llvm`
   -> `convert-func-to-llvm`
   -> `convert-cf-to-llvm`
   -> `reconcile-unrealized-casts`

如果只想先跑通端到端，也可以额外注册：

### 4. `vortex-end-to-end-mvp-pipeline`

作用：

1. 一次性串起上面所有必要阶段
