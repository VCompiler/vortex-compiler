# Vortex 到 LLVM Conversion 现有能力分析

## 1. 结论

当前结论可以先直接定下来：

1. `mlir-vortex` 仓库里现在还没有任何 `VortexToLLVM` pass 实现
2. `affine.apply` 不需要自己写 lowering，上游已有标准 `lower-affine`
3. `memref` 到 LLVM descriptor 的通用 lowering 已有，上游 `finalize-memref-to-llvm` 可以直接复用
4. `#vortex.address_space<...>` 不能直接喂给默认 `LLVMTypeConverter`
5. `vortex.barrier / vortex.fence / vortex.local_alloc / vortex.launch / vortex.core_id / vortex.subgroup_id / vortex.thread_id` 都还需要我们自己补 Vortex 专用 conversion
6. 当前 LLVM Vortex 后端已经有现成的 `vx_*` / `riscv_vx_*` 入口可接：
   1. `vx_core_id`
   2. `vx_warp_id`
   3. `vx_thread_id`
   4. `vx_num_cores`
   5. `vx_num_warps`
   6. `vx_num_threads`
   7. `vx_barrier`
   8. `vx_tmc`
7. 当前没有看到现成的 `vx_fence` 后端入口

---

## 2. `affine.apply` 现成怎么处理

### 2.1 上游已有标准 pass

上游已有：

1. `lower-affine`

它的语义是：

1. `affine.for -> scf.for`
2. `affine.if -> scf.if`
3. `affine.apply -> arith` 组合

因此对我们现在这条链来说：

```text
vortex-lower-linalg-inside-kernel
  -> 可能留下 affine.apply
  -> lower-affine
  -> 变成纯 scf + arith + memref
```

这一步不需要额外发明 Vortex pass。

### 2.2 对当前仓库的直接影响

我们刚做的 `vortex-lower-linalg-inside-kernel` 是调用上游
`linalgOpToLoops`，它会生成一些 `affine.apply`。

所以紧接在它后面的 pass 应该就是：

1. `lower-affine`

而不是自己重写一套 `affine.apply` lowering。

---

## 3. 地址空间到 LLVM 的现状

### 3.1 上游 `MemRefToLLVM` 能做什么

上游已有：

1. `finalize-memref-to-llvm`
2. `LLVMTypeConverter::getMemRefAddressSpace`

这说明：

1. `memref` descriptor 到 LLVM dialect 的通用 lowering 已经有
2. memref 的 memory space 最终会进入 LLVM pointer address space

### 3.2 默认 `LLVMTypeConverter` 不能直接识别 `#vortex.address_space<...>`

默认实现只内建了：

1. `IntegerAttr` memory space 直接映射到同号地址空间

而我们当前用的是：

1. `#vortex.address_space<global>`
2. `#vortex.address_space<local>`
3. `#vortex.address_space<private>`

这意味着如果直接跑默认 `finalize-memref-to-llvm`：

1. `LLVMTypeConverter` 不能把 `AddressSpaceAttr` 转成整数地址空间
2. conversion 会失败

### 3.3 上游已有“自定义 memory space attr -> 整数地址空间”的范式

GPU 相关 conversion 已经有现成范式：

1. `populateGpuMemorySpaceAttributeConversions`

它本质上就是给 `TypeConverter` 补一条规则：

```text
gpu.address_space attr -> IntegerAttr address space
```

Vortex 完全可以照这个模式做：

```text
#vortex.address_space<global>  -> 0
#vortex.address_space<local>   -> X
#vortex.address_space<private> -> Y
```

因此地址空间这块不需要自己从零发明框架，
但需要自己写 Vortex 版本的映射函数。

### 3.4 当前真正的难点不是“地址空间 attr 映射”，而是 `vortex.local_alloc`

`finalize-memref-to-llvm` 只认识标准 `memref.*` 分配/访存形式。

但我们当前还有：

1. `vortex.local_alloc`

所以即便把 `#vortex.address_space<...>` 映射成整数地址空间，
下面的问题还是没解决：

1. `vortex.local_alloc` 怎么 lower

这一步要么：

1. 在 `PrepareVortexToLLVM` 里先改成标准 `memref.alloca` / `memref.alloc`

要么：

1. 在 `ConvertVortexToLLVM` 里直接把它变成 LLVM descriptor

---

## 4. 同步到 LLVM 的现状

### 4.1 上游已有的同步 lowering 范式

上游 GPU 相关 conversion 已有：

1. `gpu.barrier -> nvvm.barrier0`
2. `gpu.barrier -> rocdl.barrier`

也就是说：

1. “目标相关 barrier op 单独 lower” 这个套路是现成的
2. 我们不需要把 `vortex.barrier` 硬塞成标准 `memref` 或 `cf` 语义

### 4.2 LLVM dialect 里有通用 `llvm.fence`

上游 LLVM dialect 已有：

1. `llvm.fence`

所以如果以后 `vortex.fence` 要走通用 LLVM 内存序语义，
理论上可以先落到：

1. `LLVM::FenceOp`

但这只解决“LLVM IR 里有一个 fence”这个层面，
不等于当前 Vortex 后端已经有对应硬件语义。

### 4.3 当前 LLVM Vortex 后端已经有 `barrier` 和执行 id 的入口

当前 Vortex LLVM fork 里已经有：

1. `riscv_vx_tid`
2. `riscv_vx_wid`
3. `riscv_vx_cid`
4. `riscv_vx_nt`
5. `riscv_vx_nw`
6. `riscv_vx_nc`
7. `riscv_vx_bar`
8. `riscv_vx_tmc`

同时还存在一个 `VortexIntrinsicFuncLowering` pass，
会把下面这些 wrapper call 改成上述 intrinsic：

1. `vx_thread_id`
2. `vx_warp_id`
3. `vx_core_id`
4. `vx_num_threads`
5. `vx_num_warps`
6. `vx_num_cores`
7. `vx_barrier`
8. `vx_tmc`

这说明对当前后端最稳妥的接法是：

1. 直接在 MLIR LLVM dialect 里生成 `LLVM::CallOp`
2. 调用这些 `vx_*` wrapper
3. 让现有 LLVM 后端 pass 继续把它们变成 `riscv_vx_*`

### 4.4 当前没有看到 `vx_fence`

当前代码里没有看到：

1. `vx_fence`
2. `riscv_vx_fence`

所以第一版 `vortex.fence` 不建议贸然接到现有后端。

MVP 更合理的策略是：

1. 先不在 pipeline 中生成 `vortex.fence`
2. 或在 `ConvertVortexToLLVM` 中直接拒绝
3. 等确认后端和硬件语义后再放开

---

## 5. 当前 LLVM 后端对地址空间本身的支持情况

这里有一个非常关键的现实约束：

当前 RISC-V target machine 代码里明确写着：

1. 它假设 address space cast 是 no-op

这意味着：

1. 当前后端并没有表现出 GPU 风格的“不同地址空间走不同机器指令选择”能力
2. 仅仅把 `#vortex.address_space<local>` 变成 LLVM address space，
   不代表后端已经会自动把它当成 `VX_local_mem`

另外，现有 Vortex 软件接口里，local memory 更接近：

1. 通过 `VX_CSR_LOCAL_MEM_BASE` 取基址
2. 再做显式地址计算

而不是依赖 LLVM pointer address space 做语义分流。

因此对当前项目来说要非常清楚：

1. “MLIR 里保留 local/global/private 地址空间语义” 是合理的
2. 但“只靠 LLVM address space 就能让后端正确访问 VX_local_mem” 目前没有证据成立

---

## 6. 对 Vortex pass 设计的直接结论

### 6.1 可以直接复用的 pass

可以直接放进后续 pipeline 的：

1. `lower-affine`
2. `expand-strided-metadata`
3. `convert-arith-to-llvm`
4. `convert-index-to-llvm`
5. `convert-scf-to-cf`
6. `convert-cf-to-llvm`
7. `convert-func-to-llvm`
8. `finalize-memref-to-llvm`
9. `reconcile-unrealized-casts`

### 6.2 需要我们自己实现的 Vortex pass

必须自己补的至少有：

1. `PrepareVortexToLLVMPass`
2. `ConvertVortexToLLVMPass`

其中：

#### `PrepareVortexToLLVMPass`

建议负责：

1. 跑 `lower-affine`
2. 清理 `vortex.launch` 外壳或把 region inline 掉
3. 明确 `vortex.local_alloc` 的 lowering 路线
4. 把 `#vortex.address_space<...>` 变成后续 `LLVMTypeConverter` 能接受的形式

#### `ConvertVortexToLLVMPass`

建议负责：

1. `vortex.core_id -> llvm.call @vx_core_id`
2. `vortex.subgroup_id -> llvm.call @vx_warp_id`
3. `vortex.thread_id -> llvm.call @vx_thread_id`
4. `vortex.barrier <core> -> llvm.call @vx_barrier`
5. `vortex.launch -> erase/inline`
6. `vortex.local_alloc -> LLVM dialect 表达`

### 6.3 `vortex.fence` 的处理建议

MVP 建议：

1. 先不开放 `vortex.fence` 的自动生成
2. `ConvertVortexToLLVMPass` 遇到它直接报错

原因：

1. LLVM dialect 虽然有 `llvm.fence`
2. 但当前 Vortex 后端没有现成 `vx_fence`
3. 贸然 lower 进去，语义未必对

---

## 7. 推荐的下一步实现顺序

建议按这个顺序做：

1. 先加一个小的 pipeline 文档或注册函数，明确 LLVM lowering 顺序
2. 实现 `PrepareVortexToLLVMPass`
   1. 先只做 `lower-affine`
   2. 再决定 `vortex.launch` 的 inline/擦除方式
3. 实现 `ConvertVortexToLLVMPass`
   1. 先做 `core_id / subgroup_id / thread_id`
   2. 再做 `barrier <core>`
   3. 最后做 `local_alloc`
4. 接标准 LLVM conversion：
   1. `convert-arith-to-llvm`
   2. `convert-index-to-llvm`
   3. `convert-scf-to-cf`
   4. `convert-cf-to-llvm`
   5. `expand-strided-metadata`
   6. `finalize-memref-to-llvm`
   7. `convert-func-to-llvm`
   8. `reconcile-unrealized-casts`

---

## 8. MVP 级判断

如果只看 MVP，当前最稳的判断是：

1. `affine.apply` 直接交给 `lower-affine`
2. 地址空间 attr 映射可以借上游 GPU 的 type-attribute conversion 模式
3. `barrier` 和执行 id 不要硬接通用 LLVM 语义，直接接当前 `vx_*` 入口
4. `fence` 先延期
5. `local_alloc` 是真正需要重点设计的一步
