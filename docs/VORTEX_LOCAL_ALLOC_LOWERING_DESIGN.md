# `vortex.local_alloc` Lowering 设计文档

## 1. 目标

本文档明确 `vortex.local_alloc` 的推荐 lowering 方案，目标是把当前
Vortex dialect 里的：

1. `vortex.local_alloc`
2. 对 local buffer 的 `memref.load/store`
3. `global <-> local` 的 `memref.copy`

稳定地接到 LLVM 后端，同时避免把当前语义过早绑死在一个并不成立的
“LLVM address space 自动代表 `VX_local_mem`”假设上。

---

## 2. 当前已知约束

### 2.1 `vortex.local_alloc` 的当前语义是 `per-core local memory`

当前仓库里，`vortex.local_alloc` 的定义是：

1. 分配 `per-core local memory`
2. 结果类型必须是 `#vortex.address_space<local>`

这和 `workgroup shared memory` 不是同一个抽象。

### 2.2 现有软件栈访问 local memory 的方式是“显式基址 + 显式地址计算”

当前 Vortex 软件接口里，`__local_mem(size)` 的实际展开是：

```c
#define __local_mem(size) \
  (void*)((int8_t*)csr_read(VX_CSR_LOCAL_MEM_BASE) + __local_group_id * size)
```

也就是说，当前 local memory 访问更接近：

1. 先读 `VX_CSR_LOCAL_MEM_BASE`
2. 再做字节地址计算

而不是“拿一个 LLVM local addrspace pointer，后端自动帮我们做对”。

### 2.3 `__local_mem(size)` 绑定的是 `vx_spawn` 的 group slot 语义

`__local_group_id` 在 `vx_spawn` 里是运行时算出来的：

1. `local_group_id = warp_id / warps_per_group`
2. 再把它写到 `__local_group_id`

因此：

1. `__local_mem(size)` 的 `size` 是“每个 group slot 的 local frame 大小”
2. 它隐含了“一个 core 上可以有多个 group 并发驻留”的运行时语义

这和我们当前 MLIR 里定义的 `per-core local_alloc` 并不完全等价。

### 2.4 仅靠 LLVM address space 目前没有足够证据成立

当前已有分析已经确认：

1. 默认 `LLVMTypeConverter` 只会处理整数 memory space
2. 当前 Vortex/RISCV 后端没有证据表明“把 local memory space 变成 LLVM addrspace”就能自动映射到 `VX_local_mem`

因此：

1. `#vortex.address_space<local>` 在 MLIR 里保留是合理的
2. 但不能把最终实现押宝在“纯 address space lowering”上

---

## 3. 推荐方案

### 3.1 总体决策

推荐采用：

1. `per-core local frame`
2. `explicit base + explicit byte offset`
3. `direct local access lowering`

换句话说：

1. 编译器先为每个 kernel 的所有 `vortex.local_alloc` 规划一个 local frame
2. 每个 alloc 拿到一个固定 byte offset
3. local 访问不再试图维持为“普通 memref alloc”
4. 而是改写成“local base + offset + index linearization”后的显式访问

### 3.2 为什么不直接复用 `__local_mem(size)`

第一版不建议直接把 `vortex.local_alloc` lower 成 `__local_mem(size)` 风格，
原因有两个：

1. 它绑定了 `__local_group_id / warps_per_group / groups_per_core` 的运行时 group slot 语义
2. 当前 `vortex.local_alloc` 的语义是 `per-core local memory`，不是 `per-group slot local frame`

所以更稳妥的编译器语义应是：

1. 先按 `per-core` 做 frame layout
2. 后面如果运行时 ABI 确认要支持多 group/core 并发驻留，再把“slot 复制/偏移”这一层显式并入 ABI

### 3.3 为什么不直接走“memref descriptor + finalize-memref-to-llvm”

这条路理论上更整洁，但当前第一版不推荐作为 MVP 主路径。

原因：

1. 标准 memref dialect 没有现成的“从 `VX_CSR_LOCAL_MEM_BASE + offset` 直接构造 local memref descriptor”的稳定上层表达
2. 为了保住纯 memref 形式，反而容易引入额外的桥接 op 或 runtime wrapper
3. 当前最直接、最容易验证的路径，其实是把 local access 直接 lower 成显式地址计算

因此建议：

1. 长期可以继续评估 descriptor 化
2. MVP 先采用“直接 local access lowering”

---

## 4. 建议的 pass 拆分

### 4.1 `VortexPlanLocalMemoryLayoutPass`

输入：

1. `vortex.local_alloc`
2. 当前 `LowerVortexRuntimeBuiltins` 前后的混合 IR

职责：

1. 收集 kernel 内所有 `vortex.local_alloc`
2. 计算每个 alloc 的：
   1. byte size
   2. alignment
   3. byte offset
3. 给 kernel 附加：
   1. `vortex.local_frame_bytes`
4. 给每个 `vortex.local_alloc` 附加：
   1. `vortex.local.byte_offset`
   2. `vortex.local.byte_size`
   3. `vortex.local.alignment`

第一版建议限制：

1. 只支持 static-shaped `vortex.local_alloc`
2. 只支持结果不逃逸
3. 禁止通过 `yield/branch/iter_args/call` 传递 local memref

### 4.2 `VortexLowerLocalMemoryPass`

职责：

1. 获取 local memory base
2. 把 local alloc/use 全部改写成显式地址访问
3. 擦除 `vortex.local_alloc`

推荐第一版支持的 use 形态：

1. `memref.load` from local
2. `memref.store` to local
3. `memref.copy global -> local`
4. `memref.copy local -> global`

第一版建议拒绝：

1. local memref 上的动态 `subview`
2. local memref 逃逸到 call
3. local memref 进入复杂 region 结果通道
4. tensor 语义

### 4.3 `LowerVortexRuntimeBuiltinsPass`

当前已经做掉：

1. `core_id / subgroup_id / thread_id`
2. `barrier<core>`

在 `local_alloc` 路线上，这个 pass 后续只负责：

1. 把 local lowering 里残留的 wrapper / LLVM helper 接到最终 LLVM dialect

---

## 5. 推荐的 ABI 约定

### 5.1 kernel 属性

建议给 kernel 增加：

1. `vortex.local_frame_bytes = <i64>`

含义：

1. 单个 kernel 实例、单个 core 所需的 local frame 总字节数

### 5.2 local base 获取

推荐优先新增一个明确入口：

```c
uintptr_t vx_local_mem_base();
```

语义：

1. 返回当前 core 可见的 `VX_local_mem` 基址

注意：

1. 这个 `vx_local_mem_base()` 目前还不是现有软件栈里已经提供的接口
2. 它是为 `vortex.local_alloc` lowering 建议补充的新 ABI

如果短期不补这个入口，MVP 备选方案是：

1. 在 LLVM dialect 里直接生成读取 `VX_CSR_LOCAL_MEM_BASE` 的 target-specific inline asm

但这条路更硬编码，也更难维护，因此只建议作为临时兜底。

### 5.3 host/runtime 约束

当前 runtime 已经有：

1. `VX_CAPS_LOCAL_MEM_SIZE`

所以编译器最终需要让 launch/host 侧知道：

1. `vortex.local_frame_bytes`

从而在启动 kernel 前做容量检查。

---

## 6. 推荐的 lowering 细节

### 6.1 local frame layout

对每个 `vortex.local_alloc`：

1. `byte_size = num_elements * element_byte_width`
2. `alignment = max(element_byte_width, explicit alignment if any)`
3. 按顺序做对齐分配，得到 `byte_offset`

### 6.2 local load/store

对：

```mlir
%v = memref.load %buf[%i0, %i1, ...]
memref.store %x, %buf[%i0, %i1, ...]
```

若 `%buf` 追溯到某个 `vortex.local_alloc`，则改写为：

1. 线性化下标
2. 乘 element byte width
3. 加上 alloc 的 `byte_offset`
4. 再加上 `local_mem_base`
5. 生成最终地址
6. 用目标低层 `load/store` 访问

### 6.3 local copy

对：

```mlir
memref.copy %global, %local
memref.copy %local, %global
```

第一版不保留 `memref.copy` 到最终 LLVM，而是先展开成显式 loop nest：

1. `global -> local`：`memref.load global` + local store
2. `local -> global`：local load + `memref.store global`

这样 local lowering 只需要处理：

1. local load
2. local store

不必再额外支持一个“special copy op”。

---

## 7. IR 形态示例

下面的 IR 主要用于说明“目标形态”，不是当前仓库已经完全实现的输出。

### 7.1 lowering 前

```mlir
module {
  func.func @kernel(%src: memref<16xf32>, %dst: memref<16xf32>)
      attributes {vortex.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %tid = call @vx_thread_id() : () -> i32
    %tid_idx = arith.index_cast %tid : i32 to index

    %tile = vortex.local_alloc() : memref<16xf32, #vortex.address_space<local>>

    scf.for %i = %c0 to %c16 step %c1 {
      %v = memref.load %src[%i] : memref<16xf32>
      memref.store %v, %tile[%i] : memref<16xf32, #vortex.address_space<local>>
    }

    %zero = arith.constant 0 : i32
    %nw = call @vx_num_warps() : () -> i32
    call @vx_barrier(%zero, %nw) : (i32, i32) -> ()

    %x = memref.load %tile[%tid_idx] : memref<16xf32, #vortex.address_space<local>>
    memref.store %x, %dst[%tid_idx] : memref<16xf32>
    return
  }
}
```

### 7.2 layout planning 后

```mlir
module {
  func.func @kernel(%src: memref<16xf32>, %dst: memref<16xf32>)
      attributes {vortex.kernel, vortex.local_frame_bytes = 64 : i64} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %tid = call @vx_thread_id() : () -> i32
    %tid_idx = arith.index_cast %tid : i32 to index

    %tile = vortex.local_alloc()
        {vortex.local.byte_offset = 0 : i64,
         vortex.local.byte_size = 64 : i64,
         vortex.local.alignment = 4 : i64}
        : memref<16xf32, #vortex.address_space<local>>

    scf.for %i = %c0 to %c16 step %c1 {
      %v = memref.load %src[%i] : memref<16xf32>
      memref.store %v, %tile[%i] : memref<16xf32, #vortex.address_space<local>>
    }

    %zero = arith.constant 0 : i32
    %nw = call @vx_num_warps() : () -> i32
    call @vx_barrier(%zero, %nw) : (i32, i32) -> ()

    %x = memref.load %tile[%tid_idx] : memref<16xf32, #vortex.address_space<local>>
    memref.store %x, %dst[%tid_idx] : memref<16xf32>
    return
  }
}
```

### 7.3 local lowering 后的目标形态

这里示意的是推荐的 MVP 输出形态：

1. global memref 仍然保持标准 `memref.*`
2. local access 直接变成显式地址计算
3. `vortex.local_alloc` 被擦除

```mlir
module {
  // 注意：这里的 @vx_local_mem_base 是“推荐新增的 wrapper ABI”，
  // 当前仓库/运行时里还没有现成实现。
  func.func private @vx_local_mem_base() -> i64

  func.func @kernel(%src: memref<16xf32>, %dst: memref<16xf32>)
      attributes {vortex.kernel, vortex.local_frame_bytes = 64 : i64} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c4_i64 = arith.constant 4 : i64
    %tid = call @vx_thread_id() : () -> i32
    %tid_idx = arith.index_cast %tid : i32 to index
    %lbase = call @vx_local_mem_base() : () -> i64

    scf.for %i = %c0 to %c16 step %c1 {
      %v = memref.load %src[%i] : memref<16xf32>
      %i_i64 = arith.index_cast %i : index to i64
      %byte = arith.muli %i_i64, %c4_i64 : i64
      %addr = arith.addi %lbase, %byte : i64
      %ptr = llvm.inttoptr %addr : i64 to !llvm.ptr
      llvm.store %v, %ptr : f32, !llvm.ptr
    }

    %zero = arith.constant 0 : i32
    %nw = call @vx_num_warps() : () -> i32
    call @vx_barrier(%zero, %nw) : (i32, i32) -> ()

    %tid_i64 = arith.index_cast %tid_idx : index to i64
    %byte2 = arith.muli %tid_i64, %c4_i64 : i64
    %addr2 = arith.addi %lbase, %byte2 : i64
    %ptr2 = llvm.inttoptr %addr2 : i64 to !llvm.ptr
    %x = llvm.load %ptr2 : !llvm.ptr -> f32
    memref.store %x, %dst[%tid_idx] : memref<16xf32>
    return
  }
}
```

---

## 8. 可行性评估

### 8.1 编译器侧

可行性判断：

1. 中等偏高

原因：

1. local frame layout 本身不复杂
2. 当前 promotion pass 产出的 local use 形态已经比较收敛
3. 第一版如果只接受 static shape + 不逃逸，问题边界是可控的

### 8.2 后端侧

可行性判断：

1. 中等

主要前提：

1. 需要一个稳定的 `local_mem_base` 入口
2. 或接受短期用 inline asm 直接读 CSR

### 8.3 runtime/host 侧

可行性判断：

1. 中等

关键在于：

1. 要让 launch/host 侧拿到 `local_frame_bytes`
2. 然后用 `VX_CAPS_LOCAL_MEM_SIZE` 做检查

---

## 9. 主要风险

### 风险 1：`local_mem_base` 入口缺失

这是当前最大的工程风险。

如果没有：

1. `vx_local_mem_base()`
2. 或等价 intrinsic

那么 compiler side 只能直接生成 target-specific inline asm。

### 风险 2：当前 runtime 的 occupancy 语义与 `per-core local frame` 语义还没有 ABI 对齐

当前 runtime 的 local memory 容量检查是围绕：

1. `group_size`
2. `warps_per_group`
3. `groups_per_core`

展开的。

如果未来 kernel ABI 明确支持“一核并发多个 group”，那么：

1. `vortex.local_frame_bytes` 的解释
2. local base 偏移公式
3. host 侧容量检查

都需要重新收紧。

### 风险 3：支持范围一旦放宽，复杂度会迅速上升

尤其是：

1. dynamic shape local alloc
2. local `subview`
3. local memref 逃逸
4. vector/tensor 语义

因此第一版必须强约束输入形态。

---

## 10. 最终建议

建议先按下面的顺序推进：

1. 先把本方案作为 `local_alloc` 的正式设计边界
2. 先实现 `VortexPlanLocalMemoryLayoutPass`
3. 再实现 `VortexLowerLocalMemoryPass`
4. `LowerVortexRuntimeBuiltinsPass` 继续只负责执行 id / barrier 和少量 wrapper 接口
5. 等 `local_mem_base` ABI 定下来后，再真正接通 end-to-end

一句话总结：

`vortex.local_alloc` 的 MVP 正确方向，不是“纯 address space lowering”，
而是“per-core local frame + explicit base/offset + direct local access lowering”。
