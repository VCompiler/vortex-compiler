# 当前 Vortex Compiler Pass 与执行流程梳理

## 1. 文档目标

这份文档回答 4 个问题：

1. 当前 `vortex-compiler` 里到底开发了哪些 pass
2. 这些 pass 分别做什么，作用在哪一层 IR
3. 现在仓库里注册了哪些 pipeline
4. 当前真正跑通到 `ELF/bin`、并已完成上板验证的流程，实际走了哪些 pass

这份文档描述的是“当前代码实际状态”，不是未来规划。

---

## 2. 总体分层

当前仓库里的 lowering 分层可以概括为：

```text
前端输入
  -> pre-vortex
     仍以标准 MLIR dialect 为主
  -> Vortex 结构化语义
     表达 kernel、执行层级、local memory、barrier、地址空间
  -> LLVM dialect
  -> LLVM IR
  -> Vortex LLVM backend
  -> ELF / bin
```

这里要特别区分两件事：

1. `vx-opt` 负责 MLIR pass 和 MLIR dialect lowering
2. 真正把 LLVM IR 变成 Vortex/RISCV 目标机器码的是 `third_party/llvm` 里的 Vortex LLVM backend，也就是后面的 `clang -Xclang +vortex`

---

## 3. 入口与注册

### 3.1 `vx-opt` 入口

`vx-opt` 的入口在：

- `tools/vx-opt/vx-opt.cpp`

它做了三件事：

1. 注册上游 MLIR 的通用 pass
2. 调用 `registerVortexPassesAndPipelines()`
3. 注册 Vortex 方言和当前会用到的标准方言

### 3.2 Vortex pass/pipeline 总注册

总注册入口在：

- `lib/InitAllPasses.cpp`

当前注册了：

1. 所有自研 Vortex pass
2. `vortex-onnx-matmul-to-pre-vortex-pipeline`
3. `vortex-pre-vortex-pipeline`
4. `vortex-mvp-backend-pipeline`

### 3.3 Pass 定义表

所有 pass 的命令行名字、简介、选项都定义在：

- `include/vortex/Transforms/Passes.td`

### 3.4 Vortex 方言

Vortex dialect 的核心 op/attr 在：

- `include/vortex/Dialect/Vortex/IR/VortexOps.td`
- `include/vortex/Dialect/Vortex/IR/VortexAttributes.td`

当前后半段 pass 主要围绕这些 Vortex op 工作：

1. `vortex.launch`
2. `vortex.core_id`
3. `vortex.subgroup_id`
4. `vortex.thread_id`
5. `vortex.local_alloc`
6. `vortex.barrier`
7. `vortex.fence`
8. `vortex.yield`

---

## 4. 当前已实现的 Pass 清单

当前仓库里一共实现了 14 个 Vortex pass。

### 4.1 Pre-Vortex 边界类

#### `vortex-normalize-onnx-frontend`

实现文件：

- `lib/Transforms/NormalizeONNXFrontendPass.cpp`

作用：

1. 把顶层 `"onnx.EntryPoint"` 转成目标函数上的临时 `vortex.entry`
2. 删除 `"onnx.EntryPoint"`
3. 清掉 `onnx.*` 和 `onnx-mlir.*` 属性

这是一个前端桥接 pass，本身不做 ONNX 算子 lowering。

#### `vortex-tile-matmul-for-pre-vortex`

实现文件：

- `lib/Transforms/TileMatmulForPreVortexPass.cpp`

作用：

1. 识别当前支持的窄子集 `linalg.fill + linalg.matmul`
2. 改写成 tiled 结构
3. 生成：
   `scf.for + memref.subview + tiled linalg.fill + tiled linalg.matmul`

注意：

1. 它保留 `linalg.matmul`
2. 它还没有进入 Vortex op 层
3. 它的职责是把前端输出规整到“当前后半段容易接”的 pre-vortex 形状

#### `vortex-validate-pre-vortex`

实现文件：

- `lib/Transforms/ValidatePreVortexPass.cpp`

作用：

1. 检查函数体里的 dialect 是否仍在允许的 pre-vortex 集合内
2. 当前允许的主要是：
   `affine/arith/bufferization/builtin/cf/func/linalg/math/memref/scf/tensor/vector`

它的作用是明确 pre-vortex 边界，避免低层或未知 dialect 提前混进来。

#### `vortex-summarize-pre-vortex`

实现文件：

- `lib/Transforms/SummarizePreVortexPass.cpp`

作用：

1. 统计函数里出现过的 op 名字
2. 统计出现过的 dialect
3. 统计接口和函数体中出现过的 memory space
4. 写回函数属性：
   - `vortex.pre_vortex_ops`
   - `vortex.pre_vortex_dialects`
   - `vortex.pre_vortex_memory_spaces`

它是一个“观测/摘要”pass，不改变核心语义。

### 4.2 Kernel 标记与地址空间类

#### `vortex-mark-kernel`

实现文件：

- `lib/Transforms/MarkVortexKernelPass.cpp`

作用：

1. 把函数标记为 `vortex.kernel`
2. 标记来源有两个：
   - 通过命令行 `kernel-name=...`
   - 函数上已有临时 `vortex.entry`
3. 可选移除 `vortex.entry`

它只负责“谁是 kernel”，不负责改函数体。

#### `vortex-materialize-address-spaces`

实现文件：

- `lib/Transforms/MaterializeVortexAddressSpacesPass.cpp`

作用：

1. 对已标记为 `vortex.kernel` 的函数参数
2. 把默认 memref memory space 显式补成 `#vortex.address_space<global>`
3. 同步修正以这些参数为源的 `memref.subview` / `memref.cast` 结果类型

这个 pass 很关键，因为后续 local/global 判定依赖显式地址空间。

### 4.3 执行映射与 local memory 结构化类

#### `vortex-map-parallel-loops-to-launch`

实现文件：

- `lib/Transforms/MapParallelLoopsToVortexLaunchPass.cpp`

作用：

1. 识别带 `vortex.mapping = "core" | "subgroup" | "thread"` 的 `scf.for` perfect nest
2. 约束：
   - lower bound 必须是 `0`
   - step 必须是 `1`
   - 不允许 iter_args / results
   - 必须是 perfect nest
   - 维度顺序必须是 `core -> subgroup -> thread`
3. 生成 `vortex.launch`
4. 把循环 IV 改写成：
   - `vortex.core_id`
   - `vortex.subgroup_id`
   - `vortex.thread_id`

它是把“显式映射的循环”变成“Vortex 执行区域”的 pass。

#### `vortex-promote-tiles-to-local`

实现文件：

- `lib/Transforms/PromoteTilesToVortexLocalPass.cpp`

作用：

1. 识别带 `vortex.promote_to_local` 的 `memref.subview`
2. 要求它在 `vortex.launch` 里
3. 把这个 tile 提升成 `vortex.local_alloc`
4. 插入：
   - `memref.copy global -> local`
   - 可选 `memref.copy local -> global`
5. 如果 tile 会被写，必须带 `vortex.write_back`

当前实现是非常保守的 promotion，不做自动分析。

#### `vortex-insert-barriers`

实现文件：

- `lib/Transforms/InsertVortexBarriersPass.cpp`

作用：

1. 扫描 `vortex.launch` 里的 local tile 使用模式
2. 识别：
   - copy-in
   - local use
   - 可选 copy-out
3. 在必要位置插入 `vortex.barrier <core>`
   - copy-in 之后、首次 local use 之前
   - 最后一次 local use 之后、copy-out 之前

当前只覆盖很窄的模式。

#### `vortex-plan-local-memory-layout`

实现文件：

- `lib/Transforms/PlanVortexLocalMemoryLayoutPass.cpp`

作用：

1. 给每个 `vortex.local_alloc` 规划 frame layout
2. 计算：
   - byte offset
   - byte size
   - alignment
3. 写回属性：
   - 函数上：`vortex.local_frame_bytes`
   - alloc 上：
     - `vortex.local.byte_offset`
     - `vortex.local.byte_size`
     - `vortex.local.alignment`

注意：

1. 它只规划元数据
2. 还不做真正的地址化 lowering

#### `vortex-lower-local-memory`

实现文件：

- `lib/Transforms/LowerVortexLocalMemoryPass.cpp`

作用：

1. 依赖 `vortex-plan-local-memory-layout` 的 metadata
2. 只接受 rooted at `vortex.local_alloc` 的 local memref
3. 做 4 类 lowering：
   - 把 local 相关 `memref.copy` 展成 `scf.for`
   - 把 local `memref.load` 降成 `llvm.inttoptr + llvm.load`
   - 把 local `memref.store` 降成 `llvm.inttoptr + llvm.store`
   - 把 `memref.subview` 的索引回溯到 root `vortex.local_alloc`
4. 最后擦掉 dead 的：
   - `vortex.local_alloc`
   - `memref.cast`
   - `memref.subview`

这条 pass 是 local memory 从“结构化抽象”进入“显式地址访问”的关键分水岭。

### 4.4 计算体与 LLVM 边界类

#### `vortex-lower-linalg-inside-kernel`

实现文件：

- `lib/Transforms/LowerLinalgInsideVortexKernelPass.cpp`

作用：

1. 只处理已标记为 `vortex.kernel` 的函数
2. 把剩余 buffer-semantics `linalg.*` 用上游 `linalgOpToLoops` 打散
3. 变成：
   - `scf.for`
   - `memref.load`
   - `memref.store`
   - `arith.*`

当前板测主线大量依赖它，因为 `matmul4x4` 现在就是直接靠它把 `linalg.matmul` 打散掉。

#### `vortex-legalize-for-llvm`

实现文件：

- `lib/Transforms/PrepareVortexToLLVMPass.cpp`

作用：

1. 先对所有 `func.func` 跑标准 `lower-affine`
2. 对 kernel 内的 `vortex.launch` 做 inline
3. 把 `#vortex.address_space<global/private>` 改成 LLVM type converter 可接受的数值 memory space
4. 严格拒绝残留：
   - `vortex.local_alloc`
   - `#vortex.address_space<local>`
   - `vortex.launch`
   - `vortex.yield`

它是进入 LLVM dialect 前的边界清理 pass。

#### `vortex-lower-runtime-builtins`

实现文件：

- `lib/Transforms/ConvertVortexToLLVMPass.cpp`

作用：

1. 把 runtime-facing 的 Vortex op 改成普通 wrapper 调用：
   - `vortex.core_id -> vx_core_id()`
   - `vortex.subgroup_id -> vx_warp_id()`
   - `vortex.thread_id -> vx_thread_id()`
   - `vortex.barrier<core> -> vx_barrier(0, vx_num_warps())`
2. wrapper 返回 `i32`，再 cast 回 `index`
3. 把 `vortex.kernel` 改成 `vortex.kernel_entry`

这个 pass 之后，IR 基本就只剩标准 dialect 和 wrapper call 了。

---

## 5. 当前注册的 3 条命名 Pipeline

定义文件：

- `lib/Pipeline/Pipelines.cpp`

### 5.1 `vortex-pre-vortex-pipeline`

顺序：

```text
canonicalize
-> cse
-> func.func(vortex-validate-pre-vortex)
-> func.func(vortex-summarize-pre-vortex)
```

作用：

1. 规范 pre-vortex IR
2. 校验边界
3. 给函数挂摘要

### 5.2 `vortex-onnx-matmul-to-pre-vortex-pipeline`

顺序：

```text
buffer-results-to-out-params
-> canonicalize
-> cse
-> vortex-normalize-onnx-frontend
-> func.func(vortex-tile-matmul-for-pre-vortex)
-> vortex-pre-vortex-pipeline
```

作用：

1. 接 ONNX-MLIR 的窄前端路径
2. 规整成当前项目定义的 pre-vortex 形状

### 5.3 `vortex-mvp-backend-pipeline`

顺序：

```text
canonicalize
-> cse
-> vortex-legalize-for-llvm
-> vortex-lower-runtime-builtins
-> canonicalize
-> cse
-> convert-scf-to-cf
-> convert-arith-to-llvm
-> convert-index-to-llvm
-> finalize-memref-to-llvm
-> convert-func-to-llvm
-> convert-cf-to-llvm
-> reconcile-unrealized-casts
```

作用：

1. 把“已经不再含 local-memory 抽象”的 Vortex IR
2. 收口到 LLVM dialect

注意：

1. 这条 pipeline 默认并不包含
   `map/promote/barrier/plan-local/lower-local/lower-linalg`
2. 它假定输入已经满足后半段边界条件

---

## 6. 当前实际会走的几条执行路径

这是理解当前项目状态最关键的一节。

### 6.1 `build-vortex-kernel.sh` 的默认 pipeline

默认值在：

- `scripts/build-vortex-kernel.sh`

默认 `PASS_PIPELINE` 是：

```text
builtin.module(vortex-mvp-backend-pipeline)
```

也就是说，如果用户不显式传 `--pass-pipeline`，默认只跑命名的 backend pipeline。

这要求输入已经是合适的“后半段 IR”。

### 6.2 当前 `matmul4x4` smoke / 板测主线

当前真正跑通到 `ELF/bin` 并完成上板验证的主线，在：

- `scripts/run-matmul4x4-smoke.sh`

它显式指定的 pipeline 是：

```text
builtin.module(
  func.func(
    vortex-mark-kernel{remove-entry-attr=1},
    vortex-materialize-address-spaces,
    vortex-lower-linalg-inside-kernel
  ),
  canonicalize,
  cse,
  vortex-legalize-for-llvm,
  vortex-lower-runtime-builtins,
  canonicalize,
  cse,
  convert-scf-to-cf,
  convert-arith-to-llvm,
  convert-index-to-llvm,
  finalize-memref-to-llvm,
  convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},
  convert-cf-to-llvm,
  reconcile-unrealized-casts
)
```

这意味着当前板测主线真正用到的自研 pass 只有 5 个：

1. `vortex-mark-kernel`
2. `vortex-materialize-address-spaces`
3. `vortex-lower-linalg-inside-kernel`
4. `vortex-legalize-for-llvm`
5. `vortex-lower-runtime-builtins`

没有进入当前板测主线的 pass：

1. `vortex-map-parallel-loops-to-launch`
2. `vortex-promote-tiles-to-local`
3. `vortex-insert-barriers`
4. `vortex-plan-local-memory-layout`
5. `vortex-lower-local-memory`

原因不是这些 pass 没实现，而是当前 `matmul4x4` 输入 IR 还没有先被构造成需要它们接手的形状。

### 6.3 当前 ONNX smoke 路线

入口：

- `scripts/run-onnx-matmul4x4-smoke.sh`

它是两段式：

第一段：

1. `onnx-mlir`
2. `onnx-mlir-opt`
3. `vx-opt --pass-pipeline='builtin.module(vortex-onnx-matmul-to-pre-vortex-pipeline{tile-size=...})'`

第二段：

1. 把生成的 `pre_vortex.mlir`
2. 再送进和 bare `matmul4x4` 类似的 backend pipeline

所以 ONNX 路径会额外使用：

1. `vortex-normalize-onnx-frontend`
2. `vortex-tile-matmul-for-pre-vortex`
3. `vortex-validate-pre-vortex`
4. `vortex-summarize-pre-vortex`

### 6.4 仓库里已有的“更完整 full-chain”示例

测试文件：

- `test/Pipeline/onnx-matmul-full-chain.mlir`

这条测试展示了一条更完整的链：

```text
vortex-onnx-matmul-to-pre-vortex-pipeline
-> func.func(
     vortex-mark-kernel,
     vortex-materialize-address-spaces,
     vortex-map-parallel-loops-to-launch,
     vortex-promote-tiles-to-local,
     vortex-insert-barriers,
     vortex-lower-linalg-inside-kernel
   )
-> vortex-mvp-backend-pipeline
```

注意：

1. 这是仓库里已经能表达、也有测试覆盖的“更完整 pass 串联”
2. 但它不是当前 `run-matmul4x4-smoke.sh` 的默认主线
3. 也不是当前板测命令实际采用的那条最小路径

---

## 7. 当前从 MLIR 到 ELF/bin 的完整执行顺序

以当前已跑通的 `matmul4x4` 板测主线为例，完整链路是：

```text
输入 MLIR
-> vx-opt 跑显式 pass pipeline
-> 产出 LLVM dialect MLIR
-> mlir-translate -mlir-to-llvmir
-> 产出 .ll
-> clang -Xclang +vortex
-> 产出 .s / .o / .elf
-> llvm-objcopy
-> 产出 .bin
```

对应脚本关系：

1. `scripts/run-matmul4x4-smoke.sh`
2. `scripts/build-vortex-kernel.sh`
3. `vx-opt`
4. `mlir-translate`
5. `third_party/llvm-vortex-build/bin/clang`
6. `llvm-objcopy`

其中真正进入 Vortex 专用 LLVM backend 的点是：

```text
clang -Xclang +vortex
```

也就是：

1. `vx-opt` 负责把 MLIR 降到 LLVM dialect / LLVM IR
2. `third_party/llvm` 里的 Vortex LLVM backend 负责把 `.ll` 变成目标机器码

---

## 8. 当前板测已验证通过的主线范围

当前已实跑验证通过的是：

1. 本地 `third_party/llvm` 构建出的 Vortex LLVM toolchain
2. `build-thirdparty-llvm/bin/vx-opt`
3. `matmul4x4` 经过当前 MVP 主线生成 `.elf/.bin`
4. 生成的 ELF 通过板端 JTAG runner 上板
5. 板测判定通过

这条已验证主线覆盖的是：

```text
vortex-mark-kernel
-> vortex-materialize-address-spaces
-> vortex-lower-linalg-inside-kernel
-> vortex-legalize-for-llvm
-> vortex-lower-runtime-builtins
-> upstream LLVM dialect conversion
-> mlir-translate
-> Vortex LLVM backend
-> ELF/bin
-> board run
```

所以从“当前真实能工作”的角度看，最应该先掌握的是这 5 个 pass。

---

## 9. 当前状态的一句话总结

当前 `vortex-compiler` 里已经具备两层能力：

1. 一层是“完整设计方向”的 pass 库：
   `mark -> address-space -> launch -> local promotion -> barrier -> local layout -> local lowering -> legalize -> runtime builtins`
2. 另一层是“当前实际用于出 ELF 并完成板测”的 MVP 主线：
   `mark -> address-space -> lower-linalg -> legalize -> runtime builtins`

也就是说：

1. 不是所有已实现 pass 都已经进入当前默认板测主线
2. 但完整 pass 库已经基本具备
3. 当前真正稳定打通的是一个更窄、但已经能生成并上板执行的后端子集

---

## 10. 每个阶段输入 IR / 输出 IR 长什么样

这一节不追求贴完整文件，而是抓每个阶段最关键的结构变化。

重点看 4 件事：

1. 函数属性怎么变
2. 类型，尤其是 memory space，怎么变
3. 结构化 op 怎么一步步被打散
4. 从 MLIR 到 LLVM IR，再到目标代码时，文本形态怎么变化

### 10.1 当前板测主线的逐阶段 IR 示例

当前板测主线就是 `scripts/run-matmul4x4-smoke.sh` 里的这条链：

```text
func.func(
  vortex-mark-kernel,
  vortex-materialize-address-spaces,
  vortex-lower-linalg-inside-kernel
)
-> vortex-legalize-for-llvm
-> vortex-lower-runtime-builtins
-> convert-* to LLVM dialect
-> mlir-translate
-> clang -Xclang +vortex
```

#### 阶段 0：原始输入 MLIR

文件：

- `examples/smoke/matmul4x4_f32.mlir`

这时还是很“前端态”的 MLIR：

```mlir
module {
  func.func @matmul4x4(%a: memref<4x4xf32>, %b: memref<4x4xf32>,
                       %c: memref<4x4xf32>) attributes {vortex.entry} {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<4x4xf32>)
    linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>)
        outs(%c : memref<4x4xf32>)
    return
  }
}
```

此时的特征是：

1. 入口函数靠 `vortex.entry` 暂时标识
2. 参数还是默认 memref，没有显式 Vortex address space
3. 计算主体还是高层 `linalg.fill + linalg.matmul`

#### 阶段 1：`vortex-mark-kernel + vortex-materialize-address-spaces`

这一步之后，kernel 身份和 global address space 被显式化：

```mlir
module {
  func.func @matmul4x4(
      %arg0: memref<4x4xf32, #vortex.address_space<global>>,
      %arg1: memref<4x4xf32, #vortex.address_space<global>>,
      %arg2: memref<4x4xf32, #vortex.address_space<global>>)
      attributes {vortex.kernel} {
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32)
      outs(%arg2 : memref<4x4xf32, #vortex.address_space<global>>)
    linalg.matmul
      ins(%arg0, %arg1 : memref<4x4xf32, #vortex.address_space<global>>,
                         memref<4x4xf32, #vortex.address_space<global>>)
      outs(%arg2 : memref<4x4xf32, #vortex.address_space<global>>)
    return
  }
}
```

这一阶段最关键的变化只有两个：

1. `vortex.entry -> vortex.kernel`
2. `memref<...> -> memref<..., #vortex.address_space<global>>`

注意这里 `linalg` 还在，说明这一步只是“标记 kernel + 补地址空间”，还没有真正打散计算体。

#### 阶段 2：再经过 `vortex-lower-linalg-inside-kernel`

这一阶段开始把 `linalg` 打散成循环和显式访存：

```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @matmul4x4(
      %arg0: memref<4x4xf32, #vortex.address_space<global>>,
      %arg1: memref<4x4xf32, #vortex.address_space<global>>,
      %arg2: memref<4x4xf32, #vortex.address_space<global>>)
      attributes {vortex.kernel} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        memref.store %cst, %arg2[%i, %j] :
          memref<4x4xf32, #vortex.address_space<global>>
      }
    }

    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        scf.for %k = %c0 to %c4 step %c1 {
          %a = memref.load %arg0[%i, %k] :
            memref<4x4xf32, #vortex.address_space<global>>
          %b = memref.load %arg1[%k, %j] :
            memref<4x4xf32, #vortex.address_space<global>>
          %c = memref.load %arg2[%i, %j] :
            memref<4x4xf32, #vortex.address_space<global>>
          %mul = arith.mulf %a, %b : f32
          %sum = arith.addf %c, %mul : f32
          memref.store %sum, %arg2[%i, %j] :
            memref<4x4xf32, #vortex.address_space<global>>
        }
      }
    }
    return
  }
}
```

这一阶段的本质是：

1. `linalg.fill` 变成初始化循环
2. `linalg.matmul` 变成三层 `scf.for`
3. 数据流变成显式 `memref.load/store + arith`
4. Vortex 的 global address space 仍然保留

所以从这一步开始，IR 已经很接近后端能接受的“显式访存循环程序”。

#### 阶段 3：再经过 `vortex-legalize-for-llvm + vortex-lower-runtime-builtins`

这一步是进入 LLVM dialect 之前的最后一层“Vortex 语义清场”：

```mlir
module {
  func.func @matmul4x4(%arg0: memref<4x4xf32>,
                       %arg1: memref<4x4xf32>,
                       %arg2: memref<4x4xf32>)
      attributes {vortex.kernel_entry} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        memref.store %cst, %arg2[%i, %j] : memref<4x4xf32>
      }
    }
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        scf.for %k = %c0 to %c4 step %c1 {
          %0 = memref.load %arg0[%i, %k] : memref<4x4xf32>
          %1 = memref.load %arg1[%k, %j] : memref<4x4xf32>
          %2 = memref.load %arg2[%i, %j] : memref<4x4xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg2[%i, %j] : memref<4x4xf32>
        }
      }
    }
    return
  }
}
```

这里有两个最关键的变化：

1. `vortex.kernel -> vortex.kernel_entry`
2. `#vortex.address_space<global>` 被改回 LLVM lowering 可接受的普通 memref 形状

对于当前 `matmul4x4` 主线来说，这一步里没有 `vortex.launch`、`vortex.thread_id`、`vortex.barrier` 之类 op，是因为这条板测主线本来就没有走 launch/local/barrier 那一支。

#### 阶段 4：上游 `convert-*` 后的 LLVM dialect MLIR

文件：

- `build/smoke/matmul4x4_f32/matmul4x4_f32.llvm.mlir`

到了这里，`func.func / memref / scf / arith` 基本都已经进入 LLVM dialect：

```mlir
module {
  llvm.func @matmul4x4(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {vortex.kernel_entry} {
    %39 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %40 = llvm.mlir.constant(0 : index) : i64
    %41 = llvm.mlir.constant(4 : index) : i64
    %42 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%40 : i64)
  ^bb1(%43: i64):
    %44 = llvm.icmp "slt" %43, %41 : i64
    llvm.cond_br %44, ^bb2, ^bb6
  ^bb12:
    %65 = llvm.load %64 : !llvm.ptr -> f32
    %71 = llvm.load %70 : !llvm.ptr -> f32
    %77 = llvm.load %76 : !llvm.ptr -> f32
    %78 = llvm.fmul %65, %71  : f32
    %79 = llvm.fadd %77, %78  : f32
    llvm.store %79, %84 : f32, !llvm.ptr
    llvm.br ^bb11(%85 : i64)
  ^bb15:
    llvm.return
  }
}
```

观察点：

1. `func.func -> llvm.func`
2. 参数已经是裸 `!llvm.ptr`
3. `scf.for` 已经变成基本块加 `llvm.br/llvm.cond_br`
4. `memref.load/store` 变成 `llvm.load/store`
5. 浮点计算已经是 `llvm.fmul/fadd`

#### 阶段 5：`mlir-translate` 之后的 LLVM IR

文件：

- `build/smoke/matmul4x4_f32/matmul4x4_f32.ll`

这一步已经不是 MLIR 文本，而是标准 LLVM IR：

```llvm
define void @matmul4x4(ptr %0, ptr %1, ptr %2) {
  br label %25

25:
  %26 = phi i64 [ %39, %38 ], [ 0, %3 ]
  %27 = icmp slt i64 %26, 4
  br i1 %27, label %28, label %40

52:
  %56 = getelementptr float, ptr %53, i64 %55
  %57 = load float, ptr %56, align 4
  %61 = getelementptr float, ptr %58, i64 %60
  %62 = load float, ptr %61, align 4
  %66 = getelementptr float, ptr %63, i64 %65
  %67 = load float, ptr %66, align 4
  %68 = fmul float %57, %62
  %69 = fadd float %67, %68
  %73 = getelementptr float, ptr %70, i64 %72
  store float %69, ptr %73, align 4
  br label %49

79:
  ret void
}
```

这时就已经完全进入 LLVM 世界，后面再也不是 MLIR pass 了。

#### 阶段 6：Vortex LLVM backend 看到的输入与吐出的结果

backend 的输入就是上一步 `.ll`，命令大致是：

```text
clang -Xclang +vortex matmul4x4_f32.ll ...
```

这里 `clang -Xclang +vortex` 做的事不是“再产生一种 IR”，而是：

1. 读入 LLVM IR
2. 走 Vortex LLVM backend 的 instruction selection / register allocation / asm emission
3. 产出 `.s`
4. 再产出 `.o / .elf / .bin`

`.s` 看起来已经是目标汇编了，例如：

```asm
matmul4x4:
  addi sp, sp, -16
  sw ra, 12(sp)
  sw s0, 8(sp)
  mv s0, a2
  mv s1, a1
  mv s2, a0
  li a2, 64
  mv a0, s0
  li a1, 0
  call memset
  flw fa5, 0(a2)
  flw fa4, 0(s1)
  fmul.s fa5, fa5, fa4
  fadd.s fa5, fa3, fa5
  fsw fa5, 0(a6)
  ret
```

`objdump` 看到的 ELF 已经是目标机器码：

```text
/home/user/vortex-compiler/build/smoke/matmul4x4_f32/matmul4x4_f32.elf:
file format elf32-littleriscv

80000094 <matmul4x4>:
  ...
```

所以如果你问“Vortex LLVM backend 接住的 IR 是什么”，答案就是：

1. 输入是 `.ll` 这种标准 LLVM IR
2. 输出已经不是 IR，而是 `.s/.o/.elf/.bin`

### 10.2 launch/local/barrier 路径的代表性 IR 示例

这一支不是当前 `matmul4x4` 板测默认主线，但它是当前仓库里已经实现出来的“结构化 Vortex 后端语义路径”。

它大致对应这串 pass：

```text
map-parallel-loops-to-launch
-> promote-tiles-to-local
-> insert-barriers
-> plan-local-memory-layout
-> lower-local-memory
-> legalize-for-llvm
-> lower-runtime-builtins
-> convert-* to LLVM dialect
```

下面不强行用一个超长完整例子，而是直接按每个 pass 的典型输入/输出形状来理解。

#### `vortex-map-parallel-loops-to-launch`

输入还是标准循环，只是在 `scf.for` 上挂了 `vortex.mapping`：

```mlir
scf.for %tile = %c0 to %c2 step %c1 {
  "scf.for"(%c0, %c2, %c1) ({
  ^bb0(%sg: index):
    "scf.for"(%c0, %c4, %c1) ({
    ^bb0(%th: index):
      memref.store %c7, %out[%sg, %th] :
        memref<8x8xi32, #vortex.address_space<global>>
      scf.yield
    }) {vortex.mapping = "thread"} : (index, index, index) -> ()
    scf.yield
  }) {vortex.mapping = "subgroup"} : (index, index, index) -> ()
}
```

输出就不再是映射循环，而是 `vortex.launch` 执行区域：

```mlir
scf.for %tile = %c0 to %c2 step %c1 {
  vortex.launch %c1, %c2, %c4 {
    %sg = vortex.subgroup_id : index
    %th = vortex.thread_id : index
    memref.store %c7, %out[%sg, %th] :
      memref<8x8xi32, #vortex.address_space<global>>
    vortex.yield
  }
}
```

理解重点：

1. 外层普通 tile loop 还可以保留
2. 被标记映射的 perfect nest 被折叠成一个 `vortex.launch`
3. 原来的 IV 变成 `vortex.subgroup_id / vortex.thread_id`

#### `vortex-promote-tiles-to-local`

输入是在 `vortex.launch` 里有一个带 `vortex.promote_to_local` 的 tile：

```mlir
vortex.launch %c1, %c1, %c1 {
  %tile = memref.subview %arg0[%c0] [8] [1] {vortex.promote_to_local} :
    memref<16xf32, #vortex.address_space<global>> to
    memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
  %value = memref.load %tile[%c0] :
    memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
  vortex.yield
}
```

输出会显式引入 local memory：

```mlir
vortex.launch %c1, %c1, %c1 {
  %tile = memref.subview %arg0[%c0] [8] [1] :
    memref<16xf32, #vortex.address_space<global>> to
    memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
  %local = vortex.local_alloc() : memref<8xf32, #vortex.address_space<local>>
  memref.copy %tile, %local :
    memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>> to
    memref<8xf32, #vortex.address_space<local>>
  %value = memref.load %local[%c0] :
    memref<8xf32, #vortex.address_space<local>>
  vortex.yield
}
```

理解重点：

1. 原来读 global tile，变成先 copy 到 local
2. 后续 use 改成读 `#vortex.address_space<local>`
3. `vortex.promote_to_local` 只是触发标记，真正落地后这个标记会消失

#### `vortex-insert-barriers`

输入可能是：

```mlir
memref.copy %tile, %local : ...global... to ...local...
%value = memref.load %local[%c0] : memref<8xf32, #vortex.address_space<local>>
```

输出会自动补同步：

```mlir
memref.copy %tile, %local : ...global... to ...local...
vortex.barrier <core>
%value = memref.load %local[%c0] : memref<8xf32, #vortex.address_space<local>>
```

如果是写回场景，则是：

```mlir
memref.store %cst, %local[%c0] : memref<8xf32, #vortex.address_space<local>>
vortex.barrier <core>
memref.copy %local, %tile : ...local... to ...global...
```

理解重点：

1. barrier 插在 local tile 的“生产-消费”边界
2. 当前实现是模式驱动的保守插入，不是通用数据流分析

#### `vortex-plan-local-memory-layout`

输入时 `vortex.local_alloc` 还只是“有一个 local buffer”，并不知道它在 local frame 里的物理偏移：

```mlir
func.func @kernel() attributes {vortex.kernel} {
  vortex.launch %c1, %c1, %c1 {
    %buf0 = vortex.local_alloc() : memref<3xf32, #vortex.address_space<local>>
    %buf1 = vortex.local_alloc() : memref<2xf64, #vortex.address_space<local>>
    %buf2 = vortex.local_alloc() : memref<5xi8, #vortex.address_space<local>>
    vortex.yield
  }
  return
}
```

输出会把 layout 元数据挂回去：

```mlir
func.func @kernel() attributes {
  vortex.kernel,
  vortex.local_frame_bytes = 37 : i64
} {
  vortex.launch %c1, %c1, %c1 {
    %buf0 = vortex.local_alloc()
      {vortex.local.alignment = 4 : i64,
       vortex.local.byte_offset = 0 : i64,
       vortex.local.byte_size = 12 : i64}
      : memref<3xf32, #vortex.address_space<local>>
    %buf1 = vortex.local_alloc()
      {vortex.local.alignment = 8 : i64,
       vortex.local.byte_offset = 16 : i64,
       vortex.local.byte_size = 16 : i64}
      : memref<2xf64, #vortex.address_space<local>>
    %buf2 = vortex.local_alloc()
      {vortex.local.alignment = 1 : i64,
       vortex.local.byte_offset = 32 : i64,
       vortex.local.byte_size = 5 : i64}
      : memref<5xi8, #vortex.address_space<local>>
    vortex.yield
  }
  return
}
```

理解重点：

1. 这一步只做“布局规划”
2. 还没有把 local alloc 变成地址计算和真实 load/store

#### `vortex-lower-local-memory`

输入时 IR 里还显式保留 `vortex.local_alloc` 和 local-space memref：

```mlir
func.func @kernel(
    %src: memref<4xf32, #vortex.address_space<global>>,
    %dst: memref<4xf32, #vortex.address_space<global>>,
    %dst2: memref<4xf32, #vortex.address_space<global>>)
    attributes {vortex.kernel, vortex.local_frame_bytes = 16 : i64} {
  vortex.launch %c1, %c1, %c1 {
    %buf = vortex.local_alloc()
      {vortex.local.byte_offset = 0 : i64,
       vortex.local.byte_size = 16 : i64,
       vortex.local.alignment = 4 : i64}
      : memref<4xf32, #vortex.address_space<local>>
    memref.copy %src, %buf :
      memref<4xf32, #vortex.address_space<global>> to
      memref<4xf32, #vortex.address_space<local>>
    %value = memref.load %buf[%c0] :
      memref<4xf32, #vortex.address_space<local>>
    memref.store %value, %dst[%c0] :
      memref<4xf32, #vortex.address_space<global>>
    memref.copy %buf, %dst2 :
      memref<4xf32, #vortex.address_space<local>> to
      memref<4xf32, #vortex.address_space<global>>
    vortex.yield
  }
  return
}
```

输出后，local memory 已经不再是抽象 memref，而是显式地址化访问：

```mlir
func.func private @vx_local_mem_base() -> i64

func.func @kernel(...) attributes {vortex.kernel, vortex.local_frame_bytes = 16 : i64} {
  %base = call @vx_local_mem_base() : () -> i64
  vortex.launch %c1, %c1, %c1 {
    scf.for ...
    %ptr0 = llvm.inttoptr ... : i64 to !llvm.ptr
    llvm.store %..., %ptr0 {alignment = 4 : i64} : f32, !llvm.ptr
    %ptr1 = llvm.inttoptr ... : i64 to !llvm.ptr
    %val = llvm.load %ptr1 {alignment = 4 : i64} : !llvm.ptr -> f32
    memref.store %val, %dst[%c0] :
      memref<4xf32, #vortex.address_space<global>>
    scf.for ...
    vortex.yield
  }
  return
}
```

这一阶段结束后，应该满足：

1. 没有 `vortex.local_alloc`
2. 没有 `memref.copy` local/global 这种高层搬运
3. 没有 `#vortex.address_space<local>` 类型残留
4. local 访问已经变成 `vx_local_mem_base + byte_offset + llvm.inttoptr + llvm.load/store`

#### 这一支后面如何并回主线

`vortex-lower-local-memory` 之后，后续就和当前板测主线逐渐合流：

1. `vortex-legalize-for-llvm` 收掉残余 `vortex.launch`
2. `vortex-lower-runtime-builtins` 把 `vortex.thread_id/barrier` 等变成 wrapper call
3. 再走上游 `convert-*`
4. 再走 `mlir-translate`
5. 最后交给 Vortex LLVM backend 生成 `.elf/.bin`

所以可以把两条路线理解成：

1. 当前板测主线：直接从 `linalg` 打散到显式循环，然后进 LLVM
2. 结构化 full-chain：先显式表达 launch/local/barrier，再逐步消掉这些 Vortex 结构化语义，最后同样进 LLVM

## 11. 相关源码入口索引

### 11.1 Pass 定义与注册

1. `include/vortex/Transforms/Passes.td`
2. `include/vortex/Transforms/Passes.h`
3. `lib/InitAllPasses.cpp`
4. `tools/vx-opt/vx-opt.cpp`

### 11.2 Pipeline 定义

1. `include/vortex/Pipeline/Pipelines.h`
2. `lib/Pipeline/Pipelines.cpp`

### 11.3 Vortex dialect

1. `include/vortex/Dialect/Vortex/IR/VortexOps.td`
2. `include/vortex/Dialect/Vortex/IR/VortexAttributes.td`

### 11.4 Pass 实现

1. `lib/Transforms/MarkVortexKernelPass.cpp`
2. `lib/Transforms/MaterializeVortexAddressSpacesPass.cpp`
3. `lib/Transforms/MapParallelLoopsToVortexLaunchPass.cpp`
4. `lib/Transforms/PromoteTilesToVortexLocalPass.cpp`
5. `lib/Transforms/InsertVortexBarriersPass.cpp`
6. `lib/Transforms/PlanVortexLocalMemoryLayoutPass.cpp`
7. `lib/Transforms/LowerVortexLocalMemoryPass.cpp`
8. `lib/Transforms/LowerLinalgInsideVortexKernelPass.cpp`
9. `lib/Transforms/PrepareVortexToLLVMPass.cpp`
10. `lib/Transforms/ConvertVortexToLLVMPass.cpp`
11. `lib/Transforms/NormalizeONNXFrontendPass.cpp`
12. `lib/Transforms/TileMatmulForPreVortexPass.cpp`
13. `lib/Transforms/ValidatePreVortexPass.cpp`
14. `lib/Transforms/SummarizePreVortexPass.cpp`
