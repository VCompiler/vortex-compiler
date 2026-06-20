# Board/XDMA Backend ABI Plan

## Current State

2026-06-20 当前结论：

- `vortex-platform` 的 PCIe/XDMA bitstream 已经能上板，基础 PCIe、DMA control BAR 和 DDR smoke 通过。
- 当前项目可以把 `examples/smoke/matmul4x4_f32.mlir` 编译成 Vortex ELF。
- 直接把通用 wrapper 交给 local-XDMA runner 时，硬件状态曾出现 `exit_seen` 但 `vx_busy` 未清掉，最终超时。这说明普通 `_Exit` 路径不足以作为当前 XDMA done/idle 协议。
- Guppy 生成器里已有可工作的局部实现：控制 lane 写 `0x88`，前后 `fence rw, rw`，随后 `vx_tmc_zero()` 并停住。但这段逻辑仍然埋在 Guppy 专用代码里。

## Backend Gaps

需要补齐的后端事项按优先级排序：

1. 标准 board/XDMA entry + exit ABI。
2. 通用 local-XDMA runner/manifest 接入，不再只靠 Guppy 脚本。
3. 命名的端到端 backend pipeline，避免 smoke、Guppy、board 路线各自维护长 pass 字符串。
4. parallel mapping 和 local-memory 路径闭环，包括控制 lane、worker lane、barrier 和 shared/local memory 约定。
5. ABI 鲁棒性，包括高参数入口、startup_arg 结构体、输出 buffer 描述符和错误码约定。

## Standard ABI

第一阶段在 compiler 仓库提供一个可链接 runtime shim，不直接修改
`/home/xiao/vortex-platform`：

- Header: `include/vortex/Runtime/BoardXDMAABI.h`
- Source: `runtime/board_xdma_abi.c`
- Build switch: `scripts/build-vortex-kernel.sh --board-xdma-abi`

导出的 C ABI：

```c
uintptr_t vortex_board_xdma_startup_arg(void);
int vortex_board_xdma_is_control_lane(void);
void vortex_board_xdma_host_visible_fence(void);
void vortex_board_xdma_progress(volatile int *progress, int value);
void vortex_board_xdma_exit_if(int status, int should_write) __attribute__((noreturn));
void vortex_board_xdma_exit(int status) __attribute__((noreturn));
```

语义：

- `vortex_board_xdma_startup_arg()` 读取 CSR `mscratch` (`0x340`)。当前硬件 reset 时会把 DCR `startup_arg` 装入 `mscratch`。
- `vortex_board_xdma_is_control_lane()` 固定定义为 `core_id == 0 && warp_id == 0 && thread_id == 0`。
- `vortex_board_xdma_host_visible_fence()` 发出 `fence rw, rw`，用于让 host 轮询可见。
- `vortex_board_xdma_exit_if(status, should_write)` 是底层 exit helper；`should_write` 为真时写 exit word，所有调用 lane 随后执行 `vx_tmc_zero()` 并停住。
- `vortex_board_xdma_exit(status)` 是控制 lane 版本，默认地址是 `IO_MPM_ADDR + 8`，没有平台宏时退回 `0x88`。
- `vortex_board_xdma_progress(progress, value)` 是可选进度字写入 helper，只由控制 lane 写并 fence。

## Entry Boundary

当前平台 `_start` 仍然是 `call main`，不会把 startup arg 当作 `main` 参数传入。
因此 compiler-side entry ABI 第一阶段只提供 `vortex_board_xdma_startup_arg()`。

后续如果要支持自动生成的 board entry wrapper，建议统一成：

```c
struct vortex_board_xdma_args {
  uint32_t version;
  uint32_t flags;
  uint64_t inputs;
  uint64_t outputs;
  uint64_t scratch;
};
```

runner 负责把该结构写到 DDR，并通过 DCR `startup_arg` 传地址；kernel 侧通过
`vortex_board_xdma_startup_arg()` 取回指针。

## Implementation Phases

### Phase 1 - Runtime Shim

已开始实现：

- 增加标准 header/source。
- `build-vortex-kernel.sh --board-xdma-abi` 自动把 source 加到 `--extra-source`。
- 新增一个 board/XDMA 版 smoke wrapper，避免普通 `return` 走 `_Exit`。

验收标准：

- `bash -n scripts/build-vortex-kernel.sh` 通过。
- 带 `--board-xdma-abi` 的 matmul smoke ELF 可以编译和链接。
- dump 中能看到对 `vortex_board_xdma_exit` 的调用或内联路径。

### Phase 2 - Generic XDMA Runner

第一版已完成：

- 新增当前仓库内的 `scripts/run-vortex-board-xdma.sh`。
- runner 会检查 PCI COMMAND 的 Mem Space/BusMaster 位和 `/dev/xdma*` 节点。
- runner 会调用 `vortex-platform` 的 `run_regression_manifest_xdma.py` 并打印 `FINAL_*` 状态。
- `scripts/run-matmul4x4-smoke.sh --driver local-xdma` 会自动：
  - 使用 board/XDMA wrapper。
  - 传 `--board-xdma-abi` 构建 ELF。
  - 生成 `local_xdma_manifest.json`。
  - 调用 `scripts/run-vortex-board-xdma.sh`。

2026-06-20 实板验收：

```text
scripts/run-matmul4x4-smoke.sh \
  --driver local-xdma \
  --platform-root /home/xiao/vortex-platform \
  --output-dir build/smoke/matmul4x4_f32_local_xdma

FINAL_STATUS=0x0000F90B
FINAL_FLAGS=user_lnk_up,ddr_init,busy_seen,done,exit_seen
FINAL_REASON=done
RUN_PASS=1
```

2026-06-20 Guppy 收敛：

- `gen_full_inference.py` 生成的 fast-exit 已改为调用 `vortex_board_xdma_exit_if`，不再在 Guppy 专用代码里裸写 `0x88`。
- `gen_projection_probe.py` 删除了未使用的私有 `probe_fast_exit`。
- `run_board_chat.py` 和 `run_full_inference.sh` 构建 Guppy wrapper 时传 `--board-xdma-abi`，确保标准 shim 参与链接。

仍待扩展：

- 通用 manifest 生成器支持输入段、输出段和 startup_arg descriptor。
- 让非 matmul smoke 也复用该 runner。

### Phase 3 - Compiler-Level Entry Wrapper

第一版已完成：

- 新增 `vortex-materialize-board-xdma-entry` Module pass。
- pass 运行在 `convert-func-to-llvm{use-bare-ptr-memref-call-conv=1}` 之后，寻找 `llvm.func` 上的 `vortex.kernel_entry`。
- 生成 `i32 @main()`，通过 `vortex_board_xdma_startup_arg()` 取得 descriptor 地址。
- descriptor MVP 布局：`u32 version, u32 flags, u64 arg0, u64 arg1, ...`；默认第一个参数地址在 byte offset 8，stride 8。
- wrapper 从 descriptor 逐项 load 64-bit 参数地址，`inttoptr` 成 lowered kernel 的裸指针参数，调用真实 kernel，随后调用 `vortex_board_xdma_exit(0)`。
- `scripts/build-vortex-kernel.sh --board-xdma-entry` 会在 lowering 后追加该 pass，并自动隐含 `--board-xdma-abi` 以链接 runtime shim。

当前限制：

- 只支持已经降成 bare pointer 的 LLVM dialect kernel 参数。
- kernel 必须返回 `void`，且默认要求唯一 `vortex.kernel_entry`。
- runner/manifest 侧还需要补 descriptor 写入能力，才能让非手写 wrapper 的 board kernel 直接上板。
