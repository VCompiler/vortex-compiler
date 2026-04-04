# Vortex xc7k480t 板上 GPT-2 最大可行配置

## 1. 硬件约束（xc7k480t）

| 资源 | 值 | 来源 |
|------|-----|------|
| DDR3 总容量 | 4 GB (2 × 2GB) | `hw/syn/xilinx/xc7k480t/mig_a.prj` |
| 核心配置 | 4 cores × 4 warps × 4 threads | `hw/syn/xilinx/xc7k480t/Makefile` |
| 核心时钟 | ~50 MHz | `IMPLEMENTATION.md` |
| L1 I-Cache | 16 KB / core | `VX_config.vh:547` |
| L1 D-Cache | 16 KB / core | `VX_config.vh:603` |
| L2 Cache | 1 MB | `VX_config.vh:679` |
| Local Memory | 16 KB / core | `VX_config.vh:223` |
| Stack | 8 KB / thread | `VX_config.vh:240` |
| FPU | f32 FMA (DSP48), 16 cycle latency | `VX_config.vh:449` |
| FP DIV/SQRT | 28 cycle latency | `VX_config.vh:470,491` |
| DDR 带宽 | 512-bit AXI, DDR3-1066 | `xc7k480t_jtag_ddr_engine.sv` |

## 2. JTAG 瓶颈

当前数据加载通过 JTAG-to-AXI 桥接，实测速度：

```
写入速度: ~77 words/sec ≈ 0.3 KB/s
回读验证: 同速
```

这是整个流程的绝对瓶颈。不同 ELF 尺寸的加载时间（write + verify）：

| ELF 大小 | JTAG 时间 | 可行性 |
|----------|----------|--------|
| 72 KB | ~7 min | ✅ 可用 |
| 1 MB | ~114 min | ⚠️ 需要 2h timeout |
| 3 MB | ~342 min | ❌ 超时 |
| 13 MB | ~25 h | ❌ 不现实 |

**当前 ELF 嵌入权重的方式，实际上限约 1-2 MB。**

## 3. 计算不是瓶颈

单核 50 MHz 下的 forward 执行时间极短：

| 配置 | 参数量 | FLOPs | 执行时间 |
|------|--------|-------|---------|
| d=64, 4L (当前) | 23 万 | 13 MFLOP | ~0.1 sec |
| d=64, 8L | 43 万 | 55 MFLOP | ~0.5 sec |
| d=128, 4L | 86 万 | 51 MFLOP | ~0.5 sec |
| d=256, 4L | 343 万 | 411 MFLOP | ~4.1 sec |
| GPT-2 Small | 1.6 亿 | 22 GFLOP | ~3.7 min |

DDR 容量和计算时间都不是问题，**纯粹卡在 JTAG 传输速度上**。

## 4. 推荐的最大配置

### 4.1 当前方式（权重嵌入 ELF）

受 JTAG 速度限制，2 小时 timeout 内可完成的最大配置：

```
推荐配置 A:
  seq_len  = 64
  d_model  = 64
  d_ff     = 256
  layers   = 8
  heads    = 1
  vocab    = 256

  参数量: ~43 万 (1.6 MB ELF)
  FLOPs:  55 MFLOP
  JTAG:   ~114 min (2h timeout)
  执行:   ~0.5 sec
```

```
推荐配置 B（更宽但更浅）:
  seq_len  = 32
  d_model  = 96
  d_ff     = 384
  layers   = 4
  heads    = 1
  vocab    = 256

  参数量: ~50 万 (1.9 MB ELF)
  JTAG:   ~135 min (2.5h timeout)
  执行:   ~0.3 sec
```

### 4.2 改进方式（权重外部加载）

如果把 kernel 代码和权重分离：
- 代码 ELF：~50 KB（JTAG ~1 min）
- 权重通过 manifest segments 预加载到 DDR 固定地址
- kernel 从 DDR 地址直接读权重

这需要修改 MLIR 生成方式：权重不再嵌入 C 的 `static const` 数组，而是通过 memref 参数从固定 DDR 地址传入。这样 ELF 始终很小，瓶颈变成 DDR 权重写入时间（仍然是 JTAG 速度，但不需要 verify 阶段）。

改进后可支持：

| 配置 | 参数量 | 权重大小 | JTAG 写入时间 |
|------|--------|----------|-------------|
| d=128, 8L | 165 万 | 6.3 MB | ~36 min |
| d=256, 4L | 343 万 | 13.1 MB | ~74 min |
| d=256, 8L | 658 万 | 25.1 MB | ~143 min |

**改进后推荐最大配置：**

```
推荐配置 C（需要代码改造）:
  seq_len  = 64
  d_model  = 256
  d_ff     = 1024
  layers   = 4
  heads    = 1 (或 4)
  vocab    = 512

  参数量: ~343 万 (13 MB 权重)
  FLOPs:  411 MFLOP
  DDR 权重加载: ~74 min (write-only, 无 verify)
  代码 ELF 加载: ~1 min
  执行:   ~4.1 sec
```

## 5. 与 GPT-2 标准配置的对比

| | 推荐 A | 推荐 C | GPT-2 Small | GPT-2 Medium |
|---|--------|--------|-------------|-------------|
| d_model | 64 | 256 | 768 | 1024 |
| layers | 8 | 4 | 12 | 24 |
| 参数量 | 43 万 | 343 万 | 1.17 亿 | 3.45 亿 |
| 占比 | 0.4% | 2.9% | 100% | — |

## 6. 如何运行

### 配置 A（当前方式，可直接运行）

```bash
# 生成
source .venv/bin/activate
python3 examples/gpt2/pytorch_to_vortex.py \
  --seq 64 --dim 64 --ff 256 --vocab 256 --layers 8 --seed 42 \
  --out-dir build/gpt2/board_max_A

# simx 验证
bash examples/gpt2/run_full_inference.sh build/gpt2/board_max_A

# 上板（需要 2h timeout）
ELF_B64=$(base64 -w0 build/gpt2/board_max_A/out/full_inference.elf)
curl -s --noproxy '*' -X POST http://100.125.4.76:8000/jobs/run-manifest \
  -H 'Content-Type: application/json' \
  -d "{
    \"kernel_elf_base64\": \"${ELF_B64}\",
    \"manifest\": {
      \"schema_version\": 1,
      \"name\": \"gpt2_64x64x256_8L\",
      \"segments\": [],
      \"outputs\": []
    },
    \"timeout_sec\": 7200,
    \"dump_stdout\": true
  }"
```

### 配置 C（需要改造，暂不可直接运行）

需要：
1. 修改 `gen_full_inference.py`，权重不嵌入 C 而是放到 manifest segments
2. kernel 通过固定 DDR 地址读取权重
3. manifest 中打包权重 segment

## 7. 提升上限的方向

| 方向 | 提升幅度 | 难度 |
|------|---------|------|
| 权重外部加载（manifest segments） | ELF 50KB→支持更大模型 | 中 |
| JTAG 批量传输优化 | 速度 ×2-5 | 需要改 Tcl 脚本 |
| 多核并行（vx_spawn_tasks） | 计算 ×4 | 需要 runtime 改造 |
| 半精度 f16 | 权重体积 ÷2 | 需要 FPU 支持 |
| 权重量化 int8 | 权重体积 ÷4 | 需要量化 kernel |
| PCIe/AXI DMA 加载 | 速度 ×100+ | 需要硬件改造 |
