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
| JTAG 传输速度 | 1-5 MB/s（优化后） | 实测 |

## 2. 各配置可行性分析

| 配置 | 参数量 | ELF 大小 | JTAG 加载 (1MB/s) | 计算量 | 执行时间 (单核 50MHz) | DDR |
|------|--------|---------|-------------------|--------|---------------------|-----|
| d=64, 8L (当前最大验证) | 43 万 | 1.6 MB | 3 min | 0.05 GFLOP | 0.5 sec | ✅ |
| d=128, 8L | 170 万 | 6.3 MB | 13 min | 0.21 GFLOP | 2 sec | ✅ |
| **d=256, 4L (推荐)** | **343 万** | **13 MB** | **26 min** | **0.41 GFLOP** | **4 sec** | ✅ |
| d=256, 8L | 660 万 | 25 MB | 50 min | 0.82 GFLOP | 8 sec | ✅ |
| d=256, 12L | 1000 万 | 38 MB | 76 min | 1.23 GFLOP | 12 sec | ✅ |
| d=512, 8L | 2630 万 | 100 MB | 200 min | 3.25 GFLOP | 33 sec | ✅ |
| d=512, 12L | 3990 万 | 152 MB | 305 min | 9.87 GFLOP | 99 sec | ✅ |
| GPT-2 Small | 1.62 亿 | 619 MB | 1238 min | 22 GFLOP | 221 sec | ✅ |

说明：
- 所有配置都放得进 4GB DDR
- 执行时间（单核）从 0.5 秒到 3.7 分钟都是合理范围
- JTAG 1 MB/s 下 100MB 以内都可以在合理时间加载
- 如果 JTAG 达到 5 MB/s，GPT-2 Small 也只需 4 分钟加载

## 3. 真实瓶颈分析

### 已不再是瓶颈
- **DDR 容量**：4 GB 足够 GPT-2 Small (619 MB)
- **JTAG 速度**：优化后 1-5 MB/s，100 MB 级 ELF 可在 ~20-100 min 加载
- **计算时间**：单核 50 MHz，GPT-2 Small 单次 forward ~3.7 min

### 当前实际瓶颈
1. **ELF 编译时间**：权重嵌入为 C static const 数组，>100 MB 的 .c 文件编译极慢
2. **simx 验证时间**：大模型在 simx（软件逐指令模拟）上很慢
3. **单核执行**：尚未实现 `vx_spawn_tasks` 多核并行（第四阶段待完成）
4. **单精度 f32**：所有权重和激活都是 f32，如有 f16 支持可减半内存和带宽

## 4. 推荐配置

### 近期目标（可直接运行）

```
配置: d=256, ff=1024, 4 层, seq=64, vocab=512, 1 head
参数量: 343 万 (GPT-2 Small 的 2.9%)
ELF: ~13 MB
JTAG 加载: ~13-26 min (取决于速度)
单核执行: ~4 sec
```

选择理由：
- d=256 足够展示非 trivial 的 Transformer 计算
- 4 层足够验证多层串联正确性
- 13 MB ELF 在 JTAG 优化后完全可行
- 4 秒执行意味着几乎即时得到结果

### 远期目标（需要工程改进）

```
配置: d=512, ff=2048, 12 层, seq=128, vocab=2048
参数量: 3990 万 (GPT-2 Small 的 34%)
ELF: ~152 MB
```

需要：
- 权重与代码分离（代码 ELF 小，权重通过 manifest segments 加载）
- 多核并行（vx_spawn_tasks）
- 编译工具链优化（避免编译 100+ MB 的 C 文件）

## 5. 如何运行

### 5.1 simx 验证

```bash
source .venv/bin/activate

# 生成
python3 examples/gpt2/pytorch_to_vortex.py \
  --seq 64 --dim 256 --ff 1024 --vocab 512 --layers 4 --seed 42 \
  --out-dir build/gpt2/board_d256

# simx 验证
bash examples/gpt2/run_full_inference.sh build/gpt2/board_d256
```

### 5.2 上板执行

```bash
ELF_B64=$(base64 -w0 build/gpt2/board_d256/out/full_inference.elf)
curl -s --noproxy '*' -X POST http://100.125.4.76:8000/jobs/run-manifest \
  -H 'Content-Type: application/json' \
  -d "{
    \"kernel_elf_base64\": \"${ELF_B64}\",
    \"manifest\": {
      \"schema_version\": 1,
      \"name\": \"gpt2_d256_4L\",
      \"segments\": [],
      \"outputs\": []
    },
    \"timeout_sec\": 3600,
    \"dump_stdout\": true
  }"

# 轮询结果
curl -s --noproxy '*' http://100.125.4.76:8000/jobs/<job-id>
```

## 6. 提升上限的方向

| 方向 | 效果 | 难度 | 优先级 |
|------|------|------|--------|
| 多核并行 (vx_spawn_tasks) | 计算速度 ×4 | 中（需要调研 spawn API） | 高 |
| 权重外部加载 (manifest segments) | 支持更大模型，ELF 缩到 50KB | 中 | 高 |
| 半精度 f16 | 权重/激活体积 ÷2 | 高（需要 FPU 支持） | 中 |
| 权重量化 int8/int4 | 权重体积 ÷4~8 | 高（需要量化 kernel） | 低 |
| 编译优化（权重不嵌入 .c） | 加速编译 | 低 | 高 |
| multi-head attention | 更接近真实 GPT-2 | 中 | 中 |
