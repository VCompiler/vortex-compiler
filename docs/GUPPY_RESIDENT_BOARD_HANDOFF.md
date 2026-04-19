# Guppy Resident Board Handoff

## 1. 当前状态

这次中断前，本地代码已经补到下面这个状态：

- `remote_vivado_service` 已新增 resident 执行接口：
  - `POST /jobs/run-resident-manifest`
- resident 模式语义已经实现：
  - 第 1 步 `run-manifest`
  - 后续步 `run-resident-manifest`
  - 后续步不再重新上传 ELF，也不再重新装载 kernel / rodata
- `chat_driver.py` 已默认切到 resident 路线，并保留回退到旧 `run-manifest` 的逻辑
- 本地验证已通过：
  - `PYTHONPATH=/home/user/remote_vivado_service/src python3 -m unittest discover -s /home/user/remote_vivado_service/tests -p 'test_*.py'`
  - `python3 -m py_compile /home/user/remote_vivado_service/src/remote_vivado_service/*.py /home/user/vortex-compiler/examples/guppy/chat_driver.py /home/user/vortex-compiler/examples/guppy/run_board_chat.py`
  - `python3 /home/user/vortex-compiler/examples/guppy/run_board_chat.py --prepare-only --prompt-text hi --max-new-tokens 2`

2026-04-15 新增确认的板级结论：

- `xc7k480t_vortex_board_top.sv` 的 exit monitor 已定位到真实 RTL 问题并修复：
  - 旧逻辑只在 `exit_mon_word_q == 32'h0000_005a` 时拉起 `vx_exit_seen`
  - 现已改成任意完整 exit word 写入都拉起 `vx_exit_seen`
- 已在 Windows 侧重建并验证新的 `sdbg2` bit：
  - `E:/fpga/repo/vx_xc7k480t_m1_20260324/sched_dbg_lsu_20260415_175657/sdbg2/sdbg2.runs/impl_1/xc7k480t_vortex_board_top.bit`
  - `E:/fpga/repo/vx_xc7k480t_m1_20260324/sched_dbg_lsu_20260415_175657/sdbg2/sdbg2.ltx`
- 直接板测已通过：
  - `VIO_STATUS=00d9aa7d`
  - `VX_EXIT_SEEN=1`
  - `FINAL_EXIT_WORD=0x00000000`
- `remote_vivado_service` 真机 `run-manifest` 板测已通过：
  - service 实际监听地址是 `http://100.125.4.76:18001`
  - 成功 job: `12bb8db17d52b176`
  - `run_pass=1`
  - `compare_errors=0`
  - `output_failures=0`
  - `final_exit_word=0x00000000`
- 但是当前 `Guppy seq16` 的 cold warm-up 仍然不现实：
  - `guppy ... warm_s00_load_manifest.txt` 总装载量是 `8,763,689` 个 32-bit word
  - 当前 `sdbg2` 路线仍然是 VIO 单词写 DDR
  - 实测 `guppy-warm` job `7f0710f18a067dfe` 在约 13 分钟时只写到 `21632/8763689`
  - 这个量级对应的冷启动预计是多天，不适合拿它做 resident 冒烟

2026-04-16 新增确认的快路径结论：

- `jtag_axi` bit 已确认会走 `LOAD_MODE=hw_axi`，不是 `vio_ddr`
- 首个 `vec_add8` 快路径 smoke `2febafd94d7ac519` 的现象是：
  - `DDR_LOAD_PASS=1`
  - dump / compare 全部通过
  - 但 `run_pass=0`
  - 根因不是 `hw_axi` loader，而是旧 bit 上 `vx_exit_seen` 仍未拉起，`jtag_run_diag_exit_vio.tcl` 之前又把它当成硬条件
- 已在本地更新：
  - `/home/user/vortex-platform/hw/syn/xilinx/xc7k480t/jtag_run_diag_exit_vio.tcl`
  - 改为在 `require_exit_seen=1` 时接受 `vx_exit_touch_seen + FINAL_EXIT_WORD` 命中，不再强依赖 `vx_exit_seen`
- 已同步到 Windows：
  - `E:\fpga\out\jtag_run_diag_exit_vio.tcl`
- 同一份 `vec_add8` 请求复测成功：
  - 成功 job: `5f941e7e54a1a651`
  - `LOAD_MODE=hw_axi`
  - `run_pass=1`
  - `compare_errors=0`
  - `output_failures=0`
  - `final_exit_word=0x00000000`
- 这说明当前真正可用的快速板级链路是：
  - `jtag_axi bit + hw_axi DDR load + generic exit 判定`

2026-04-16 新增确认的 Guppy resident 现状：

- 已在本地和 Windows service 同步 `remote_vivado_service/src/remote_vivado_service/manager.py`：
  - `guppy` 每个 generation step 现在强制写入 `manifest.require_exit_seen = true`
  - 目的不是修板子，而是禁止 host 侧把“没有真正退出的 resident rerun”误判成成功
- 真机已验证这个 host 侧修复生效：
  - service 重启后，用旧 resident `0ac4ecdf661bca22/steps/step_00` 做临时 prompt 验证
  - 验证 job: `26b7a893da24cfb6`
  - 现象不再是“立刻返回旧的 hi”
  - 而是：
    - `manifest_load` 很快通过
    - `manifest_run` 持续轮询
    - `RUN_POLL_INDEX=209`
    - `VIO_STATUS=0099aa17`
    - `VX_STARTED=1`
    - `VX_BUSY=1`
    - `VX_DONE_LATCHED=0`
    - `VX_EXIT_SEEN=0`
    - `FINAL_EXIT_WORD=UNKNOWN`
    - `argmax_actual.mem = FFFFFFFF`
- 这说明当前 `guppy? -> hi hi hi...` 的直接根因已经确认：
  - 旧 service 会把异常 resident rerun 当成功，随后复用上一轮 session metadata / 旧输出
  - 新 service 不再复用旧 `hi`
  - 现在暴露出来的是真实板级问题：
    - resident rerun 已经启动
    - 但长时间没有 exit，也没有产生新的 token
- 另外，full-model warm-up 为什么看起来“像卡住”也已确认：
  - `jtag_load_multi_ddr.tcl` 在真正连板前，会先执行 `build_segment_ops`
  - 这一步会把大 mem 段展开成 Tcl 列表，CPU 会长时间占用但日志几乎不前进
  - 所以 full warm 的慢，不只是 DDR 写慢，host 侧预处理本身也很重

这次没做完的唯一原因是：

- 当前会话对 `ssh/scp/curl` 的提权审批超时
- 不是代码问题
- 不是 Windows 主机不可用

## 2. 本地改动文件

### 2.1 需要同步到 Windows 的 service 运行时文件

只同步下面这 4 个文件：

- `/home/user/remote_vivado_service/src/remote_vivado_service/api.py`
- `/home/user/remote_vivado_service/src/remote_vivado_service/manager.py`
- `/home/user/remote_vivado_service/src/remote_vivado_service/models.py`
- `/home/user/remote_vivado_service/src/remote_vivado_service/runner.py`

### 2.2 只在本地 Linux 侧使用的文件

这些不需要同步到 Windows，但下一会话需要知道已经改过：

- `/home/user/vortex-compiler/examples/guppy/chat_driver.py`
- `/home/user/vortex-compiler/examples/guppy/README.md`
- `/home/user/vortex-compiler/docs/GUPPY_CHAT_MVP_CHECKLIST.md`
- `/home/user/remote_vivado_service/README.md`
- `/home/user/remote_vivado_service/docs/WINDOWS_DEPLOYMENT.md`
- `/home/user/vortex-platform/hw/syn/xilinx/xc7k480t/jtag_run_diag_exit_vio.tcl`

其中有一个例外需要记住：

- 上面这份 `jtag_run_diag_exit_vio.tcl` 虽然源文件在 Linux 侧仓库，但真机跑之前必须同步到 Windows：
  - `E:\fpga\out\jtag_run_diag_exit_vio.tcl`

## 3. Windows 侧要先检查什么

重启会话后，先按这个顺序做：

1. 确认 SSH 连通
2. 确认 Windows 上 `remote_vivado_service` 仓库路径还在
3. 确认实际 Python 包来源仍然是 `src/remote_vivado_service`
4. 确认 service 当前是否已运行
5. 如果没运行或版本旧，先同步 4 个 Python 文件并重启 service

优先检查的命令：

```bash
ssh Administrator@100.125.4.76 hostname
```

```bash
curl -s --noproxy '*' http://100.125.4.76:18001/healthz
```

```bash
ssh Administrator@100.125.4.76 "powershell -NoProfile -ExecutionPolicy Bypass -File E:\fpga\repo\remote_vivado_service\windows\status-service.ps1"
```

```bash
ssh Administrator@100.125.4.76 "powershell -NoProfile -Command \"Get-Content 'E:\fpga\repo\remote_vivado_service\pyproject.toml'\""
```

这里要特别确认：

- `pyproject.toml` 里应看到：
  - `packages = ["src/remote_vivado_service"]`

如果 Windows 仓库里同时存在：

- `E:\fpga\repo\remote_vivado_service\src\remote_vivado_service`
- `E:\fpga\repo\remote_vivado_service\remote_vivado_service`

那么这次只同步 `src/remote_vivado_service` 这条树，不要先去覆盖另一份。

## 4. 需要同步到 Windows 的目标路径

Windows 目标目录：

- `E:\fpga\repo\remote_vivado_service\src\remote_vivado_service\api.py`
- `E:\fpga\repo\remote_vivado_service\src\remote_vivado_service\manager.py`
- `E:\fpga\repo\remote_vivado_service\src\remote_vivado_service\models.py`
- `E:\fpga\repo\remote_vivado_service\src\remote_vivado_service\runner.py`

## 5. service 重启方式

先停，再用 `-RefreshInstall` 起服务，确保 editable install 吃到最新改动。

注意：

- 2026-04-15 验证时，真正可用的 service 不是 `:8000`，而是 `100.125.4.76:18001`
- `127.0.0.1:8000` 当时被 VSCode 的 `SimpleHTTP` 占用，不是 FPGA service
- 如果继续沿用 `:8000`，Windows 本机或外部浏览器都可能看到误导性的页面或 502

先停：

```bash
ssh Administrator@100.125.4.76 "powershell -NoProfile -ExecutionPolicy Bypass -File E:\fpga\repo\remote_vivado_service\windows\stop-service.ps1"
```

再起：

```bash
ssh Administrator@100.125.4.76 "powershell -NoProfile -ExecutionPolicy Bypass -File E:\fpga\repo\remote_vivado_service\windows\start-service.ps1 -RefreshInstall -BindAddress '100.125.4.76' -Port 18001 -BoardScriptsDir 'E:\fpga\out' -DefaultBitPath 'E:\fpga\repo\vx_xc7k480t_m1_20260324\sched_dbg_lsu_20260415_175657\sdbg2\sdbg2.runs\impl_1\xc7k480t_vortex_board_top.bit' -DefaultLtxPath 'E:\fpga\repo\vx_xc7k480t_m1_20260324\sched_dbg_lsu_20260415_175657\sdbg2\sdbg2.ltx' -DefaultHwServerUrl 'TCP:localhost:3121' -DefaultHwServerBind 'TCP:localhost:3121'"
ssh Administrator@100.125.4.76 "powershell -NoProfile -ExecutionPolicy Bypass -File E:\fpga\repo\remote_vivado_service\windows\start-service.ps1 -RefreshInstall -BindAddress '100.125.4.76' -Port 18001 -BoardScriptsDir 'E:\fpga\out' -DefaultBitPath 'E:\fpga\repo\vx_xc7k480t_jtag_axi_20260405\jtag_axi_build\jtag_axi_build.runs\impl_1\xc7k480t_vortex_board_top.bit' -DefaultLtxPath 'E:\fpga\repo\vx_xc7k480t_jtag_axi_20260405\jtag_axi_build\jtag_axi_build.runs\impl_1\xc7k480t_vortex_board_top.ltx' -DefaultHwServerUrl 'TCP:localhost:3121' -DefaultHwServerBind 'TCP:localhost:3121'"
```

然后检查：

```bash
curl -s --noproxy '*' http://100.125.4.76:18001/healthz
```

```bash
ssh Administrator@100.125.4.76 "powershell -NoProfile -ExecutionPolicy Bypass -File E:\fpga\repo\remote_vivado_service\windows\status-service.ps1"
```

如果起不来，先看：

- `E:\fpga\repo\remote_vivado_service\.runtime\logs\remote-vivado-service.stdout.log`
- `E:\fpga\repo\remote_vivado_service\.runtime\logs\remote-vivado-service.stderr.log`

## 6. 先做一个小的 resident 冒烟

不要一上来就跑 `seq16_l6` / `seq128_l6` resident warm。
在当前 `sdbg2 + VIO 单词写 DDR` 路线下，它们的 cold start 都太大。

先做一个已验证通过的 service 真机 smoke：

- ELF: `/home/user/vortex-compiler/build/smoke/vec_add8_i32/vec_add8_i32.elf`
- bit:
  - `E:/fpga/repo/vx_xc7k480t_m1_20260324/sched_dbg_lsu_20260415_175657/sdbg2/sdbg2.runs/impl_1/xc7k480t_vortex_board_top.bit`
- ltx:
  - `E:/fpga/repo/vx_xc7k480t_m1_20260324/sched_dbg_lsu_20260415_175657/sdbg2/sdbg2.ltx`
- service:
  - `http://100.125.4.76:18001`

推荐直接复查成功 job：

```bash
curl -s --noproxy '*' http://100.125.4.76:18001/jobs/12bb8db17d52b176
```

这一步先确认三件事：

1. `status == "succeeded"`
2. `run_pass == "1"`
3. `final_exit_word == "0x00000000"`

只有在这个基础 smoke 稳定后，才应该继续碰 Guppy resident。

当前 Guppy resident 的判断标准不是“功能坏了”，而是“冷启动装载量远超当前 VIO 路线可接受范围”。

## 7. 再跑 full Guppy

如果要继续推进 full Guppy，先不要直接点 session activate。
必须先满足下面至少一条：

1. 把加载路径切到更快的板级通道：
   - 例如 `jtag_axi` / `hw_axi` / 更粗粒度 burst loader
2. 或者保留一个已经 warm 好的 resident，不要重新 cold load
3. 或者换更小的 stage / 更少层数 / 更短序列，显著缩小 `kernel_seg0`

在当前 `sdbg2` 路线上，`seq16_l6` warm 都要装 `8,763,689` 个 word，已经不适合作为“冒烟”。

只有上面条件满足后，再跑：

```bash
python3 /home/user/vortex-compiler/examples/guppy/run_board_chat.py \
  --stage-dir /home/user/vortex-compiler/build/guppy/full_inference_seq128_l6 \
  --sequence-length 128 \
  --layer-limit 6 \
  --prompt-text "hi" \
  --max-new-tokens 2
```

当前 full stage 已存在：

- `/home/user/vortex-compiler/build/guppy/full_inference_seq128_l6`
- ELF:
  - `/home/user/vortex-compiler/build/guppy/full_inference_seq128_l6/out/full_inference.elf`

ELF 大小大约：

- `36M`

这正是 resident 路线必须验证的原因：

- 如果仍然每 token 重传 ELF，性能和可用性都不成立

## 8. 成功标准

最小成功标准：

1. service 健康检查通过：
   - `http://100.125.4.76:18001/healthz`
2. `vec_add8_i32` 这类小 `run-manifest` 板测稳定通过
3. resident 路线使用了更快 loader 或已有 warm resident
4. 在上面前提下，Guppy session activate 不再卡在 cold load
5. full `seq128_l6` 至少成功回出 2 个 token

## 9. 如果失败，要抓什么

先记下：

- `job_id`
- `status`
- `exit_code`
- `summary.json`
- `manifest_run.log`
- `manifest_load.log`
- `runner.log`
- `stdout.log`

服务侧拉日志的方式：

```bash
curl -s --noproxy '*' http://100.125.4.76:8000/jobs/<JOB_ID>/logs
```

```bash
curl -s --noproxy '*' http://100.125.4.76:8000/jobs/<JOB_ID>/logs/summary.json
```

```bash
curl -s --noproxy '*' http://100.125.4.76:8000/jobs/<JOB_ID>/logs/manifest_run.log
```

重点看这几个故障点：

- `run-resident-manifest` 是否返回 404
  - 说明 Windows service 没同步到新版本
- `resident state not found`
  - 说明 step0 没成功，或 service 进程重启后 resident 状态丢了
- `run_pass=0`
  - 说明板上运行本身没完成
- 第二步仍然走 `run-manifest`
  - 说明 host 侧触发了回退路径，需要检查 resident endpoint 或 HTTP 错误

## 10. 下一会话的第一句话该做什么

重启会话后，直接做这三件事：

1. 读取本文件
2. 先 SSH / `healthz` 检查 Windows service
3. 同步 4 个 service Python 文件并重启 service

不要先重新分析 Guppy 模型，不要先改编译链，不要先改 pass。
当前唯一缺的是 Windows service 同步和真实板测。
这个结论已过时。

截至 2026-04-15，真实板测已经补充到：

- service `run-manifest` 已真实通过
- `vx_exit_seen` 的板级 RTL 问题已修复并验证
- 当前最大阻塞点已经变成：
  - `Guppy cold start` 的 resident kernel 首次装载量过大
  - 现有 VIO 单词写 DDR 路线带宽过低
