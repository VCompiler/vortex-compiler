# GuppyLM 满血 Case Git 上传整理

## 1. 这次要整理的范围

这里的“满血 GuppyLM case”指的是：

- 使用原始 Guppy 导出配置：
  - `n_layers = 6`
  - `max_seq_len = 128`
- 生成 `seq128_l6` full inference stage
- 主机侧 decode loop
- resident warm / prompt session API
- 最小 Web 面板
- 板级 `run-manifest` / `run-resident-manifest` 支撑脚本

注意：

- `build/guppy/full_inference_seq128_l6/**` 是构建产物，不应该直接进 git
- 应该进 git 的是：
  - 导出器
  - stage 生成器
  - host chat driver
  - service 端 session / prompt / web 代码
  - 板级 Tcl / RTL 修复

## 2. 推荐的仓库拆分

建议按 3 个仓库上传，不要混成一个大提交。

### 2.1 `vortex-compiler`

这部分放“模型导出 + 编译 + host 侧 chat”：

- `examples/guppy/README.md`
- `examples/guppy/chat_driver.py`
- `examples/guppy/download_assets.py`
- `examples/guppy/dump_reference_logits.py`
- `examples/guppy/fixed_prompt_messages.json`
- `examples/guppy/gen_full_inference.py`
- `examples/guppy/guppy_to_vortex.py`
- `examples/guppy/run_board_chat.py`
- `examples/guppy/run_full_inference.sh`
- `examples/guppy/test_chat_driver.py`
- `docs/GUPPY_CHAT_MVP_CHECKLIST.md`
- `docs/GUPPY_RESIDENT_BOARD_HANDOFF.md`
- `docs/GUPPY_FULL_CASE_GIT_UPLOAD.md`
- `scripts/build-vortex-kernel.sh`

这次 `git status` 里还有一些不属于 Guppy 满血 case 的文件，例如：

- `examples/smoke/*`
- `scripts/run-matmul4x4-smoke.sh`
- `scripts/run-bare-ptr-smoke.sh`
- `scripts/run-smoke-suite.sh`

如果目标是先把 Guppy case 单独上传，这些应拆到另一组提交，不要混入 Guppy 提交。

### 2.2 `remote_vivado_service`

这部分放“session API + resident + prompt + web 面板 + Windows service 启停”。

当前本地目录 `/home/user/remote_vivado_service` 还不是 git 仓库，所以有两种做法：

1. 最推荐：把它当独立仓库上传
2. 次选：作为子目录并入现有 monorepo

如果按独立仓库上传，建议至少包含：

- `.gitignore`
- `pyproject.toml`
- `README.md`
- `docs/**`
- `windows/**`
- `src/remote_vivado_service/api.py`
- `src/remote_vivado_service/config.py`
- `src/remote_vivado_service/file_lock.py`
- `src/remote_vivado_service/guppy.py`
- `src/remote_vivado_service/job_store.py`
- `src/remote_vivado_service/main.py`
- `src/remote_vivado_service/manager.py`
- `src/remote_vivado_service/models.py`
- `src/remote_vivado_service/runner.py`
- `src/remote_vivado_service/session_store.py`
- `src/remote_vivado_service/web.py`
- `tests/test_api_validation.py`
- `tests/test_file_lock.py`
- `tests/test_guppy.py`
- `tests/test_job_store.py`
- `tests/test_manager_guppy.py`
- `tests/test_runner_hw_target.py`
- `tests/test_runner_manifest.py`
- `tests/test_session_store.py`

说明：

- Windows 真机运行时只需要同步少量文件到 `src/remote_vivado_service`
- 但 git 上传应提交完整可复现项目，不要只传 4 个热修文件

### 2.3 `vortex-platform`

这部分放“板级 loader / stdout drain / exit 判定 / RTL 修复”。

当前 `/home/user/vortex-platform` 工作区很脏，不能直接 `git add .`。
先按“确认跟 Guppy resident/full case 直接相关”的最小集合切分：

- `hw/syn/xilinx/xc7k480t/jtag_load_multi_ddr.tcl`
- `hw/syn/xilinx/xc7k480t/jtag_ddr_min_rw_vio.tcl`
- `hw/syn/xilinx/xc7k480t/jtag_run_diag_exit_vio.tcl`
- `hw/syn/xilinx/xc7k480t/jtag_stdout_drain_vio.tcl`
- `hw/syn/xilinx/xc7k480t/xc7k480t_vortex_board_top.sv`

这些文件分别覆盖了：

- `hw_axi` 快速 DDR load/readback
- `generic exit` 判定
- stuck 检测
- stdout drain
- exit monitor / rerun 启动时序 / debug probe

其余 RTL 改动很多，先不要和 Guppy 上传混在一起，除非你确认它们是这次 bitstream 必须依赖的闭包。

## 3. 明确不要进 git 的内容

### 3.1 `vortex-compiler`

不要提交：

- `build/**`
- `.venv-guppy-stagea/`
- `ramulator.stats.log`
- `trace/`
- `third_party/llvm-vortex-build/`
- `__pycache__/`

`seq128_l6` 的 `.mlir/.elf/.bin/.ll/.mem` 等产物都应该通过脚本重建。

### 3.2 `remote_vivado_service`

不要提交：

- `.venv/`
- `.runtime/`
- `__pycache__/`
- Windows 侧运行出来的 `jobs/`、`sessions/`、日志目录

### 3.3 `vortex-platform`

不要提交：

- `kernel/*.o`
- `tests/kernel/**/*.elf`
- `tests/kernel/**/*.bin`
- `tests/kernel/**/*.dump`
- `project_sched_dbg/` 一类综合产物
- 本地仿真/构建缓存

## 4. 推荐上传顺序

建议按下面顺序传：

1. `vortex-compiler`
2. `remote_vivado_service`
3. `vortex-platform`

原因：

- `vortex-compiler` 先定义“如何生成满血 case”
- `remote_vivado_service` 再定义“如何长期运行 resident / prompt”
- `vortex-platform` 最后补“板子为什么能跑得通”

这样 review 起来最清楚。

## 5. 建议的提交切分

### 5.1 `vortex-compiler`

建议 2 个 commit：

1. `examples/guppy + docs`
2. `build-vortex-kernel compatibility fix`

可参考：

```bash
git -C /home/user/vortex-compiler add \
  examples/guppy \
  docs/GUPPY_CHAT_MVP_CHECKLIST.md \
  docs/GUPPY_RESIDENT_BOARD_HANDOFF.md \
  docs/GUPPY_FULL_CASE_GIT_UPLOAD.md \
  .gitignore

git -C /home/user/vortex-compiler commit -m "Add GuppyLM export and board chat flow"

git -C /home/user/vortex-compiler add scripts/build-vortex-kernel.sh
git -C /home/user/vortex-compiler commit -m "Fix MLIR-to-LLVM translation for Guppy kernels"
```

### 5.2 `remote_vivado_service`

如果你准备把它正式上传成独立仓库：

```bash
git -C /home/user/remote_vivado_service init
git -C /home/user/remote_vivado_service add \
  .gitignore \
  pyproject.toml \
  README.md \
  docs \
  windows \
  src/remote_vivado_service \
  tests
git -C /home/user/remote_vivado_service commit -m "Add Guppy resident session API and web panel"
```

然后再配置远端：

```bash
git -C /home/user/remote_vivado_service remote add origin <your-remote-url>
git -C /home/user/remote_vivado_service branch -M main
git -C /home/user/remote_vivado_service push -u origin main
```

### 5.3 `vortex-platform`

不要整仓上传，先只加确认过的 Guppy 依赖文件：

```bash
git -C /home/user/vortex-platform add \
  hw/syn/xilinx/xc7k480t/jtag_load_multi_ddr.tcl \
  hw/syn/xilinx/xc7k480t/jtag_ddr_min_rw_vio.tcl \
  hw/syn/xilinx/xc7k480t/jtag_run_diag_exit_vio.tcl \
  hw/syn/xilinx/xc7k480t/jtag_stdout_drain_vio.tcl \
  hw/syn/xilinx/xc7k480t/xc7k480t_vortex_board_top.sv

git -C /home/user/vortex-platform commit -m "Improve xc7k480t resident run flow for Guppy chat"
```

如果重新生成 bitstream 还依赖更多 RTL，再额外补一提交，不要把不确定依赖一次性塞进来。

## 6. 是否要提交满血 stage 本身

不建议提交下面这些生成物：

- `build/guppy/export/**`
- `build/guppy/full_inference_seq128_l6/**`

原因：

- 它们体积大
- 属于可再生工件
- 后续 prompt / tolerance / wrapper 变动后容易频繁失效

更合理的做法是：

1. 提交生成脚本
2. 在 README 里固定生成命令
3. 如确实需要交付二进制，放 release asset，不放 git 历史

## 7. 当前结论

当前这套“满血 GuppyLM case”可以上传，但应该拆成 3 份：

- `vortex-compiler`: 生成链和 host chat
- `remote_vivado_service`: resident session API 和 Web 面板
- `vortex-platform`: 板级 loader / exit / stdout / RTL 修复

最需要避免的错误是：

- 把 `build/` 产物一起传上去
- 把 `vortex-platform` 当前整坨脏工作区直接 `git add .`
- 只传 Windows 热修文件而不传完整 service 工程
