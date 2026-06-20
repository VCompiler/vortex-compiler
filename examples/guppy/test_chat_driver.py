from __future__ import annotations

import random
import tempfile
from pathlib import Path
import unittest

import chat_driver


class ChatDriverResidentStateTest(unittest.TestCase):
    def test_first_full_run_becomes_resident_session(self) -> None:
        resident_job_id = chat_driver.update_resident_job_id(
            None,
            {
                "job_id": "job-step0",
                "status": "succeeded",
                "used_resident": False,
            },
            reuse_resident=True,
        )
        self.assertEqual(resident_job_id, "job-step0")

    def test_successful_resident_step_keeps_existing_session(self) -> None:
        resident_job_id = chat_driver.update_resident_job_id(
            "job-step0",
            {
                "job_id": "job-step1",
                "status": "succeeded",
                "used_resident": True,
            },
            reuse_resident=True,
        )
        self.assertEqual(resident_job_id, "job-step0")

    def test_fallback_full_run_replaces_resident_session(self) -> None:
        resident_job_id = chat_driver.update_resident_job_id(
            "job-step0",
            {
                "job_id": "job-step1-fallback",
                "status": "succeeded",
                "used_resident": False,
            },
            reuse_resident=True,
        )
        self.assertEqual(resident_job_id, "job-step1-fallback")

    def test_failed_full_run_clears_resident_session(self) -> None:
        resident_job_id = chat_driver.update_resident_job_id(
            "job-step0",
            {
                "job_id": "job-step1-failed",
                "status": "failed",
                "used_resident": False,
            },
            reuse_resident=True,
        )
        self.assertIsNone(resident_job_id)

    def test_disabled_reuse_always_clears_resident_session(self) -> None:
        resident_job_id = chat_driver.update_resident_job_id(
            "job-step0",
            {
                "job_id": "job-step1",
                "status": "succeeded",
                "used_resident": True,
            },
            reuse_resident=False,
        )
        self.assertIsNone(resident_job_id)

    def test_run_generation_step_passes_resident_full_reload_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stage_dir = Path(tmp_dir)
            elf_path = stage_dir / "full_inference.elf"
            elf_path.write_bytes(b"ELF")

            observed_post: dict[str, object] = {}

            class _FakeClient:
                def post_json(self, url, payload):  # type: ignore[no-untyped-def]
                    observed_post["url"] = url
                    observed_post["payload"] = payload
                    return {"job_id": "resident-job-1"}

            class _FakeTokenizer:
                def decode(self, ids):  # type: ignore[no-untyped-def]
                    return ",".join(str(item) for item in ids)

            original_poll_job = chat_driver.poll_job
            original_fetch_log_list = chat_driver.fetch_log_list
            original_try_fetch_log_content = chat_driver.try_fetch_log_content
            try:
                chat_driver.poll_job = lambda *args, **kwargs: {  # type: ignore[assignment]
                    "status": "succeeded",
                    "exit_code": 0,
                    "metadata": {},
                }
                chat_driver.fetch_log_list = lambda *args, **kwargs: [  # type: ignore[assignment]
                    "artifacts/argmax_actual.mem",
                    "summary.json",
                ]
                chat_driver.try_fetch_log_content = lambda _client, _service_url, _job_id, name: {  # type: ignore[assignment]
                    "artifacts/argmax_actual.mem": "0000002A\n",
                    "summary.json": "{}",
                }.get(name, "")

                chat_driver.run_generation_step(
                    runner_mode="service",
                    client=_FakeClient(),
                    tokenizer=_FakeTokenizer(),
                    service_url="http://service",
                    elf_path=elf_path,
                    stage_dir=stage_dir,
                    request_base={
                        "board_scripts_dir": "E:/fpga/out",
                        "ltx_path": "E:/fpga/out/demo.ltx",
                        "device_index": 0,
                        "debug_jtag_freq_hz": 10_000_000,
                        "hw_server_url": "TCP:localhost:3121",
                        "hw_server_bind": "TCP:localhost:3121",
                        "dump_stdout": False,
                        "persistent_vivado_session": False,
                        "stdout_max_chars": 4096,
                        "timeout_sec": 60,
                    },
                    symbols={
                        "guppy_input_token_ids": 0x1000,
                        "guppy_runtime_prompt_length": 0x2000,
                        "guppy_runtime_expect_golden": 0x3000,
                        "guppy_output_last_token_argmax": 0x4000,
                    },
                    seq_len=4,
                    pad_id=0,
                    vocab_size=64,
                    token_ids=[1, 2],
                    run_name="guppy_test",
                    step_index=1,
                    timeout_sec=60,
                    poll_interval_sec=0.01,
                    read_logits=False,
                    top_k=8,
                    decode_mode="greedy",
                    temperature=1.0,
                    sample_top_k=8,
                    rng=random.Random(0),
                    resident_job_id="resident-job-0",
                    reload_all_kernel_segments=True,
                )
            finally:
                chat_driver.poll_job = original_poll_job  # type: ignore[assignment]
                chat_driver.fetch_log_list = original_fetch_log_list  # type: ignore[assignment]
                chat_driver.try_fetch_log_content = original_try_fetch_log_content  # type: ignore[assignment]

        self.assertEqual(observed_post["url"], "http://service/jobs/run-resident-manifest")
        self.assertTrue(observed_post["payload"]["reload_all_kernel_segments"])

    def test_local_jtag_step_writes_manifest_and_skips_reprogram_after_step0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stage_dir = Path(tmp_dir)
            elf_path = stage_dir / "full_inference.elf"
            board_scripts_dir = stage_dir / "board"
            board_runner = board_scripts_dir / "run_regression_manifest_jtag.py"
            elf_path.write_bytes(b"ELF")
            board_scripts_dir.mkdir()
            board_runner.write_text("#!/usr/bin/env python3\n")

            observed_calls: list[list[str]] = []

            def fake_run_and_log(cmd, *, cwd=None, env=None, log_path=None):  # type: ignore[no-untyped-def]
                observed_calls.append(list(cmd))
                assert log_path is not None
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("stub\n")
                if "--stage-dir" in cmd:
                    step_dir = Path(cmd[cmd.index("--stage-dir") + 1])
                    (step_dir / "argmax_actual.mem").write_text("0000002A\n")
                    (step_dir / "run.log").write_text("RUN_PASS=1\nFINAL_EXIT_WORD=0x00000000\n")
                    (step_dir / "stdout.log").write_text("STDOUT_TEXT_BEGIN\nhello\nSTDOUT_TEXT_END\n")
                return 0

            original_run_and_log = chat_driver.run_and_log
            try:
                chat_driver.run_and_log = fake_run_and_log  # type: ignore[assignment]
                step_result = chat_driver.run_generation_step(
                    runner_mode="local-jtag",
                    client=chat_driver.JsonHttpClient(),
                    tokenizer=type("Tok", (), {"decode": staticmethod(lambda ids: ",".join(str(i) for i in ids))})(),
                    service_url="http://unused",
                    elf_path=elf_path,
                    stage_dir=stage_dir,
                    request_base={
                        "bit_path": str(stage_dir / "demo.bit"),
                        "ltx_path": str(stage_dir / "demo.ltx"),
                        "board_scripts_dir": str(board_scripts_dir),
                        "board_runner": str(board_runner),
                        "vivado_settings_sh": "/tmp/settings64.sh",
                        "device_index": 0,
                        "program_jtag_freq_hz": 30_000_000,
                        "debug_jtag_freq_hz": 10_000_000,
                        "hw_server_url": "TCP:127.0.0.1:53121",
                        "hw_server_bind": "TCP:127.0.0.1:53121",
                        "verify": False,
                        "dump_stdout": True,
                        "persistent_vivado_session": False,
                        "stdout_max_chars": 4096,
                        "timeout_sec": 60,
                    },
                    symbols={
                        "guppy_input_token_ids": 0x1000,
                        "guppy_runtime_prompt_length": 0x2000,
                        "guppy_runtime_expect_golden": 0x3000,
                        "guppy_output_last_token_argmax": 0x4000,
                    },
                    seq_len=4,
                    pad_id=0,
                    vocab_size=64,
                    token_ids=[1, 2],
                    run_name="guppy_test",
                    step_index=1,
                    timeout_sec=60,
                    poll_interval_sec=0.01,
                    read_logits=False,
                    top_k=8,
                    decode_mode="greedy",
                    temperature=1.0,
                    sample_top_k=8,
                    rng=random.Random(0),
                    resident_job_id=None,
                    reload_all_kernel_segments=False,
                )
            finally:
                chat_driver.run_and_log = original_run_and_log  # type: ignore[assignment]

        self.assertEqual(step_result["board_argmax_id"], 42)
        self.assertEqual(step_result["stdout_text"], "hello")
        self.assertIn("--skip-program", observed_calls[0])


if __name__ == "__main__":
    unittest.main()
