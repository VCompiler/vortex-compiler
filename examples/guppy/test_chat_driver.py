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


if __name__ == "__main__":
    unittest.main()
