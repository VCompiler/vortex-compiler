#!/usr/bin/env python3
"""Download or collect the minimum Guppy assets required for stage A."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path


HF_REPO = "arman-bd/guppylm-9M"
HF_PRIMARY_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"
HF_MIRROR_BASE = f"https://hf-mirror.com/{HF_REPO}/resolve/main"
DOWNLOAD_TIMEOUT_SEC = float(os.environ.get("GUPPY_DOWNLOAD_TIMEOUT_SEC", "30"))


def download_bases() -> list[str]:
    bases = []
    override = os.environ.get("GUPPY_HF_BASE", "").strip()
    if override:
        bases.append(override.rstrip("/"))
    bases.extend([HF_PRIMARY_BASE, HF_MIRROR_BASE])

    deduped = []
    for base in bases:
        if base and base not in deduped:
            deduped.append(base)
    return deduped


def synthesize_local_config(out_path: Path, guppy_root: Path) -> dict:
    sys.path.insert(0, str(guppy_root))
    try:
        from guppylm.config import GuppyConfig
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "failed to import local guppylm.config while synthesizing config.json"
        ) from exc

    cfg = GuppyConfig()
    raw_config = {
        "model_type": "guppylm",
        "architectures": ["GuppyLM"],
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_seq_len,
        "hidden_size": cfg.d_model,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "intermediate_size": cfg.ffn_hidden,
        "hidden_dropout_prob": cfg.dropout,
        "pad_token_id": cfg.pad_id,
        "bos_token_id": cfg.bos_id,
        "eos_token_id": cfg.eos_id,
    }
    out_path.write_text(json.dumps(raw_config, indent=2, ensure_ascii=False) + "\n")
    return {
        "name": "config.json",
        "path": str(out_path),
        "source": "synthesized-local",
        "source_path": str(guppy_root / "guppylm" / "config.py"),
        "size_bytes": out_path.stat().st_size,
    }


def copy_or_download(name: str, out_path: Path, local_candidates: list[Path]) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for candidate in local_candidates:
        if candidate.is_file():
            shutil.copy2(candidate, out_path)
            return {
                "name": name,
                "path": str(out_path),
                "source": "local-copy",
                "source_path": str(candidate),
                "size_bytes": out_path.stat().st_size,
            }

    errors = []
    for base in download_bases():
        url = f"{base}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
                with out_path.open("wb") as f:
                    shutil.copyfileobj(response, f)
            return {
                "name": name,
                "path": str(out_path),
                "source": "download",
                "source_url": url,
                "size_bytes": out_path.stat().st_size,
            }
        except (OSError, TimeoutError, urllib.error.URLError, socket.timeout) as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        f"failed to download {name}; attempted bases:\n" + "\n".join(errors)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the minimum Guppy stage-A assets."
    )
    parser.add_argument(
        "--out-dir",
        default="build/guppy/assets",
        help="Output directory for pytorch_model.bin/config.json/tokenizer.json",
    )
    parser.add_argument(
        "--guppylm-root",
        default=os.path.expanduser("~/guppylm"),
        help="Optional local guppylm checkout used as a local asset source",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    guppy_root = Path(args.guppylm_root).expanduser().resolve()

    manifest = {
        "schema_version": 1,
        "repo": HF_REPO,
        "out_dir": str(out_dir),
        "download_bases": download_bases(),
        "files": [],
    }

    specs = [
        (
            "pytorch_model.bin",
            [
                guppy_root / "checkpoints" / "pytorch_model.bin",
                guppy_root / "pytorch_model.bin",
            ],
        ),
        (
            "config.json",
            [
                guppy_root / "checkpoints" / "config.json",
                guppy_root / "config.json",
            ],
        ),
        (
            "tokenizer.json",
            [
                guppy_root / "data" / "tokenizer.json",
                guppy_root / "docs" / "tokenizer.json",
                guppy_root / "tokenizer.json",
            ],
        ),
    ]

    for name, candidates in specs:
        out_path = out_dir / name
        try:
            info = copy_or_download(name, out_path, candidates)
        except RuntimeError:
            if name != "config.json":
                raise
            info = synthesize_local_config(out_path, guppy_root)
        manifest["files"].append(info)
        size_mb = info["size_bytes"] / 1e6
        print(f"{name}: {size_mb:.1f} MB [{info['source']}]")

    manifest_path = out_dir / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
