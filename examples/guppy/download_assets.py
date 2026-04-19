#!/usr/bin/env python3
"""Download or collect the minimum Guppy assets required for stage A."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path


HF_REPO = "arman-bd/guppylm-9M"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"


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

    url = f"{HF_BASE}/{name}"
    urllib.request.urlretrieve(url, out_path)
    return {
        "name": name,
        "path": str(out_path),
        "source": "download",
        "source_url": url,
        "size_bytes": out_path.stat().st_size,
    }


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
        info = copy_or_download(name, out_path, candidates)
        manifest["files"].append(info)
        size_mb = info["size_bytes"] / 1e6
        print(f"{name}: {size_mb:.1f} MB [{info['source']}]")

    manifest_path = out_dir / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
