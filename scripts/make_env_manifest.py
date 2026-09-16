#!/usr/bin/env python3
"""Record the RL environment this milestone's numbers were produced on (spec §8, M2).

Written once at M2 start and re-run whenever a package moves. The numpy version is called out
separately because projectairsim 1.0.2 is built against 1.26.4 and a silent upgrade would break the
simulator client long after the fact.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ("numpy", "torch", "stable-baselines3", "gymnasium", "tensorboard", "projectairsim", "opencv-python")


def _driver_version() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_manifest() -> dict:
    import torch

    available = bool(torch.cuda.is_available())
    return {
        "description": "The Python environment M2's throughput and training numbers were measured on.",
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {p: md.version(p) for p in PACKAGES},
        "cuda": {
            "available": available,
            "device_name": torch.cuda.get_device_name(0) if available else None,
            "torch_cuda": torch.version.cuda,
            "driver": _driver_version(),
        },
        "numpy_pinned": "1.26.4 — projectairsim 1.0.2 is built against it; do not upgrade",
    }


def main() -> int:
    from autofly_ue5.paths import ROOT

    out = ROOT / "docs" / "gates" / "m2_env_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_manifest(), indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
