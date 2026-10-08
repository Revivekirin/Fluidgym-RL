"""Verification of generated PNG/GIF artifacts (file exists, decodes, is not blank / not a static GIF)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fluidgym_rl.report import CheckFailure


def verify_png(path, min_std: float = 1e-3) -> dict:
    from PIL import Image
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise CheckFailure(f"{p} missing or empty")
    if p.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise CheckFailure(f"{p} is not a PNG")
    arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    info = {"path": str(p), "bytes": p.stat().st_size, "shape": list(arr.shape), "pixel_std": float(arr.std())}
    if arr.std() < min_std:
        raise CheckFailure("PNG decodes but is (almost) constant/blank", info)
    return info


def verify_gif(path, min_frames: int = 2) -> dict:
    from PIL import Image
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise CheckFailure(f"{p} missing or empty")
    if p.read_bytes()[:4] != b"GIF8":
        raise CheckFailure(f"{p} is not a GIF")
    im = Image.open(p)
    n = getattr(im, "n_frames", 1)
    info = {"path": str(p), "bytes": p.stat().st_size, "frames": n, "size": list(im.size)}
    if n < min_frames:
        raise CheckFailure(f"GIF has {n} frames (< {min_frames})", info)
    first = np.asarray(im.convert("RGB"))
    im.seek(n - 1)
    last = np.asarray(im.convert("RGB"))
    info["first_vs_last_mean_abs_diff"] = float(np.abs(first.astype(float) - last.astype(float)).mean())
    if info["first_vs_last_mean_abs_diff"] == 0.0:
        raise CheckFailure("GIF frames are identical (no visible evolution)", info)
    return info
