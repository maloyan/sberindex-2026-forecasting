"""Zero-shot forecasts of the foundation models for every origin (rolling origin).

For each origin o (context = months 0..o of each series) the model predicts h = 1..12.
Results: results/cache/fm_<name>.npz with cubes p10, p50, p90 (see src/common.py).
Origins with fewer than `min_context` months get the naive forecast.

--space level  the model sees the raw series in rubles (plain zero-shot use)
--space local  the model sees exp(d), the local component of the decomposition
               log y = national(category, month) + d (src/models/decomp.py). This removes
               the common seasonality, which 24 months are too short to learn.
               Output: results/cache/fmlocal_<name>.npz (forecasts of exp(d)).

Usage:
  .venv/bin/python src/fm/run_fm.py --models chronos2 timesfm25 [--space local] [--device mps]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import make  # noqa: E402
from common import CACHE, HMAX, empty_cube, get_panel, load_config  # noqa: E402


def run(name: str, Y: np.ndarray, meta, origins, min_context: int, device: str, prefix: str = "fm"):
    n, T = Y.shape
    cubes = {k: empty_cube(n, T) for k in ("p10", "p50", "p90")}
    t0 = time.time()
    model = make(name, device)
    groups = None
    if name.endswith("_mv"):
        groups = [list(ix) for ix in meta.groupby("territory_id").indices.values()]
    for o in origins:
        if o + 1 < min_context:
            for k in cubes:
                cubes[k][:, o, :] = Y[:, o:o + 1]
            continue
        ctx = [Y[i, :o + 1] for i in range(n)]
        q = model(ctx, HMAX, groups=groups) if groups is not None else model(ctx, HMAX)
        q = np.maximum(q, 0.0)
        cubes["p10"][:, o], cubes["p50"][:, o], cubes["p90"][:, o] = q[:, 0], q[:, 1], q[:, 2]
        print(f"  {name} origin {o:2d}: {time.time() - t0:6.0f}s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"{prefix}_{name}.npz", **cubes, seconds=time.time() - t0)
    return time.time() - t0


def main():
    cfg = load_config()
    fcfg = cfg["foundation_models"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=fcfg["models"])
    ap.add_argument("--device", default=fcfg["device"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--space", choices=["level", "local"], default="level")
    a = ap.parse_args()
    meta, Y, months = get_panel(cfg)
    origins = list(range(len(months)))
    prefix = "fm"
    if a.space == "local":
        from models.decomp import national_paths
        L = np.log(Y)
        Y = np.exp(L - national_paths(L, meta["category"].to_numpy()))
        prefix = "fmlocal"
    for name in a.models:
        if (CACHE / f"{prefix}_{name}.npz").exists() and not a.force:
            print(f"{name}: cached, skip"); continue
        sec = run(name, Y, meta, origins, fcfg["min_context"], a.device, prefix)
        print(f"{name}: done in {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
