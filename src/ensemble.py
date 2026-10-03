"""Ensemble selection (Caruana et al., 2004) per horizon, on the VALIDATION window only.

For each horizon h, greedy forward selection with replacement: start from nothing, add
the model that most reduces the MAE of the simple average, `steps` times. The weight of a
model = how often it was picked / steps. Validation pairs of the neighbour horizons h-1
and h+1 are pooled with h, which makes the weights smoother across h.

The candidate pool is frozen in configs/config.yaml (ensemble.pool): the 35 cubes of the
headline ensemble v1. Cubes added later do not change v1.

Outputs:
  results/cache/<name>.npz              cube with validation-selected weights (honest test)
  results/cache/<name>_final.npz        weights re-selected on validation + test (for 2025)
  results/metrics/<name>_weights.csv    weights per horizon, both versions

Usage: .venv/bin/python src/ensemble.py                    # v1 -> ensemble, ensemble_weights.csv
       .venv/bin/python src/ensemble.py --variant ensemble_nat3   # post-hoc variant (LOG)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, HMAX, ROOT, empty_cube, gather, get_panel, load_config, save_cube  # noqa: E402
from evaluate import list_models, load_p50, windows  # noqa: E402

# zero-shot level FMs and raw baselines stay as candidates; the selection decides
NEVER = ("ensemble", "ensemble_final", "snaive", "drift")


def select(cubes: dict, Y, targets, h, steps=20):
    hs = [x for x in (h - 1, h, h + 1) if 1 <= x <= HMAX]
    pairs = [(T - x, x) for x in hs for T in targets if T - x >= 0]
    y = None
    P = {}
    for nm, C in cubes.items():
        yy, p = gather(C, Y, pairs)[:2]
        if not np.isfinite(p).all():
            continue
        P[nm] = p.astype(np.float64); y = yy
    names = list(P)
    M = np.stack([P[n] for n in names])            # (models, rows)
    counts = np.zeros(len(names))
    cur = np.zeros_like(y, dtype=np.float64)
    for s in range(steps):
        cand = (cur[None, :] * s + M) / (s + 1)
        mae = np.abs(cand - y[None, :]).mean(axis=1)
        j = int(np.argmin(mae))
        counts[j] += 1
        cur = cand[j]
    w = counts / counts.sum()
    return {n: float(wi) for n, wi in zip(names, w) if wi > 0}


def combine(cubes, W: dict[int, dict], shape):
    out = empty_cube(*shape)
    for h, w in W.items():
        acc = np.zeros(shape, dtype=np.float64)
        for nm, wi in w.items():
            acc += wi * cubes[nm][:, :, h - 1]
        out[:, :, h - 1] = acc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="*")
    ap.add_argument("--variant", default=None, help="a key of ensemble.variants in the config")
    a = ap.parse_args()
    cfg = load_config()
    ec = cfg["ensemble"]
    steps = ec.get("steps", 20)
    meta, Y, months = get_panel(cfg)
    names = a.candidates or ec.get("pool") or [m for m in list_models() if m not in NEVER]
    base = "ensemble"
    if a.variant:
        base = a.variant
        pre = tuple(ec["variants"][a.variant]["extra_prefix"])
        names = list(names) + [m for m in list_models() if m.startswith(pre) and m not in names]
    print(f"{base}: {len(names)} candidates")
    cubes = {n: load_p50(n) for n in names}
    W = windows(cfg, months)
    rows = []
    for tag, targets, out in (("validation", W["validation"], base),
                              ("validation+test", W["validation"] + W["test"], f"{base}_final")):
        weights = {h: select(cubes, Y, targets, h, steps) for h in range(1, HMAX + 1)}
        save_cube(out, combine(cubes, weights, Y.shape))
        for h, w in weights.items():
            for nm, wi in w.items():
                rows.append({"fit_on": tag, "h": h, "model": nm, "weight": wi})
        print(f"{out}: weights chosen on {tag}")
        for h in (1, 3, 6, 12):
            print(f"   h={h:2d}:", ", ".join(f"{k} {v:.2f}" for k, v in sorted(weights[h].items(), key=lambda kv: -kv[1])))
    (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "metrics" / f"{base}_weights.csv", index=False)


if __name__ == "__main__":
    main()
