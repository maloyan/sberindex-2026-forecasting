"""Simple and classical baselines, computed for every origin o and horizon h = 1..12.

  naive               y(o)
  snaive              y(T-12); before a full year of history: y(o)
  snaive_growth       y(T-12) * exp(mean YoY log growth of the series over the last k months)
  snaive_growth_nat   y(T-12) * exp(median YoY log growth of the category, last k months)
                      (robust to one noisy series; same cross-section info as LightGBM)
  drift               y(o) * exp(h * mean monthly log change over the history)
  prophet             organizer's base model, one Prophet per series (src/models/prophet_model.py)
  ets/theta/arima     statsforecast AutoETS / AutoTheta / AutoARIMA (src/models/statsf.py)

Usage: .venv/bin/python src/models/baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HMAX, empty_cube, get_panel, load_config, save_cube  # noqa: E402


def yoy_growth(L: np.ndarray, o: int, k: int, codes=None) -> np.ndarray | None:
    """Mean YoY log growth over the k months up to o (None if no year-ago data)."""
    kk = min(k, o - 11)
    if kk <= 0:
        return None
    g = (L[:, o - kk + 1:o + 1] - L[:, o - kk + 1 - 12:o + 1 - 12]).mean(axis=1)
    if codes is None:
        return g
    out = np.empty_like(g)
    for c in np.unique(codes):
        m = codes == c
        out[m] = np.median(g[m])
    return out


def cubes(Y: np.ndarray, codes: np.ndarray, k: int = 3) -> dict[str, np.ndarray]:
    n, T = Y.shape
    L = np.log(Y)
    out = {m: empty_cube(n, T) for m in ("naive", "snaive", "snaive_growth", "snaive_growth_nat", "drift")}
    for o in range(T):
        g_own = yoy_growth(L, o, k)
        g_nat = yoy_growth(L, o, k, codes)
        slope = (L[:, o] - L[:, 0]) / o if o > 0 else np.zeros(n)
        for h in range(1, HMAX + 1):
            t = o + h
            out["naive"][:, o, h - 1] = Y[:, o]
            base = Y[:, t - 12] if 0 <= t - 12 <= o else Y[:, o]
            out["snaive"][:, o, h - 1] = base
            out["snaive_growth"][:, o, h - 1] = base * np.exp(g_own) if g_own is not None else base
            out["snaive_growth_nat"][:, o, h - 1] = base * np.exp(g_nat) if g_nat is not None else base
            out["drift"][:, o, h - 1] = Y[:, o] * np.exp(h * slope)
    return out


def main():
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    codes = meta["category"].astype("category").cat.codes.to_numpy()
    for name, c in cubes(Y, codes, cfg["models"]["snaive_growth"]["k"]).items():
        save_cube(name, c)
        print("saved", name)


if __name__ == "__main__":
    main()
