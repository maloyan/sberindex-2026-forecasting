"""Hierarchical decomposition: national (category) component x local (municipal) component.

    log y[i,t] = n[c(i), t] + d[i,t]

n[c,t] is the median log spending of category c over all municipalities in month t. It
holds the seasonality and the inflation that are common to all of Russia. It uses only
month-t data, so it is known at month t (no look-ahead). d[i,t] is the local deviation:
no common seasonality, slow dynamics, local shocks.

Forecast at origin o for target T = o + h:
  national:  n_hat[c,T] = n[c,T-12] + G[c,o]                 (seasonal naive + growth)
             G = annual log growth = w * own YoY of n (mean of last k months)
                                    + (1 - w) * Rosstat retail YoY (lag `lag` months)
             Before a full year of SberIndex data (o < 12), G = Rosstat prior only.
  local:     d_hat[i,T] from a local model:
             persist   d[i,o]
             mean3     mean of d[i,o-2..o]
             seas      d[i,T-12] + mean of (d[i,o-j] - d[i,o-j-12]), j < k   (o >= 12)
             fm_<name> foundation model forecast of exp(d) (src/fm/run_fm.py --space local)
             gbm       src/models/gbm.py (target = d[i,T] - d[i,o])
  forecast:  y_hat = exp(n_hat + d_hat)

The grid for (k, w) is searched on the VALIDATION window only (see `tune`).

Usage: .venv/bin/python src/models/decomp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, HMAX, ROOT, empty_cube, gather, get_panel, load_config, metrics, save_cube, window_pairs  # noqa: E402,E501


def national_paths(L: np.ndarray, cats: np.ndarray) -> np.ndarray:
    nat = np.zeros_like(L)
    for c in np.unique(cats):
        m = cats == c
        nat[m] = np.median(L[m], axis=0)
    return nat


def rosstat_yoy(months: list[str], lag: int, k: int = 3) -> np.ndarray:
    """Rosstat retail turnover YoY log growth, mean of k months ending at o - lag, per origin o."""
    ros = pd.read_csv(ROOT / "data" / "external" / "rosstat_monthly.csv").set_index("date")
    rl = np.log(ros["retail_total"])
    yoy = rl - rl.shift(12)
    out = np.zeros(len(months))
    for o, m in enumerate(months):
        per = pd.Period(m, "M")
        out[o] = np.mean([yoy[str(per - lag - j)] for j in range(k)])
    return out


def national_forecast(nat: np.ndarray, months, k=3, w=1.0, lag=2) -> np.ndarray:
    """Log national forecast cube (n_series, T, HMAX)."""
    n, T = nat.shape
    prior = rosstat_yoy(months, lag)
    out = empty_cube(n, T)
    for o in range(T):
        if o >= 12:
            kk = min(k, o - 11)
            own = (nat[:, o - kk + 1:o + 1] - nat[:, o - kk + 1 - 12:o + 1 - 12]).mean(1)
            G = w * own + (1 - w) * prior[o]
        else:
            G = np.full(n, prior[o])
        for h in range(1, HMAX + 1):
            t = o + h
            # t - 12 < 0 only for origins before 2023-12 at short horizons (not evaluated)
            out[:, o, h - 1] = nat[:, t - 12] + G if t - 12 >= 0 else nat[:, o] + G * h / 12
    return out


def local_cubes(d: np.ndarray, k: int = 3) -> dict[str, np.ndarray]:
    n, T = d.shape
    out = {m: empty_cube(n, T) for m in ("persist", "mean3", "seas")}
    for o in range(T):
        for h in range(1, HMAX + 1):
            t = o + h
            out["persist"][:, o, h - 1] = d[:, o]
            out["mean3"][:, o, h - 1] = d[:, max(0, o - 2):o + 1].mean(1)
            if o >= 12 and t - 12 <= o:
                kk = min(k, o - 11)
                drift = (d[:, o - kk + 1:o + 1] - d[:, o - kk + 1 - 12:o + 1 - 12]).mean(1)
                out["seas"][:, o, h - 1] = d[:, t - 12] + drift
            else:
                out["seas"][:, o, h - 1] = d[:, o]
    return out


def tune(Y, nat, d, months, targets, horizons):
    """Grid over (k, w) on the validation window.

    Only validation pairs with origin >= 12 depend on (k, w). For a horizon where at least
    half of the validation pairs have origin >= 12 (h <= 3), (k, w) is chosen for that
    horizon. For longer horizons the choice would rest on 1-2 origins, so one pooled pair
    of values (all horizons together) is used.
    Returns {h: (k, w)} and the grid table.
    """
    loc = d[:, :, None]
    rows = []
    for k in (1, 2, 3, 6):
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            N = national_forecast(nat, months, k=k, w=w)
            C = np.exp(N + np.repeat(loc, HMAX, axis=2))      # persist local model
            for h in ["pooled"] + list(horizons):
                hs = list(horizons) if h == "pooled" else [h]
                pairs = [(o, x) for (o, x) in window_pairs(targets, hs) if o >= 12]
                if not pairs:
                    continue
                y, p = gather(C, Y, pairs)[:2]
                share = len(pairs) / len(window_pairs(targets, hs))
                rows.append({"h": h, "k": k, "w": w, "MAE": np.abs(y - p).mean(), "n": len(y),
                             "share_o_ge_12": share})
    r = pd.DataFrame(rows)
    pooled = r[r.h == "pooled"].sort_values("MAE").iloc[0]
    best = {}
    for h in horizons:
        rh = r[r.h == h]
        if len(rh) and rh.share_o_ge_12.iloc[0] >= 0.5:
            b = rh.sort_values("MAE").iloc[0]
            best[h] = (int(b.k), float(b.w))
        else:
            best[h] = (int(pooled.k), float(pooled.w))
    return best, r


def main():
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    L = np.log(Y)
    cats = meta["category"].to_numpy()
    nat = national_paths(L, cats)
    d = L - nat
    idx = {m: i for i, m in enumerate(months)}
    val = [idx[m] for m in cfg["backtest"]["validation_targets"]]
    H = list(range(1, HMAX + 1))
    best, grid = tune(Y, nat, d, months, val, H)
    grid.to_csv(ROOT / "results" / "metrics" / "decomp_grid_validation.csv", index=False)
    print("chosen on validation (h: k, w):", best)
    N = empty_cube(*Y.shape)
    for kw in set(best.values()):
        Nk = national_forecast(nat, months, k=kw[0], w=kw[1])
        for h, v in best.items():
            if v == kw:
                N[:, :, h - 1] = Nk[:, :, h - 1]
    np.savez_compressed(CACHE / "feat_national_log.npz", p50=N)
    for name, D in local_cubes(d).items():
        save_cube(f"dec_{name}", np.exp(N + D))
        print("saved", f"dec_{name}")
    # foundation models on the local component (if their cubes exist)
    for p in sorted(CACHE.glob("fmlocal_*.npz")):
        with np.load(p) as z:
            D = np.log(np.maximum(z["p50"], 1e-6))
        save_cube(f"dec_{p.stem}", np.exp(N + D))
        print("saved", f"dec_{p.stem}")


if __name__ == "__main__":
    main()
