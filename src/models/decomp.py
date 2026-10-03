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

Usage: .venv/bin/python src/models/decomp.py [--version v2]
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


# v2: each category gets the Rosstat series closest to it (v1 used retail_total for all)
ROSSTAT_FOR = {"Все категории": "retail_total", "Здоровье": "retail_nonfood",
               "Маркетплейсы": "retail_nonfood", "Общественное питание": "food_service",
               "Продовольствие": "retail_food", "Транспорт": "retail_total"}


def rosstat_yoy(months: list[str], lag: int, k: int = 3, col: str = "retail_total") -> np.ndarray:
    """Rosstat YoY log growth of `col`, mean of k months ending at o - lag, per origin o."""
    ros = pd.read_csv(ROOT / "data" / "external" / "rosstat_monthly.csv").set_index("date")
    rl = np.log(ros[col])
    yoy = rl - rl.shift(12)
    out = np.zeros(len(months))
    for o, m in enumerate(months):
        per = pd.Period(m, "M")
        out[o] = np.mean([yoy[str(per - lag - j)] for j in range(k)])
    return out


def national_forecast(nat: np.ndarray, months, k=3, w=1.0, lag=2, cats=None) -> np.ndarray:
    """Log national forecast cube (n_series, T, HMAX).

    cats=None: Rosstat retail_total prior for every series (v1). With cats, each category
    uses the Rosstat series in ROSSTAT_FOR (v2)."""
    n, T = nat.shape
    if cats is None:
        prior = np.repeat(rosstat_yoy(months, lag)[None, :], n, axis=0)
    else:
        prior = np.zeros((n, T))
        for c in np.unique(cats):
            prior[cats == c] = rosstat_yoy(months, lag, col=ROSSTAT_FOR[c])[None, :]
    out = empty_cube(n, T)
    for o in range(T):
        if o >= 12:
            kk = min(k, o - 11)
            own = (nat[:, o - kk + 1:o + 1] - nat[:, o - kk + 1 - 12:o + 1 - 12]).mean(1)
            G = w * own + (1 - w) * prior[:, o]
        else:
            G = prior[:, o]
        for h in range(1, HMAX + 1):
            t = o + h
            # t - 12 < 0 only for origins before 2023-12 at short horizons (not evaluated)
            out[:, o, h - 1] = nat[:, t - 12] + G if t - 12 >= 0 else nat[:, o] + G * h / 12
    return out


def portal_forecast(nat: np.ndarray, months, cats, years: int = 3) -> np.ndarray:
    """v3: log national forecast from the SberIndex portal national series (src/exog/portal.py).

    For category c with portal series p (log, months <= origin o only):
      A  n[T-12] + G,  G = YoY of p at o                       (one-year panel seasonality)
      B  n[o] + (p_hat[T] - p[o]),  p_hat[T] = p[o] + mean over the last `years` years of the
         seasonal increment p[T-12y] - p[o-12y], corrected to the current growth:
         + h/12 * (G - mean annual growth of those years)     (multi-year seasonality)
      forecast = mean(A, B); B alone if T < 12 (no panel month T-12).
    """
    from exog.portal import PORTAL_FOR, load_portal
    P = np.log(load_portal())
    base = pd.Period(months[0], "M")
    n, T = nat.shape
    out = empty_cube(n, T)
    for c in np.unique(cats):
        m = cats == c
        i0 = np.where(m)[0][0]
        nc, p = nat[i0], P[PORTAL_FOR[c]]
        pv = lambda t: p[str(base + t)]  # noqa: E731  (t = month index, may be < 0)
        A = np.full((T, HMAX), np.nan)
        for o in range(T):
            G = pv(o) - pv(o - 12)
            gy = np.mean([pv(o - 12 * (y - 1)) - pv(o - 12 * y) for y in range(1, years + 1)])
            for h in range(1, HMAX + 1):
                t = o + h
                inc = np.mean([pv(t - 12 * y) - pv(o - 12 * y) for y in range(1, years + 1)])
                b = nc[o] + inc + h / 12 * (G - gy)
                A[o, h - 1] = 0.5 * (nc[t - 12] + G) + 0.5 * b if t - 12 >= 0 else b
        out[m] = A[None]
    return out


def local_season_shift(d: np.ndarray) -> np.ndarray:
    """S[i, o, h-1] = s_i(month of o+h) - s_i(month of o), where s_i is the local seasonal
    deviation of series i over the 12 months o-11..o (d minus its 12-month mean). Zero for
    o < 11 (no full year yet). A local forecast plus lam * S puts back a share lam of the
    municipality's own seasonal pattern, which the national component does not remove."""
    n, T = d.shape
    S = np.zeros((n, T, HMAX), dtype=np.float32)
    for o in range(11, T):
        dev = d[:, o - 11:o + 1] - d[:, o - 11:o + 1].mean(1, keepdims=True)
        for h in range(1, HMAX + 1):
            S[:, o, h - 1] = dev[:, (h - 1) % 12] - dev[:, 11]
    return S


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


def tune(Y, nat, d, months, targets, horizons, cats=None):
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
            N = national_forecast(nat, months, k=k, w=w, cats=cats)
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2", "v3"], default="v1",
                    help="v1: Rosstat retail_total prior, cubes dec_*; v2: category Rosstat prior, "
                         "cubes dec2_*; v3: v2 + SberIndex portal national series, cubes dec3_*")
    a = ap.parse_args()
    v2 = a.version in ("v2", "v3")
    pre, nat_file = {"v1": ("dec_", "feat_national_log"), "v2": ("dec2_", "feat_national_log_v2"),
                     "v3": ("dec3_", "feat_national_log_v3")}[a.version]
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    L = np.log(Y)
    cats = meta["category"].to_numpy()
    nat = national_paths(L, cats)
    d = L - nat
    idx = {m: i for i, m in enumerate(months)}
    val = [idx[m] for m in cfg["backtest"]["validation_targets"]]
    H = list(range(1, HMAX + 1))
    cv = cats if v2 else None
    best, grid = tune(Y, nat, d, months, val, H, cats=cv)
    grid.to_csv(ROOT / "results" / "metrics" / f"decomp_grid_validation{'_v2' if v2 else ''}.csv", index=False)
    print("chosen on validation (h: k, w):", best)
    N = empty_cube(*Y.shape)
    for kw in set(best.values()):
        Nk = national_forecast(nat, months, k=kw[0], w=kw[1], cats=cv)
        for h, v in best.items():
            if v == kw:
                N[:, :, h - 1] = Nk[:, :, h - 1]
    if a.version == "v3":
        # chosen on validation (LOG 2026-10-03, ensemble-v2): origins < 12 (no own YoY yet) use
        # the portal forecast; origins >= 12 use the mean of v2 and the portal forecast
        Np = portal_forecast(nat, months, cats)
        N[:, :12] = Np[:, :12]
        N[:, 12:] = 0.5 * (N[:, 12:] + Np[:, 12:])
    np.savez_compressed(CACHE / f"{nat_file}.npz", p50=N)
    for name, D in local_cubes(d).items():
        save_cube(f"{pre}{name}", np.exp(N + D))
        print("saved", f"{pre}{name}")
    if a.version == "v3":
        # persistence plus a share lam of the local seasonal pattern; lam in {0.2, 0.4} from a
        # validation grid 0..1 (LOG 2026-10-03); the stack chooses between them
        S = local_season_shift(d)
        for lam in (0.2, 0.4):
            save_cube(f"{pre}persist_sa{int(lam * 10)}", np.exp(N + d[:, :, None] + lam * S))
            print("saved", f"{pre}persist_sa{int(lam * 10)}")
    # foundation models on the local component (if their cubes exist)
    for p in sorted(CACHE.glob("fmlocal_*.npz")):
        with np.load(p) as z:
            D = np.log(np.maximum(z["p50"], 1e-6))
        save_cube(f"{pre}{p.stem}", np.exp(N + D))
        print("saved", f"{pre}{p.stem}")


if __name__ == "__main__":
    main()
