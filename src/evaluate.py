"""Evaluate all cached forecast cubes on the validation and test windows.

Outputs (results/metrics/):
  metrics_<window>.csv          model x horizon: MAE, R2, R2_change, WAPE, MdAPE, Bias%
  metrics_<window>_bycat.csv    the same per category
  mae_curve_<window>.csv        MAE for every h = 1..12
  bootstrap_test_vs_prophet.csv MAE(model) - MAE(Prophet) on test, every model and h:
                                95% CI of the cluster bootstrap over municipalities (the 6
                                categories of one МО stay together) and of the TWO-WAY
                                bootstrap (municipalities x target months; the main CI),
                                DM-HLN test on the 6 monthly differences, months won
  significance_pairs.csv        the same for selected pairs (v1 vs v2, v1 vs nat3, ...)

Usage: .venv/bin/python src/evaluate.py [--models a b c]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, HMAX, ROOT, gather, get_panel, load_config, metrics, window_pairs  # noqa: E402

OUT = ROOT / "results" / "metrics"
EXCLUDE_PREFIX = ("feat_", "tmp_", "fmlocal_", "prophet_grid")  # fmlocal_ = local space, not rubles


def list_models() -> list[str]:
    """Rubles-space backtest cubes. *_final cubes (weights refitted on test, for 2025) are excluded."""
    return sorted(p.stem for p in CACHE.glob("*.npz")
                  if not p.stem.startswith(EXCLUDE_PREFIX) and not p.stem.endswith("_final"))


def load_p50(name: str) -> np.ndarray:
    with np.load(CACHE / f"{name}.npz") as z:
        return z["p50"]


def windows(cfg, months):
    idx = {m: i for i, m in enumerate(months)}
    bt = cfg["backtest"]
    return {"validation": [idx[m] for m in bt["validation_targets"]],
            "test": [idx[m] for m in bt["test_targets"]]}


def table(cubes: dict, Y, meta, targets, horizons, by_cat=False) -> pd.DataFrame:
    rows = []
    cats = meta["category"].to_numpy()
    for name, C in cubes.items():
        for h in horizons:
            pairs = window_pairs(targets, [h])
            y, p, hh, T, sid = gather(C, Y, pairs)
            yo = Y[sid, T - hh]
            ok = np.isfinite(p)
            if ok.mean() < 0.99:
                continue
            groups = [("all", np.ones_like(ok))] if not by_cat else [(c, cats[sid] == c) for c in np.unique(cats)]
            for g, msk in groups:
                m = ok & msk
                rows.append({"model": name, "h": h, "category": g, **metrics(y[m], p[m], yo[m])})
    return pd.DataFrame(rows)


def _draws(n_mo, n_t, reps, seed):
    rng = np.random.default_rng(seed)
    cm = np.stack([np.bincount(rng.integers(0, n_mo, n_mo), minlength=n_mo) for _ in range(reps)])
    tm = np.stack([np.bincount(rng.integers(0, n_t, n_t), minlength=n_t) for _ in range(reps)])
    return cm.astype(float), tm.astype(float)


def compare(C_a, C_b, Y, meta, targets, h, draws):
    """dMAE = MAE(a) - MAE(b) on (targets, h) with three views of its uncertainty:
      cluster   bootstrap over municipalities (the 6 categories of an МО stay together)
      two-way   municipalities and target months resampled independently. Errors of all
                series in one month share the national component, so this CI is wider and
                is the one we report
      DM-HLN    Diebold-Mariano test on the 6 monthly mean loss differentials
    Returns None if a cube has missing values on the window."""
    pairs = window_pairs(targets, [h])
    terr = meta["territory_id"].to_numpy()
    uniq, inv = np.unique(terr, return_inverse=True)
    y, pa, _, T, sid = gather(C_a, Y, pairs)
    pb = gather(C_b, Y, pairs)[1]
    d = np.abs(pa - y) - np.abs(pb - y)
    if not np.isfinite(d).all():
        return None
    nT = len(targets)
    ti = np.searchsorted(np.array(sorted(targets)), T)
    S = np.zeros((nT, len(uniq))); np.add.at(S, (ti, inv[sid]), d)
    N = np.zeros((nT, len(uniq))); np.add.at(N, (ti, inv[sid]), 1)
    cm, tm = draws
    cl = (cm @ S.sum(0)) / (cm @ N.sum(0))
    tw = np.einsum("rt,tm,rm->r", tm, S, cm) / np.einsum("rt,tm,rm->r", tm, N, cm)
    monthly = S.sum(1) / N.sum(1)
    stat, pv = dm_hln(monthly, h)
    return {"h": h, "MAE_a": np.abs(pa - y).mean(), "MAE_b": np.abs(pb - y).mean(), "dMAE": d.mean(),
            "dMAE_pct": d.mean() / np.abs(pb - y).mean() * 100,
            "cluster_lo95": np.quantile(cl, 0.025), "cluster_hi95": np.quantile(cl, 0.975),
            "twoway_lo95": np.quantile(tw, 0.025), "twoway_hi95": np.quantile(tw, 0.975),
            "DM_HLN": stat, "DM_p": pv, "months_a_better": int((monthly < 0).sum()), "n_months": nT,
            "share_series_better": (d < 0).mean()}


def vs_reference(cubes, Y, meta, targets, H, ref="prophet", reps=1000, seed=0):
    """Every model against `ref` (Prophet) on the test window."""
    n_mo = meta["territory_id"].nunique()
    draws = _draws(n_mo, len(targets), reps, seed)
    out = []
    for name, C in cubes.items():
        if name == ref:
            continue
        for h in H:
            r = compare(C, cubes[ref], Y, meta, targets, h, draws)
            if r is not None:
                out.append({"model": name, "ref": ref, **r})
    return pd.DataFrame(out)


PAIRS = [("ensemble", "prophet"), ("ensemble_v2", "ensemble"), ("ensemble_nat3", "ensemble"),
         ("ensemble_v2", "prophet"), ("ensemble_nat3", "prophet"), ("ensemble", "lgbm_v1"),
         ("ensemble", "gbm_full"), ("ensemble", "dec_fmlocal_timesfm25"), ("ensemble", "dec_persist")]


def pairs_table(cubes, Y, meta, targets, H, reps=1000, seed=0):
    n_mo = meta["territory_id"].nunique()
    draws = _draws(n_mo, len(targets), reps, seed)
    out = []
    for a, b in PAIRS:
        if a in cubes and b in cubes:
            for h in H:
                r = compare(cubes[a], cubes[b], Y, meta, targets, h, draws)
                if r is not None:
                    out.append({"a": a, "b": b, **r})
    return pd.DataFrame(out)


def dm_hln(dT: np.ndarray, h: int):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction on the series of
    monthly mean loss differentials dT (one value per target month). Newey-West variance with
    h-1 lags (capped at len-1). Returns (statistic, two-sided p-value, Student t, len-1 df).
    With 6 test months the HLN correction is undefined for h >= 6 (NaN)."""
    from scipy import stats
    n = len(dT)
    d = dT - dT.mean()
    L = min(h - 1, n - 1)
    var = (d @ d) / n + 2 * sum((1 - k / (L + 1)) * (d[k:] @ d[:-k]) / n for k in range(1, L + 1))
    var = max(var, (d @ d) / n * 1e-3)
    dm = dT.mean() / np.sqrt(var / n)
    corr = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if corr <= 0.05:          # HLN is undefined when h is close to the number of months
        return np.nan, np.nan
    stat = dm * np.sqrt(corr)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--no-bootstrap", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    names = a.models or list_models()
    cubes = {n: load_p50(n) for n in names}
    W = windows(cfg, months)
    H = cfg["backtest"]["horizons"]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    for w, targets in W.items():
        t = table(cubes, Y, meta, targets, H)
        t.to_csv(OUT / f"metrics_{w}.csv", index=False)
        table(cubes, Y, meta, targets, H, by_cat=True).to_csv(OUT / f"metrics_{w}_bycat.csv", index=False)
        curve = table(cubes, Y, meta, targets, list(range(1, HMAX + 1)))
        curve.pivot(index="model", columns="h", values="MAE").to_csv(OUT / f"mae_curve_{w}.csv")
        print(f"\n=== {w}: MAE, rubles per resident per month ===")
        print(t.pivot(index="model", columns="h", values="MAE").round(1).sort_values(6).to_string())
        print(f"--- {w}: R2 on the log change vs origin ---")
        print(t.pivot(index="model", columns="h", values="R2_change").round(3).sort_values(6, ascending=False).to_string())
    reps = cfg["backtest"]["bootstrap_reps"]
    if not a.no_bootstrap and "prophet" in cubes:
        bs = vs_reference(cubes, Y, meta, W["test"], H, reps=reps)
        bs.to_csv(OUT / "bootstrap_test_vs_prophet.csv", index=False)
        print("\n=== test: dMAE vs Prophet, two-way 95% CI ===")
        print(bs[bs.model.str.startswith("ensemble")].round(2).to_string(index=False))
        pt = pairs_table(cubes, Y, meta, W["test"], H, reps=reps)
        pt.to_csv(OUT / "significance_pairs.csv", index=False)
        print(pt.round(2).to_string(index=False))

if __name__ == "__main__":
    main()
