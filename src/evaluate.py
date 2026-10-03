"""Evaluate all cached forecast cubes on the validation and test windows.

Outputs (results/metrics/):
  metrics_<window>.csv          model x horizon: MAE, R2, R2_change, WAPE, MdAPE, Bias%
  metrics_<window>_bycat.csv    the same per category
  mae_curve_<window>.csv        MAE for every h = 1..12
  bootstrap_test_vs_prophet.csv MAE(model) - MAE(Prophet) with a 95% cluster-bootstrap CI
                                (resampling municipalities, so the 6 categories of one МО
                                stay together)

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
EXCLUDE_PREFIX = ("feat_", "tmp_", "fmlocal_", "ensemble_final", "ensemble_v2_final")  # fmlocal_ = local space, not rubles


def list_models() -> list[str]:
    return sorted(p.stem for p in CACHE.glob("*.npz") if not p.stem.startswith(EXCLUDE_PREFIX))


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


def cluster_bootstrap(cubes, Y, meta, targets, h, ref="prophet", reps=1000, seed=0):
    """MAE(model) - MAE(ref) with 95% CI; clusters = municipalities."""
    pairs = window_pairs(targets, [h])
    terr = meta["territory_id"].to_numpy()
    uniq, inv = np.unique(terr, return_inverse=True)
    yr, pr, _, _, sid = gather(cubes[ref], Y, pairs)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    counts = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws])   # (reps, n_terr)
    cl = inv[sid]
    e_ref = np.abs(pr - yr)
    out = []
    for name, C in cubes.items():
        if name == ref:
            continue
        y, p, _, _, _ = gather(C, Y, pairs)
        d = np.abs(p - y) - e_ref
        if not np.isfinite(d).all():
            continue
        s = np.bincount(cl, weights=d, minlength=len(uniq))
        c = np.bincount(cl, minlength=len(uniq))
        bs = (counts @ s) / (counts @ c)
        out.append({"model": name, "h": h, "dMAE": d.mean(), "lo95": np.quantile(bs, 0.025),
                    "hi95": np.quantile(bs, 0.975), "share_better": (d < 0).mean()})
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


def significance_v2(cubes, Y, meta, targets, H, reps=1000, seed=0):
    """ensemble_v2 against Prophet, the v1 ensemble and the best single base model of each
    horizon on TEST (chosen after the fact: the hardest fair reference).

    Three views of the uncertainty of dMAE = MAE(v2) - MAE(ref):
      cluster   bootstrap over municipalities (the 6 categories of an МО stay together)
      two-way   municipalities and target months resampled independently; errors of all
                series in one month share the national component, so this CI is wider
      DM-HLN    Diebold-Mariano test on the 6 monthly mean loss differentials
    Output: results/metrics/significance_v2.csv"""
    t = pd.read_csv(OUT / "metrics_test.csv")
    single = t[~t.model.str.startswith("ensemble")]
    terr = meta["territory_id"].to_numpy()
    uniq, inv = np.unique(terr, return_inverse=True)
    rng = np.random.default_rng(seed)
    nT = len(targets)
    rows = []
    for h in H:
        best = single[single.h == h].sort_values("MAE").model.iloc[0]
        pairs = window_pairs(targets, [h])
        y, p2, _, T, sid = gather(cubes["ensemble_v2"], Y, pairs)
        for ref, kind in ((best, "best single (test)"), ("ensemble", "ensemble v1"), ("prophet", "Prophet")):
            if ref not in cubes:
                continue
            pr = gather(cubes[ref], Y, pairs)[1]
            d = np.abs(p2 - y) - np.abs(pr - y)
            # matrix month x municipality of summed differentials and counts
            ti = np.searchsorted(np.array(sorted(targets)), T)
            S = np.zeros((nT, len(uniq))); np.add.at(S, (ti, inv[sid]), d)
            N = np.zeros((nT, len(uniq))); np.add.at(N, (ti, inv[sid]), 1)
            cm = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
                           for _ in range(reps)])
            tm = np.stack([np.bincount(rng.integers(0, nT, nT), minlength=nT) for _ in range(reps)])
            cl = (cm @ S.sum(0)) / (cm @ N.sum(0))
            tw = np.einsum("rt,tm,rm->r", tm, S, cm) / np.einsum("rt,tm,rm->r", tm, N, cm)
            stat, pv = dm_hln(S.sum(1) / N.sum(1), h)
            rows.append({"h": h, "ref": ref, "ref_kind": kind, "MAE_v2": np.abs(p2 - y).mean(),
                         "MAE_ref": np.abs(pr - y).mean(), "dMAE": d.mean(),
                         "dMAE_pct": d.mean() / np.abs(pr - y).mean() * 100,
                         "cluster_lo95": np.quantile(cl, 0.025), "cluster_hi95": np.quantile(cl, 0.975),
                         "twoway_lo95": np.quantile(tw, 0.025), "twoway_hi95": np.quantile(tw, 0.975),
                         "DM_HLN": stat, "DM_p": pv, "share_series_better": (d < 0).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "significance_v2.csv", index=False)
    print(out.round(3).to_string(index=False))


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
    if not a.no_bootstrap and "prophet" in cubes:
        bs = pd.concat([cluster_bootstrap(cubes, Y, meta, W["test"], h, reps=cfg["backtest"]["bootstrap_reps"])
                        for h in H])
        bs.to_csv(OUT / "bootstrap_test_vs_prophet.csv", index=False)
    if not a.no_bootstrap and "ensemble_v2" in cubes:
        significance_v2(cubes, Y, meta, W["test"], H, cfg["backtest"]["bootstrap_reps"])

if __name__ == "__main__":
    main()
