"""Recompute the headline numbers from a small committed cache and check them (fresh clone).

The full forecast cache (results/cache/, 2.7 GB, 4-5 h to rebuild) is not in git. This script
needs only the contest data (`make data`) and results/verify/cubes.npz (about 8 MB): the
forecasts of 8 key models on the TEST window (h = 1, 3, 6, 12; the headline ensemble for all
h = 1..12) and the final 2025 forecast cube, float32 with the mantissa cut to 14 bits
(relative precision 6e-5; the values compress well).

It recomputes and compares with the committed tables:
  1. MAE, R2, R2 on change (results/metrics/metrics_test.csv)
  2. MAE by category, incl. "Все категории" (metrics_test_bycat.csv)
  3. dMAE vs Prophet with the two-way bootstrap CI and DM-HLN (bootstrap_test_vs_prophet.csv)
  4. v1 vs v2 / nat3 pairs (significance_pairs.csv)
  5. the 2025 forecast with 80% intervals (results/forecast_2025.csv)

Usage:
  .venv/bin/python src/verify.py            # check (about 1 minute); exit code 1 on a mismatch
  .venv/bin/python src/verify.py --build    # write results/verify/cubes.npz from results/cache/
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HMAX, ROOT, get_panel, load_config, load_cube  # noqa: E402

OUT = ROOT / "results" / "verify" / "cubes.npz"
MET = ROOT / "results" / "metrics"
MODELS = ["ensemble", "prophet", "ensemble_v2", "ensemble_nat3", "dec_persist", "gbm_full",
          "dec_fmlocal_timesfm25", "lgbm_v1"]
H4 = [1, 3, 6, 12]
TOL_MAE = 0.05          # rubles; forecasts are stored with relative precision 6e-5
TOL_CI = 0.5            # rubles; bootstrap bounds move a little with the rounding
MASK = np.uint32(0xFFFFFFFF << 9 & 0xFFFFFFFF)   # keep 14 of 23 mantissa bits


def pack(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return (u + np.uint32(1 << 8)) & MASK          # round to nearest, then cut


def unpack(a):
    return a.view(np.float32)


def panel_key(meta) -> str:
    s = "|".join(f"{t}:{c}" for t, c in zip(meta["territory_id"], meta["category"]))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def build(cfg, meta, Y, months):
    idx = {m: i for i, m in enumerate(months)}
    test = [idx[m] for m in cfg["backtest"]["test_targets"]]
    o_final = months.index(cfg["forecast"]["origin"])
    arrs = {}
    for m in MODELS:
        C = load_cube(m)
        H = range(1, HMAX + 1) if m == "ensemble" else H4
        for h in H:
            arrs[f"{m}__h{h}"] = pack(np.stack([C[:, T - h, h - 1] for T in test], 1))
    arrs["ensemble_final__o"] = pack(load_cube("ensemble_final")[:, o_final, :])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, panel_key=np.array(panel_key(meta)), test=np.array(test), **arrs)
    print(f"wrote {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.1f} MB")


def cubes_from_npz(z, n, T, test, o_final):
    cubes = {}
    for k in z.files:
        if "__h" not in k:
            continue
        m, h = k.split("__h")
        C = cubes.setdefault(m, np.full((n, T, HMAX), np.nan, dtype=np.float32))
        a = unpack(z[k])
        for j, Tt in enumerate(test):
            C[:, Tt - int(h), int(h) - 1] = a[:, j]
    F = np.full((n, T, HMAX), np.nan, dtype=np.float32)
    F[:, o_final, :] = unpack(z["ensemble_final__o"])
    return cubes, F


def check(name, got: pd.Series, ref: pd.Series, tol, rel=False, atol=0.0) -> bool:
    """|got - ref| <= tol (absolute), or with rel=True <= max(atol, tol * |ref|)."""
    diff = (got.to_numpy() - ref.to_numpy())
    lim = np.maximum(atol, tol * np.abs(ref.to_numpy())) if rel else tol
    ok = bool((np.abs(diff) <= lim).all())
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max |diff| = {np.abs(diff).max():.3g}"
          f" (tolerance {'%g x |value|' % tol if rel else tol}{', min %g' % atol if atol else ''})")
    return ok


def verify(cfg, meta, Y, months):
    from evaluate import pairs_table, table, vs_reference
    from forecast_final import build as build_forecast
    z = np.load(OUT)
    if str(z["panel_key"]) != panel_key(meta):
        sys.exit("panel differs from the one the cache was built on (check `make data`)")
    test = [int(t) for t in z["test"]]
    n, T = Y.shape
    o_final = months.index(cfg["forecast"]["origin"])
    cubes, F = cubes_from_npz(z, n, T, test, o_final)
    reps = cfg["backtest"]["bootstrap_reps"]
    ok = []
    print("1. Metrics on test (MAE, R2, R2 on change)")
    t = table(cubes, Y, meta, test, H4).set_index(["model", "h"])
    ref = pd.read_csv(MET / "metrics_test.csv").set_index(["model", "h"]).loc[t.index]
    ok.append(check("MAE", t.MAE, ref.MAE, TOL_MAE))
    ok.append(check("R2 (levels)", t.R2, ref.R2, 1e-4))
    ok.append(check("R2 on change", t.R2_change, ref.R2_change, 1e-3, rel=True, atol=1e-3))
    print("2. MAE by category")
    tb = table(cubes, Y, meta, test, H4, by_cat=True).set_index(["model", "h", "category"])
    rb = pd.read_csv(MET / "metrics_test_bycat.csv").set_index(["model", "h", "category"]).loc[tb.index]
    ok.append(check("MAE by category", tb.MAE, rb.MAE, TOL_MAE))
    print("3. dMAE vs Prophet, two-way bootstrap CI, DM-HLN")
    b = vs_reference(cubes, Y, meta, test, H4, reps=reps).set_index(["model", "h"])
    rb = pd.read_csv(MET / "bootstrap_test_vs_prophet.csv").set_index(["model", "h"]).loc[b.index]
    ok.append(check("dMAE", b.dMAE, rb.dMAE, TOL_MAE))
    ok.append(check("two-way CI low", b.twoway_lo95, rb.twoway_lo95, TOL_CI))
    ok.append(check("two-way CI high", b.twoway_hi95, rb.twoway_hi95, TOL_CI))
    ok.append(check("DM-HLN p", b.DM_p.fillna(-1), rb.DM_p.fillna(-1), 1e-2))
    print("4. Pairs (v1 vs v2, v1 vs nat3, ...)")
    p = pairs_table(cubes, Y, meta, test, H4, reps=reps).set_index(["a", "b", "h"])
    rp = pd.read_csv(MET / "significance_pairs.csv").set_index(["a", "b", "h"]).loc[p.index]
    ok.append(check("pair dMAE", p.dMAE, rp.dMAE, TOL_MAE))
    ok.append(check("pair two-way CI", pd.concat([p.twoway_lo95, p.twoway_hi95]),
                    pd.concat([rp.twoway_lo95, rp.twoway_hi95]), TOL_CI))
    print("5. Forecast 2025")
    f = build_forecast(cfg, meta, Y, months, F, cubes["ensemble"])
    rf = pd.read_csv(ROOT / cfg["forecast"]["out"])
    ok.append(check("forecast, lo80, hi80 (rubles)", pd.concat([f.forecast, f.lo80, f.hi80]),
                    pd.concat([rf.forecast, rf.lo80, rf.hi80]), 1e-4, rel=True, atol=0.11))   # rounded to 0.1
    s = b.loc["ensemble"]
    print("\nHeadline: ensemble v1 vs Prophet on test (MAE, two-way 95% CI of the difference)")
    for h in H4:
        r = s.loc[h]
        print(f"  h={h:2d}: MAE {r.MAE_a:6.1f} vs {r.MAE_b:6.1f}  dMAE {r.dMAE:7.1f} "
              f"[{r.twoway_lo95:7.1f}; {r.twoway_hi95:6.1f}]  DM-HLN p={r.DM_p:.3f}")
    print("\nALL CHECKS PASSED" if all(ok) else "\nSOME CHECKS FAILED")
    return all(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    if a.build:
        build(cfg, meta, Y, months)
    else:
        sys.exit(0 if verify(cfg, meta, Y, months) else 1)


if __name__ == "__main__":
    main()
