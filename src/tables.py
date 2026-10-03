"""Tables for the jury-facing report (results/metrics/*.csv, results/cpd/decision_table.csv).

  selection_table.csv   per horizon: the model the rule picks on VALIDATION (the ensemble),
                        its validation and test MAE, its test rank among the 35 candidates +
                        the ensemble, the best single model on test (known only after the
                        fact), dMAE vs Prophet with the two-way 95% CI, R2 and R2 on change
  model_table_test.csv  every candidate: validation MAE, test MAE, test rank, R2, R2 on change,
                        dMAE vs Prophet with the two-way CI (h = 1, 3, 6, 12)
  intervals_test.csv    80% intervals from the VALIDATION log errors (per h and category),
                        scored on TEST: coverage, mean width, WIS (one interval + median)
  prophet_grid_ci.csv   ensemble vs the best of 10 Prophet configurations (chosen on test, per h)
                        on the 400-МО sample, two-way bootstrap CI (after src/models/prophet_grid.py)
  decision_table.csv    change-point detectors: synthetic F1, delay, outliers, false alarms,
                        real data (Orsk, news events, flood registry), one production choice

Usage: .venv/bin/python src/tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, gather, get_panel, load_config, window_pairs  # noqa: E402
from evaluate import load_p50, windows  # noqa: E402

MET = ROOT / "results" / "metrics"
CPD = ROOT / "results" / "cpd"


def selection(cfg):
    pool = cfg["ensemble"]["pool"]
    v = pd.read_csv(MET / "metrics_validation.csv")
    t = pd.read_csv(MET / "metrics_test.csv")
    bs = pd.read_csv(MET / "bootstrap_test_vs_prophet.csv")
    cand = pool + ["ensemble"]
    rows, full = [], []
    for h in cfg["backtest"]["horizons"]:
        vh = v[(v.h == h) & v.model.isin(cand)].set_index("model")
        th = t[(t.h == h) & t.model.isin(cand)].set_index("model")
        trank = th.MAE.rank(method="min")
        vrank = vh.loc[pool].MAE.rank(method="min")           # the ensemble's validation MAE is in-sample
        b = bs[bs.h == h].set_index("model")
        for m in cand:
            r = {"h": h, "model": m, "MAE_val": vh.MAE.get(m), "rank_val_single": vrank.get(m),
                 "MAE_test": th.MAE[m], "rank_test": int(trank[m]), "R2_test": th.R2[m],
                 "R2_change_test": th.R2_change[m]}
            if m in b.index:
                r.update(dMAE_vs_prophet=b.dMAE[m], twoway_lo95=b.twoway_lo95[m], twoway_hi95=b.twoway_hi95[m],
                         DM_p=b.DM_p[m], months_better=b.months_a_better[m])
            full.append(r)
        best_val = vh.loc[pool].MAE.idxmin()
        best_test = th.loc[pool].MAE.idxmin()
        e = b.loc["ensemble"]
        rows.append({"h": h, "chosen": "ensemble", "best_single_val": best_val,
                     "MAE_val_best_single": vh.MAE[best_val], "MAE_val_ensemble_insample": vh.MAE["ensemble"],
                     "MAE_test_ensemble": th.MAE["ensemble"], "rank_test_ensemble": int(trank["ensemble"]),
                     "n_candidates": len(cand), "MAE_test_of_best_val_single": th.MAE[best_val],
                     "best_single_test": best_test, "MAE_test_best_single": th.MAE[best_test],
                     "MAE_test_prophet": th.MAE["prophet"], "dMAE_vs_prophet": e.dMAE,
                     "twoway_lo95": e.twoway_lo95, "twoway_hi95": e.twoway_hi95, "DM_p": e.DM_p,
                     "months_better": e.months_a_better, "R2_test": th.R2["ensemble"],
                     "R2_change_test": th.R2_change["ensemble"]})
    pd.DataFrame(rows).to_csv(MET / "selection_table.csv", index=False)
    pd.DataFrame(full).to_csv(MET / "model_table_test.csv", index=False)
    print(pd.DataFrame(rows).round(3).T.to_string())


def intervals(cfg, meta, Y, months, models=("ensemble", "prophet", "dec_persist", "gbm_full")):
    """Split-conformal style 80% interval: quantiles 10/90% of log(y / p) on VALIDATION per
    (h, category), applied to the TEST forecasts. WIS with K = 1 interval (alpha = 0.2):
    WIS = (0.5 |y - m| + alpha/2 * IS) / 1.5, IS = (u - l) + 2/alpha (l - y)+ + 2/alpha (y - u)+."""
    W = windows(cfg, months)
    cats = meta["category"].to_numpy()
    a = 0.2
    rows = []
    for m in models:
        C = np.maximum(load_p50(m), 1.0)      # Prophet can forecast <= 0 at long h; clip to 1 ruble
        for h in cfg["backtest"]["horizons"]:
            yv, pv, _, _, sv = gather(C, Y, window_pairs(W["validation"], [h]))
            yt, pt, _, _, st = gather(C, Y, window_pairs(W["test"], [h]))
            ev = np.log(yv / pv)
            lo, hi = np.zeros_like(pt), np.zeros_like(pt)
            for c in np.unique(cats):
                ql, qh = np.quantile(ev[cats[sv] == c], [a / 2, 1 - a / 2])
                k = cats[st] == c
                lo[k], hi[k] = pt[k] * np.exp(ql), pt[k] * np.exp(qh)
            IS = (hi - lo) + 2 / a * np.maximum(lo - yt, 0) + 2 / a * np.maximum(yt - hi, 0)
            wis = (0.5 * np.abs(yt - pt) + a / 2 * IS) / 1.5
            rows.append({"model": m, "h": h, "coverage80": ((yt >= lo) & (yt <= hi)).mean(),
                         "width_rel_median": np.median((hi - lo) / pt), "WIS": wis.mean(),
                         "MAE": np.abs(yt - pt).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(MET / "intervals_test.csv", index=False)
    print(out.round(3).to_string(index=False))


NAMES_CPD = {"spike": "Скачок (spike)", "cusum": "CUSUM", "pelt_online": "PELT", "binseg_online": "BinSeg",
             "window_online": "Window", "bocpd": "BOCPD", "bocpd_confirm": "BOCPD-подтв.",
             "resid_cusum": "CUSUM по остаткам", "bocpd_news": "BOCPD + новости", "cusum_news": "CUSUM + новости"}


def decision_table():
    s = pd.read_csv(CPD / "synthetic_online.csv")
    ne = pd.read_csv(CPD / "real_news_events.csv").set_index("detector")
    rc = pd.read_csv(CPD / "real_cases.csv")
    orsk = rc[rc.case == "Орск"]
    flood = pd.read_csv(CPD / "registry_events.csv").set_index("detector") if (CPD / "registry_events.csv").exists() else None
    semi = pd.read_csv(CPD / "news_semi_synthetic.csv").set_index("detector")
    rows = []
    for det in NAMES_CPD:
        sd = s[s.detector == det]
        r = {"detector": det, "name": NAMES_CPD[det]}
        if len(sd):
            sh = sd[sd.scenario == "shift"].set_index("size")
            r.update(F1_5=sh.F1.get(0.05), F1_10=sh.F1.get(0.1), F1_20=sh.F1.get(0.2),
                     recall_10=sh.recall.get(0.1), delay_10=sh.mean_delay.get(0.1),
                     false_alarm=sh.clean_FAR.iloc[0])
            sp = sd[sd.scenario == "spike"]
            if len(sp):
                r["outlier_alarm"] = sp.recall.iloc[0]
            tr = sd[(sd.scenario == "trend") & (sd["size"] == 0.04)]
            if len(tr):
                r["F1_trend4"] = tr.F1.iloc[0]
        elif det in semi.index:     # news-aware detectors: only the semi-synthetic benchmark
            r.update(F1_10_semi=semi.F1[det], delay_10=semi.mean_delay[det], false_alarm=semi.clean_FAR[det])
        al = orsk[orsk[det]].month.tolist() if det in orsk else []
        r["orsk_first_alarm"] = al[0] if al else "нет"
        r["news_lift"] = ne.lift.get(det)
        if flood is not None:
            r["flood_hit"] = flood.hit_rate.get(det)
            r["flood_lift"] = flood.lift.get(det)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(CPD / "decision_table.csv", index=False)
    print(out.round(2).to_string(index=False))


def prophet_grid_ci(cfg, meta, Y, months):
    from common import CACHE
    from evaluate import _draws, compare
    p = CACHE / "prophet_grid.npz"
    if not p.exists():
        return
    z = np.load(p)
    rows = z["rows"]
    W = windows(cfg, months)
    g = pd.read_csv(MET / "prophet_grid_test.csv")
    grid = g[g.kind == "prophet grid"].pivot(index="config", columns="h", values="MAE")
    sub = meta.iloc[rows].reset_index(drop=True)
    Ys = Y[rows]
    E = load_p50("ensemble")[rows]
    draws = _draws(sub["territory_id"].nunique(), len(W["test"]), cfg["backtest"]["bootstrap_reps"], 0)
    out = []
    for h in cfg["backtest"]["horizons"]:
        best = grid[h].idxmin()
        for ref in (best, "ours_m3"):
            r = compare(E, z[ref], Ys, sub, W["test"], h, draws)
            out.append({"ref": ref, "is_best_on_test": ref == best, **r})
    out = pd.DataFrame(out)
    out.to_csv(MET / "prophet_grid_ci.csv", index=False)
    print(out.round(2).to_string(index=False))


def main():
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    pd.set_option("display.width", 250)
    selection(cfg)
    intervals(cfg, meta, Y, months)
    decision_table()
    prophet_grid_ci(cfg, meta, Y, months)


if __name__ == "__main__":
    main()
