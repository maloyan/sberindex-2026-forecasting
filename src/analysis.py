"""Extra analysis tables for the report (results/metrics/analysis_*.csv).

  analysis_subsets.csv   test MAE of GBM variants on subsets: series with news in the
                         3 months before the origin, series with a recent CUSUM alarm
  analysis_fm.csv        foundation models: test MAE (level vs local), 80% interval
                         coverage and width on the local component, run time
  analysis_bycat.csv     test MAE by category for the key models

Usage: .venv/bin/python src/analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, ROOT, gather, get_panel, load_config, window_pairs  # noqa: E402
from evaluate import load_p50, windows  # noqa: E402
from models.decomp import national_paths  # noqa: E402

MET = ROOT / "results" / "metrics"


def subsets(cfg, meta, Y, months, W):
    from cpd.features import cpd_array
    from news.features import news_array
    news, names = news_array(cfg, meta, months)
    docs = np.expm1(news[:, :, names.index("mo_docs_log")])
    F, fn = cpd_array(cfg, meta, Y, months)
    recent_alarm = F[:, :, fn.index("cpd_alarm_recent")] > 0
    models = [m for m in ("gbm_base", "gbm_news", "gbm_cpd", "gbm_fm", "gbm_full", "cat_full", "ensemble", "prophet")
              if (CACHE / f"{m}.npz").exists()]
    cubes = {m: load_p50(m) for m in models}
    rows = []
    for h in cfg["backtest"]["horizons"]:
        pairs = window_pairs(W["test"], [h])
        for m, C in cubes.items():
            y, p, hh, T, sid = gather(C, Y, pairs)
            o = T - hh
            has_news = np.array([docs[s, max(0, oo - 2):oo + 1].sum() > 0 for s, oo in zip(sid, o)])
            al = recent_alarm[sid, o]
            e = np.abs(p - y)
            rows.append({"model": m, "h": h, "MAE_all": e.mean(), "MAE_news_MO": e[has_news].mean(),
                         "share_news": has_news.mean(), "MAE_recent_alarm": e[al].mean(),
                         "share_alarm": al.mean()})
    return pd.DataFrame(rows)


def fm_table(cfg, meta, Y, W):
    L = np.log(Y)
    d = L - national_paths(L, meta["category"].to_numpy())
    test = pd.read_csv(MET / "metrics_test.csv")
    rows = []
    for p in sorted(CACHE.glob("fmlocal_*.npz")):
        nm = p.stem.replace("fmlocal_", "")
        with np.load(p) as z:
            lo, hi, sec = np.log(np.maximum(z["p10"], 1e-6)), np.log(np.maximum(z["p90"], 1e-6)), float(z["seconds"])
        r = {"fm": nm, "seconds_local": sec}
        lev = CACHE / f"fm_{nm}.npz"
        if lev.exists():
            with np.load(lev) as z:
                r["seconds_level"] = float(z["seconds"])
        for h in cfg["backtest"]["horizons"]:
            pairs = window_pairs(W["test"], [h])
            dd = np.concatenate([d[:, o + x] for o, x in pairs])
            l_ = np.concatenate([lo[:, o, x - 1] for o, x in pairs])
            u_ = np.concatenate([hi[:, o, x - 1] for o, x in pairs])
            r[f"cov80_h{h}"] = ((dd >= l_) & (dd <= u_)).mean()
            r[f"width80_h{h}"] = np.median(u_ - l_)
            for space, pre in (("level", "fm_"), ("local", "dec_fmlocal_")):
                t = test[(test.model == pre + nm) & (test.h == h)]
                if len(t):
                    r[f"MAE_{space}_h{h}"] = float(t.MAE.iloc[0])
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    W = windows(cfg, months)
    pd.set_option("display.width", 220)
    s = subsets(cfg, meta, Y, months, W)
    s.to_csv(MET / "analysis_subsets.csv", index=False)
    print(s.round(1).to_string(index=False))
    f = fm_table(cfg, meta, Y, W)
    f.to_csv(MET / "analysis_fm.csv", index=False)
    print(f.round(3).to_string(index=False))
    bc = pd.read_csv(MET / "metrics_test_bycat.csv")
    key = ["prophet", "snaive_growth", "lgbm_v1", "dec_persist", "gbm_full", "ensemble"]
    t = bc[bc.model.isin(key)].pivot_table(index=["category", "model"], columns="h", values="MAE")
    t.to_csv(MET / "analysis_bycat.csv")
    print(t.round(1).to_string())


if __name__ == "__main__" and "--fm-contribution" not in sys.argv:
    main()


def ensemble_without_fm():
    """Contribution of foundation models: the same ensemble selection without any FM-based cube."""
    from common import HMAX
    from ensemble import NEVER, combine, select
    from evaluate import list_models
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    W = windows(cfg, months)
    fm_based = lambda m: m.startswith(("fm_", "dec_fmlocal_")) or m in ("gbm_fm", "gbm_full", "cat_full")  # noqa: E731
    rows = []
    for tag, keep in (("with FM", lambda m: True), ("without FM", lambda m: not fm_based(m))):
        names = [m for m in (cfg["ensemble"].get("pool") or [m for m in list_models() if m not in NEVER]) if keep(m)]
        cubes = {n: load_p50(n) for n in names}
        weights = {h: select(cubes, Y, W["validation"], h) for h in range(1, HMAX + 1)}
        C = combine(cubes, weights, Y.shape)
        for h in cfg["backtest"]["horizons"]:
            y, p = gather(C, Y, window_pairs(W["test"], [h]))[:2]
            rows.append({"ensemble": tag, "h": h, "MAE_test": np.abs(p - y).mean()})
    r = pd.DataFrame(rows).pivot(index="ensemble", columns="h", values="MAE_test")
    r.to_csv(MET / "analysis_ensemble_fm_contribution.csv")
    print(r.round(1).to_string())


if __name__ == "__main__" and "--fm-contribution" in sys.argv:
    ensemble_without_fm()
