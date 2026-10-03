"""Break-aware seasonal naive: snaive_growth that corrects the history after a detected break.

At origin o the online CUSUM detector (calibrated in src/cpd/benchmark.py) runs on the
local component d of every series. If it raised an alarm a in months o-11..o, the break
month b is localised (best two-mean split in [a-3, a]) and the jump J = mean d after b -
mean d over the 4 months before b. The year-ago values before b are moved to the new
level: y(t) * exp(J) for t < b. Then the usual snaive_growth formula is applied.
Series without an alarm get exactly snaive_growth. Everything is causal.

Output: results/cache/snaive_growth_ba.npz and results/metrics/break_aware_real.csv
(test MAE on the series with an alarm, with and without the correction).

Usage: .venv/bin/python src/models/break_aware.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpd"))
import detectors as D  # noqa: E402
from benchmark import localise  # noqa: E402
from common import HMAX, ROOT, empty_cube, gather, get_panel, load_config, load_cube, save_cube, window_pairs  # noqa: E402,E501
from evaluate import windows  # noqa: E402
from models.decomp import national_paths  # noqa: E402


def main():
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    k = cfg["models"]["snaive_growth"]["k"]
    par = pd.read_csv(ROOT / "results" / "cpd" / "calibrated_params.csv", index_col=0).iloc[:, 0].to_dict()
    L = np.log(Y)
    d = L - national_paths(L, meta["category"].to_numpy())
    z, _ = D.standardize(d)
    A = D.cusum(z, 12, h=float(par["cusum"]))
    n, T = Y.shape
    base = load_cube("snaive_growth")
    cube = base.copy()
    flagged = np.zeros((n, T), dtype=bool)
    for o in range(12, T):
        for i in np.where(A[:, max(12, o - 11):o + 1].any(axis=1))[0]:
            a = max(12, o - 11) + np.where(A[i, max(12, o - 11):o + 1])[0][-1]
            b = localise(d[i], a, o)
            J = d[i, b:o + 1].mean() - d[i, max(0, b - 4):b].mean()
            La = L[i].copy()
            La[:b] += J
            kk = min(k, o - 11)
            g = (La[o - kk + 1:o + 1] - La[o - kk + 1 - 12:o + 1 - 12]).mean()
            for h in range(1, HMAX + 1):
                t = o + h
                if t - 12 <= o:
                    cube[i, o, h - 1] = np.exp(La[t - 12] + g)
            flagged[i, o] = True
    save_cube("snaive_growth_ba", cube)
    W = windows(cfg, months)
    rows = []
    for h in cfg["backtest"]["horizons"]:
        pairs = window_pairs(W["test"], [h])
        yb, pb, hh, TT, sid = gather(base, Y, pairs)
        ya, pa = gather(cube, Y, pairs)[:2]
        f = flagged[sid, TT - hh]
        rows.append({"h": h, "flagged_share": f.mean(), "MAE_plain_flagged": np.abs(pb - yb)[f].mean(),
                     "MAE_break_aware_flagged": np.abs(pa - ya)[f].mean(),
                     "MAE_plain_all": np.abs(pb - yb).mean(), "MAE_break_aware_all": np.abs(pa - ya).mean()})
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "results" / "metrics" / "break_aware_real.csv", index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
