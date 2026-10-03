"""Online change-point features per series and origin (for the forecasting models).

Computed on the local component d (src/models/decomp.py) with the online detectors of
src/cpd/detectors.py and the thresholds calibrated in src/cpd/benchmark.py. Every value
at origin o uses data up to o only.

  cpd_prob          BOCPD posterior probability of a new regime (run length <= 2)
  cpd_alarm_recent  1 if CUSUM raised an alarm in months o-2..o
  cpd_since         months since the last CUSUM alarm (12 if none)
  cpd_jump          d[o] minus the mean of d over months o-6..o-3 (size of a recent step)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectors as D  # noqa: E402
from common import ROOT  # noqa: E402
from models.decomp import national_paths  # noqa: E402


def cpd_array(cfg, meta, Y, months, start=12):
    p = ROOT / "results" / "cpd" / "calibrated_params.csv"
    par = pd.read_csv(p, index_col=0).iloc[:, 0].to_dict() if p.exists() else {"cusum": 4.0, "bocpd": 0.5}
    L = np.log(Y)
    d = L - national_paths(L, meta["category"].to_numpy())
    z, _ = D.standardize(d)
    A = D.cusum(z, start, h=float(par["cusum"]))
    _, P = D.bocpd(z, start, threshold=float(par["bocpd"]))
    n, T = d.shape
    recent = np.zeros((n, T)); since = np.full((n, T), 12.0); jump = np.full((n, T), np.nan)
    last = np.full(n, -100)
    for t in range(T):
        last = np.where(A[:, t], t, last)
        since[:, t] = np.minimum(t - last, 12)
        recent[:, t] = A[:, max(0, t - 2):t + 1].any(axis=1)
        if t >= 6:
            jump[:, t] = d[:, t] - d[:, t - 6:t - 2].mean(axis=1)
    P[:, :start] = 0.0
    F = np.stack([P, recent, since, jump], axis=2).astype(np.float32)
    return F, ["cpd_prob", "cpd_alarm_recent", "cpd_since", "cpd_jump"]
