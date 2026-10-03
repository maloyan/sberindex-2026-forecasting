"""Change-point (structural break) detectors for the local component d[i,t].

d = log y - national(category, month) (src/models/decomp.py). National seasonality and
inflation are removed, so a local shock is a step (or a slope change) in d.

All series are standardized by a robust noise scale of the monthly changes in the
reference period (2023): z = (d - mean_ref) / s, s = 1.4826 * MAD(diff(d_ref)).

ONLINE detectors return a boolean alarm matrix A (n_series, T): A[i,t] = alarm raised at
month t using data up to t only. They run from month `start` (2024-01).
  spike          |z_t - z_{t-1}| > k                    (one-month jump, no memory)
  cusum          two-sided CUSUM on z, drift k, threshold h, restart after an alarm
  pelt_online    ruptures PELT (l2) re-fitted on z[0..t]; alarm if a break in the last months
  binseg_online  ruptures Binseg (l2), same rule
  window_online  ruptures Window (l2), same rule
  bocpd          Bayesian online change-point detection (Adams & MacKay 2007), Normal model
                 with unknown mean, alarm if P(run length <= 2 | data) > threshold
  resid_cusum    CUSUM on standardized one-step forecast errors of a forecaster

OFFLINE detectors return a list of break months per series (the whole series is seen):
  pelt, binseg, bottomup, window, kernel (rbf), bocpd_map (MAP segmentation).
"""
from __future__ import annotations

import numpy as np
import ruptures as rpt
from scipy.special import gammaln


def standardize(d: np.ndarray, ref: int = 12):
    dx = np.diff(d[:, :ref], axis=1)
    s = 1.4826 * np.median(np.abs(dx - np.median(dx, axis=1, keepdims=True)), axis=1) + 1e-4
    mu = d[:, :ref].mean(axis=1)
    return (d - mu[:, None]) / s[:, None], s


# ----------------------------------------------------------------------------- online
def spike(z, start=12, k=3.0):
    A = np.zeros_like(z, dtype=bool)
    A[:, 1:] = np.abs(np.diff(z, axis=1)) > k
    A[:, :start] = False
    return A


def cusum(z, start=12, k=0.5, h=4.0, ref=12, thr_scale=None):
    """thr_scale: optional (n, T) multiplier of h (news-aware variant lowers h in news months)."""
    n, T = z.shape
    # reference level/scale of z itself in 2023 (z is a level, its spread can exceed 1)
    mu = z[:, :ref].mean(1)
    sd = z[:, :ref].std(1) + 1e-6
    x = (z - mu[:, None]) / np.maximum(sd[:, None], 1.0)
    A = np.zeros_like(z, dtype=bool)
    sp = np.zeros(n); sn = np.zeros(n)
    for t in range(start, T):
        sp = np.maximum(0, sp + x[:, t] - k)
        sn = np.maximum(0, sn - x[:, t] - k)
        hh = h * (thr_scale[:, t] if thr_scale is not None else 1.0)
        A[:, t] = (sp > hh) | (sn > hh)
        # restart after an alarm, with the new level as reference
        r = A[:, t]
        sp[r] = 0; sn[r] = 0
        mu[r] = z[r, t]
        x[r, t:] = (z[r, t:] - mu[r, None]) / np.maximum(sd[r, None], 1.0)
    return A


def _online_ruptures(z, algo, start=12, penalty=8.0, min_size=2, recent=1, **kw):
    n, T = z.shape
    A = np.zeros((n, T), dtype=bool)
    for i in range(n):
        for t in range(start, T):
            seg = z[i, :t + 1]
            if algo == "pelt":
                bk = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(seg).predict(pen=penalty)[:-1]
            elif algo == "binseg":
                bk = rpt.Binseg(model="l2", min_size=min_size, jump=1).fit(seg).predict(pen=penalty)[:-1]
            else:
                w = kw.get("width", 4)
                if len(seg) < 2 * w:
                    continue
                bk = rpt.Window(width=w, model="l2", jump=1).fit(seg).predict(pen=penalty)[:-1]
            # a break at b (first month of the new regime) is visible from t = b + min_size - 1
            A[i, t] = any(0 <= t - b < min_size - 1 + recent for b in bk)
    return A


def pelt_online(z, start=12, penalty=8.0, min_size=2):
    return _online_ruptures(z, "pelt", start, penalty, min_size)


def binseg_online(z, start=12, penalty=8.0, min_size=2):
    return _online_ruptures(z, "binseg", start, penalty, min_size)


def window_online(z, start=12, width=4, penalty=8.0):
    return _online_ruptures(z, "window", start, penalty, min_size=width // 2 + 1, width=width)


def bocpd_runlength(x, hazard=0.05, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0):
    """Run-length posterior R[t, r] for one series (Normal-Gamma prior, Student-t predictive)."""
    T = len(x)
    R = np.zeros((T + 1, T + 1)); R[0, 0] = 1.0
    mu, kappa, alpha, beta = (np.array([v]) for v in (mu0, kappa0, alpha0, beta0))
    for t in range(T):
        # Student-t predictive for every current run length
        scale = beta * (kappa + 1) / (alpha * kappa)
        nu = 2 * alpha
        lp = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu * scale)
              - (nu + 1) / 2 * np.log1p((x[t] - mu) ** 2 / (nu * scale)))
        pred = np.exp(lp)
        growth = R[t, :t + 1] * pred * (1 - hazard)
        cp = (R[t, :t + 1] * pred * hazard).sum()
        R[t + 1, 1:t + 2] = growth
        R[t + 1, 0] = cp
        R[t + 1] /= R[t + 1].sum()
        # posterior update of the sufficient statistics
        mu_n = (kappa * mu + x[t]) / (kappa + 1)
        beta_n = beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_n]); kappa = np.concatenate([[kappa0], kappa + 1])
        alpha = np.concatenate([[alpha0], alpha + 0.5]); beta = np.concatenate([[beta0], beta_n])
    return R[1:]


def bocpd(z, start=12, hazard=0.05, threshold=0.5, short=2, hazard_scale=None, lo=1):
    """Alarm at t if the posterior mass of run lengths lo..short exceeds threshold.

    lo=1, short=2: fast mode (alarm on the first point of a new regime).
    lo=2, short=3: confirmed mode ("bocpd_confirm"): the new regime must hold for 2 months,
    so a one-month outlier does not raise an alarm; the cost is one month of delay.

    hazard_scale: optional (n, T) multiplier of the hazard (news-aware variant).
    Run length r at time t counts the points of the current regime, including x_t; r = 1
    means x_t is the first point of a new regime. A shift at month b is seen at t >= b.
    """
    n, T = z.shape
    A = np.zeros((n, T), dtype=bool)
    P = np.zeros((n, T))
    for i in range(n):
        x = z[i] - z[i, :start].mean()
        if hazard_scale is None:
            R = bocpd_runlength(x, hazard=hazard)
        else:
            R = _bocpd_varhaz(x, hazard * hazard_scale[i])
        # R[t, r]: r = 0 is "change before x_t is seen", r = 1 means x_t starts a new regime
        P[i] = R[:, lo:short + 1].sum(axis=1)
    A[:, start:] = P[:, start:] > threshold
    return A, P


def _bocpd_varhaz(x, hz, mu0=0.0, kappa0=0.1, alpha0=2.0, beta0=1.0):
    """BOCPD with a time-varying hazard hz[t]."""
    T = len(x)
    R = np.zeros((T + 1, T + 1)); R[0, 0] = 1.0
    mu, kappa, alpha, beta = (np.array([v]) for v in (mu0, kappa0, alpha0, beta0))
    for t in range(T):
        h = float(np.clip(hz[t], 1e-4, 0.9))
        scale = beta * (kappa + 1) / (alpha * kappa)
        nu = 2 * alpha
        lp = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu * scale)
              - (nu + 1) / 2 * np.log1p((x[t] - mu) ** 2 / (nu * scale)))
        pred = np.exp(lp)
        R[t + 1, 1:t + 2] = R[t, :t + 1] * pred * (1 - h)
        R[t + 1, 0] = (R[t, :t + 1] * pred * h).sum()
        R[t + 1] /= R[t + 1].sum()
        mu_n = (kappa * mu + x[t]) / (kappa + 1)
        beta_n = beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_n]); kappa = np.concatenate([[kappa0], kappa + 1])
        alpha = np.concatenate([[alpha0], alpha + 0.5]); beta = np.concatenate([[beta0], beta_n])
    return R[1:]


def resid_cusum(e, start=12, k=0.5, h=4.0, ref=12):
    """CUSUM on standardized one-step log forecast errors e (n, T); NaN errors are skipped."""
    n, T = e.shape
    s = 1.4826 * np.nanmedian(np.abs(e[:, :start] - np.nanmedian(e[:, :start], 1, keepdims=True)), 1) + 1e-4
    x = np.nan_to_num(e / s[:, None])
    A = np.zeros_like(e, dtype=bool)
    sp = np.zeros(n); sn = np.zeros(n)
    for t in range(start, T):
        sp = np.maximum(0, sp + x[:, t] - k)
        sn = np.maximum(0, sn - x[:, t] - k)
        A[:, t] = (sp > h) | (sn > h)
        sp[A[:, t]] = 0; sn[A[:, t]] = 0
    return A


# ----------------------------------------------------------------------------- offline
def offline(z: np.ndarray, method: str, penalty: float = 25.0, min_size: int = 3) -> list[list[int]]:
    out = []
    for i in range(z.shape[0]):
        x = z[i]
        if method == "pelt":
            bk = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(x).predict(pen=penalty)[:-1]
        elif method == "binseg":
            bk = rpt.Binseg(model="l2", min_size=min_size, jump=1).fit(x).predict(pen=penalty)[:-1]
        elif method == "bottomup":
            bk = rpt.BottomUp(model="l2", min_size=min_size, jump=1).fit(x).predict(pen=penalty)[:-1]
        elif method == "window":
            bk = rpt.Window(width=6, model="l2", jump=1).fit(x).predict(pen=penalty)[:-1]
        elif method == "kernel":
            bk = rpt.KernelCPD(kernel="rbf", min_size=min_size).fit(x[:, None]).predict(pen=penalty / 10)[:-1]
        elif method == "bocpd_map":
            R = bocpd_runlength(x - x[:12].mean(), hazard=1 / 24)
            # MAP run length at the end, walked back: a change where the MAP run length resets
            rl = R.argmax(axis=1)
            bk = [t - rl[t] + 1 for t in range(1, len(x)) if rl[t] < rl[t - 1] and rl[t] <= 3]
            bk = sorted({b for b in bk if min_size <= b <= len(x) - 1})
        else:
            raise ValueError(method)
        out.append(list(bk))
    return out
