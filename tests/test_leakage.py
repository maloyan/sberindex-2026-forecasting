"""Leakage tests on synthetic data (no contest data needed, seconds).

Rule under test: a forecast made at origin o, and an online alarm raised in month t, must not
change when the data after o (after t) change. Run: make smoke  (or .venv/bin/pytest tests)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import window_pairs  # noqa: E402
from cpd import detectors as D  # noqa: E402
from models import baselines, decomp  # noqa: E402

T = 24
MONTHS = [str(p) for p in pd.period_range("2023-01", periods=T, freq="M")]
rng = np.random.default_rng(0)


def panel(n=60):
    season = np.tile(np.sin(np.arange(12) / 12 * 2 * np.pi) * 0.1, 2)
    level = rng.normal(8, 0.5, size=(n, 1))
    return np.exp(level + season + 0.01 * np.arange(T) + rng.normal(0, 0.03, size=(n, T)))


def perturb_after(X, o):
    Z = X.copy()
    Z[:, o + 1:] = Z[:, o + 1:] * rng.uniform(0.5, 2.0, size=Z[:, o + 1:].shape)
    return Z


@pytest.mark.parametrize("o", [3, 11, 12, 17, 22])
def test_national_component_uses_month_t_only(o):
    L = np.log(panel())
    cats = np.repeat(np.arange(6), 10)
    a = decomp.national_paths(L, cats)
    b = decomp.national_paths(np.log(perturb_after(np.exp(L), o)), cats)
    assert np.allclose(a[:, :o + 1], b[:, :o + 1])


@pytest.mark.parametrize("o", [5, 12, 18, 22])
def test_national_forecast_is_causal(o):
    L = np.log(panel())
    cats = np.repeat(np.arange(6), 10)
    nat = decomp.national_paths(L, cats)
    nat2 = decomp.national_paths(np.log(perturb_after(np.exp(L), o)), cats)
    a = decomp.national_forecast(nat, MONTHS, k=1, w=0.5, lag=2)
    b = decomp.national_forecast(nat2, MONTHS, k=1, w=0.5, lag=2)
    assert np.allclose(a[:, :o + 1], b[:, :o + 1], equal_nan=True)


def test_rosstat_prior_respects_publication_lag(monkeypatch):
    """The Rosstat growth used at origin o must come from months <= o - lag."""
    real = pd.read_csv
    ros = real(Path(decomp.ROOT) / "data" / "external" / "rosstat_monthly.csv")
    cut = ros.index[ros.date == "2024-03"][0]

    def fake(path, *a, **k):
        df = real(path, *a, **k)
        if str(path).endswith("rosstat_monthly.csv"):
            num = df.select_dtypes("number").columns
            df.loc[cut:, num] = df.loc[cut:, num] * 3.0
        return df

    base = decomp.rosstat_yoy(MONTHS, lag=2)
    monkeypatch.setattr(decomp.pd, "read_csv", fake)
    pert = decomp.rosstat_yoy(MONTHS, lag=2)
    o_last_clean = MONTHS.index("2024-04")       # 2024-04 - 2 = 2024-02 < 2024-03
    assert np.allclose(base[:o_last_clean + 1], pert[:o_last_clean + 1])
    assert not np.allclose(base[o_last_clean + 1:], pert[o_last_clean + 1:])


@pytest.mark.parametrize("o", [2, 12, 15, 22])
def test_local_models_are_causal(o):
    d = rng.normal(0, 0.1, size=(40, T)).cumsum(1)
    a = decomp.local_cubes(d)
    b = decomp.local_cubes(_pert_add(d, o))
    for k in a:
        assert np.allclose(a[k][:, :o + 1], b[k][:, :o + 1], equal_nan=True), k


def _pert_add(d, o):
    z = d.copy()
    z[:, o + 1:] += rng.normal(0, 1, size=z[:, o + 1:].shape)
    return z


@pytest.mark.parametrize("o", [4, 13, 20])
def test_baselines_are_causal(o):
    Y = panel()
    codes = np.repeat(np.arange(6), 10)
    a = baselines.cubes(Y, codes)
    b = baselines.cubes(perturb_after(Y, o), codes)
    for k in a:
        assert np.allclose(a[k][:, :o + 1], b[k][:, :o + 1], equal_nan=True), k


@pytest.mark.parametrize("t", [13, 16, 20])
def test_online_detectors_do_not_look_ahead(t):
    z = rng.normal(0, 1, size=(30, T))
    z[:10, 15:] += 3.0
    z2 = z.copy()
    z2[:, t + 1:] += rng.normal(0, 5, size=z2[:, t + 1:].shape)
    for f in (lambda x: D.spike(x, 12), lambda x: D.cusum(x, 12), lambda x: D.bocpd(x, 12)[0]):
        a, b = f(z), f(z2)
        assert np.array_equal(np.asarray(a)[:, :t + 1], np.asarray(b)[:, :t + 1])


def test_window_pairs_target_after_origin():
    for o, h in window_pairs(list(range(18, 24)), [1, 3, 6, 12]):
        assert o >= 0 and 18 <= o + h <= 23
