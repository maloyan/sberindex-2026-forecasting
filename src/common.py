"""Shared helpers: config, panel cache, forecast cubes, metrics.

A forecast "cube" is a float32 array C of shape (n_series, n_months, HMAX).
C[i, o, h-1] is the forecast for month o+h of series i, made at origin o
(the model saw months 0..o only). NaN means "not computed".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
HMAX = 12


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_panel(cfg: dict | None = None):
    """(meta, Y, months). Cached in data/interim/panel.pkl after the first call."""
    import pickle

    from data import load_panel
    cfg = cfg or load_config()
    p = ROOT / "data" / "interim" / "panel.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    p.parent.mkdir(parents=True, exist_ok=True)
    res = load_panel(cfg)
    with open(p, "wb") as f:
        pickle.dump(res, f)
    return res


def empty_cube(n: int, T: int) -> np.ndarray:
    return np.full((n, T, HMAX), np.nan, dtype=np.float32)


def save_cube(name: str, cube: np.ndarray, **extra) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"{name}.npz", p50=cube.astype(np.float32), **extra)


def load_cube(name: str, key: str = "p50") -> np.ndarray:
    with np.load(CACHE / f"{name}.npz") as z:
        return z[key]


def has_cube(name: str) -> bool:
    return (CACHE / f"{name}.npz").exists()


# ----------------------------------------------------------------------------- windows
def window_pairs(targets: list[int], horizons: list[int]) -> list[tuple[int, int]]:
    """(origin, h) pairs for the given target months and horizons."""
    return [(T - h, h) for h in horizons for T in targets if T - h >= 0]


def gather(cube: np.ndarray, Y: np.ndarray, pairs, rows=None):
    """Long arrays (y, p, h, T, sid) for the (o, h) pairs."""
    n = Y.shape[0]
    rows = np.arange(n) if rows is None else rows
    ys, ps, hs, Ts, ss = [], [], [], [], []
    for o, h in pairs:
        T = o + h
        ys.append(Y[rows, T]); ps.append(cube[rows, o, h - 1])
        hs.append(np.full(len(rows), h)); Ts.append(np.full(len(rows), T)); ss.append(rows)
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(hs),
            np.concatenate(Ts), np.concatenate(ss))


# ----------------------------------------------------------------------------- metrics
def metrics(y, p, y_origin=None) -> dict:
    """MAE (mandatory), R2 on levels, R2 on log change vs origin, WAPE, MdAPE, bias."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    out = {"MAE": np.abs(err).mean(),
           "R2": 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum(),
           "WAPE": np.abs(err).sum() / np.abs(y).sum(),
           "MdAPE": np.median(np.abs(err) / np.abs(y)),
           "Bias%": err.sum() / y.sum() * 100,
           "n": len(y)}
    if y_origin is not None:
        g, gp = np.log(y / y_origin), np.log(np.maximum(p, 1e-6) / y_origin)
        out["R2_change"] = 1 - ((g - gp) ** 2).sum() / ((g - g.mean()) ** 2).sum()
    return out
