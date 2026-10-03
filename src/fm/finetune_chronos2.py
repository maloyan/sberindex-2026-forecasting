"""Fine-tune Chronos-2 on the local component, strictly causally (no target after the cutoff).

Chronos-2 is fine-tuned (LoRA or full) on exp(d[:, :c+1]) of all series, where c is a
cutoff month. The fine-tuning windows are sampled inside months 0..c, so no value after
the cutoff is used. A model fine-tuned at cutoff c serves origins c..(next cutoff - 1).
Origins before the first cutoff use the zero-shot model.

  cutoffs 11 (2023-12), 17 (2024-06), 23 (2024-12, final forecast only)

Output: results/cache/fmlocal_chronos2_ft.npz (same format as src/fm/run_fm.py, local space)

Usage: .venv/bin/python src/fm/finetune_chronos2.py [--steps 300] [--mode lora]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import Chronos2, _device  # noqa: E402
from common import CACHE, HMAX, ROOT, empty_cube, get_panel, load_config  # noqa: E402
from models.decomp import national_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--mode", default="full", choices=["full", "lora"])
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--cutoffs", nargs="+", type=int, default=[11, 17, 23])
    a = ap.parse_args()
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    L = np.log(Y)
    X = np.exp(L - national_paths(L, meta["category"].to_numpy())).astype(np.float32)
    n, T = X.shape
    zs_path = CACHE / "fmlocal_chronos2.npz"
    with np.load(zs_path) as z:
        cubes = {k: z[k].copy() for k in ("p10", "p50", "p90")}   # zero-shot as the default
    dev = _device(cfg["foundation_models"]["device"])
    t0 = time.time()
    log = []
    for j, c in enumerate(a.cutoffs):
        base = Chronos2(dev)
        # MPS attention kernels do not support dropout in training; fine-tune without dropout
        base.p.model.config.dropout_rate = 0.0
        train = [{"target": X[i, :c + 1]} for i in range(n)]
        ft = base.p.fit(train, prediction_length=6, min_past=3, finetune_mode=a.mode,
                        learning_rate=a.lr, num_steps=a.steps, batch_size=256,
                        output_dir=str(ROOT / "results" / "cache" / "chronos2_ft" / f"c{c}"),
                        remove_printer_callback=True, report_to="none",
                        disable_tqdm=True, save_strategy="no")
        model = Chronos2.__new__(Chronos2)
        model.dev, model.p = dev, ft
        last = a.cutoffs[j + 1] if j + 1 < len(a.cutoffs) else T
        for o in range(c, last):
            q = model([X[i, :o + 1] for i in range(n)], HMAX)
            q = np.maximum(q, 1e-6)
            cubes["p10"][:, o], cubes["p50"][:, o], cubes["p90"][:, o] = q[:, 0], q[:, 1], q[:, 2]
        log.append(f"cutoff {c}: origins {c}..{last - 1}, {time.time() - t0:.0f}s")
        print(log[-1], flush=True)
        del base, ft, model
        if dev == "mps":
            torch.mps.empty_cache()
    np.savez_compressed(CACHE / "fmlocal_chronos2_ft.npz", **cubes, seconds=time.time() - t0)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
