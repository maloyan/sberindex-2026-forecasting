"""Uniform zero-shot interface for open-weight time-series foundation models.

Every adapter takes a list of 1-D numpy arrays (history in rubles, oldest first) and a
horizon H, and returns an array of shape (n_series, 3, H) with the 0.1, 0.5 and 0.9
quantiles. All models are Apache-2.0 (see LICENSES section of the report).

  chronos_bolt_small / chronos_bolt_base   amazon/chronos-bolt-*        (2024)
  chronos2                                 amazon/chronos-2             (2025)
  chronos2_mv   Chronos-2 with the 6 categories of one municipality as one multivariate group
  timesfm25                                google/timesfm-2.5-200m-pytorch (2025)
  tirex2                                   NX-AI/TiRex-2                (2026)
  toto2_22m / toto2_313m                   Datadog/Toto-2.0-*           (2026)
"""
from __future__ import annotations

import numpy as np
import torch

Q = [0.1, 0.5, 0.9]


def _device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ChronosBolt:
    def __init__(self, size="small", device="auto"):
        from chronos import BaseChronosPipeline
        self.dev = _device(device)
        self.p = BaseChronosPipeline.from_pretrained(f"amazon/chronos-bolt-{size}",
                                                     device_map=self.dev, torch_dtype=torch.float32)

    def __call__(self, ctx, H, bs=1024):
        out = []
        for i in range(0, len(ctx), bs):
            b = [torch.tensor(c, dtype=torch.float32) for c in ctx[i:i + bs]]
            q, _ = self.p.predict_quantiles(b, prediction_length=H, quantile_levels=Q)
            out.append(q.permute(0, 2, 1).cpu().numpy())        # (b, 3, H)
        return np.concatenate(out)


class Chronos2:
    """Univariate by default. groups: list of index lists -> one multivariate input per group."""

    def __init__(self, device="auto", model_id="amazon/chronos-2"):
        from chronos import BaseChronosPipeline
        self.dev = _device(device)
        self.p = BaseChronosPipeline.from_pretrained(model_id, device_map=self.dev,
                                                     torch_dtype=torch.float32)

    def __call__(self, ctx, H, bs=512, groups=None, past_cov=None, future_cov=None):
        """past_cov / future_cov: optional dict name -> array (n_series, len) for covariates."""
        if groups is None:
            groups = [[i] for i in range(len(ctx))]
        inputs = []
        for g in groups:
            tgt = np.stack([ctx[i] for i in g]).astype(np.float32)
            item = {"target": tgt if len(g) > 1 else tgt[0]}
            if past_cov is not None or future_cov is not None:
                pc = {}
                fc = {}
                for name, arr in (past_cov or {}).items():
                    pc[name] = arr[g[0], :tgt.shape[-1]].astype(np.float32)
                for name, arr in (future_cov or {}).items():
                    pc[name] = arr[g[0], :tgt.shape[-1]].astype(np.float32)
                    fc[name] = arr[g[0], tgt.shape[-1]:tgt.shape[-1] + H].astype(np.float32)
                item["past_covariates"] = pc
                if fc:
                    item["future_covariates"] = fc
            inputs.append(item)
        preds = self.p.predict(inputs, prediction_length=H, batch_size=bs)
        qidx = [self.p.quantiles.index(q) for q in Q]
        res = np.zeros((len(ctx), 3, H), dtype=np.float32)
        for g, pr in zip(groups, preds):
            pr = pr.cpu().numpy()                    # (n_variates, n_quantiles, H)
            for j, i in enumerate(g):
                res[i] = pr[j, qidx, :]
        return res


class TimesFM25:
    def __init__(self, device="auto"):
        import timesfm
        torch.set_float32_matmul_precision("high")
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        self.m.compile(timesfm.ForecastConfig(max_context=64, max_horizon=16, normalize_inputs=True,
                                              use_continuous_quantile_head=True,
                                              force_flip_invariance=True, infer_is_positive=True,
                                              fix_quantile_crossing=True, per_core_batch_size=256))

    def __call__(self, ctx, H, bs=2048):
        out = []
        for i in range(0, len(ctx), bs):
            point, quant = self.m.forecast(horizon=H, inputs=[np.asarray(c, float) for c in ctx[i:i + bs]])
            # quant: (b, H, 10) = mean, q0.1..q0.9
            out.append(np.stack([quant[:, :, 1], quant[:, :, 5], quant[:, :, 9]], axis=1))
        return np.concatenate(out)


class TiRex2:
    def __init__(self, device="auto"):
        from tirex2 import load_model
        self.dev = _device(device)
        self.m = load_model("NX-AI/TiRex-2", device=self.dev)

    def __call__(self, ctx, H, bs=512):
        from tirex2 import TimeseriesType
        ts = [TimeseriesType(target=torch.tensor(np.asarray(c, np.float32))[None, :],
                             past_covariates=None, future_covariates=None) for c in ctx]
        fc = self.m.forecast(ts, prediction_length=H, batch_size=bs)   # list of [1, Q, H]
        ql = [round(float(q), 4) for q in self.m.quantiles.cpu().numpy()]
        qidx = [ql.index(q) for q in Q]
        return np.stack([f.cpu().numpy()[0, qidx, :] for f in fc]).astype(np.float32)


class Toto2:
    def __init__(self, size="22m", device="auto"):
        from toto2 import Toto2Model
        self.dev = _device(device)
        self.m = Toto2Model.from_pretrained(f"Datadog/Toto-2.0-{size}").to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, ctx, H, bs=512):
        # Toto needs equal lengths inside a batch; all our contexts in one call have equal length.
        out = []
        ps = self.m.config.patch_size
        for i in range(0, len(ctx), bs):
            x = np.stack(ctx[i:i + bs]).astype(np.float32)
            T = x.shape[1]
            pad = (-T) % ps                            # left-pad to a whole number of patches
            xp = np.concatenate([np.repeat(x[:, :1], pad, 1), x], 1)
            mask = np.ones_like(xp, dtype=bool)
            mask[:, :pad] = False
            xt = torch.tensor(xp, device=self.dev)[:, None, :]
            mt = torch.tensor(mask, device=self.dev)[:, None, :]
            q = self.m.forecast({"target": xt, "target_mask": mt,
                                 "series_ids": torch.zeros(xt.shape[0], 1, dtype=torch.long, device=self.dev)},
                                horizon=H, has_missing_values=pad > 0)
            q = q.float().cpu().numpy()                # (9, b, 1, H)
            out.append(np.stack([q[0, :, 0], q[4, :, 0], q[8, :, 0]], axis=1))
        return np.concatenate(out)


def make(name: str, device="auto"):
    return {
        "chronos_bolt_small": lambda: ChronosBolt("small", device),
        "chronos_bolt_base": lambda: ChronosBolt("base", device),
        "chronos2": lambda: Chronos2(device),
        "chronos2_mv": lambda: Chronos2(device),
        "timesfm25": lambda: TimesFM25(device),
        "tirex2": lambda: TiRex2(device),
        "toto2_22m": lambda: Toto2("22m", device),
        "toto2_313m": lambda: Toto2("313m", device),
    }[name]()
