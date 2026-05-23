"""
X-MultiVLA Rev4 — Inference Wrapper

Three guarantees:
  1. predict_at(t) uses only feat_arr[t-seq_len : t] — zero future peek
  2. 21 timestamps processed sequentially (no batch scoring)
  3. Portfolio value updated with realized returns at each step

Usage:
    from inference_wrapper import XMultiVLAInferenceWrapper, build_model
    model = build_model(checkpoint_path="checkpoints/v2_R5_phase2_T0.001.pt")
    wrapper = XMultiVLAInferenceWrapper(model, feat_arr, price_arr, timestamps)
    result = wrapper.run(cutoff_datetime="2026-04-05 00:00:00", n_steps=21)
    XMultiVLAInferenceWrapper.print_report(result)
"""

import os, json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.models.itransformer import iTransformer
from src.models.action_head import ActionHead
from config import (
    ASSETS, N_ASSETS, SEQ_LEN, D_MODEL, N_HEADS, N_LAYERS, DROPOUT, COMMISSION
)

ASSETS_CASH = ASSETS + ["Cash"]


class VLAModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.encoder = iTransformer(
            n_coins=N_ASSETS, n_features=n_features, seq_len=SEQ_LEN,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT,
        )
        self.head = ActionHead(D_MODEL, N_ASSETS, DROPOUT)

    def forward(self, x):
        _, _, v_emb = self.encoder(x)
        return self.head(v_emb)


def build_model(n_features: int, phase1_ckpt: str = None, phase2_ckpt: str = None,
                device: str = "cuda") -> VLAModel:
    """Load VLAModel from checkpoint files."""
    model = VLAModel(n_features).to(device)
    if phase1_ckpt and os.path.exists(phase1_ckpt):
        model.load_state_dict(torch.load(phase1_ckpt, map_location=device))
        print(f"Phase 1 weights loaded: {phase1_ckpt}")
    if phase2_ckpt and os.path.exists(phase2_ckpt):
        model.head.load_state_dict(torch.load(phase2_ckpt, map_location=device))
        print(f"Phase 2 weights loaded: {phase2_ckpt}")
    model.eval()
    return model


class XMultiVLAInferenceWrapper:
    """
    Sequential inference over n_steps timestamps.

    Args:
        model:          VLAModel (encoder + head)
        all_feat_arr:   np.ndarray (T, N, F) — feature array, full history
        all_price_arr:  np.ndarray (T, N)    — close prices, full history
        all_timestamps: pd.DatetimeIndex     — timestamps corresponding to rows
        seq_len:        lookback window (default 60)
        n_assets:       number of coins (default 5)
        commission:     rebalancing cost (default 0.001)
        btc_min:        minimum BTC weight enforced at each step (default 0.15)
        device:         "cuda" or "cpu"
    """

    def __init__(self, model: VLAModel,
                 all_feat_arr: np.ndarray,
                 all_price_arr: np.ndarray,
                 all_timestamps: pd.DatetimeIndex,
                 seq_len: int = SEQ_LEN,
                 n_assets: int = N_ASSETS,
                 commission: float = COMMISSION,
                 btc_min: float = 0.15,
                 device: str = "cuda"):
        self.model      = model
        self.all_feat   = all_feat_arr
        self.all_price  = all_price_arr
        self.all_ts     = all_timestamps
        self.seq_len    = seq_len
        self.n_assets   = n_assets
        self.commission = commission
        self.btc_min    = btc_min
        self.device     = device
        self.model.eval()

    def _predict_at(self, t_global_idx: int) -> np.ndarray:
        """
        Predict portfolio weights at time t.
        Uses feat_arr[t-seq_len : t] only — strict no-future-peek.
        """
        window = self.all_feat[t_global_idx - self.seq_len : t_global_idx]
        x = torch.FloatTensor(window).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, v_emb = self.model.encoder(x)
            w = self.model.head(v_emb).squeeze(0).cpu().numpy()

        # Enforce BTC minimum weight
        if w[0] < self.btc_min:
            w[0] = self.btc_min
            rest = w[1:].sum()
            if rest > 1e-8:
                w[1:] *= (1 - self.btc_min) / rest

        w = np.clip(w, 0, 1)
        w /= w.sum() + 1e-8
        return w

    def run(self, cutoff_datetime: str, n_steps: int = 21) -> dict:
        """
        Run sequential inference from cutoff_datetime for n_steps.

        Args:
            cutoff_datetime: ISO string, e.g. "2026-04-05 00:00:00"
            n_steps:         number of steps to evaluate

        Returns:
            dict with keys: cutoff, n_steps, records, equity_curve, total_return
        """
        cutoff_ts  = pd.Timestamp(cutoff_datetime)
        cutoff_idx = np.searchsorted(self.all_ts, cutoff_ts)

        print(f"Blind cutoff : {cutoff_ts}")
        print(f"Eval range   : {self.all_ts[cutoff_idx]} ~ "
              f"{self.all_ts[min(cutoff_idx + n_steps - 1, len(self.all_ts) - 1)]}")
        print("-" * 55)

        equity  = 1.0
        prev_w  = np.zeros(self.n_assets + 1)
        prev_w[-1] = 1.0  # start in cash
        records = []

        for step in range(n_steps):
            t_cur  = cutoff_idx + step
            t_next = t_cur + 1

            if t_cur >= len(self.all_ts):
                break
            ts_cur = self.all_ts[t_cur]

            w = self._predict_at(t_cur)

            if t_next < len(self.all_price):
                coin_rets = ((self.all_price[t_next] - self.all_price[t_cur])
                             / (self.all_price[t_cur] + 1e-9))
                port_ret  = float(np.dot(w[:self.n_assets], coin_rets))
                cost      = self.commission * float(np.abs(w - prev_w).sum())
                step_ret  = port_ret - cost
            else:
                coin_rets = np.zeros(self.n_assets)
                step_ret  = 0.0

            equity *= (1 + step_ret)
            prev_w  = w.copy()

            records.append({
                "step"     : step + 1,
                "timestamp": str(ts_cur),
                "action"   : ASSETS_CASH[int(np.argmax(w))],
                "weights"  : {a: round(float(w[j]) * 100, 1)
                              for j, a in enumerate(ASSETS_CASH)},
                "coin_rets": {ASSETS[j]: round(float(coin_rets[j]) * 100, 2)
                              for j in range(self.n_assets)},
                "step_ret" : round(step_ret * 100, 3),
                "equity"   : round(equity, 6),
            })

        return {
            "cutoff"      : cutoff_datetime,
            "n_steps"     : n_steps,
            "records"     : records,
            "equity_curve": [r["equity"] for r in records],
            "total_return": round((equity - 1) * 100, 2),
        }

    @staticmethod
    def print_report(result: dict):
        print(f"\n{'='*72}")
        print(f"  X-MultiVLA Rev4  Inference Validation")
        print(f"  Cutoff: {result['cutoff']}  ({result['n_steps']} steps)")
        print(f"{'='*72}")
        print(f"  {'St':>3}  {'Timestamp':>22}  {'Action':>5}  "
              f"{'BTC%':>5} {'Cash%':>5}  {'StepRet':>8}  {'CumRet':>8}")
        print(f"  {'-'*68}")
        for r in result["records"]:
            w = r["weights"]
            print(f"  {r['step']:>3}  {r['timestamp']:>22}  {r['action']:>5}  "
                  f"{w.get('BTC', 0):>5.1f} {w.get('Cash', 0):>5.1f}  "
                  f"{r['step_ret']:>+7.3f}%  {(r['equity'] - 1) * 100:>+7.2f}%")
        print(f"  {'-'*68}")
        print(f"  Total return: {result['total_return']:+.2f}%")
        print(f"{'='*72}")

    def save_result(self, result: dict, path: str,
                    btc_price_arr: np.ndarray = None, cutoff_idx: int = None):
        """Save result JSON with optional BTC benchmark."""
        extra = {}
        if btc_price_arr is not None and cutoff_idx is not None:
            n = result["n_steps"]
            btc_start = float(btc_price_arr[cutoff_idx, 0])
            btc_end   = float(btc_price_arr[min(cutoff_idx + n - 1, len(btc_price_arr) - 1), 0])
            btc_ret   = (btc_end / btc_start - 1) * 100
            extra = {
                "btc_return_pct" : round(btc_ret, 2),
                "alpha_pct"      : round(result["total_return"] - btc_ret, 2),
                "btc_price_start": btc_start,
                "btc_price_end"  : btc_end,
            }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({**result, **extra}, f, indent=2, default=str)
        print(f"Saved: {path}")
