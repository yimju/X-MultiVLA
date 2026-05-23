"""
X-MultiVLA Rev4 — Walk-Forward Training Script

Two-stage training per round:
  Phase 1: Oracle DP Supervised Fine-Tuning (iTransformer + ActionHead)
  Phase 2: GRPO Reinforcement Learning (ActionHead only, encoder frozen)

Usage:
    python train.py --round R1
    python train.py --round R5 --skip_phase1  # load existing phase1 checkpoint
"""

import argparse
import copy
import gc
import importlib
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── path setup for src/ ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from src.models.itransformer import iTransformer
from src.models.action_head import ActionHead
from src.training.oracle_labels import OracleLabelGenerator
from config import (
    ASSETS, N_ASSETS, SYM_MAP, NEWS_COLS, META_COLS,
    SEQ_LEN, D_MODEL, N_HEADS, N_LAYERS, DROPOUT,
    COMMISSION, BATCH_SIZE, CKPT_DIR, DATASET_FILE,
    GRPO_GROUP_SIZE, GRPO_LR, GRPO_CLIP_EPS, GRPO_DIRICHLET_CONC,
    ROUNDS, TRAIN_START,
)

# ── hyperparameters ───────────────────────────────────────────────────────────
P1_EPOCHS    = 80
P1_LR        = 1e-3
ORACLE_TEMP  = 0.001
GRPO_STEPS   = 30_000
KL_BETA      = 0.001
BTC_MIN      = 0.15   # minimum BTC weight post-processing


# ── model ─────────────────────────────────────────────────────────────────────
class VLASimple(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = iTransformer(
            n_coins=N_ASSETS, n_features=n_features, seq_len=SEQ_LEN,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT,
        )
        self.head = ActionHead(D_MODEL, N_ASSETS, DROPOUT)

    def get_v_emb(self, x):
        _, _, v = self.encoder(x)
        return v

    def forward(self, x):
        return self.head(self.get_v_emb(x))

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def precompute_v_embs(self, windows, batch_size=64):
        self.eval()
        embs = []
        dev  = next(self.parameters()).device
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                embs.append(self.get_v_emb(
                    torch.FloatTensor(windows[i:i + batch_size]).to(dev)
                ).cpu().numpy())
        return np.vstack(embs)

    def forward_from_v_emb(self, v):
        return self.head(v)


def load_data():
    """Load and align multi-coin dataset from parquet."""
    df = pd.read_parquet(DATASET_FILE)
    df["open_time"] = pd.to_datetime(df["open_time"])
    feature_cols = [c for c in df.columns if c not in set(NEWS_COLS + META_COLS)]

    coin_dfs = {}
    for sym, coin in SYM_MAP.items():
        sub = df[df["symbol"] == sym].sort_values("open_time").reset_index(drop=True)
        sub.index = sub["open_time"]
        coin_dfs[coin] = sub

    timestamps = pd.DatetimeIndex(
        sorted(set.intersection(*[set(d.index) for d in coin_dfs.values()])))
    price_arr = np.stack(
        [coin_dfs[c]["close"].reindex(timestamps).values for c in ASSETS], axis=1)
    feat_arr = np.stack(
        [coin_dfs[c][feature_cols].reindex(timestamps).fillna(0).values
         for c in ASSETS], axis=1).astype(np.float32)

    return feat_arr, price_arr, timestamps, feature_cols


def train_round(run_round: str, skip_phase1: bool = False, device: str = "cuda"):
    feat_arr, price_arr, timestamps, feature_cols = load_data()
    n_features = len(feature_cols)

    windows   = np.stack([feat_arr[i:i + SEQ_LEN] for i in range(len(feat_arr) - SEQ_LEN + 1)])
    prices_w  = price_arr[SEQ_LEN - 1:]
    win_ts    = timestamps[SEQ_LEN - 1:]

    round_info = next(r for r in ROUNDS if r["name"] == run_round)
    test_start = pd.Timestamp(round_info["test_start"])
    test_end   = pd.Timestamp(round_info["test_end"])
    train_start = pd.Timestamp(TRAIN_START)

    tr_idx = np.where((win_ts >= train_start) & (win_ts < test_start))[0]
    te_idx = np.where((win_ts >= test_start)  & (win_ts < test_end))[0]

    X_tr, P_tr = windows[tr_idx], prices_w[tr_idx]
    X_te, P_te = windows[te_idx], prices_w[te_idx]

    btc_s   = price_arr[(timestamps >= test_start) & (timestamps < test_end), 0]
    btc_ret = (btc_s[-1] / btc_s[0] - 1) * 100 if len(btc_s) > 1 else 0.0
    print(f"\n{'='*55}\n  {run_round}: {test_start.date()} ~ {test_end.date()}  BTC={btc_ret:+.1f}%\n{'='*55}")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── Oracle cache ──────────────────────────────────────────────────────────
    oracle_cache = f"{CKPT_DIR}/v2_{run_round}_oracle_T{ORACLE_TEMP}.pkl"
    if os.path.exists(oracle_cache):
        d = joblib.load(oracle_cache)
        oracle_labels, focal_weights = d["labels"], d["focal"]
        print(f"  Oracle cache loaded: {oracle_cache}")
    else:
        oracle = OracleLabelGenerator(
            commission=COMMISSION, n_assets=N_ASSETS, gamma=0.99,
            temperature=ORACLE_TEMP, volatility_penalty=0.3,
            focal_weight=10.0, focal_percentile=95.0,
        )
        oracle_labels, focal_weights = oracle.generate(
            np.vstack([P_tr, P_tr[-1:]]), verbose=True)
        joblib.dump({"labels": oracle_labels, "focal": focal_weights}, oracle_cache)

    # ── Phase 1: Oracle SFT ───────────────────────────────────────────────────
    p1_ckpt = f"{CKPT_DIR}/v2_{run_round}_phase1_T{ORACLE_TEMP}.pt"
    model   = VLASimple(n_features).to(device)

    if skip_phase1 and os.path.exists(p1_ckpt):
        model.load_state_dict(torch.load(p1_ckpt, map_location=device))
        model.freeze_encoder()
        print(f"  Phase 1 loaded: {p1_ckpt}")
    elif not os.path.exists(p1_ckpt):
        ds1  = TensorDataset(torch.FloatTensor(X_tr),
                             torch.FloatTensor(oracle_labels),
                             torch.FloatTensor(focal_weights))
        dl1  = DataLoader(ds1, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
        opt1 = torch.optim.AdamW(model.parameters(), lr=P1_LR, weight_decay=1e-4)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=P1_EPOCHS)
        best = float("inf")

        for epoch in range(P1_EPOCHS):
            model.train()
            total = 0
            for xb, yb, fw in dl1:
                w    = model(xb.to(device))
                loss = (-(yb.to(device) * torch.log(w.clamp(1e-7))).sum(-1)
                        * fw.to(device)).mean()
                if not torch.isfinite(loss):
                    continue
                opt1.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
                opt1.step()
                total += loss.item()
            sch1.step()
            avg = total / len(dl1)
            if avg < best:
                best = avg
                torch.save(model.state_dict(), p1_ckpt)

        model.load_state_dict(torch.load(p1_ckpt, map_location=device))
        model.freeze_encoder()
        print(f"  Phase 1 done  CE={best:.4f}")
    else:
        model.load_state_dict(torch.load(p1_ckpt, map_location=device))
        model.freeze_encoder()
        print(f"  Phase 1 loaded: {p1_ckpt}")

    # ── Phase 2: GRPO ─────────────────────────────────────────────────────────
    p2_ckpt  = f"{CKPT_DIR}/v2_{run_round}_phase2_T{ORACLE_TEMP}.pt"
    vemb_tr  = model.precompute_v_embs(X_tr)

    if os.path.exists(p2_ckpt):
        model.head.load_state_dict(torch.load(p2_ckpt, map_location=device))
        print(f"  Phase 2 loaded: {p2_ckpt}")
    else:
        ref_head = copy.deepcopy(model.head)
        for p in ref_head.parameters():
            p.requires_grad_(False)
        ref_head.eval()

        opt2   = torch.optim.AdamW(model.head.parameters(), lr=GRPO_LR, weight_decay=1e-4)
        T, G   = len(vemb_tr) - 1, GRPO_GROUP_SIZE
        prev_w = np.zeros(N_ASSETS + 1)
        prev_w[-1] = 1.0
        model.head.train()

        for step in range(GRPO_STEPS):
            t     = np.random.randint(0, T)
            h_t   = torch.FloatTensor(vemb_tr[t]).unsqueeze(0).to(device)
            w     = model.head(h_t).squeeze(0)
            conc  = (w * GRPO_DIRICHLET_CONC).clamp(min=1.0)
            dist  = Dirichlet(conc)
            samp  = dist.sample((G,))
            lp    = dist.log_prob(samp)

            with torch.no_grad():
                rw  = ref_head(h_t).squeeze(0)
                rlp = Dirichlet((rw * GRPO_DIRICHLET_CONC).clamp(1.0)).log_prob(samp)

            rets = ((P_tr[t + 1] - P_tr[t]) / (P_tr[t] + 1e-9)
                    if t + 1 < len(P_tr) else np.zeros(N_ASSETS))
            rews = [float(np.dot(ww[:N_ASSETS], rets)) - COMMISSION * np.abs(ww - prev_w).sum()
                    for ww in samp.cpu().numpy()]

            rew  = torch.tensor(rews, dtype=torch.float32, device=device)
            adv  = (rew - rew.mean()) / (rew.std().clamp(1e-3) + 1e-8)
            ratio   = torch.exp((lp - rlp).clamp(-10, 10))
            clipped = ratio.clamp(1 - GRPO_CLIP_EPS, 1 + GRPO_CLIP_EPS)
            loss    = (-torch.min(ratio * adv, clipped * adv).mean()
                       + KL_BETA * (lp - rlp).mean())

            if not torch.isfinite(loss):
                continue
            opt2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            opt2.step()
            prev_w = w.detach().cpu().numpy()

        torch.save(model.head.state_dict(), p2_ckpt)
        print(f"  Phase 2 done  checkpoint: {p2_ckpt}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    vemb_te = model.precompute_v_embs(X_te)
    equity  = 1.0
    prev_w  = np.zeros(N_ASSETS + 1)
    prev_w[-1] = 1.0

    with torch.no_grad():
        for i in range(len(vemb_te) - 1):
            h_t = torch.FloatTensor(vemb_te[i]).unsqueeze(0).to(device)
            w   = model.forward_from_v_emb(h_t).squeeze(0).cpu().numpy().copy()

            # BTC minimum weight
            if w[0] < BTC_MIN:
                w[0]  = BTC_MIN
                rest  = w[1:].sum()
                if rest > 1e-8:
                    w[1:] *= (1 - BTC_MIN) / rest

            w = np.clip(w, 0, 1)
            w /= w.sum() + 1e-8
            rets   = (P_te[i + 1] - P_te[i]) / (P_te[i] + 1e-9)
            equity *= (1 + float(np.dot(w[:N_ASSETS], rets))
                       - COMMISSION * np.abs(w - prev_w).sum())
            prev_w  = w

    total_ret = equity - 1.0
    result_path = f"{CKPT_DIR}/wf_result_{run_round}.json"
    result = {
        "round"       : run_round,
        "total_return": total_ret,
        "btc_ret_pct" : float(btc_ret),
        "alpha_pct"   : total_ret * 100 - btc_ret,
        "test_start"  : str(test_start.date()),
        "test_end"    : str(test_end.date()),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    flag = "✅" if total_ret * 100 > btc_ret else ("🔶" if total_ret * 100 > 0 else "❌")
    print(f"\n{flag} {run_round} | Strategy={total_ret*100:+.2f}%  BTC={btc_ret:+.1f}%  Alpha={total_ret*100-btc_ret:+.2f}%")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=[r["name"] for r in ROUNDS] + ["all"],
                        default="all")
    parser.add_argument("--skip_phase1", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    rounds_to_run = [r["name"] for r in ROUNDS] if args.round == "all" else [args.round]
    for rnd in rounds_to_run:
        train_round(rnd, skip_phase1=args.skip_phase1, device=device)
