"""
X-MultiVLA Rev4 — Inference Runner

Loads the trained model (R5 by default) and runs 21-step sequential inference
starting from a specified cutoff datetime.

Usage:
    python run_inference.py
    python run_inference.py --round R4 --cutoff "2026-01-01 00:00:00" --steps 30
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from inference_wrapper import XMultiVLAInferenceWrapper, build_model
from config import (
    ASSETS, N_ASSETS, SYM_MAP, NEWS_COLS, META_COLS,
    SEQ_LEN, CKPT_DIR, DATASET_FILE,
)


def load_data():
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round",   default="R5",
                        help="Which round checkpoint to load (R1-R5)")
    parser.add_argument("--cutoff",  default=None,
                        help="Blind cutoff datetime, e.g. '2026-04-05 00:00:00'. "
                             "Defaults to 21 steps before end of data.")
    parser.add_argument("--steps",   type=int, default=21)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    feat_arr, price_arr, timestamps, feature_cols = load_data()
    n_features = len(feature_cols)
    print(f"Data: {len(timestamps):,} timestamps  ({timestamps[0].date()} ~ {timestamps[-1].date()})")

    oracle_temp = 0.001
    p1_ckpt = f"{CKPT_DIR}/v2_{args.round}_phase1_T{oracle_temp}.pt"
    p2_ckpt = f"{CKPT_DIR}/v2_{args.round}_phase2_T{oracle_temp}.pt"

    model = build_model(n_features, phase1_ckpt=p1_ckpt, phase2_ckpt=p2_ckpt, device=device)

    cutoff_str = args.cutoff or str(timestamps[-args.steps])
    print(f"Cutoff: {cutoff_str}  ({args.steps} steps)")

    wrapper = XMultiVLAInferenceWrapper(
        model          = model,
        all_feat_arr   = feat_arr,
        all_price_arr  = price_arr,
        all_timestamps = timestamps,
        seq_len        = SEQ_LEN,
        n_assets       = N_ASSETS,
        btc_min        = 0.15,
        device         = device,
    )

    result = wrapper.run(cutoff_datetime=cutoff_str, n_steps=args.steps)
    XMultiVLAInferenceWrapper.print_report(result)

    # BTC benchmark
    cutoff_idx = int(pd.Index(timestamps).get_loc(pd.Timestamp(cutoff_str), method="nearest"))
    btc_start  = float(price_arr[cutoff_idx, 0])
    btc_end    = float(price_arr[min(cutoff_idx + args.steps - 1, len(price_arr) - 1), 0])
    btc_ret    = (btc_end / btc_start - 1) * 100
    alpha      = result["total_return"] - btc_ret

    print(f"\n  Strategy : {result['total_return']:+.2f}%")
    print(f"  BTC B&H  : {btc_ret:+.2f}%")
    print(f"  Alpha    : {alpha:+.2f}%")

    out_path = os.path.join(args.out_dir, f"inference_{args.round}_result.json")
    wrapper.save_result(result, out_path, price_arr, cutoff_idx)


if __name__ == "__main__":
    main()
