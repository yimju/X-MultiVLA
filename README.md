# X-MultiVLA Rev4

**VLA (Vision-Language-Action) framework for multi-asset cryptocurrency portfolio management.**

Applies the embodied-agent architecture from robotics to daily portfolio rebalancing across 5 crypto coins.

---

## Architecture

```
Market Data (60-bar window)
       │
       ▼
  iTransformer          ← Vision: cross-coin attention (ICLR 2024)
  (encoder)
       │  v_emb (256d)
       ▼
  ActionHead            ← Language→Action: MLP + Softmax
  (3-layer MLP)
       │  portfolio weights
       ▼
  {BTC, ETH, SOL, XRP, DOGE, Cash}   (BTC ≥ 15% enforced)
```

**Two-stage training** (analogous to LLM SFT → RLHF):
- **Phase 1** — Oracle DP Supervised Fine-Tuning: backward DP generates near-optimal labels (τ=0.001, EW-relative reward)
- **Phase 2** — GRPO Reinforcement Learning: Group Relative Policy Optimisation with G=8 Dirichlet samples per step, ActionHead only (encoder frozen)

---

## Walk-Forward Results

| Round | Test Period | BTC B&H | X-MultiVLA | Alpha |
|-------|-------------|---------|-----------|-------|
| R1    | 2025 Q1     | +24.6%  | +6.33%    | −18.3%p |
| R2    | 2025 Q2     | +4.3%   | +23.59%   | **+19.3%p** |
| R3    | 2025 Q3     | −15.2%  | −18.63%   | −3.4%p |
| R4    | 2025 Q4     | −25.2%  | −24.09%   | **+1.1%p** |
| R5    | 2026 Q1     | +9.3%   | +2.04%    | −7.3%p |
| **Avg** | 5 rounds  | −0.4%   | −2.15%    | −1.75%p |

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Data

The model expects a parquet file at `config.DATASET_FILE` with schema:

```
columns: symbol, open_time, open, high, low, close, volume, [technical indicators]
```

One row per (symbol, timestamp). Timestamps must align across all 5 coins.

### Checkpoints

Download from Google Drive (`X-MultiVLA_rev4/checkpoints/`):

```
v2_R{1-5}_phase1_T0.001.pt    ← Phase 1 full model weights
v2_R{1-5}_phase2_T0.001.pt    ← Phase 2 ActionHead weights only
v2_R{1-5}_oracle_T0.001.pkl   ← Oracle DP label cache
```

---

## Training

```bash
# Train all 5 rounds sequentially
python train.py --round all

# Train a single round
python train.py --round R5

# Skip Phase 1 (load existing checkpoint)
python train.py --round R5 --skip_phase1
```

Requires GPU (A100 recommended). Phase 1: ~80 epochs, Phase 2: 30,000 GRPO steps.

---

## Inference

```bash
# Run 21-step inference from last available date (uses R5 checkpoint)
python run_inference.py

# Custom cutoff and round
python run_inference.py --round R4 --cutoff "2026-04-05 00:00:00" --steps 21
```

### Programmatic Usage

```python
import numpy as np
import pandas as pd
from inference_wrapper import XMultiVLAInferenceWrapper, build_model

# 1. Load model
model = build_model(
    n_features=94,
    phase1_ckpt="checkpoints/v2_R5_phase1_T0.001.pt",
    phase2_ckpt="checkpoints/v2_R5_phase2_T0.001.pt",
    device="cuda",
)

# 2. Prepare data arrays
# feat_arr:  (T, N_coins, N_features)  float32
# price_arr: (T, N_coins)              float64
# timestamps: pd.DatetimeIndex

# 3. Run inference
wrapper = XMultiVLAInferenceWrapper(
    model=model,
    all_feat_arr=feat_arr,
    all_price_arr=price_arr,
    all_timestamps=timestamps,
    btc_min=0.15,
    device="cuda",
)
result = wrapper.run(cutoff_datetime="2026-04-05 00:00:00", n_steps=21)
XMultiVLAInferenceWrapper.print_report(result)
```

---

## Repository Structure

```
X-MultiVLA/
├── config.py                   ← global constants (paths, hyperparams)
├── train.py                    ← walk-forward training (Phase 1 + Phase 2)
├── run_inference.py            ← inference runner script
├── inference_wrapper.py        ← XMultiVLAInferenceWrapper class
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── itransformer.py     ← Vision encoder (ICLR 2024)
│   │   └── action_head.py      ← Action MLP
│   └── training/
│       └── oracle_labels.py    ← Oracle DP label generator
└── data/
    ├── collector.py            ← Binance OHLCV downloader
    ├── preprocessor.py         ← feature engineering
    └── news_fetcher.py         ← news sentiment (optional)
```

---

## Known Limitations

- **v_emb collapse**: iTransformer outputs near-constant embedding (std ≈ 0.0025) → ActionHead learns a fixed allocation rather than context-adaptive weights. Proposed fix: VICReg loss + Reconstruction head.
- **BTC underweight**: Oracle DP favours altcoins in bull markets → explicit BTC ≥ 15% constraint added.
- Static allocation degrades in strong directional markets where single-asset concentration would dominate.

---

## Citation

If you use this code, please cite:

```
X-MultiVLA Rev4: VLA-Inspired Multi-Asset Crypto Portfolio Management
Graduate project, Hanyang University, 2026
```
