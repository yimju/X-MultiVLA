# X-MultiVLA Rev4 Configuration
# ============================================================
# Edit paths for your environment before running train.py / run_inference.py
# ============================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Google Colab (default)
REV4_ROOT  = "/content/drive/MyDrive/X-MultiVLA_rev4"
DATA_DIR   = "/content/drive/MyDrive/X-MultiVLA_rev2_5Simbols/data"   # shared data from Rev3
CKPT_DIR   = f"{REV4_ROOT}/checkpoints"
OUT_DIR    = f"{REV4_ROOT}/outputs"
LOG_DIR    = f"{REV4_ROOT}/logs"

# Dataset parquet (OHLCV + technical indicators, all 5 coins combined)
DATASET_FILE = f"{DATA_DIR}/dataset_8hr_full.parquet"

# ── Assets ────────────────────────────────────────────────────────────────────
ASSETS  = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
SYM_MAP = {
    "BTCUSDT" : "BTC",
    "ETHUSDT" : "ETH",
    "SOLUSDT" : "SOL",
    "XRPUSDT" : "XRP",
    "DOGEUSDT": "DOGE",
}
N_ASSETS = len(ASSETS)

# Columns to exclude when building feature matrix
NEWS_COLS = ["news_count", "news_sentiment_mean", "news_pos_mean", "news_neg_mean"]
META_COLS = ["symbol", "open_time", "date", "target", "close", "symbol_encoded"]

# ── iTransformer (Vision encoder) ─────────────────────────────────────────────
SEQ_LEN  = 60     # lookback window (bars)
D_MODEL  = 256    # embedding dimension
N_HEADS  = 8
N_LAYERS = 4
DROPOUT  = 0.1

# ── Training ──────────────────────────────────────────────────────────────────
COMMISSION  = 0.001   # 0.1% per-side rebalancing cost
BATCH_SIZE  = 64

# ── Walk-Forward Rounds ───────────────────────────────────────────────────────
TRAIN_START = "2024-03-01"
ROUNDS = [
    {"name": "R1", "test_start": "2025-03-01", "test_end": "2025-06-01"},
    {"name": "R2", "test_start": "2025-06-01", "test_end": "2025-09-01"},
    {"name": "R3", "test_start": "2025-09-01", "test_end": "2025-12-01"},
    {"name": "R4", "test_start": "2025-12-01", "test_end": "2026-03-01"},
    {"name": "R5", "test_start": "2026-03-01", "test_end": "2026-04-13"},
]

# ── GRPO (Phase 2 RL) ─────────────────────────────────────────────────────────
GRPO_STEPS          = 30_000
GRPO_GROUP_SIZE     = 8        # G — samples per GRPO step
GRPO_LR             = 1e-4
GRPO_KL_BETA        = 0.001    # KL penalty coefficient
GRPO_CLIP_EPS       = 0.2      # PPO clip epsilon
GRPO_DIRICHLET_CONC = 10.0     # Dirichlet concentration scale

# ── API Keys (optional, for data collection) ──────────────────────────────────
try:
    from google.colab import userdata
    BINANCE_API_KEY     = userdata.get("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY  = userdata.get("BINANCE_SECRET_KEY", "")
    CRYPTOPANIC_API_KEY = userdata.get("CRYPTOPANIC_API_KEY", "")
except Exception:
    BINANCE_API_KEY     = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY  = os.getenv("BINANCE_SECRET_KEY", "")
    CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
