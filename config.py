# X-MultiVLA v15 — Config

SYM_MAP = {
    'BTC-USD': 'BTC', 'ETH-USD': 'ETH', 'SOL-USD': 'SOL',
    'XRP-USD': 'XRP', 'DOGE-USD': 'DOGE',
}
BTCUSDT_MAP = {
    'BTCUSDT': 'BTC', 'ETHUSDT': 'ETH', 'SOLUSDT': 'SOL',
    'XRPUSDT': 'XRP', 'DOGEUSDT': 'DOGE',
}
ASSETS    = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']
N_ASSETS  = 5
META_COLS = ['symbol', 'open_time']
NEWS_COLS = []
SEQ_LEN   = 60
D_MODEL   = 256
N_HEADS   = 8
N_LAYERS  = 2
DROPOUT   = 0.1
BATCH_SIZE  = 32
COMMISSION  = 0.001
TRAIN_START = '2024-06-01'

# Walk-Forward rounds
ROUNDS = [
    {'name': 'R1', 'test_start': '2025-03-01', 'test_end': '2025-06-01'},
    {'name': 'R2', 'test_start': '2025-06-01', 'test_end': '2025-09-01'},
    {'name': 'R3', 'test_start': '2025-09-01', 'test_end': '2025-12-01'},
    {'name': 'R4', 'test_start': '2025-12-01', 'test_end': '2026-03-01'},
    {'name': 'R5', 'test_start': '2026-03-01', 'test_end': '2026-06-01'},
]

# Phase 1
P1_EPOCHS     = 80
P1_LR         = 1e-3
P1_RANK_W     = 0.05
P1_VIC_W      = 0.05
P1_PRICE_W    = 0.3

# Sortino GT
SHARPE_HORIZON = 12     # 12 x 8h = 4 days
LONG_HORIZON   = 36     # 36 x 8h = 12 days
GT_BLEND       = 0.6
SMOOTH_ALPHA   = 0.35
DD_THRESHOLD   = 0.03   # 3% drawdown → start boost
DD_MAX_BOOST   = 4.0

# GRPO
GRPO_EPOCHS = 200
GRPO_LR     = 3e-5
GRPO_G      = 8
GRPO_CONC   = 0.3
