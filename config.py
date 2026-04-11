# config.py
# ============================================================
# X-MultiVLA 전역 설정 파일
# Google Colab Secrets 탭에 아래 키를 등록하거나 직접 입력하세요.
# ============================================================

import os
from dataclasses import dataclass, field
from typing import List

# ─────────────────────────────────────────
# API Keys (Colab Secrets 사용 권장)
# ─────────────────────────────────────────
try:
    from google.colab import userdata
    FRED_API_KEY       = userdata.get("FRED_API_KEY")
    BINANCE_API_KEY    = userdata.get("BINANCE_API_KEY")     # 선택 (퍼블릭 데이터 불필요)
    BINANCE_SECRET_KEY = userdata.get("BINANCE_SECRET_KEY")  # 실거래 시만 필요
    CRYPTOPANIC_API_KEY = userdata.get("CRYPTOPANIC_API_KEY")  # 무료 키 발급
except Exception:
    # 로컬 실행 시 환경변수로 대체
    FRED_API_KEY       = os.getenv("FRED_API_KEY", "")
    BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
    CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")


# ─────────────────────────────────────────
# 데이터 설정
# ─────────────────────────────────────────
@dataclass
class DataConfig:
    symbol: str             = "BTC/USDT"
    timeframe: str          = "1h"          # 기본 타임프레임
    timeframes: List[str]   = field(default_factory=lambda: ["1h", "1d"])
    history_days: int       = 730           # 과거 2년치
    window_size: int        = 60            # 모델 입력 윈도우 (시간 단위)
    train_ratio: float      = 0.8
    val_ratio: float        = 0.1
    # test_ratio = 1 - train_ratio - val_ratio

    # Yahoo Finance 심볼
    nasdaq_symbol: str      = "^IXIC"
    gold_symbol: str        = "GC=F"
    dxy_symbol: str         = "DX-Y.NYB"

    # FRED 시리즈 ID
    fred_series: List[str]  = field(default_factory=lambda: ["FEDFUNDS", "T10Y2Y", "CPIAUCSL"])

    # 저장 경로
    raw_data_dir: str       = "/content/drive/MyDrive/X-MultiVLA/data/raw"
    processed_data_dir: str = "/content/drive/MyDrive/X-MultiVLA/data/processed"


# ─────────────────────────────────────────
# 모델 설정
# ─────────────────────────────────────────
@dataclass
class ModelConfig:
    # PatchTST (시계열 인코더)
    patch_size: int         = 16
    num_patches: int        = 4             # window_size // patch_size
    d_model: int            = 128
    nhead: int              = 8
    num_encoder_layers: int = 3
    dim_feedforward: int    = 256
    dropout: float          = 0.1

    # FinBERT (뉴스 인코더)
    finbert_model: str      = "ProsusAI/finbert"
    news_embed_dim: int     = 768           # FinBERT 출력 차원
    news_proj_dim: int      = 128           # 프로젝션 후 차원

    # Fusion (Cross-Attention)
    fusion_dim: int         = 256           # chart_dim + news_proj_dim
    fusion_heads: int       = 4

    # Action Head
    action_dim: int         = 3             # [Long, Short, Hold]
    hidden_dim: int         = 128


# ─────────────────────────────────────────
# 강화학습 설정
# ─────────────────────────────────────────
@dataclass
class RLConfig:
    algorithm: str          = "PPO"
    total_timesteps: int    = 500_000
    learning_rate: float    = 3e-4
    n_steps: int            = 2048          # PPO rollout 길이
    batch_size: int         = 64
    n_epochs: int           = 10
    gamma: float            = 0.99          # 할인율
    gae_lambda: float       = 0.95
    clip_range: float       = 0.2
    ent_coef: float         = 0.01          # 탐색 장려

    # 보상 함수 가중치
    reward_profit_weight: float    = 1.0
    reward_sharpe_weight: float    = 0.5
    reward_mdd_penalty: float      = 2.0    # 최대 낙폭 패널티 계수
    reward_fee_rate: float         = 0.001  # 바이낸스 거래 수수료 0.1%
    reward_align_weight: float     = 0.3    # 설명 일관성 보상

    # 환경 설정
    initial_balance: float  = 10_000.0     # 시뮬레이션 초기 잔고 (USDT)
    max_position: float     = 1.0          # 최대 포지션 비율


# ─────────────────────────────────────────
# 학습 설정
# ─────────────────────────────────────────
@dataclass
class TrainConfig:
    # 지도학습 사전 학습
    pretrain_epochs: int    = 30
    pretrain_lr: float      = 1e-3
    pretrain_batch: int     = 32

    # 체크포인트
    checkpoint_dir: str     = "/content/drive/MyDrive/X-MultiVLA/checkpoints"
    log_dir: str            = "/content/drive/MyDrive/X-MultiVLA/logs"

    # 디바이스 (Colab GPU 자동 감지)
    device: str             = "auto"        # "cuda" / "cpu" / "auto"

    # 시드 고정 (재현성)
    seed: int               = 42


# ─────────────────────────────────────────
# 단일 인스턴스 (임포트 즉시 사용 가능)
# ─────────────────────────────────────────
data_cfg  = DataConfig()
model_cfg = ModelConfig()
rl_cfg    = RLConfig()
train_cfg = TrainConfig()
