# data/preprocessor.py
# ============================================================
# 수집된 원시 데이터를 모델 입력 형태로 전처리
#   1. 로그 수익률 변환 (가격 → 비율)
#   2. 기술 지표 추가 (RSI, MACD, Bollinger)
#   3. 정규화 (StandardScaler)
#   4. 시퀀스 슬라이딩 윈도우 생성 → PyTorch Dataset
# ============================================================

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib
import os

from config import data_cfg, train_cfg


# ───────────────────────────────────────
# 기술 지표 계산 헬퍼
# ───────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    ma    = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    # %B (현재 위치) 와 bandwidth
    pct_b = (series - lower) / (upper - lower + 1e-8)
    bw    = (upper - lower) / (ma + 1e-8)
    return pct_b, bw


# ───────────────────────────────────────
# 메인 전처리 클래스
# ───────────────────────────────────────

class Preprocessor:
    """
    사용법:
        pp = Preprocessor()
        train_ds, val_ds, test_ds = pp.fit_transform(merged_df)
        pp.save_scaler("scaler.pkl")
    """

    def __init__(self):
        self.scaler      = StandardScaler()
        self.feature_cols: list = []

    # ───────────────────────────────────────
    # 1. 피처 엔지니어링
    # ───────────────────────────────────────
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        원시 DataFrame에 기술 지표 + 로그 수익률 컬럼을 추가합니다.
        """
        df = df.copy()

        # 로그 수익률 (BTC 및 거시지표)
        for col in ["BTC_close", "NASDAQ", "GOLD", "DXY"]:
            if col in df.columns:
                df[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))

        # 기술 지표 (BTC 기반)
        price = df["BTC_close"]
        df["RSI_14"]             = _rsi(price, 14)
        macd, sig, hist          = _macd(price)
        df["MACD"]               = macd
        df["MACD_signal"]        = sig
        df["MACD_hist"]          = hist
        df["BB_pctB"], df["BB_bw"] = _bollinger(price, 20)

        # 거래량 변화율
        if "BTC_volume" in df.columns:
            df["volume_ret"] = np.log(df["BTC_volume"] / df["BTC_volume"].shift(1))

        # 뉴스 감성 (있는 경우)
        news_cols = [c for c in df.columns if c.startswith("news_")]

        # 최종 피처 목록 (학습에 사용할 컬럼)
        base_features = [
            "BTC_open",   "BTC_high",  "BTC_low",   "BTC_close", "BTC_volume",
            "BTC_close_ret", "NASDAQ_ret", "GOLD_ret", "DXY_ret",
            "RSI_14", "MACD", "MACD_signal", "MACD_hist",
            "BB_pctB", "BB_bw", "volume_ret",
        ]
        macro_cols  = [c for c in ["FEDFUNDS", "T10Y2Y", "CPIAUCSL"] if c in df.columns]
        self.feature_cols = [c for c in base_features + macro_cols + news_cols if c in df.columns]

        df = df.dropna()
        return df

    # ───────────────────────────────────────
    # 2. 정규화 + 학습/검증/테스트 분리
    # ───────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame, window: int = None):
        """
        Returns:
            train_ds, val_ds, test_ds  (TradingDataset 인스턴스)
        """
        window = window or data_cfg.window_size
        df     = self.engineer_features(df)

        X = df[self.feature_cols].values
        y = df["BTC_close_ret"].values   # 예측 타깃: 다음 스텝 수익률

        # 학습 집합으로만 scaler fit (data leakage 방지)
        n       = len(X)
        n_train = int(n * data_cfg.train_ratio)
        n_val   = int(n * data_cfg.val_ratio)

        self.scaler.fit(X[:n_train])
        X_scaled = self.scaler.transform(X)

        train_ds = TradingDataset(X_scaled[:n_train],        y[:n_train],        window)
        val_ds   = TradingDataset(X_scaled[n_train:n_train+n_val], y[n_train:n_train+n_val], window)
        test_ds  = TradingDataset(X_scaled[n_train+n_val:],  y[n_train+n_val:],  window)

        print(f"[preprocessor] 피처 수: {len(self.feature_cols)}")
        print(f"[preprocessor] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        return train_ds, val_ds, test_ds

    def transform(self, df: pd.DataFrame, window: int = None) -> "TradingDataset":
        """이미 fit된 scaler로 새 데이터 변환 (추론용)"""
        window = window or data_cfg.window_size
        df     = self.engineer_features(df)
        X      = self.scaler.transform(df[self.feature_cols].values)
        y      = df["BTC_close_ret"].values
        return TradingDataset(X, y, window)

    # ───────────────────────────────────────
    # 3. Scaler 저장/로드
    # ───────────────────────────────────────
    def save_scaler(self, path: str = "scaler.pkl"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_cols": self.feature_cols}, path)
        print(f"[preprocessor] Scaler 저장: {path}")

    def load_scaler(self, path: str = "scaler.pkl"):
        data = joblib.load(path)
        self.scaler       = data["scaler"]
        self.feature_cols = data["feature_cols"]
        print(f"[preprocessor] Scaler 로드: {path} ({len(self.feature_cols)} 피처)")


# ───────────────────────────────────────
# PyTorch Dataset
# ───────────────────────────────────────

class TradingDataset(Dataset):
    """
    슬라이딩 윈도우 기반 시계열 Dataset.

    Args:
        X      : (T, num_features) 정규화된 피처 배열
        y      : (T,) 타깃 수익률
        window : 과거 window 스텝을 하나의 샘플로 묶음

    __getitem__ 반환:
        x_seq  : (window, num_features) float tensor
        y_val  : scalar float tensor (다음 스텝 수익률)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window: int):
        self.X      = torch.tensor(X, dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.float32)
        self.window = window

    def __len__(self) -> int:
        return len(self.X) - self.window

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.window]          # (window, F)
        y_val = self.y[idx + self.window]                 # scalar
        return x_seq, y_val


# ─────────────────────────────────────────
# 단독 실행 (테스트용)
# ─────────────────────────────────────────
if __name__ == "__main__":
    # 더미 데이터로 파이프라인 테스트
    dates  = pd.date_range("2022-01-01", periods=500, freq="h", tz="UTC")
    dummy  = pd.DataFrame({
        "BTC_open":   np.random.uniform(20000, 50000, 500),
        "BTC_high":   np.random.uniform(20000, 50000, 500),
        "BTC_low":    np.random.uniform(20000, 50000, 500),
        "BTC_close":  np.random.uniform(20000, 50000, 500),
        "BTC_volume": np.random.uniform(1000, 5000, 500),
        "NASDAQ":     np.random.uniform(12000, 16000, 500),
        "GOLD":       np.random.uniform(1800, 2000, 500),
        "DXY":        np.random.uniform(95, 115, 500),
    }, index=dates)

    pp = Preprocessor()
    train_ds, val_ds, test_ds = pp.fit_transform(dummy)
    x, y = train_ds[0]
    print(f"샘플 형태: x={x.shape}, y={y.shape}")
