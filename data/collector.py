# data/collector.py
# ============================================================
# 바이낸스(BTC) + 나스닥/금/DXY(yfinance) + 금리(FRED) 수집
# 모두 날짜 인덱스로 정렬 후 하나의 DataFrame으로 병합
# ============================================================

import time
import ccxt
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from fredapi import Fred

from config import data_cfg, FRED_API_KEY


class MarketDataCollector:
    """
    멀티소스 시장 데이터 수집기
    
    출력 컬럼:
        BTC_open, BTC_high, BTC_low, BTC_close, BTC_volume   (바이낸스)
        NASDAQ, GOLD, DXY                                     (yfinance)
        FEDFUNDS, T10Y2Y, CPIAUCSL                            (FRED)
    """

    def __init__(self):
        self.exchange = ccxt.binance({"enableRateLimit": True})
        self.fred     = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
        self.cfg      = data_cfg

    # ───────────────────────────────────────
    # 1. 바이낸스 BTC OHLCV 수집
    # ───────────────────────────────────────
    def fetch_binance(
        self,
        symbol: str = None,
        timeframe: str = None,
        days: int = None,
    ) -> pd.DataFrame:
        symbol    = symbol    or self.cfg.symbol
        timeframe = timeframe or self.cfg.timeframe
        days      = days      or self.cfg.history_days

        since = self.exchange.parse8601(
            (datetime.utcnow() - timedelta(days=days)).isoformat()
        )

        all_ohlcv = []
        print(f"[collector] Binance {symbol} {timeframe} 수집 시작...")
        while True:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(0.5)         # rate limit 방지
            if len(ohlcv) < 1000:
                break

        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "BTC_open", "BTC_high", "BTC_low", "BTC_close", "BTC_volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        print(f"[collector] Binance 완료: {len(df)} rows")
        return df

    # ───────────────────────────────────────
    # 2. Yahoo Finance (나스닥·금·DXY)
    # ───────────────────────────────────────
    def fetch_yahoo(self, days: int = None) -> pd.DataFrame:
        days  = days or self.cfg.history_days
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        end   = datetime.utcnow().strftime("%Y-%m-%d")

        symbols = {
            self.cfg.nasdaq_symbol: "NASDAQ",
            self.cfg.gold_symbol:   "GOLD",
            self.cfg.dxy_symbol:    "DXY",
        }

        print(f"[collector] Yahoo Finance 수집 시작: {list(symbols.keys())}")
        raw = yf.download(list(symbols.keys()), start=start, end=end, progress=False)["Close"]
        raw = raw.rename(columns=symbols)
        raw.index = pd.to_datetime(raw.index, utc=True)
        print(f"[collector] Yahoo Finance 완료: {len(raw)} rows")
        return raw

    # ───────────────────────────────────────
    # 3. FRED 거시경제 지표
    # ───────────────────────────────────────
    def fetch_fred(self, days: int = None) -> pd.DataFrame:
        if not self.fred:
            print("[collector] FRED API 키가 없어 거시경제 데이터를 건너뜁니다.")
            return pd.DataFrame()

        days  = days or self.cfg.history_days
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        dfs = []
        print(f"[collector] FRED 수집 시작: {self.cfg.fred_series}")
        for series_id in self.cfg.fred_series:
            try:
                s = self.fred.get_series(series_id, observation_start=start)
                s.name = series_id
                dfs.append(s)
            except Exception as e:
                print(f"[collector] FRED {series_id} 실패: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index, utc=True)
        print(f"[collector] FRED 완료: {len(df)} rows")
        return df

    # ───────────────────────────────────────
    # 4. 전체 병합 (메인 메서드)
    # ───────────────────────────────────────
    def collect_all(self, timeframe: str = "1d", days: int = None) -> pd.DataFrame:
        """
        모든 소스를 일봉(1d) 또는 시간봉(1h) 기준으로 병합.
        
        - 코인: 365일 24시간 거래
        - 주식/금리: 평일만 존재 → forward-fill 처리
        """
        days = days or self.cfg.history_days

        btc   = self.fetch_binance(timeframe=timeframe, days=days)
        yahoo = self.fetch_yahoo(days=days)
        fred  = self.fetch_fred(days=days)

        # 일봉 리샘플링 (시간봉일 경우 야후·FRED는 일봉이므로 reindex 필요)
        if timeframe == "1h":
            # 야후/FRED를 시간 단위로 확장 후 forward-fill
            yahoo = yahoo.reindex(btc.index, method="ffill")
            if not fred.empty:
                fred = fred.reindex(btc.index, method="ffill")

        # 병합
        dfs = [btc, yahoo]
        if not fred.empty:
            dfs.append(fred)

        merged = pd.concat(dfs, axis=1)
        merged = merged.ffill().dropna()

        print(f"[collector] 최종 병합 완료: shape={merged.shape}, range={merged.index[0]} ~ {merged.index[-1]}")
        return merged

    # ───────────────────────────────────────
    # 5. CSV 저장
    # ───────────────────────────────────────
    def save(self, df: pd.DataFrame, path: str = None) -> str:
        import os
        path = path or f"{self.cfg.raw_data_dir}/market_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
        print(f"[collector] 저장 완료: {path}")
        return path


# ─────────────────────────────────────────
# 단독 실행 (테스트용)
# ─────────────────────────────────────────
if __name__ == "__main__":
    collector = MarketDataCollector()
    df = collector.collect_all(timeframe="1d", days=730)
    collector.save(df)
    print(df.tail())
