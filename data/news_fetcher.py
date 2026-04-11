# data/news_fetcher.py
# ============================================================
# CryptoPanic API로 뉴스 수집 → FinBERT로 감성 벡터 변환
# ============================================================

import time
import requests
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import CRYPTOPANIC_API_KEY, model_cfg


class NewsFetcher:
    """
    CryptoPanic에서 암호화폐 뉴스를 가져와
    FinBERT 감성 점수(긍정/부정/중립 확률)로 변환합니다.
    
    출력 컬럼:
        news_title, news_published_at,
        news_pos, news_neg, news_neu   (FinBERT 감성 확률)
        news_sentiment_score           (pos - neg 단일 점수)
    """

    CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self):
        self.api_key = CRYPTOPANIC_API_KEY
        self._load_finbert()

    # ───────────────────────────────────────
    # FinBERT 모델 로드 (처음 1회만)
    # ───────────────────────────────────────
    def _load_finbert(self):
        print("[news] FinBERT 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.finbert_model)
        self.finbert   = AutoModelForSequenceClassification.from_pretrained(
            model_cfg.finbert_model
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finbert.to(self.device).eval()
        print(f"[news] FinBERT 로드 완료 (device={self.device})")

    # ───────────────────────────────────────
    # 1. CryptoPanic API 호출
    # ───────────────────────────────────────
    def fetch_raw(self, pages: int = 10, currencies: str = "BTC") -> pd.DataFrame:
        """
        pages: 가져올 페이지 수 (1페이지 = 최신 20개 뉴스)
        """
        if not self.api_key:
            print("[news] CRYPTOPANIC_API_KEY 없음 → 더미 데이터 반환")
            return self._dummy_news()

        all_news = []
        next_url = self.CRYPTOPANIC_URL

        params = {
            "auth_token": self.api_key,
            "currencies": currencies,
            "public": "true",
            "filter": "important",
        }

        print(f"[news] CryptoPanic 뉴스 수집 ({pages}페이지)...")
        for page_idx in range(pages):
            try:
                resp = requests.get(next_url, params=params if page_idx == 0 else {}, timeout=10)
                if resp.status_code != 200:
                    print(f"[news] API 오류: {resp.status_code}")
                    break

                data     = resp.json()
                results  = data.get("results", [])
                next_url = data.get("next")

                for item in results:
                    all_news.append({
                        "news_title":        item.get("title", ""),
                        "news_published_at": item.get("published_at", ""),
                    })

                if not next_url:
                    break

                time.sleep(0.5)     # rate limit

            except Exception as e:
                print(f"[news] 페이지 {page_idx} 오류: {e}")
                break

        df = pd.DataFrame(all_news)
        if df.empty:
            return self._dummy_news()

        df["news_published_at"] = pd.to_datetime(df["news_published_at"], utc=True)
        df = df.drop_duplicates("news_title").sort_values("news_published_at")
        print(f"[news] 수집 완료: {len(df)}개")
        return df

    # ───────────────────────────────────────
    # 2. FinBERT 감성 분석 (배치 처리)
    # ───────────────────────────────────────
    @torch.no_grad()
    def analyze_sentiment(self, df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """
        news_title 컬럼을 FinBERT에 통과시켜 감성 확률을 추가합니다.
        레이블 순서: [positive, negative, neutral]  (FinBERT 기준)
        """
        titles   = df["news_title"].fillna("").tolist()
        all_pos, all_neg, all_neu = [], [], []

        print(f"[news] FinBERT 감성 분석 중... ({len(titles)}개)")
        for i in range(0, len(titles), batch_size):
            batch = titles[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            ).to(self.device)

            logits = self.finbert(**inputs).logits       # (B, 3)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

            all_pos.extend(probs[:, 0].tolist())
            all_neg.extend(probs[:, 1].tolist())
            all_neu.extend(probs[:, 2].tolist())

        df = df.copy()
        df["news_pos"]             = all_pos
        df["news_neg"]             = all_neg
        df["news_neu"]             = all_neu
        df["news_sentiment_score"] = df["news_pos"] - df["news_neg"]  # -1 ~ +1

        print("[news] 감성 분석 완료")
        return df

    # ───────────────────────────────────────
    # 3. 차트 데이터와 시간 정렬 (merge_asof)
    # ───────────────────────────────────────
    def align_with_chart(
        self,
        chart_df: pd.DataFrame,
        news_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        차트의 각 타임스탬프에 직전에 발생한 뉴스 감성을 붙입니다.
        뉴스가 없는 시간대는 0으로 채웁니다 (Zero Vector).
        """
        chart = chart_df.reset_index().rename(columns={"timestamp": "ts"})
        news  = news_df[["news_published_at", "news_pos", "news_neg", "news_neu", "news_sentiment_score"]].copy()
        news  = news.rename(columns={"news_published_at": "ts"})

        chart["ts"] = pd.to_datetime(chart["ts"], utc=True)
        news["ts"]  = pd.to_datetime(news["ts"],  utc=True)

        merged = pd.merge_asof(
            chart.sort_values("ts"),
            news.sort_values("ts"),
            on="ts",
            direction="backward",   # 뉴스 발생 직후 차트에 매핑
        )

        # 뉴스 없는 구간 0으로 채움 (ZeroVector → 모델이 뉴스 없음으로 인식)
        for col in ["news_pos", "news_neg", "news_neu", "news_sentiment_score"]:
            merged[col] = merged[col].fillna(0.0)

        merged = merged.set_index("ts")
        print(f"[news] 차트-뉴스 정렬 완료: shape={merged.shape}")
        return merged

    # ───────────────────────────────────────
    # 더미 데이터 (API 키 없을 때)
    # ───────────────────────────────────────
    def _dummy_news(self) -> pd.DataFrame:
        import numpy as np
        dates = pd.date_range("2023-01-01", periods=100, freq="6h", tz="UTC")
        return pd.DataFrame({
            "news_title":        [f"Dummy news {i}" for i in range(100)],
            "news_published_at": dates,
        })


# ─────────────────────────────────────────
# 단독 실행 (테스트용)
# ─────────────────────────────────────────
if __name__ == "__main__":
    fetcher  = NewsFetcher()
    raw_news = fetcher.fetch_raw(pages=5)
    news_df  = fetcher.analyze_sentiment(raw_news)
    print(news_df[["news_title", "news_sentiment_score"]].head(10))
