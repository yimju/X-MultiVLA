# models/xai_explainer.py
# ============================================================
# XAI(설명 가능한 AI) 모듈
#   - Cross-Attention 가중치 시각화
#   - 피처 중요도 (간소화 SHAP 대용: permutation importance)
#   - 자연어 설명 생성 (규칙 기반)
# ============================================================

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, List


class VLAExplainer:
    """
    사용법:
        explainer = VLAExplainer(pretrain_model, feature_cols)
        explanation = explainer.explain(chart_seq, news_vec, action)
        explainer.plot_attention(chart_seq, news_vec)
    """

    ACTION_NAMES = {0: "Hold (관망)", 1: "Long (매수)", 2: "Short (매도)"}
    ACTION_EMOJIS = {0: "⏸", 1: "📈", 2: "📉"}

    def __init__(self, model, feature_cols: List[str]):
        self.model        = model
        self.feature_cols = feature_cols
        self._attn_cache: Optional[torch.Tensor] = None   # 마지막 attention weight

    # ───────────────────────────────────────
    # 1. 자연어 설명 생성 (규칙 기반 CoT)
    # ───────────────────────────────────────
    def explain(
        self,
        chart_seq: torch.Tensor,   # (1, window, F)
        news_score: float,         # news_sentiment_score (-1 ~ +1)
        action: int,               # 0/1/2
        confidence: float = None,  # PPO 정책 확률
    ) -> str:
        """
        모델 결정에 대한 자연어 설명을 생성합니다.
        """
        x = chart_seq[0, -1, :]   # 마지막 타임스텝 피처

        # 주요 피처 읽기 (인덱스는 feature_cols 순서에 의존)
        def _get(name: str, default: float = 0.0) -> float:
            if name in self.feature_cols:
                idx = self.feature_cols.index(name)
                return float(x[idx]) if idx < len(x) else default
            return default

        rsi    = _get("RSI_14", 50)
        macd_h = _get("MACD_hist", 0)
        btc_ret = _get("BTC_close_ret", 0)
        nas_ret = _get("NASDAQ_ret", 0)

        lines = [f"🤖 X-MultiVLA 결정: {self.ACTION_EMOJIS[action]} {self.ACTION_NAMES[action]}"]
        if confidence:
            lines.append(f"   확신도: {confidence:.1%}")
        lines.append("")
        lines.append("📊 [V] 차트 분석:")

        # RSI 해석
        if rsi > 70:
            lines.append(f"  · RSI {rsi:.1f} → 과매수 구간 (하락 가능성)")
        elif rsi < 30:
            lines.append(f"  · RSI {rsi:.1f} → 과매도 구간 (반등 가능성)")
        else:
            lines.append(f"  · RSI {rsi:.1f} → 중립 구간")

        # MACD 해석
        if macd_h > 0:
            lines.append(f"  · MACD Histogram 양수 → 상승 모멘텀")
        else:
            lines.append(f"  · MACD Histogram 음수 → 하락 모멘텀")

        # BTC 수익률
        lines.append(f"  · 직전 BTC 수익률: {btc_ret:+.2%}")

        lines.append("")
        lines.append("📰 [L] 뉴스·거시 분석:")

        # 나스닥
        if nas_ret > 0.005:
            lines.append(f"  · 나스닥 상승({nas_ret:+.2%}) → 위험 선호 심리 ↑")
        elif nas_ret < -0.005:
            lines.append(f"  · 나스닥 하락({nas_ret:+.2%}) → 위험 회피 심리 ↑")
        else:
            lines.append(f"  · 나스닥 보합 → 시장 관망세")

        # 뉴스 감성
        if news_score > 0.2:
            lines.append(f"  · 뉴스 감성: 긍정({news_score:+.2f}) → 시장 낙관")
        elif news_score < -0.2:
            lines.append(f"  · 뉴스 감성: 부정({news_score:+.2f}) → 시장 우려")
        else:
            lines.append(f"  · 뉴스 감성: 중립({news_score:+.2f})")

        lines.append("")
        lines.append(f"🎯 [A] 결론: {self.ACTION_NAMES[action]}")

        return "\n".join(lines)

    # ───────────────────────────────────────
    # 2. Attention 가중치 시각화
    # ───────────────────────────────────────
    @torch.no_grad()
    def plot_attention(
        self,
        chart_seq: torch.Tensor,   # (1, window, F)
        news_vec: torch.Tensor,    # (1, 4)
        news_labels: List[str]     = None,
        save_path: str             = None,
    ):
        """
        Cross-Attention 가중치를 히트맵으로 시각화합니다.
        """
        self.model.eval()
        chart_vec = self.model.chart_encoder(chart_seq)
        news_p    = self.model.news_projector(news_vec)
        _, attn_w = self.model.fusion.cross_attn(chart_vec, news_p)

        attn = attn_w.squeeze().cpu().numpy()   # scalar or (H,)

        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        # Chart → News attention weight
        ax.bar(
            ["뉴스 감성 (News → Chart Attention)"],
            [float(attn.mean()) if attn.ndim > 0 else float(attn)],
            color="#00d4ff", width=0.4,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Attention Weight", color="white")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.set_title("Cross-Attention: 차트 ↔ 뉴스 영향도", color="white", fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.show()

    # ───────────────────────────────────────
    # 3. 피처 중요도 (Permutation Importance)
    # ───────────────────────────────────────
    @torch.no_grad()
    def feature_importance(
        self,
        chart_seq_batch: torch.Tensor,    # (N, window, F)
        news_vec_batch: torch.Tensor,     # (N, 4)
        n_repeat: int = 5,
        top_k: int    = 10,
    ) -> pd.DataFrame:
        """
        각 피처를 랜덤으로 섞었을 때 예측 변화량으로 중요도 추정.
        """
        self.model.eval()
        base_pred = self.model(chart_seq_batch, news_vec_batch).squeeze()

        importances = []
        F = chart_seq_batch.shape[-1]

        for f_idx in range(F):
            scores = []
            for _ in range(n_repeat):
                perturbed          = chart_seq_batch.clone()
                perm               = torch.randperm(perturbed.size(0))
                perturbed[:, :, f_idx] = perturbed[perm, :, f_idx]
                pert_pred          = self.model(perturbed, news_vec_batch).squeeze()
                diff               = (base_pred - pert_pred).abs().mean().item()
                scores.append(diff)

            fname = self.feature_cols[f_idx] if f_idx < len(self.feature_cols) else f"feat_{f_idx}"
            importances.append({"feature": fname, "importance": np.mean(scores)})

        df_imp = pd.DataFrame(importances).sort_values("importance", ascending=False)

        # Top-K 시각화
        top_df  = df_imp.head(top_k)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        colors  = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_df)))
        ax.barh(top_df["feature"], top_df["importance"], color=colors)
        ax.set_xlabel("Permutation Importance (예측 변화량)", color="white")
        ax.set_title(f"Top {top_k} 피처 중요도", color="white")
        ax.tick_params(colors="white")
        ax.invert_yaxis()
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        plt.tight_layout()
        plt.show()

        return df_imp


# ─────────────────────────────────────────
# 단독 실행 (설명 텍스트 테스트)
# ─────────────────────────────────────────
if __name__ == "__main__":
    feature_cols = ["BTC_close_ret", "NASDAQ_ret", "RSI_14", "MACD_hist"]

    class DummyModel:
        chart_encoder = None
        news_projector = None
        fusion = None

    explainer = VLAExplainer(DummyModel(), feature_cols)

    dummy_chart = torch.zeros(1, 60, 4)
    dummy_chart[0, -1, 2] = 65.0   # RSI > 70 흉내
    dummy_chart[0, -1, 3] = 0.5    # MACD 양수

    text = explainer.explain(dummy_chart, news_score=0.35, action=1)
    print(text)
