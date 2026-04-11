# models/fusion.py
# ============================================================
# 차트 벡터 + 뉴스 벡터 → Cross-Attention 융합 → 통합 표현
# 추가로 Chain-of-Thought 스타일의 "설명 점수" 산출
# ============================================================

import torch
import torch.nn as nn
from config import model_cfg


# ───────────────────────────────────────
# Cross-Attention Fusion Block
# ───────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    차트 벡터가 Query, 뉴스 벡터가 Key/Value가 되는 Cross-Attention.

    "금리 인상 뉴스(K)"가 "가격 하락 추세(Q)"의 가중치를 얼마나 높이는지 학습합니다.

    Args:
        chart_dim  : PatchTST 출력 차원 (d_model)
        news_dim   : NewsProjector 출력 차원 (proj_dim)
        fusion_dim : Cross-Attention 내부 차원
        nhead      : Attention head 수
    """

    def __init__(
        self,
        chart_dim: int  = None,
        news_dim: int   = None,
        fusion_dim: int = None,
        nhead: int      = None,
        dropout: float  = None,
    ):
        super().__init__()
        cfg        = model_cfg
        chart_dim  = chart_dim  or cfg.d_model
        news_dim   = news_dim   or cfg.news_proj_dim
        fusion_dim = fusion_dim or cfg.fusion_dim
        nhead      = nhead      or cfg.fusion_heads
        dropout    = dropout    or cfg.dropout

        # 차원 맞추기 (chart·news → fusion_dim)
        self.q_proj = nn.Linear(chart_dim,  fusion_dim)
        self.k_proj = nn.Linear(news_dim,   fusion_dim)
        self.v_proj = nn.Linear(news_dim,   fusion_dim)

        # Multi-Head Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # 잔차 연결 + 정규화
        self.norm1    = nn.LayerNorm(fusion_dim)
        self.norm2    = nn.LayerNorm(fusion_dim)
        self.ff       = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.dropout  = nn.Dropout(dropout)

        # 최종 출력 차원 조정 (chart_dim + fusion_dim → hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(chart_dim + fusion_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )

    def forward(
        self,
        chart_vec: torch.Tensor,   # (B, chart_dim)
        news_vec: torch.Tensor,    # (B, news_dim)
    ):
        """
        Returns:
            fused     : (B, hidden_dim)  융합된 표현
            attn_w    : (B, 1, 1)        Attention 가중치 (XAI 시각화용)
        """
        # (B, D) → (B, 1, D) : MultiheadAttention은 시퀀스 입력
        Q = self.q_proj(chart_vec).unsqueeze(1)   # (B, 1, fusion_dim)
        K = self.k_proj(news_vec).unsqueeze(1)    # (B, 1, fusion_dim)
        V = self.v_proj(news_vec).unsqueeze(1)    # (B, 1, fusion_dim)

        # Cross-Attention
        attn_out, attn_w = self.attn(Q, K, V)    # (B, 1, fusion_dim), (B, 1, 1)

        # Pre-LN Residual #1
        attn_out = self.norm1(Q + self.dropout(attn_out))

        # FFN + Residual #2
        ff_out   = self.norm2(attn_out + self.dropout(self.ff(attn_out)))  # (B, 1, fusion_dim)
        ff_out   = ff_out.squeeze(1)                                        # (B, fusion_dim)

        # 차트 벡터와 Concat → 최종 표현
        fused = torch.cat([chart_vec, ff_out], dim=-1)   # (B, chart_dim + fusion_dim)
        fused = self.out_proj(fused)                      # (B, hidden_dim)

        return fused, attn_w


# ───────────────────────────────────────
# 뉴스 부재 시 Zero Vector 처리
# ───────────────────────────────────────

class MultiModalFusion(nn.Module):
    """
    뉴스가 없을 때(zero vector)도 안정적으로 동작하는 멀티모달 융합 레이어.
    chart_vec만 있을 때는 단순 MLP로 폴백합니다.
    """

    def __init__(self, chart_dim: int = None, news_dim: int = None, **kwargs):
        super().__init__()
        cfg       = model_cfg
        chart_dim = chart_dim or cfg.d_model
        news_dim  = news_dim  or cfg.news_proj_dim

        self.cross_attn = CrossAttentionFusion(
            chart_dim=chart_dim, news_dim=news_dim, **kwargs
        )

        # chart only fallback
        self.chart_only = nn.Sequential(
            nn.Linear(chart_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )

    def forward(
        self,
        chart_vec: torch.Tensor,
        news_vec: torch.Tensor | None = None,
    ):
        """
        news_vec가 None이거나 zero tensor면 chart_only 경로 사용.
        """
        if news_vec is None:
            return self.chart_only(chart_vec), None

        # 뉴스가 Zero Vector인지 확인 (뉴스 없는 타임스텝)
        is_zero = (news_vec.abs().sum(dim=-1, keepdim=True) < 1e-6)   # (B, 1)

        fused, attn_w = self.cross_attn(chart_vec, news_vec)

        # Zero vector 타임스텝은 chart_only 출력으로 대체
        fallback = self.chart_only(chart_vec)
        fused    = torch.where(is_zero, fallback, fused)

        return fused, attn_w


# ─────────────────────────────────────────
# 단독 실행 (shape 확인용)
# ─────────────────────────────────────────
if __name__ == "__main__":
    B = 4
    chart = torch.randn(B, model_cfg.d_model)         # (4, 128)
    news  = torch.randn(B, model_cfg.news_proj_dim)   # (4, 128)

    fusion = MultiModalFusion()
    out, w = fusion(chart, news)
    print(f"Fused: {out.shape}")      # (4, 128)
    print(f"Attn weights: {w.shape if w is not None else None}")   # (4, 1, 1)

    out_only, _ = fusion(chart, None)
    print(f"Chart only: {out_only.shape}")
