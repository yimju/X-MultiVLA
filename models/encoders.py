# models/encoders.py
# ============================================================
# V 모듈: 시계열 데이터 → 잠재 벡터
#   - PatchTST : 시계열을 패치 단위로 토큰화하는 Transformer 인코더
#
# L 모듈 (프로젝션): 뉴스 감성 벡터(FinBERT 출력) → 같은 차원으로 압축
# ============================================================

import math
import torch
import torch.nn as nn
from config import model_cfg


# ───────────────────────────────────────
# 위치 인코딩 (Sinusoidal)
# ───────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


# ───────────────────────────────────────
# PatchTST 시계열 인코더
# ───────────────────────────────────────

class PatchTSTEncoder(nn.Module):
    """
    시계열 (B, window, num_features) 를 입력받아
    패치 단위로 토큰화한 뒤 Transformer Encoder로 처리.

    출력: (B, d_model) 전체 시퀀스를 대표하는 벡터 (CLS-토큰 방식 대신 평균 풀링)

    Args:
        num_features : 입력 피처 수 (채널 수)
        window       : 입력 시퀀스 길이
        patch_size   : 각 패치의 타임스텝 수
        d_model      : Transformer 임베딩 차원
    """

    def __init__(
        self,
        num_features: int,
        window: int       = None,
        patch_size: int   = None,
        d_model: int      = None,
        nhead: int        = None,
        num_layers: int   = None,
        dim_ff: int       = None,
        dropout: float    = None,
    ):
        super().__init__()
        cfg        = model_cfg
        window     = window     or cfg.num_patches * cfg.patch_size  # 기본 window
        patch_size = patch_size or cfg.patch_size
        d_model    = d_model    or cfg.d_model
        nhead      = nhead      or cfg.nhead
        num_layers = num_layers or cfg.num_encoder_layers
        dim_ff     = dim_ff     or cfg.dim_feedforward
        dropout    = dropout    or cfg.dropout

        self.patch_size  = patch_size
        self.num_patches = window // patch_size          # 패치 개수
        self.d_model     = d_model

        # 패치 선형 임베딩 (patch_size * num_features → d_model)
        self.patch_embed = nn.Linear(patch_size * num_features, d_model)

        # 위치 인코딩
        self.pos_enc = PositionalEncoding(d_model, max_len=self.num_patches + 1, dropout=dropout)

        # Transformer Encoder
        enc_layer   = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN (학습 안정성)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, window, num_features)
        returns : (B, d_model)
        """
        B, T, F = x.shape

        # ── 패치 분할 ──────────────────────────
        # 마지막 T가 patch_size의 배수가 아닌 경우 잘라냄
        n_patch = T // self.patch_size
        x = x[:, : n_patch * self.patch_size, :]             # (B, n*p, F)
        x = x.reshape(B, n_patch, self.patch_size * F)       # (B, n, p*F)

        # ── 임베딩 ─────────────────────────────
        x = self.patch_embed(x)                              # (B, n, d_model)

        # ── 위치 인코딩 ────────────────────────
        x = self.pos_enc(x)

        # ── Transformer ────────────────────────
        x = self.transformer(x)                              # (B, n, d_model)
        x = self.norm(x)

        # ── 평균 풀링 → 고정 크기 벡터 ──────────
        out = x.mean(dim=1)                                  # (B, d_model)
        return out


# ───────────────────────────────────────
# 뉴스 감성 프로젝션 (FinBERT 출력 압축)
# ───────────────────────────────────────

class NewsProjector(nn.Module):
    """
    news_fetcher.py에서 추출한 감성 피처 벡터를 d_model 차원으로 압축.

    입력:
        news_vec : (B, news_input_dim)  예) [pos, neg, neu, score, ...] 등 감성 관련 컬럼
    출력:
        (B, proj_dim)

    news_input_dim : 감성 피처 수 (기본 4 : pos·neg·neu·score)
    """

    def __init__(self, news_input_dim: int = 4, proj_dim: int = None, dropout: float = None):
        super().__init__()
        proj_dim = proj_dim or model_cfg.news_proj_dim
        dropout  = dropout  or model_cfg.dropout

        self.net = nn.Sequential(
            nn.Linear(news_input_dim, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, news_vec: torch.Tensor) -> torch.Tensor:
        # news_vec : (B, news_input_dim)
        return self.net(news_vec)   # (B, proj_dim)


# ─────────────────────────────────────────
# 단독 실행 (shape 확인용)
# ─────────────────────────────────────────
if __name__ == "__main__":
    B, T, F = 8, 60, 20   # 배치·윈도우·피처 수

    enc  = PatchTSTEncoder(num_features=F, window=T)
    x    = torch.randn(B, T, F)
    out  = enc(x)
    print(f"PatchTSTEncoder: {x.shape} → {out.shape}")   # (8, 128)

    proj     = NewsProjector(news_input_dim=4)
    news_vec = torch.randn(B, 4)
    out_n    = proj(news_vec)
    print(f"NewsProjector:   {news_vec.shape} → {out_n.shape}")   # (8, 128)
