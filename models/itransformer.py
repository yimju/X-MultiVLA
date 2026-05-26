"""X-MultiVLA v15 — 2-Level iTransformer + ActionHead + PriceHead"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import D_MODEL, N_HEADS, N_LAYERS, DROPOUT, SEQ_LEN, N_ASSETS


class _FFN(nn.Module):
    def __init__(self, d, do):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(do), nn.Linear(d * 4, d))

    def forward(self, x):
        return self.net(x)


class _ITLayer(nn.Module):
    def __init__(self, d, h, do):
        super().__init__()
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=do, batch_first=True)
        self.ff   = _FFN(d, do)
        self.drop = nn.Dropout(do)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.n1(x + self.drop(a))
        return self.n2(x + self.drop(self.ff(x)))


class iTransformer(nn.Module):
    """2-Level iTransformer.

    Level 1 (feat_layers): F feature tokens per coin, attend across features.
    Level 2 (coin_layers): N coin tokens, attend across coins.

    Returns:
        v_emb     : (B, N*D) — ActionHead input
        coin_repr : (B, N, D) — PriceHead + regime_emb input
    """

    def __init__(self, seq_len, n_features, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.feat_proj   = nn.Linear(seq_len, d_model)
        self.feat_layers = nn.ModuleList(
            [_ITLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.feat_norm   = nn.LayerNorm(d_model)
        self.coin_layers = nn.ModuleList(
            [_ITLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.coin_norm   = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, N, F = x.shape
        x_feat      = x.permute(0, 2, 3, 1).reshape(B * N, F, T)
        feat_tokens = self.feat_proj(x_feat)
        for l in self.feat_layers:
            feat_tokens = l(feat_tokens)
        feat_tokens = self.feat_norm(feat_tokens)
        coin_repr   = feat_tokens.mean(dim=1).reshape(B, N, -1)
        for l in self.coin_layers:
            coin_repr = l(coin_repr)
        coin_repr = self.coin_norm(coin_repr)
        v_emb     = coin_repr.reshape(B, N * coin_repr.shape[-1])
        return v_emb, coin_repr


class ActionHead(nn.Module):
    """Portfolio weight head with regime signal and drawdown-aware cash bias.

    Inputs:
        h      : (B, V_DIM) — projected v_emb
        regime : (B, D_MODEL) — mean coin_repr
        w      : (B, N+1) — previous portfolio weights including cash

    No rule-based constraints — pure softmax output.
    """

    def __init__(self, in_dim, regime_dim, n_assets, dropout):
        super().__init__()
        self.n_out = n_assets + 1
        self.w_proj = nn.Linear(self.n_out, in_dim)
        self.r_proj = nn.Linear(regime_dim, in_dim)
        self.cash_bias_net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, self.n_out))

    def forward(self, h, regime, w):
        h2        = h + self.w_proj(w) + self.r_proj(regime)
        logits    = self.net(h2)
        cash_bias = self.cash_bias_net(h2)
        logits[:, -1] += cash_bias.squeeze(-1)
        return F.softmax(logits, dim=-1)


class PriceHead(nn.Module):
    """Per-coin horizon MIN/MAX return prediction (auxiliary loss)."""

    def __init__(self, d_model, n_assets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, coin_emb):
        return self.net(coin_emb)  # (B, N, 2)


def make_model(n_features, device='cpu'):
    V_DIM    = N_ASSETS * D_MODEL
    D_REGIME = D_MODEL

    class VLA(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc        = iTransformer(SEQ_LEN, n_features, D_MODEL,
                                           N_HEADS, N_LAYERS, DROPOUT)
            self.proj       = nn.Sequential(
                nn.Linear(V_DIM, V_DIM * 2), nn.GELU(),
                nn.Dropout(DROPOUT), nn.Linear(V_DIM * 2, V_DIM))
            self.head       = ActionHead(V_DIM, D_REGIME, N_ASSETS, DROPOUT)
            self.price_head = PriceHead(D_MODEL, N_ASSETS)

        def forward(self, x, w):
            v_emb, coin_repr = self.enc(x)
            regime_emb = coin_repr.mean(dim=1)
            w_star     = self.head(self.proj(v_emb), regime_emb, w)
            price_pred = self.price_head(coin_repr)
            return w_star, v_emb, price_pred

    return VLA().to(device)
