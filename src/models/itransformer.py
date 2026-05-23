import torch
import torch.nn as nn
import torch.nn.functional as F


class iTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class iTransformer(nn.Module):
    """
    iTransformer — Vision component (ICLR 2024)
    Cross-coin attention: treats each coin as a token, attends across coins (not across time).

    Outputs:
      heading_mu:  (B, N)       predicted returns per coin
      heading_sig: (B, N)       uncertainty (higher = less confident)
      v_emb:       (B, d_model) market embedding (Language input)
    """
    def __init__(self, n_coins=5, n_features=94, seq_len=60,
                 d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.n_coins = n_coins
        self.d_model = d_model

        self.token_embed = nn.Linear(seq_len * n_features, d_model)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_coins, d_model) * 0.02)
        self.drop        = nn.Dropout(dropout)
        self.layers      = nn.ModuleList([
            iTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm        = nn.LayerNorm(d_model)

        self.heading_mu  = nn.Linear(d_model, 1)
        self.heading_sig = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.v_proj      = nn.Linear(d_model * n_coins, d_model)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, N, F)  — batch, time, coins, features
        """
        B, T, N, F = x.shape
        tokens = x.permute(0, 2, 1, 3).reshape(B, N, T * F)
        tokens = self.drop(self.token_embed(tokens) + self.pos_embed)

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)  # (B, N, d)

        heading_mu  = self.heading_mu(tokens).squeeze(-1)   # (B, N)
        heading_sig = self.heading_sig(tokens).squeeze(-1)  # (B, N)
        v_emb = self.v_proj(tokens.reshape(B, N * self.d_model))  # (B, d)
        return heading_mu, heading_sig, v_emb
