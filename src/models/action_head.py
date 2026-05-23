import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    """
    Action component: market embedding (v_emb) -> portfolio weights.
    3-layer MLP + Softmax. Supports NAV constraints (min cash, max single).
    """
    def __init__(self, in_dim: int, n_assets: int = 5, dropout: float = 0.1):
        super().__init__()
        self.n_assets = n_assets
        self.decoder  = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, n_assets + 1),  # +1 for cash
        )

    def forward(self, x: torch.Tensor, nav: dict = None) -> torch.Tensor:
        """
        x:   (B, in_dim)
        nav: optional constraints dict with keys min_cash, max_single
        returns: (B, n_assets+1) portfolio weights summing to 1
        """
        weights = F.softmax(self.decoder(x), dim=-1)
        if nav:
            weights = self._apply_nav(weights, nav)
        return weights

    def _apply_nav(self, w, nav):
        w = w.clone()
        min_cash = nav.get("min_cash", 0.0)
        if min_cash > 0:
            deficit = (min_cash - w[:, -1]).clamp(min=0)
            w[:, -1] += deficit
            s = w[:, :self.n_assets].sum(-1, keepdim=True).clamp(min=1e-8)
            w[:, :self.n_assets] -= deficit.unsqueeze(-1) * w[:, :self.n_assets] / s
        max_s = nav.get("max_single", 1.0)
        if max_s < 1.0:
            w = w.clamp(max=max_s)
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-8)
        return w
