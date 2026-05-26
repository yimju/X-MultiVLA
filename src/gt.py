"""Hindsight Sortino GT  v15 — multi-horizon + drawdown-boosted cash signal."""
import numpy as np
from config import (SHARPE_HORIZON, LONG_HORIZON, GT_BLEND,
                    SMOOTH_ALPHA, DD_THRESHOLD, DD_MAX_BOOST, N_ASSETS)


def _sortino_raw(prices, horizon, n_assets=N_ASSETS):
    T      = len(prices) - 1
    labels = np.zeros((T, n_assets + 1), dtype=np.float32)
    for t in range(T):
        end  = min(t + horizon, T)
        p    = prices[t:end + 1]
        rets = (p[1:] - p[:-1]) / (p[:-1] + 1e-9)
        mu           = rets.mean(axis=0)
        downside     = np.minimum(rets, 0.0)
        downside_dev = np.sqrt(np.mean(downside ** 2, axis=0)) + 1e-8
        sortino      = mu / downside_dev
        market_trend = rets.mean()

        if market_trend > 0:
            cash_score = 0.0
        else:
            base_cash = (abs(market_trend) / (rets.std() + 1e-8)) * 10.0
            dd = np.maximum(0, (p[0] - p.min(axis=0)) / (p[0] + 1e-9)).mean()
            dd_boost   = float(np.clip(dd / DD_THRESHOLD, 1.0, DD_MAX_BOOST))
            cash_score = base_cash * dd_boost

        sv  = np.append(sortino, cash_score)
        pos = np.maximum(sv, 0.0)
        if pos.sum() < 1e-8:
            labels[t, -1] = 1.0
        else:
            labels[t] = pos / pos.sum()
    return labels


def generate_sortino_labels(prices, horizon=SHARPE_HORIZON, n_assets=N_ASSETS,
                            long_horizon=LONG_HORIZON, blend=GT_BLEND,
                            smooth_alpha=SMOOTH_ALPHA):
    raw_short = _sortino_raw(prices, horizon, n_assets)
    raw_long  = _sortino_raw(prices, long_horizon, n_assets)

    blended = blend * raw_short + (1.0 - blend) * raw_long
    s = blended.sum(axis=1, keepdims=True)
    blended = np.where(s < 1e-8,
                       np.eye(n_assets + 1)[[-1]].repeat(len(blended), axis=0),
                       blended / s)

    labels = blended.copy()
    for t in range(1, len(labels)):
        labels[t] = (1.0 - smooth_alpha) * labels[t - 1] + smooth_alpha * blended[t]
        s = labels[t].sum()
        labels[t] = labels[t] / s if s > 1e-8 else np.eye(n_assets + 1)[-1]

    return labels.astype(np.float32)
