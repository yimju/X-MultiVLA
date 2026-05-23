import numpy as np
import torch
import torch.nn.functional as F


class OracleLabelGenerator:
    """
    Phase 1 supervised label generator using backward Dynamic Programming.

    Reward per step:
      R[s,a] = effective_port_ret - ew_ret - switching_cost
      effective_port_ret = raw_port_ret - vol_penalty * intraday_mdd

    Focal weighting: top-focal_percentile% high-volatility steps get focal_weight x gradient.
    """

    def __init__(self, commission: float = 0.001, n_assets: int = 5,
                 gamma: float = 0.99, label_type: str = "soft",
                 temperature: float = 0.001,
                 volatility_penalty: float = 0.3,
                 focal_weight: float = 10.0,
                 focal_percentile: float = 95.0):
        self.comm             = commission
        self.n_assets         = n_assets
        self.gamma            = gamma
        self.label_type       = label_type
        self.temperature      = temperature
        self.vol_penalty      = volatility_penalty
        self.focal_weight     = focal_weight
        self.focal_percentile = focal_percentile

        self.n_actions       = n_assets + 1  # coins + cash
        self._action_weights = np.array(
            [self._one_hot(i) for i in range(n_assets)] + [self._cash()],
            dtype=np.float32)

    def _cash(self):
        w = np.zeros(self.n_assets + 1, dtype=np.float32)
        w[-1] = 1.0
        return w

    def _one_hot(self, idx):
        w = np.zeros(self.n_assets + 1, dtype=np.float32)
        w[idx] = 1.0
        return w

    def _reward_matrix(self, prices, t, high_prices=None, low_prices=None):
        rets = (prices[t + 1] - prices[t]) / (prices[t] + 1e-9)
        ew_ret = rets.mean()

        if high_prices is not None and low_prices is not None:
            intraday_mdd = (high_prices[t] - low_prices[t]) / (prices[t] + 1e-9)
        else:
            intraday_mdd = np.abs(rets)

        R = np.zeros((self.n_actions, self.n_actions), dtype=np.float32)
        for s in range(self.n_actions):
            w_prev = self._action_weights[s]
            for a in range(self.n_actions):
                w_new   = self._action_weights[a]
                cost    = self.comm * np.abs(w_new - w_prev).sum()
                raw_ret = float(np.dot(w_new[:self.n_assets], rets))

                if a < self.n_assets:
                    effective_ret = raw_ret - self.vol_penalty * float(intraday_mdd[a])
                else:
                    effective_ret = raw_ret

                R[s, a] = effective_ret - ew_ret - cost
        return R

    def compute_focal_weights(self, high_prices, low_prices, prices):
        T = len(prices) - 1
        if high_prices is None or low_prices is None:
            return np.ones(T, dtype=np.float32)

        mdd_mean  = np.mean((high_prices[:T] - low_prices[:T]) / (prices[:T] + 1e-9), axis=1)
        threshold = np.percentile(mdd_mean, self.focal_percentile)
        return np.where(mdd_mean >= threshold, self.focal_weight, 1.0).astype(np.float32)

    def generate(self, prices, high_prices=None, low_prices=None,
                 init_action=None, verbose=True):
        """
        Backward DP over entire price sequence.

        Args:
            prices: (T+1, n_assets) closing prices
            high_prices, low_prices: (T+1, n_assets) optional OHLC

        Returns:
            labels:        (T, n_assets+1) soft portfolio weight targets
            focal_weights: (T,) per-step loss weights
        """
        T  = len(prices) - 1
        A  = self.n_actions
        s0 = init_action if init_action is not None else self.n_assets  # start in cash

        if verbose:
            print(f"  Oracle DP: computing reward matrices... T={T}")

        R_all = np.zeros((T, A, A), dtype=np.float32)
        for t in range(T):
            R_all[t] = self._reward_matrix(prices, t, high_prices, low_prices)

        # Backward DP
        V      = np.zeros(A, dtype=np.float32)
        policy = np.zeros((T, A), dtype=np.int32)
        Q_all  = np.zeros((T, A, A), dtype=np.float32)

        for t in range(T - 1, -1, -1):
            Q          = R_all[t] + self.gamma * V[np.newaxis, :]
            Q_all[t]   = Q
            policy[t]  = Q.argmax(axis=1)
            V          = Q.max(axis=1)

        # Forward pass — generate labels
        labels    = np.zeros((T, self.n_assets + 1), dtype=np.float32)
        cur_state = s0

        for t in range(T):
            best_a = policy[t, cur_state]
            if self.label_type == "soft":
                q_vals = Q_all[t, cur_state]
                probs  = F.softmax(
                    torch.tensor(q_vals * self.temperature), dim=0).numpy()
                label  = (probs[:, np.newaxis] * self._action_weights).sum(axis=0)
                label /= label.sum() + 1e-8
            else:
                label = self._action_weights[best_a].copy()
            labels[t]  = label
            cur_state  = best_a

        focal_weights = self.compute_focal_weights(high_prices, low_prices, prices)

        if verbose:
            cash_r  = (labels[:, -1] > 0.5).mean() * 100
            n_focal = (focal_weights > 1.0).sum()
            print(f"  Done: cash={cash_r:.1f}%  focal={n_focal}/{T} ({n_focal/T*100:.1f}%, {self.focal_weight}x weight)")

        return labels, focal_weights
