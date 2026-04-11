# utils/gym_env.py
# ============================================================
# Gymnasium 커스텀 트레이딩 환경
#
# State   : 과거 window 타임스텝의 정규화된 피처 + 잔고 정보
# Action  : 0=Hold, 1=Long(매수), 2=Short(매도)
# Reward  : 수익률 + Sharpe 보너스 - MDD 패널티 - 수수료
# Done    : 데이터 끝 or 잔고 0 이하
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import rl_cfg, data_cfg


class CryptoTradingEnv(gym.Env):
    """
    사용법:
        env = CryptoTradingEnv(df_scaled, feature_cols)
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_scaled: np.ndarray,          # (T, num_features) 정규화된 피처 배열
        prices: np.ndarray,             # (T,) 실제 BTC 종가 (보상 계산용)
        window: int      = None,
        initial_balance: float = None,
        fee_rate: float  = None,
        render_mode      = None,
    ):
        super().__init__()

        self.df         = df_scaled
        self.prices     = prices
        self.window     = window          or data_cfg.window_size
        self.balance_0  = initial_balance or rl_cfg.initial_balance
        self.fee_rate   = fee_rate        or rl_cfg.reward_fee_rate
        self.render_mode = render_mode

        self.num_features = df_scaled.shape[1]
        self.T            = len(df_scaled)

        # ── Action Space: 0=Hold, 1=Long, 2=Short ──
        self.action_space = spaces.Discrete(3)

        # ── Observation Space ───────────────────────
        # [window × num_features] + [잔고비율, 포지션, 보유스텝수]
        obs_dim = self.window * self.num_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_state()

    # ───────────────────────────────────────
    # 상태 초기화 헬퍼
    # ───────────────────────────────────────
    def _reset_state(self):
        self.current_step = self.window
        self.balance      = self.balance_0
        self.position     = 0              # 0: 없음, 1: Long, -1: Short
        self.entry_price  = 0.0
        self.hold_steps   = 0
        self.peak_balance = self.balance_0
        self.returns_log  = []             # 스텝별 수익률 기록 (Sharpe 계산용)

    # ───────────────────────────────────────
    # 관측값 생성
    # ───────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        window_data = self.df[self.current_step - self.window : self.current_step]
        window_flat = window_data.flatten()                       # (window × F,)

        # 정규화된 상태 정보
        balance_ratio = np.clip(self.balance / self.balance_0, 0, 3)
        pos_enc       = float(self.position)                       # -1, 0, 1
        hold_steps_n  = np.clip(self.hold_steps / 100, 0, 1)

        return np.concatenate([window_flat, [balance_ratio, pos_enc, hold_steps_n]]).astype(np.float32)

    # ───────────────────────────────────────
    # reset
    # ───────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    # ───────────────────────────────────────
    # step
    # ───────────────────────────────────────
    def step(self, action: int):
        """
        action: 0=Hold, 1=Long(매수 또는 유지), 2=Short(매도 또는 공매도)
        """
        current_price = self.prices[self.current_step - 1]
        next_price    = self.prices[min(self.current_step, self.T - 1)]
        price_return  = (next_price - current_price) / (current_price + 1e-8)

        reward = 0.0
        fee    = 0.0

        # ── 포지션 변경 ─────────────────────────
        if action == 1:    # Long
            if self.position == 0:          # 신규 매수
                fee = self.balance * self.fee_rate
                self.balance    -= fee
                self.entry_price = current_price
                self.position    = 1
                self.hold_steps  = 0
            elif self.position == -1:       # Short 청산 후 Long
                pnl = -price_return * self.balance
                fee = abs(pnl) * self.fee_rate * 2
                self.balance    += pnl - fee
                self.entry_price = current_price
                self.position    = 1
                self.hold_steps  = 0

        elif action == 2:  # Short
            if self.position == 0:          # 신규 Short
                fee = self.balance * self.fee_rate
                self.balance    -= fee
                self.entry_price = current_price
                self.position    = -1
                self.hold_steps  = 0
            elif self.position == 1:        # Long 청산 후 Short
                pnl = price_return * self.balance
                fee = abs(pnl) * self.fee_rate * 2
                self.balance    += pnl - fee
                self.entry_price = current_price
                self.position    = -1
                self.hold_steps  = 0

        else:              # Hold
            self.hold_steps += 1

        # ── 포지션 P&L ──────────────────────────
        if self.position == 1:
            step_pnl = price_return * self.balance
        elif self.position == -1:
            step_pnl = -price_return * self.balance
        else:
            step_pnl = 0.0

        self.balance += step_pnl
        step_return  = step_pnl / (self.balance_0 + 1e-8)
        self.returns_log.append(step_return)

        # ── 보상 계산 ────────────────────────────
        reward = self._compute_reward(step_return, fee)

        # ── MDD 업데이트 ─────────────────────────
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # ── 종료 조건 ────────────────────────────
        self.current_step += 1
        terminated = (
            self.current_step >= self.T
            or self.balance <= self.balance_0 * 0.1    # 잔고 90% 손실 시 강제 종료
        )
        truncated = False

        info = {
            "balance":   self.balance,
            "position":  self.position,
            "price":     current_price,
            "step_return": step_return,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ───────────────────────────────────────
    # 보상 함수 (reward.py와 연동)
    # ───────────────────────────────────────
    def _compute_reward(self, step_return: float, fee: float) -> float:
        cfg = rl_cfg

        # 1. 수익 보상
        profit_reward = step_return * cfg.reward_profit_weight

        # 2. Sharpe 보너스 (최소 10스텝 이후)
        sharpe_reward = 0.0
        if len(self.returns_log) >= 10:
            r_arr  = np.array(self.returns_log[-50:])   # 최근 50스텝
            mean_r = r_arr.mean()
            std_r  = r_arr.std() + 1e-8
            sharpe_reward = (mean_r / std_r) * cfg.reward_sharpe_weight * 0.01

        # 3. MDD 패널티
        mdd      = (self.peak_balance - self.balance) / (self.peak_balance + 1e-8)
        mdd_pen  = -mdd * cfg.reward_mdd_penalty * 0.01

        # 4. 거래 수수료 패널티
        fee_pen  = -(fee / self.balance_0) * 10

        reward = profit_reward + sharpe_reward + mdd_pen + fee_pen
        return float(reward)

    # ───────────────────────────────────────
    # 렌더
    # ───────────────────────────────────────
    def render(self):
        mdd = (self.peak_balance - self.balance) / (self.peak_balance + 1e-8)
        print(
            f"Step {self.current_step:4d} | "
            f"Balance: ${self.balance:,.0f} | "
            f"Position: {['Hold','Long','Short'][self.position]} | "
            f"MDD: {mdd:.2%}"
        )

    def get_portfolio_summary(self) -> dict:
        """에피소드 종료 후 성과 요약"""
        returns = np.array(self.returns_log)
        total_return = (self.balance - self.balance_0) / self.balance_0
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252 * 24)  # 연환산
        mdd    = (self.peak_balance - self.balance) / (self.peak_balance + 1e-8)
        return {
            "total_return":  total_return,
            "sharpe_ratio":  sharpe,
            "max_drawdown":  mdd,
            "final_balance": self.balance,
        }


# ─────────────────────────────────────────
# 단독 실행 (환경 테스트)
# ─────────────────────────────────────────
if __name__ == "__main__":
    T, F = 500, 20
    df_scaled = np.random.randn(T, F).astype(np.float32)
    prices    = np.linspace(30000, 35000, T)

    env = CryptoTradingEnv(df_scaled, prices)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")

    total_reward = 0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"Total reward: {total_reward:.4f}")
    summary = env.get_portfolio_summary()
    print(summary)
