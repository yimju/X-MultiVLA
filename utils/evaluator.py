# utils/evaluator.py
# ============================================================
# 학습된 에이전트로 백테스트 실행 + 성능 지표 계산
#   - 총 수익률, Sharpe Ratio, MDD, Win Rate, 거래 횟수
#   - 결과 시각화 (matplotlib)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional
import os

from utils.gym_env import CryptoTradingEnv
from config import train_cfg


class Backtester:
    """
    사용법:
        bt = Backtester(agent, df_scaled, prices)
        result = bt.run()
        bt.plot(result)
    """

    def __init__(self, agent, df_scaled: np.ndarray, prices: np.ndarray, window: int = 60):
        self.agent      = agent
        self.df_scaled  = df_scaled
        self.prices     = prices
        self.window     = window

    # ───────────────────────────────────────
    # 백테스트 실행
    # ───────────────────────────────────────
    def run(self, n_episodes: int = 1) -> dict:
        """
        Returns:
            dict with keys:
                balance_curve   : (T,) 자산 곡선
                actions         : (T,) 행동 기록
                metrics         : 성과 지표 dict
        """
        env = CryptoTradingEnv(self.df_scaled, self.prices, window=self.window)
        obs, _ = env.reset()

        balance_curve  = [env.balance_0]
        action_log     = []
        reward_log     = []
        trades         = {"wins": 0, "losses": 0}

        prev_balance   = env.balance_0

        while True:
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            balance_curve.append(info["balance"])
            action_log.append(action)
            reward_log.append(reward)

            # 거래 성사 판별 (포지션이 닫힐 때)
            if info["balance"] > prev_balance:
                trades["wins"]  += 1
            elif info["balance"] < prev_balance:
                trades["losses"] += 1
            prev_balance = info["balance"]

            if terminated or truncated:
                break

        metrics = self._compute_metrics(
            balance_curve = np.array(balance_curve),
            trades        = trades,
            reward_log    = np.array(reward_log),
        )

        return {
            "balance_curve": np.array(balance_curve),
            "actions":       np.array(action_log),
            "rewards":       np.array(reward_log),
            "metrics":       metrics,
        }

    # ───────────────────────────────────────
    # 성과 지표 계산
    # ───────────────────────────────────────
    def _compute_metrics(
        self,
        balance_curve: np.ndarray,
        trades: dict,
        reward_log: np.ndarray,
    ) -> dict:
        initial  = balance_curve[0]
        final    = balance_curve[-1]

        # 총 수익률
        total_return = (final - initial) / initial

        # 일별 수익률 (시간봉 기준 → 24개 = 1일)
        daily_rets  = np.diff(balance_curve) / (balance_curve[:-1] + 1e-8)

        # Sharpe Ratio (연환산, 24h 기준)
        if daily_rets.std() > 0:
            sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(24 * 365)
        else:
            sharpe = 0.0

        # MDD (Max DrawDown)
        peak = np.maximum.accumulate(balance_curve)
        dd   = (peak - balance_curve) / (peak + 1e-8)
        mdd  = dd.max()

        # Win Rate
        total_trades = trades["wins"] + trades["losses"]
        win_rate     = trades["wins"] / max(total_trades, 1)

        # CAGR (Compound Annual Growth Rate)
        n_years = len(balance_curve) / (24 * 365)
        cagr    = (final / initial) ** (1 / max(n_years, 1e-3)) - 1

        return {
            "total_return":  f"{total_return:.2%}",
            "sharpe_ratio":  f"{sharpe:.2f}",
            "max_drawdown":  f"{mdd:.2%}",
            "win_rate":      f"{win_rate:.2%}",
            "cagr":          f"{cagr:.2%}",
            "total_trades":  total_trades,
            "final_balance": f"${final:,.0f}",
        }

    # ───────────────────────────────────────
    # 시각화
    # ───────────────────────────────────────
    def plot(self, result: dict, save_path: str = None):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.patch.set_facecolor("#0e1117")

        curve   = result["balance_curve"]
        actions = result["actions"]
        rewards = result["rewards"]
        metrics = result["metrics"]

        step_axis = np.arange(len(curve))

        # ── Panel 1: 자산 곡선 ─────────────────
        ax1 = axes[0]
        ax1.set_facecolor("#0e1117")
        ax1.plot(step_axis, curve, color="#00d4ff", linewidth=1.5, label="Balance")
        ax1.fill_between(step_axis, curve[0], curve, alpha=0.15, color="#00d4ff")
        ax1.axhline(curve[0], color="#555", linestyle="--", linewidth=0.8, label="Initial")

        # 매수/매도 마커 (actions: 1=Long, 2=Short)
        long_idx  = np.where(actions == 1)[0]
        short_idx = np.where(actions == 2)[0]
        if len(long_idx):
            ax1.scatter(long_idx,  curve[long_idx],  color="#00ff88", s=10, zorder=5, label="Long")
        if len(short_idx):
            ax1.scatter(short_idx, curve[short_idx], color="#ff4466", s=10, zorder=5, label="Short")

        ax1.set_ylabel("Balance (USDT)", color="white")
        ax1.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e")
        ax1.tick_params(colors="white")
        ax1.spines["bottom"].set_color("#333")
        ax1.spines["left"].set_color("#333")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # ── Panel 2: 수익률 분포 ──────────────
        ax2 = axes[1]
        ax2.set_facecolor("#0e1117")
        daily = np.diff(curve) / (curve[:-1] + 1e-8)
        colors = ["#00ff88" if r >= 0 else "#ff4466" for r in daily]
        ax2.bar(np.arange(len(daily)), daily * 100, color=colors, width=1.0, alpha=0.7)
        ax2.axhline(0, color="#555", linewidth=0.8)
        ax2.set_ylabel("Step Return (%)", color="white")
        ax2.tick_params(colors="white")
        ax2.spines["bottom"].set_color("#333")
        ax2.spines["left"].set_color("#333")

        # ── Panel 3: Drawdown ─────────────────
        ax3 = axes[2]
        ax3.set_facecolor("#0e1117")
        peak = np.maximum.accumulate(curve)
        dd   = (peak - curve) / (peak + 1e-8) * 100
        ax3.fill_between(step_axis, 0, -dd, color="#ff4466", alpha=0.6)
        ax3.set_ylabel("Drawdown (%)", color="white")
        ax3.set_xlabel("Steps", color="white")
        ax3.tick_params(colors="white")
        ax3.spines["bottom"].set_color("#333")
        ax3.spines["left"].set_color("#333")

        # ── 지표 텍스트 박스 ───────────────────
        m = result["metrics"]
        info_text = (
            f"Return: {m['total_return']}  |  Sharpe: {m['sharpe_ratio']}  |  "
            f"MDD: {m['max_drawdown']}  |  WinRate: {m['win_rate']}  |  "
            f"CAGR: {m['cagr']}  |  Trades: {m['total_trades']}"
        )
        fig.text(0.5, 0.01, info_text, ha="center", fontsize=9, color="#aaaaaa")

        plt.suptitle("X-MultiVLA Backtest Results", color="white", fontsize=13, y=0.99)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"[evaluator] 차트 저장: {save_path}")

        plt.show()
        return fig

    def print_metrics(self, result: dict):
        print("\n" + "=" * 50)
        print("  X-MultiVLA 백테스트 성과 요약")
        print("=" * 50)
        for k, v in result["metrics"].items():
            print(f"  {k:20s}: {v}")
        print("=" * 50)
