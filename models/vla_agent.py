# models/vla_agent.py
# ============================================================
# X-MultiVLA 에이전트
#
# Phase 1: 지도학습 사전 학습 (VLAPretrainer)
#          차트+뉴스 → 다음 수익률 예측
#
# Phase 2: PPO 강화학습 (stable-baselines3 CustomPolicy)
#          Gym 환경에서 매수/매도/홀딩 결정
# ============================================================

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from models.encoders import PatchTSTEncoder, NewsProjector
from models.fusion   import MultiModalFusion
from config import model_cfg, rl_cfg, train_cfg


# ───────────────────────────────────────
# Phase 1: 지도학습 사전 학습 모델
# ───────────────────────────────────────

class VLAPretrainer(nn.Module):
    """
    지도학습으로 차트 인코더 + 뉴스 인코더 + 융합 레이어를 사전 학습합니다.
    
    입력:
        chart_seq : (B, window, num_features)  정규화된 차트 시퀀스
        news_vec  : (B, 4)                     뉴스 감성 벡터 (또는 zeros)
    출력:
        pred      : (B, 1)                     다음 스텝 BTC 수익률 예측값
    """

    def __init__(self, num_features: int, window: int = None):
        super().__init__()
        window = window or data_cfg_window()

        self.chart_encoder = PatchTSTEncoder(num_features=num_features, window=window)
        self.news_projector = NewsProjector(news_input_dim=4)
        self.fusion         = MultiModalFusion()

        self.head = nn.Sequential(
            nn.Linear(model_cfg.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, chart_seq: torch.Tensor, news_vec: torch.Tensor = None):
        chart_vec = self.chart_encoder(chart_seq)          # (B, d_model)
        news_p    = self.news_projector(news_vec) if news_vec is not None else None
        fused, _  = self.fusion(chart_vec, news_p)        # (B, hidden_dim)
        pred      = self.head(fused)                       # (B, 1)
        return pred

    def encode(self, chart_seq: torch.Tensor, news_vec: torch.Tensor = None) -> torch.Tensor:
        """RL 에이전트에서 임베딩 추출용"""
        chart_vec = self.chart_encoder(chart_seq)
        news_p    = self.news_projector(news_vec) if news_vec is not None else None
        fused, _  = self.fusion(chart_vec, news_p)
        return fused


def data_cfg_window():
    from config import data_cfg
    return data_cfg.window_size


# ───────────────────────────────────────
# Phase 2: SB3 Custom Features Extractor
# ───────────────────────────────────────

class VLAFeaturesExtractor(BaseFeaturesExtractor):
    """
    stable-baselines3의 PPO와 연결되는 커스텀 피처 추출기.
    
    Gym 환경의 flat observation을 다시 시퀀스 형태로 복원한 뒤
    사전 학습된 VLA 인코더를 통과시킵니다.

    Args:
        observation_space : gym 환경의 obs space (flat)
        num_features      : 원래 피처 수 (window * num_features 분해용)
        window            : 입력 윈도우 크기
        pretrained_model  : 사전 학습된 VLAPretrainer (optional)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        num_features: int,
        window: int,
        pretrained_model: VLAPretrainer = None,
    ):
        features_dim = model_cfg.hidden_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.num_features = num_features
        self.window       = window

        # 사전 학습 모델 재활용 또는 신규 생성
        if pretrained_model is not None:
            self.chart_encoder  = pretrained_model.chart_encoder
            self.news_projector = pretrained_model.news_projector
            self.fusion         = pretrained_model.fusion
        else:
            self.chart_encoder  = PatchTSTEncoder(num_features=num_features, window=window)
            self.news_projector = NewsProjector(news_input_dim=4)
            self.fusion         = MultiModalFusion()

        # 잔고/포지션 정보 처리 MLP
        self.state_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
        )

        # 최종 통합 레이어
        self.merge = nn.Sequential(
            nn.Linear(model_cfg.hidden_dim + 16, model_cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(model_cfg.hidden_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations : (B, obs_dim)  → flat
            obs_dim = window * num_features + 3
        """
        seq_len = self.window * self.num_features

        # ── 분리 ──────────────────────────────
        chart_flat = observations[:, :seq_len]                         # (B, W*F)
        state_info = observations[:, seq_len:]                         # (B, 3)

        chart_seq  = chart_flat.reshape(-1, self.window, self.num_features)   # (B, W, F)

        # 뉴스 컬럼: 피처 배열 안에 news_pos·neg·neu·score가 포함된 경우 추출
        # (없으면 zeros 사용)
        news_vec = torch.zeros(chart_seq.shape[0], 4, device=chart_seq.device)

        # ── 인코딩 + 융합 ─────────────────────
        chart_vec        = self.chart_encoder(chart_seq)
        news_p           = self.news_projector(news_vec)
        fused, _         = self.fusion(chart_vec, news_p)

        # ── 잔고 정보 통합 ────────────────────
        state_enc = self.state_mlp(state_info)
        out       = self.merge(torch.cat([fused, state_enc], dim=-1))
        return out


# ───────────────────────────────────────
# PPO 에이전트 빌더
# ───────────────────────────────────────

class VLAAgent:
    """
    PPO 기반 강화학습 에이전트 래퍼.
    
    사용법:
        agent = VLAAgent(env, num_features=20, window=60)
        agent.train(total_timesteps=500_000)
        agent.save("vla_agent.zip")
        action = agent.predict(obs)
    """

    def __init__(
        self,
        env,
        num_features: int,
        window: int         = None,
        pretrained_model    = None,
        learning_rate: float = None,
        device: str          = "auto",
    ):
        from config import data_cfg
        window = window or data_cfg.window_size
        lr     = learning_rate or rl_cfg.learning_rate

        policy_kwargs = {
            "features_extractor_class": VLAFeaturesExtractor,
            "features_extractor_kwargs": {
                "num_features":    num_features,
                "window":          window,
                "pretrained_model": pretrained_model,
            },
            "net_arch": [dict(pi=[128, 64], vf=[128, 64])],
        }

        self.model = PPO(
            policy         = "MlpPolicy",
            env            = env,
            learning_rate  = lr,
            n_steps        = rl_cfg.n_steps,
            batch_size     = rl_cfg.batch_size,
            n_epochs        = rl_cfg.n_epochs,
            gamma          = rl_cfg.gamma,
            gae_lambda     = rl_cfg.gae_lambda,
            clip_range     = rl_cfg.clip_range,
            ent_coef       = rl_cfg.ent_coef,
            policy_kwargs  = policy_kwargs,
            verbose        = 1,
            device         = device,
            tensorboard_log = train_cfg.log_dir,
        )

    def train(self, total_timesteps: int = None, callback=None):
        ts = total_timesteps or rl_cfg.total_timesteps
        print(f"[VLAAgent] PPO 학습 시작 ({ts:,} timesteps)")
        self.model.learn(total_timesteps=ts, callback=callback)
        print("[VLAAgent] 학습 완료")

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        print(f"[VLAAgent] 저장: {path}")

    @classmethod
    def load(cls, path: str, env):
        instance = cls.__new__(cls)
        instance.model = PPO.load(path, env=env)
        return instance

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)


# ─────────────────────────────────────────
# 단독 실행 (shape 확인용)
# ─────────────────────────────────────────
if __name__ == "__main__":
    num_features, window = 20, 60

    model = VLAPretrainer(num_features=num_features, window=window)
    chart = torch.randn(4, window, num_features)
    news  = torch.randn(4, 4)

    pred = model(chart, news)
    print(f"VLAPretrainer 예측: {pred.shape}")    # (4, 1)

    emb  = model.encode(chart, news)
    print(f"임베딩 벡터: {emb.shape}")            # (4, 128)
