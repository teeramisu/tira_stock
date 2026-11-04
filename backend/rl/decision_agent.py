"""DecisionAgent — RL-based decision layer for RL-MACS.

Implements PPO Actor-Critic with Safety-Critic constraint mechanism.
The Decision Agent learns to fuse multi-agent opinion vectors into
trading actions via reinforcement learning.

Architecture:
- Shared Feature Extractor: MLP(256, 128) + LayerNorm + Regime Embedding
- PPO Actor: outputs mean/std for continuous action distribution
- PPO Critic: estimates state value V(s)
- Safety-Critic: predicts action risk C(s,a), constrains policy when risk > threshold

Compatible with Stable-Baselines3 via custom network architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DecisionConfig:
    """Configuration for the Decision Agent network."""

    # Network architecture
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    regime_embed_dim: int = 16
    n_regime_classes: int = 4
    dropout: float = 0.1

    # Safety-Critic
    safety_threshold: float = 0.3  # Risk tolerance θ
    safety_weight: float = 1.0  # λ for conditional safety penalty

    # Action space
    action_scale: float = 0.3  # Scale factor for raw action output

    # Training
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_epochs: int = 10
    batch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


if TORCH_AVAILABLE:

    class RegimeEmbedding(nn.Module):
        """Embeds regime one-hot into a dense vector."""

        def __init__(self, n_classes: int = 4, embed_dim: int = 16):
            super().__init__()
            self.embedding = nn.Embedding(n_classes, embed_dim)
            nn.init.xavier_uniform_(self.embedding.weight)

        def forward(self, regime_onehot: torch.Tensor) -> torch.Tensor:
            """Convert one-hot to embedding.

            Args:
                regime_onehot: (batch, n_classes) one-hot regime vector.

            Returns:
                (batch, embed_dim) regime embedding.
            """
            # Convert one-hot to index
            regime_idx = regime_onehot.argmax(dim=-1)
            return self.embedding(regime_idx)

    class SharedFeatureExtractor(nn.Module):
        """Shared feature extractor for Actor and Critic.

        Processes the full observation vector through MLP layers with
        regime embedding injection.
        """

        def __init__(
            self,
            obs_dim: int,
            hidden_dims: tuple[int, ...] = (256, 128),
            regime_embed_dim: int = 16,
            n_regime_classes: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.regime_embed = RegimeEmbedding(n_regime_classes, regime_embed_dim)

            # Input: obs_dim (includes 4-dim regime one-hot) + regime_embed_dim
            input_dim = obs_dim + regime_embed_dim

            layers: list[nn.Module] = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = h_dim

            self.network = nn.Sequential(*layers)
            self.output_dim = hidden_dims[-1]

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            """Extract features from observation.

            Args:
                obs: (batch, obs_dim) observation vector.

            Returns:
                (batch, output_dim) feature representation.
            """
            # Extract regime one-hot (dims 10-13 in the observation)
            regime_onehot = obs[:, 10:14]  # 4-dim regime encoding
            regime_emb = self.regime_embed(regime_onehot)

            # Concatenate obs with regime embedding
            x = torch.cat([obs, regime_emb], dim=-1)
            return self.network(x)

    class ActorNetwork(nn.Module):
        """PPO Actor network — outputs action distribution parameters."""

        def __init__(
            self,
            feature_dim: int,
            action_dim: int,
            hidden_dim: int = 64,
            action_scale: float = 0.3,
        ):
            super().__init__()
            self.action_scale = action_scale

            self.mu_net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),  # Output in [-1, 1]
            )

            # Learnable log standard deviation
            self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute action mean and std.

            Args:
                features: (batch, feature_dim) from shared extractor.

            Returns:
                (mu, std) each of shape (batch, action_dim).
            """
            mu = self.mu_net(features) * self.action_scale
            std = self.log_std.exp().expand_as(mu)
            return mu, std

        def get_action(
            self, features: torch.Tensor, deterministic: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Sample action from policy.

            Returns:
                (action, log_prob) tuple.
            """
            mu, std = self.forward(features)
            dist = Normal(mu, std)

            if deterministic:
                action = mu
            else:
                action = dist.rsample()

            # Clamp to valid range
            action = action.clamp(-1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob

    class CriticNetwork(nn.Module):
        """PPO Critic network — estimates state value V(s)."""

        def __init__(self, feature_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.value_net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            """Estimate state value.

            Args:
                features: (batch, feature_dim) from shared extractor.

            Returns:
                (batch, 1) value estimate.
            """
            return self.value_net(features)

    class SafetyCritic(nn.Module):
        """Safety-Critic network — predicts action risk C(s, a).

        When predicted risk exceeds threshold θ, applies conditional
        safety penalty (CSP) to the policy gradient, constraining
        the agent from taking dangerous actions.

        Reference: MARS (AAAI 2026) Safety-Critic design.
        """

        def __init__(
            self,
            feature_dim: int,
            action_dim: int,
            hidden_dim: int = 64,
            safety_threshold: float = 0.3,
        ):
            super().__init__()
            self.safety_threshold = safety_threshold

            self.risk_net = nn.Sequential(
                nn.Linear(feature_dim + action_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),  # Risk in [0, 1]
            )

        def forward(
            self, features: torch.Tensor, actions: torch.Tensor
        ) -> torch.Tensor:
            """Predict risk of taking action in given state.

            Args:
                features: (batch, feature_dim) state features.
                actions: (batch, action_dim) proposed actions.

            Returns:
                (batch, 1) predicted risk level in [0, 1].
            """
            x = torch.cat([features, actions], dim=-1)
            return self.risk_net(x)

        def compute_safety_penalty(
            self, features: torch.Tensor, actions: torch.Tensor
        ) -> torch.Tensor:
            """Compute conditional safety penalty (CSP).

            CSP = max(0, C(s,a) - θ) — only penalizes when risk exceeds threshold.

            Returns:
                (batch,) penalty values.
            """
            risk = self.forward(features, actions).squeeze(-1)
            penalty = F.relu(risk - self.safety_threshold)
            return penalty

    class DecisionNetwork(nn.Module):
        """Complete Decision Agent network combining all components.

        Architecture:
            Observation → SharedFeatureExtractor → {Actor, Critic, SafetyCritic}

        The Actor produces actions, the Critic estimates value,
        and the SafetyCritic constrains risky actions.
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            config: DecisionConfig = DecisionConfig(),
        ):
            super().__init__()
            self.config = config
            self.obs_dim = obs_dim
            self.action_dim = action_dim

            # Shared feature extractor
            self.feature_extractor = SharedFeatureExtractor(
                obs_dim=obs_dim,
                hidden_dims=config.hidden_dims[:2],  # (256, 128)
                regime_embed_dim=config.regime_embed_dim,
                n_regime_classes=config.n_regime_classes,
                dropout=config.dropout,
            )

            feature_dim = self.feature_extractor.output_dim

            # Actor (policy)
            self.actor = ActorNetwork(
                feature_dim=feature_dim,
                action_dim=action_dim,
                hidden_dim=config.hidden_dims[-1],  # 64
                action_scale=config.action_scale,
            )

            # Critic (value)
            self.critic = CriticNetwork(
                feature_dim=feature_dim,
                hidden_dim=config.hidden_dims[-1],
            )

            # Safety-Critic (risk)
            self.safety_critic = SafetyCritic(
                feature_dim=feature_dim,
                action_dim=action_dim,
                hidden_dim=config.hidden_dims[-1],
                safety_threshold=config.safety_threshold,
            )

            self._init_weights()

        def _init_weights(self):
            """Xavier initialization for all linear layers."""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(
            self, obs: torch.Tensor, deterministic: bool = False
        ) -> dict[str, torch.Tensor]:
            """Full forward pass.

            Args:
                obs: (batch, obs_dim) observation.
                deterministic: If True, use mean action (no sampling).

            Returns:
                Dict with 'action', 'log_prob', 'value', 'risk'.
            """
            features = self.feature_extractor(obs)
            action, log_prob = self.actor.get_action(features, deterministic)
            value = self.critic(features)
            risk = self.safety_critic(features, action)

            return {
                "action": action,
                "log_prob": log_prob,
                "value": value.squeeze(-1),
                "risk": risk.squeeze(-1),
                "features": features,
            }

        def get_value(self, obs: torch.Tensor) -> torch.Tensor:
            """Get state value only (for GAE computation)."""
            features = self.feature_extractor(obs)
            return self.critic(features).squeeze(-1)

        def evaluate_actions(
            self, obs: torch.Tensor, actions: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            """Evaluate given actions (for PPO update).

            Args:
                obs: (batch, obs_dim) observations.
                actions: (batch, action_dim) actions taken.

            Returns:
                Dict with 'log_prob', 'value', 'entropy', 'risk_penalty'.
            """
            features = self.feature_extractor(obs)
            mu, std = self.actor(features)
            dist = Normal(mu, std)

            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            value = self.critic(features).squeeze(-1)

            # Safety penalty
            safety_penalty = self.safety_critic.compute_safety_penalty(features, actions)

            return {
                "log_prob": log_prob,
                "value": value,
                "entropy": entropy,
                "safety_penalty": safety_penalty,
            }

        def count_parameters(self) -> dict[str, int]:
            """Count parameters per component."""
            return {
                "feature_extractor": sum(p.numel() for p in self.feature_extractor.parameters()),
                "actor": sum(p.numel() for p in self.actor.parameters()),
                "critic": sum(p.numel() for p in self.critic.parameters()),
                "safety_critic": sum(p.numel() for p in self.safety_critic.parameters()),
                "total": sum(p.numel() for p in self.parameters()),
            }

else:
    # Stub classes when PyTorch is not available
    class DecisionNetwork:  # type: ignore[no-redef]
        """Stub DecisionNetwork when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                "PyTorch is required for DecisionNetwork. "
                "Install with: pip install torch"
            )


def create_decision_network(
    obs_dim: int,
    action_dim: int,
    config: DecisionConfig = DecisionConfig(),
) -> Any:
    """Factory function to create a DecisionNetwork.

    Args:
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        config: Network configuration.

    Returns:
        DecisionNetwork instance.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    return DecisionNetwork(obs_dim=obs_dim, action_dim=action_dim, config=config)


def save_decision_network(model: Any, path: str | Path) -> None:
    """Save DecisionNetwork weights to file."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "action_dim": model.action_dim,
        "config": model.config,
    }, path)
    logger.info(f"DecisionNetwork saved to {path}")


def load_decision_network(path: str | Path, device: str = "cpu") -> Any:
    """Load DecisionNetwork from file."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = DecisionNetwork(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        config=config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"DecisionNetwork loaded from {path}")
    return model
