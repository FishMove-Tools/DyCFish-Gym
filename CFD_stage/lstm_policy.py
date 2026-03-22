"""LSTM-augmented actor-critic policy for Stable-Baselines3.

This module provides :class:`LstmExtractor`, a lightweight LSTM feature
extractor, and :class:`MlpLstmPolicy`, an actor-critic policy that
combines MLP layers with LSTM hidden states for temporal reasoning.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy, register_policy


class LstmExtractor(nn.Module):
    """LSTM-based feature extractor for sequential observations.

    Wraps a multi-layer LSTM that processes a single time-step at each
    call, maintaining hidden states across steps within an episode.
    """

    def __init__(
        self,
        feature_dim: int,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.lstm_hidden_states: Optional[
            Tuple[th.Tensor, th.Tensor]
        ] = None

    def forward(self, features: th.Tensor) -> th.Tensor:
        """Run one forward step through the LSTM."""
        batch_size = features.shape[0]
        seq_features = features.reshape(batch_size, 1, -1)

        # Re-initialise hidden states when missing or batch size changes
        if (
            self.lstm_hidden_states is None
            or self.lstm_hidden_states[0].shape[1] != batch_size
        ):
            self.lstm_hidden_states = (
                th.zeros(
                    self.lstm_layers, batch_size, self.lstm_hidden_size,
                    device=features.device,
                ),
                th.zeros(
                    self.lstm_layers, batch_size, self.lstm_hidden_size,
                    device=features.device,
                ),
            )

        lstm_out, self.lstm_hidden_states = self.lstm(
            seq_features, self.lstm_hidden_states,
        )

        # Return the output from the last (only) time-step
        return lstm_out[:, -1, :]

    def reset_states(
        self,
        batch_size: int = 1,
        device: th.device = th.device("cpu"),
    ) -> None:
        """Reset the LSTM hidden and cell states to zeros."""
        self.lstm_hidden_states = (
            th.zeros(
                self.lstm_layers, batch_size, self.lstm_hidden_size,
                device=device,
            ),
            th.zeros(
                self.lstm_layers, batch_size, self.lstm_hidden_size,
                device=device,
            ),
        )


class MlpLstmPolicy(ActorCriticPolicy):
    """Actor-critic policy combining MLP and LSTM feature extractors.

    The policy augments the standard MLP feature extraction pipeline
    with a per-network LSTM that captures temporal dependencies in the
    observation sequence.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        lr_schedule: Callable[[float], float],
        *args: Any,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        enable_critic_lstm: bool = True,
        **kwargs: Any,
    ) -> None:
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.enable_critic_lstm = enable_critic_lstm

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        # Actor LSTM
        self.actor_lstm = LstmExtractor(
            self.features_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
        )

        # Critic LSTM (optional)
        if enable_critic_lstm:
            self.critic_lstm = LstmExtractor(
                self.features_dim,
                lstm_hidden_size=lstm_hidden_size,
                lstm_layers=lstm_layers,
            )
        else:
            self.critic_lstm = None

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Compute actions, value estimates, and log-probabilities."""
        features = self.extract_features(obs)

        actor_features = self.actor_lstm(features)
        critic_features = (
            self.critic_lstm(features)
            if self.enable_critic_lstm
            else features
        )

        latent_pi = self.mlp_extractor.policy_net(actor_features)
        action_distribution = self._get_action_dist_from_latent(latent_pi)

        latent_vf = self.mlp_extractor.value_net(critic_features)
        values = self.value_net(latent_vf)

        actions = action_distribution.get_actions(deterministic=deterministic)
        log_probs = action_distribution.log_prob(actions)

        return actions, values, log_probs

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Evaluate value, log-probability, and entropy for given actions."""
        features = self.extract_features(obs)

        actor_features = self.actor_lstm(features)
        critic_features = (
            self.critic_lstm(features)
            if self.enable_critic_lstm
            else features
        )

        latent_pi = self.mlp_extractor.policy_net(actor_features)
        action_distribution = self._get_action_dist_from_latent(latent_pi)

        latent_vf = self.mlp_extractor.value_net(critic_features)
        values = self.value_net(latent_vf)

        log_probs = action_distribution.log_prob(actions)
        entropy = action_distribution.entropy()

        return values, log_probs, entropy

    def reset_lstm_states(self) -> None:
        """Reset LSTM hidden states for both actor and critic networks."""
        device = next(self.parameters()).device
        self.actor_lstm.reset_states(device=device)
        if self.enable_critic_lstm:
            self.critic_lstm.reset_states(device=device)


# Register the custom policy so SB3 can look it up by string name.
register_policy("MlpLstmPolicy", MlpLstmPolicy)