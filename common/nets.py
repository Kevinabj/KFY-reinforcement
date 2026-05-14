"""Network modules shared across the four algorithms.

We use small MLPs with orthogonal initialisation, which is the standard recipe
in modern deep-RL implementations and considerably more stable than the
default Kaiming uniform.

Naming convention:
  - MLPQNet:                Q(s, .) for discrete-action DQN
  - MLPCritic:              Q(s, a) for off-policy continuous critics (SAC/TD3)
  - MLPVCritic:             V(s)    for on-policy value function (PPO)
  - CategoricalActor:       discrete softmax policy (PPO)
  - GaussianActor:          continuous diagonal-Gaussian policy with state-
                            independent log-std (PPO)
  - SquashedGaussianActor:  continuous tanh-squashed Gaussian with state-
                            dependent log-std (SAC)
  - DeterministicActor:     continuous deterministic tanh policy (TD3)
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _orth(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(sizes: Sequence[int], activation=nn.ReLU, hidden_gain: float = math.sqrt(2.0)) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(_orth(nn.Linear(sizes[i], sizes[i + 1]), gain=hidden_gain))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------- DQN
class MLPQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        sizes = [obs_dim, *hidden, n_actions]
        self.net = _mlp(sizes)
        # Final layer gets a smaller gain to start near-zero Q values.
        _orth(self.net[-1], gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# -------------------------------------------------------------- PPO critic
class MLPVCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        sizes = [obs_dim, *hidden, 1]
        self.net = _mlp(sizes)
        _orth(self.net[-1], gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# -------------------------------------------------------------- SAC / TD3 critic
class MLPCritic(nn.Module):
    """Q(s, a) where action is concatenated to the input."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        sizes = [obs_dim + act_dim, *hidden, 1]
        self.net = _mlp(sizes)
        _orth(self.net[-1], gain=1.0)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)


# -------------------------------------------------------------- PPO actors
class CategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        sizes = [obs_dim, *hidden, n_actions]
        self.net = _mlp(sizes)
        # Small init on the policy head to keep the initial policy near-uniform.
        _orth(self.net[-1], gain=0.01)

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def log_prob_and_entropy(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(obs)
        return dist.log_prob(action), dist.entropy()


class GaussianActor(nn.Module):
    """Continuous PPO actor: Gaussian with state-independent log-std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        self.act_dim = act_dim
        sizes = [obs_dim, *hidden, act_dim]
        self.mean_net = _mlp(sizes)
        _orth(self.mean_net[-1], gain=0.01)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def _dist(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def log_prob_and_entropy(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# -------------------------------------------------------------- SAC actor
class SquashedGaussianActor(nn.Module):
    """SAC actor: Gaussian over pre-tanh space, action = tanh(u).

    The log-prob carries the tanh change-of-variables correction
        log pi(a|s) = log N(u|mu,sigma) - sum_i log(1 - tanh(u_i)^2 + eps)
    which is critical for correct entropy regularisation.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.trunk = _mlp([obs_dim, *hidden])
        self.mean_head = _orth(nn.Linear(hidden[-1], act_dim), gain=0.01)
        self.log_std_head = _orth(nn.Linear(hidden[-1], act_dim), gain=0.01)
        self.trunk.append(nn.ReLU())  # one extra activation so trunk output is non-linear

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        u = normal.rsample()                          # reparameterised sample
        a = torch.tanh(u)
        # log-prob with tanh correction
        log_prob = normal.log_prob(u).sum(dim=-1)
        log_prob = log_prob - torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1)
        return a, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


# -------------------------------------------------------------- TD3 actor
class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        sizes = [obs_dim, *hidden, act_dim]
        self.net = _mlp(sizes)
        _orth(self.net[-1], gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
