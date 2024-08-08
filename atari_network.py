from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.actor import ActorFactory
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.module.intermediate import (
    IntermediateModule,
    IntermediateModuleFactory,
)
from tianshou.utils.net.discrete import Actor, NoisyLinear
from torch import nn


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ScaledObsInputModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, denom: float = 255.0) -> None:
        super().__init__()
        self.module = module
        self.denom = denom
        # This is required such that the value can be retrieved by downstream modules (see usages of get_output_dim)
        self.output_dim = module.output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        if info is None:
            info = {}
        return self.module.forward(obs / self.denom, state, info)


def scale_obs(module: nn.Module, denom: float = 255.0) -> nn.Module:
    return ScaledObsInputModule(module, denom=denom)


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: str | int | torch.device = "cpu",
        features_only: bool = False,
        output_dim: int | None = None,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.output_dim = int(np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:]))
        if not features_only:
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(self.output_dim, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, int(np.prod(action_shape)))),
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(self.output_dim, output_dim)),
                nn.ReLU(inplace=True),
            )
            self.output_dim = output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        if info is None:
            info = {}
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state

class ActorFactoryAtariDQN(ActorFactory):
    def __init__(
        self,
        hidden_size: int | Sequence[int],
        scale_obs: bool,
        features_only: bool,
    ) -> None:
        self.hidden_size = hidden_size
        self.scale_obs = scale_obs
        self.features_only = features_only

    def create_module(self, envs: Environments, device: TDevice) -> Actor:
        net = DQN(
            *envs.get_observation_shape(),
            envs.get_action_shape(),
            device=device,
            features_only=self.features_only,
            output_dim=self.hidden_size,
            layer_init=layer_init,
        )
        if self.scale_obs:
            net = scale_obs(net)
        return Actor(
            net, envs.get_action_shape(), device=device, softmax_output=False
        ).to(device)


class IntermediateModuleFactoryAtariDQN(IntermediateModuleFactory):
    def __init__(self, features_only: bool = False, net_only: bool = False) -> None:
        self.features_only = features_only
        self.net_only = net_only

    def create_intermediate_module(
        self, envs: Environments, device: TDevice
    ) -> IntermediateModule:
        dqn = DQN(
            *envs.get_observation_shape(),
            envs.get_action_shape(),
            device=device,
            features_only=self.features_only,
        ).to(device)
        module = dqn.net if self.net_only else dqn
        return IntermediateModule(module, dqn.output_dim)


class IntermediateModuleFactoryAtariDQNFeatures(IntermediateModuleFactoryAtariDQN):
    def __init__(self) -> None:
        super().__init__(features_only=True, net_only=True)
