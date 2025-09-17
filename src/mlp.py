import numpy as np
import torch
from torch import nn


class ScalingLayer(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :]


class NTPLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, zero_init: bool = False
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        factor: float = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(factor * torch.randn(1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 / np.sqrt(self.in_features)) * (x @ self.weight) + self.bias


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(torch.tanh(nn.functional.softplus(x)))


class RealMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.model: nn.Sequential = nn.Sequential(
            ScalingLayer(n_features=input_dim),
            NTPLinear(in_features=input_dim, out_features=256),
            Mish(),
            NTPLinear(in_features=256, out_features=256),
            Mish(),
            NTPLinear(in_features=256, out_features=256),
            nn.Tanh(),
            NTPLinear(in_features=256, out_features=output_dim, zero_init=True),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
