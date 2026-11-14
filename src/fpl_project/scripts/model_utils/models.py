"""Neural network architectures for FPL points regression."""

import torch
from torch import nn


class MLP(nn.Module):
    """Multi-Layer Perceptron regressor for tabular FPL features."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int = 1,
        dropout_rates: list[float] | None = None,
        dropout: float | None = None,
    ):
        super().__init__()
        if dropout_rates is None:
            if dropout is not None:
                dropout_rates = [dropout] * len(hidden_sizes)
            else:
                dropout_rates = [0.2] + [0.1] * (len(hidden_sizes) - 1)

        layers = []
        in_dim = input_size
        for h, dr in zip(hidden_sizes, dropout_rates):
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.LeakyReLU(0.01), nn.Dropout(dr)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv1DRegressor(nn.Module):
    """1D Convolutional regressor for sequential gameweek windows (bez połączeń resztkowych)."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [128, 64]

        self.seq_len = seq_len
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_features, channels[0], kernel_size=1),
            nn.BatchNorm1d(channels[0]),
            nn.LeakyReLU(0.01),
        )

        self.conv_blocks = nn.ModuleList([
            self._conv_block(channels[i], channels[i + 1], kernel_size, dropout)
            for i in range(len(channels) - 1)
        ])

        pooled_dim = channels[-1] * 3
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, ks: int, dr: float) -> nn.Sequential:
        pad = ks // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.01),
            nn.Dropout(dr),
            nn.Conv1d(out_ch, out_ch, kernel_size=ks, padding=pad),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_proj(x.transpose(1, 2))
        for block in self.conv_blocks:
            out = nn.functional.leaky_relu(block(out), negative_slope=0.01)
        pooled = torch.cat([out[:, :, -1], out.mean(dim=-1), out.amax(dim=-1)], dim=1)
        return self.head(pooled)
