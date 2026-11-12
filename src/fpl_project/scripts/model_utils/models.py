import numpy as np
import pandas as pd
import torch
from torch import nn

from fpl_project.scripts import config


def compute_fpl_weights(
    df: pd.DataFrame,
    p_low: float,
    p_high: float,
    price_col: str = "value_ema_3",
    minutes_col: str = "minutes_ema_3",
    w_mid: float = config.W_MID,
    w_premium: float = config.W_PREMIUM,
    eps_min: float = config.EPS_MIN,
) -> np.ndarray:
    """Compute FPLLoss sample weights: w_i = w_cena_i * (ln(1 + m_i) + eps_min)."""
    price = df[price_col].to_numpy(dtype=np.float64)
    minutes = np.clip(df[minutes_col].to_numpy(dtype=np.float64) / 90.0, 0.0, 1.0)

    w_price = np.ones_like(price, dtype=np.float64)
    w_price = np.where(price >= p_low, w_mid, w_price)
    w_price = np.where(price >= p_high, w_premium, w_price)

    w_min = np.log1p(minutes) + eps_min
    weights = w_price * w_min
    return weights.astype(np.float64)


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
    """Residual 1D Convolutional regressor for sequential gameweek windows."""

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
            channels = [64, 128, 64]

        self.input_proj = nn.Conv1d(n_features, channels[0], kernel_size=1)

        blocks = [
            self._conv_block(channels[i], channels[i + 1], kernel_size, dropout)
            for i in range(len(channels) - 1)
        ]
        self.conv_blocks = nn.ModuleList(blocks)

        self.skips = nn.ModuleList([
            nn.Conv1d(channels[i], channels[i + 1], kernel_size=1)
            if channels[i] != channels[i + 1]
            else nn.Identity()
            for i in range(len(channels) - 1)
        ])

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, ks: int, dr: float) -> nn.Sequential:
        pad = ks // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Conv1d(out_ch, out_ch, kernel_size=ks, padding=pad),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_proj(x.transpose(1, 2))
        for block, skip in zip(self.conv_blocks, self.skips):
            out = torch.relu(block(out) + skip(out))
        return self.head(self.gap(out))


class LSTMRegressor(nn.Module):
    """Stacked LSTM regressor for sequential gameweek windows."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        self.input_proj = nn.Linear(n_features, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * self.directions
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Linear(lstm_out_size, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        _, (h_n, _) = self.lstm(x)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.bidirectional else h_n[-1]
        return self.head(h_last)


class FPLLoss(nn.Module):
    """Weighted MSE loss for FPL: mean(w_i * (y_i - y_hat_i)^2) with w_i = w_cena_i * (ln(1 + m_i) + eps_min)."""

    def __init__(
        self,
        p_low: float,
        p_high: float,
        value_idx: int = 0,
        minutes_idx: int = 1,
        model_type: str = "mlp",
        w_mid: float = config.W_MID,
        w_premium: float = config.W_PREMIUM,
        eps_min: float = config.EPS_MIN,
        use_ema_minutes: bool = False,
    ):
        super().__init__()
        if model_type not in ("mlp", "lstm", "conv1d"):
            raise ValueError(f"Nieznany model_type: {model_type}")
        self.p_low = p_low
        self.p_high = p_high
        self.value_idx = value_idx
        self.minutes_idx = minutes_idx
        self.model_type = model_type
        self.w_mid = w_mid
        self.w_premium = w_premium
        self.eps_min = eps_min
        self.use_ema_minutes = use_ema_minutes

    def _get_feature(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if self.model_type == "mlp" or x.ndim == 2:
            return x[:, idx]
        return x[:, -1, idx]

    def _get_minutes(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_ema_minutes and self.model_type in ("lstm", "conv1d") and x.ndim == 3:
            seq = x[:, -3:, self.minutes_idx]
            m = seq[:, 0]
            for t in range(1, seq.shape[1]):
                m = 0.5 * seq[:, t] + 0.5 * m
            return m
        return self._get_feature(x, self.minutes_idx)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        x: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        if weights is not None:
            w = weights.flatten().to(y_pred.device, dtype=y_pred.dtype).detach()
            return (w * (y_true - y_pred) ** 2).mean()

        if x is not None and x.ndim == 1:
            w = x.flatten().to(y_pred.device, dtype=y_pred.dtype).detach()
            return (w * (y_true - y_pred) ** 2).mean()

        price = self._get_feature(x, self.value_idx)
        minutes = self._get_minutes(x)

        w_price = torch.ones_like(price)
        w_price = torch.where(price >= self.p_low, torch.full_like(price, self.w_mid), w_price)
        w_price = torch.where(price >= self.p_high, torch.full_like(price, self.w_premium), w_price)

        w_min = torch.log1p(minutes) + self.eps_min
        w = (w_price * w_min).detach()
        return (w * (y_true - y_pred) ** 2).mean()
