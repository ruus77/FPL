"""Loss functions and FPLLoss sample weights shared by tabular and neural models."""

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
        minutes_mean: float = 0.0,
        minutes_scale: float = 1.0,
    ):
        super().__init__()
        if model_type not in ("mlp", "conv1d"):
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
        self.minutes_mean = minutes_mean
        self.minutes_scale = minutes_scale

    def _get_feature(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if self.model_type == "mlp" or x.ndim == 2:
            return x[:, idx]
        return x[:, -1, idx]

    def _get_minutes(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_ema_minutes and self.model_type == "conv1d" and x.ndim == 3:
            seq = x[:, -3:, self.minutes_idx]
            m = seq[:, 0]
            for t in range(1, seq.shape[1]):
                m = 0.5 * seq[:, t] + 0.5 * m
        else:
            m = self._get_feature(x, self.minutes_idx)
        m = m * self.minutes_scale + self.minutes_mean
        return torch.clamp(m, 0.0, 1.0)

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


class MSELossWrapper(nn.Module):
    """Standard MSE loss wrapper accepting optional auxiliary inputs."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mse(y_pred.flatten(), y_true.flatten())
