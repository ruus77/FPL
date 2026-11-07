from abc import ABC, abstractclassmethod
from torch import nn
import torch


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int = 1,
                 dropout_rates: list[float] | None = None, dropout: float | None = None):
        super().__init__()
        if dropout_rates is None:
            if dropout is not None:
                dropout_rates = [dropout] * len(hidden_sizes)
            else:
                dropout_rates = [0.2] + [0.1] * (len(hidden_sizes) - 1)

        layers = []
        in_dim = input_size
        self.feature_selection = nn.Parameter(torch.zeros(input_size))

        for h, dr in zip(hidden_sizes, dropout_rates):
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.LeakyReLU(0.01), nn.Dropout(dr)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv1DRegressor(nn.Module):
    def __init__(self, n_features: int, seq_len: int, channels: list[int] = None, 
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        
        if channels is None:
            channels = [64, 128, 64]
            
        self.feature_selection = nn.Parameter(torch.zeros(n_features))
        
        self.input_proj = nn.Conv1d(n_features, channels[0], kernel_size=1)
        
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(self._conv_block(channels[i], channels[i + 1], kernel_size, dropout))
        self.conv_blocks = nn.ModuleList(blocks)
        
        self.skips = nn.ModuleList([
            nn.Conv1d(channels[i], channels[i + 1], kernel_size=1) 
            if channels[i] != channels[i + 1] else nn.Identity()
            for i in range(len(channels) - 1)
        ])
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
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
        x = x.transpose(1, 2)
        
        out = self.input_proj(x)
        
        for block, skip in zip(self.conv_blocks, self.skips):
            residual = skip(out)
            out = block(out) + residual
            out = torch.relu(out)
            
        out = self.gap(out)
        return self.head(out)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        
        self.feature_selection = nn.Parameter(torch.zeros(n_features))
        
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
            bidirectional=bidirectional
        )

        lstm_out_size = hidden_size * self.directions

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Linear(lstm_out_size, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.input_proj(x)
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]

        return self.head(h_last)


class FPLLoss(nn.Module):
    r"""Jawna, ważona strata MSE dla FPL.

    L = mean( w_i * (y_i - yhat_i)^2 ),
    w_i = w_price(price_i) * (ln(1 + m_i) + eps_min)

    gdzie w_price = 1          dla price <  p_low,
                    w_mid      dla p_low <= price < p_high,
                    w_premium  dla price >= p_high   (twarde progi).

    price i m to cena i minuty (przeskalowane) z cech MLP albo z ostatniego kroku
    okna (lstm/conv1d); przy use_ema_minutes=True m to EMA3 minut z 3 ostatnich
    kroków okna (tylko sekwencje). Wagi dzielone przez stałą `norm_const`
    (domyślnie 1.0 = brak normalizacji).
    """

    def __init__(self, p_low: float, p_high: float, value_idx: int, minutes_idx: int,
                 model_type: str = "mlp",
                 w_mid: float = 1.25,
                 w_premium: float = 2.0,
                 eps_min: float = 0.1,
                 norm_const: float = 1.0,
                 use_ema_minutes: bool = False):
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
        self.norm_const = norm_const
        self.use_ema_minutes = use_ema_minutes

    def _get_feature(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if self.model_type == "mlp":
            return x[:, idx]
        elif self.model_type in ["lstm", "conv1d"]:
            return x[:, -1, idx]
        else:
            raise ValueError(f"Nieznany model_type: {self.model_type}")

    def _get_minutes(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_ema_minutes and self.model_type in ("lstm", "conv1d"):
            seq = x[:, -3:, self.minutes_idx]
            alpha = 0.5  # EMA3: span=3
            m = seq[:, 0]
            for t in range(1, seq.shape[1]):
                m = alpha * seq[:, t] + (1 - alpha) * m
            return m
        return self._get_feature(x, self.minutes_idx)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        price = self._get_feature(x, self.value_idx)
        minutes = self._get_minutes(x)

        w_price = torch.ones_like(price)
        w_price = torch.where(price >= self.p_low, torch.full_like(price, self.w_mid), w_price)
        w_price = torch.where(price >= self.p_high, torch.full_like(price, self.w_premium), w_price)

        w_min = torch.log1p(minutes.clamp(min=0.0)) + self.eps_min
        weights = (w_price * w_min / self.norm_const).detach()

        return (weights * (y_true - y_pred) ** 2).mean()