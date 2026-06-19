from torch import nn
import torch


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int = 1,
                 dropout_rates: list[float] | None = None):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = [0.2] + [0.1] * (len(hidden_sizes) - 1)

        layers = []
        in_dim = input_size
        self.feature_selection = nn.Parameter(torch.ones(input_size) * .01)

        for h, dr in zip(hidden_sizes, dropout_rates):
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.LeakyReLU(0.01), nn.Dropout(dr)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(self.feature_selection)

        return self.net(x)


class Conv1DRegressor(nn.Module):
    def __init__(self, n_features: int, seq_len: int, channels: list[int] = None, 
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        
        if channels is None:
            channels = [64, 128, 64]
            
        self.feature_selection = nn.Parameter(torch.ones(n_features))
        
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
        x = x * self.feature_selection
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
        
        self.feature_selection = nn.Parameter(torch.ones(n_features))
        
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
        x = x * self.feature_selection
        
        x = self.input_proj(x)
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]

        return self.head(h_last)


class FPLLoss(nn.Module):
    def __init__(self, p_low: float, p_high: float, value_idx: int, minutes_idx: int,
                 model_type: str = "mlp",
                 w_premium: float = 3.0, under_predict_penalty: float = 1.5):
        super().__init__()
        self.p_low = p_low
        self.p_high = p_high
        self.value_idx = value_idx
        self.minutes_idx = minutes_idx
        self.model_type = model_type

        self.w_premium = w_premium
        self.under_predict_penalty = under_predict_penalty

        # Używamy bazowego MSE bez redukcji, aby kontrolować wagi per-sample
        self.mse = nn.MSELoss(reduction='none')

    def _get_feature(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if self.model_type == "mlp":
            return x[:, idx]
        elif self.model_type in ["lstm", "conv1d"]:
            return x[:, -1, idx]
        else:
            raise ValueError(f"Nieznany model_type: {self.model_type}")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        price = self._get_feature(x, self.value_idx)
        minutes = self._get_feature(x, self.minutes_idx)

        # 1. Bazowy błąd kwadratowy
        base_loss = self.mse(y_pred, y_true)

        # 2. GŁADKA MASKA PREMIUM (zamiast twardego True/False)
        # Używamy sigmoidy, aby płynnie przechodzić w wagę premium wokół progu p_high
        # Mnożnik *20 kontroluje stromość przejścia (można dostosować)
        premium_soft_mask = torch.sigmoid((price - self.p_high) * 20)
        price_weights = 1.0 + (self.w_premium - 1.0) * premium_soft_mask

        # 3. GŁADKA KARA ZA NIEDOSZACOWANIE (Asymetryczne wygładzenie)
        # Zamiast twardego `y_true > y_pred`, sprawdzamy różnicę (y_true - y_pred).
        # Softplus aktywuje się płynnie tylko dla wartości dodatnich (gdy niedoszacujemy).
        error_diff = y_true - y_pred
        under_predict_signal = nn.functional.softplus(error_diff, beta=2.0)
        # Mnożymy wagę premium przez asymetryczną karę, proporcjonalnie do stopnia niedoszacowania
        asymmetry_weight = 1.0 + (self.under_predict_penalty - 1.0) * (
                    under_predict_signal / (under_predict_signal + 1.0))
        price_weights = price_weights * asymmetry_weight

        # 4. Ważenie minutami i finalna ŚREDNIA (Krytyczna poprawka z .mean())
        final_loss = base_loss * price_weights * (minutes + 0.1)

        return final_loss.mean()  # <-- Zmieniono z .sum() na .mean() dla stabilności gradientów