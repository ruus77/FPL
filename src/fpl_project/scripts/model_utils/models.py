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
        self.feature_selection = nn.Parameter(torch.ones(input_size))
        
        for h, dr in zip(hidden_sizes, dropout_rates):
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.LeakyReLU(0.01), nn.Dropout(dr)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.feature_selection
        
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
        mask = self.feature_selection.view(1, -1, 1)
        x = x * mask
        
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
