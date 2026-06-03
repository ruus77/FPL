from torchmetrics import Metric
import torch


class BaseMetric(Metric):
    def __init__(self, p_low: float, p_high: float, value_idx: int, model_type: str = "mlp"):
        """
        Parameters
        ----------
        p_low:       dolny próg cenowy w [0, 1]
        p_high:      górny próg cenowy w [0, 1]
        value_idx:   indeks kolumny 'value' w wymiarze cech
        model_type:  'mlp'    → X: (B, F)
                     'lstm'   → X: (B, T, F)
                     'conv1d' → X: (B, F, T)
        """
        super().__init__()
        assert model_type in ("mlp", "lstm", "conv1d"), f"Nieznany model_type: {model_type}"

        self.p_low = p_low
        self.p_high = p_high
        self.value_idx = value_idx
        self.model_type = model_type

        self.add_state("err_sum_b", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("err_sum_m", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("err_sum_p", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_b",       torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_m",       torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_p",       torch.tensor(0.0), dist_reduce_fx="sum")

    def _get_price(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "mlp":
            return x[:, self.value_idx]          # (B, F)
        elif self.model_type == "lstm":
            return x[:, -1, self.value_idx]      # (B, T, F) — ostatni timestep
        elif self.model_type == "conv1d":
            return x[:, self.value_idx, -1]      # (B, F, T) — ostatni timestep

    def _apply_masks(self, price: torch.Tensor):
        return (
            price < self.p_low,
            (price >= self.p_low) & (price < self.p_high),
            price >= self.p_high,
        )

    def set_config(self, value_idx: int, model_type: str):
        assert model_type in ("mlp", "lstm", "conv1d"), f"Nieznany model_type: {model_type}"
        self.value_idx = value_idx
        self.model_type = model_type


class CustomMSE(BaseMetric):
    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor, x: torch.Tensor):
        preds, target = preds.flatten(), target.flatten()
        price = self._get_price(x)
        err   = (preds - target) ** 2
        b, m, p = self._apply_masks(price)
        if b.any(): self.err_sum_b += err[b].sum(); self.n_b += b.sum()
        if m.any(): self.err_sum_m += err[m].sum(); self.n_m += m.sum()
        if p.any(): self.err_sum_p += err[p].sum(); self.n_p += p.sum()

    def compute(self) -> dict[str, torch.Tensor]:
        return {
            "mse@budget":  self.err_sum_b / self.n_b.clamp(min=1),
            "mse@mid":     self.err_sum_m / self.n_m.clamp(min=1),
            "mse@premium": self.err_sum_p / self.n_p.clamp(min=1),
        }


class CustomMAE(BaseMetric):
    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor, x: torch.Tensor):
        preds, target = preds.flatten(), target.flatten()
        price = self._get_price(x)
        err   = torch.abs(preds - target)
        b, m, p = self._apply_masks(price)
        if b.any(): self.err_sum_b += err[b].sum(); self.n_b += b.sum()
        if m.any(): self.err_sum_m += err[m].sum(); self.n_m += m.sum()
        if p.any(): self.err_sum_p += err[p].sum(); self.n_p += p.sum()

    def compute(self) -> dict[str, torch.Tensor]:
        return {
            "mae@budget":  self.err_sum_b / self.n_b.clamp(min=1),
            "mae@mid":     self.err_sum_m / self.n_m.clamp(min=1),
            "mae@premium": self.err_sum_p / self.n_p.clamp(min=1),
        }