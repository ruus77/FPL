from collections import defaultdict
import copy
from dataclasses import dataclass
import random

import numpy as np
import sklearn
import torch
from torchmetrics import R2Score
from tqdm.auto import tqdm

from .custom_metrics import CustomMAE, CustomMSE


def set_seed(seed: int = 77) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class EpochMetrics:
    """Structured metrics computed over a single training or evaluation epoch."""

    loss: float
    mse: float
    mae: float
    adj_r2: float
    mse_budget: float
    mse_mid: float
    mse_premium: float
    mae_budget: float
    mae_mid: float
    mae_premium: float
    predictions: torch.Tensor | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "mse": self.mse,
            "mae": self.mae,
            "adj_r2": self.adj_r2,
            "mse@budget": self.mse_budget,
            "mse@mid": self.mse_mid,
            "mse@premium": self.mse_premium,
            "mae@budget": self.mae_budget,
            "mae@mid": self.mae_mid,
            "mae@premium": self.mae_premium,
        }

    def get_monitored_value(self, monitor: str) -> float:
        monitor_map = {
            "loss": self.loss,
            "mse": self.mse,
            "mae": self.mae,
            "adj_r2": self.adj_r2,
        }
        if monitor not in monitor_map:
            raise ValueError(f"Nieznana metryka monitorowania: {monitor}")
        return monitor_map[monitor]


class Trainer:
    """Trainer and evaluator for tabular and sequential PyTorch regression models."""

    def __init__(
        self,
        device: str,
        num_features: int,
        random_state: int = 77,
        p_low: float = 0.3,
        p_high: float = 0.8,
        value_idx: int = 0,
        model_type: str = "mlp",
        scaler: sklearn.preprocessing.StandardScaler | None = None,
    ):
        set_seed(random_state)

        self.device = device
        self.num_features = num_features
        self.scaler = scaler

        self.mse_at = CustomMSE(
            p_low=p_low,
            p_high=p_high,
            value_idx=value_idx,
            model_type=model_type,
        ).to(device)

        self.mae_at = CustomMAE(
            p_low=p_low,
            p_high=p_high,
            value_idx=value_idx,
            model_type=model_type,
        ).to(device)

        self.r2_metric = R2Score().to(device)

    def _adjusted_r2(self, r2: float, n_samples: int) -> float:
        if n_samples <= self.num_features + 1:
            return 0.0
        return 1.0 - ((1.0 - r2) * (n_samples - 1) / (n_samples - self.num_features - 1))

    def _inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaler is None:
            return x
        arr = x.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        inv = self.scaler.inverse_transform(arr)
        return torch.tensor(inv, device=self.device)

    def set_model_config(self, value_idx: int, model_type: str) -> None:
        self.mse_at.set_config(value_idx=value_idx, model_type=model_type)
        self.mae_at.set_config(value_idx=value_idx, model_type=model_type)

    def _reset_metrics(self) -> None:
        self.mse_at.reset()
        self.mae_at.reset()
        self.r2_metric.reset()

    def _update_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor) -> None:
        y_pred_inv = self._inverse(y_pred)
        y_true_inv = self._inverse(y_true)

        self.mse_at(preds=y_pred_inv, target=y_true_inv, x=x)
        self.mae_at(preds=y_pred_inv, target=y_true_inv, x=x)
        self.r2_metric(y_pred_inv.flatten(), y_true_inv.flatten())

    def _build_epoch_metrics(
        self,
        avg_loss: float,
        n_samples: int,
        predictions: torch.Tensor | None = None,
    ) -> EpochMetrics:
        mse = self.mse_at.compute()
        mae = self.mae_at.compute()
        r2 = self.r2_metric.compute().item()

        total_n_mse = (self.mse_at.n_b + self.mse_at.n_m + self.mse_at.n_p).clamp(min=1)
        overall_mse = ((self.mse_at.err_sum_b + self.mse_at.err_sum_m + self.mse_at.err_sum_p) / total_n_mse).item()

        total_n_mae = (self.mae_at.n_b + self.mae_at.n_m + self.mae_at.n_p).clamp(min=1)
        overall_mae = ((self.mae_at.err_sum_b + self.mae_at.err_sum_m + self.mae_at.err_sum_p) / total_n_mae).item()

        return EpochMetrics(
            loss=avg_loss,
            mse=overall_mse,
            mae=overall_mae,
            adj_r2=self._adjusted_r2(r2, n_samples),
            mse_budget=mse["mse@budget"].item(),
            mse_mid=mse["mse@mid"].item(),
            mse_premium=mse["mse@premium"].item(),
            mae_budget=mae["mae@budget"].item(),
            mae_mid=mae["mae@mid"].item(),
            mae_premium=mae["mae@premium"].item(),
            predictions=predictions,
        )

    @staticmethod
    def _unpack_batch(batch, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) == 3:
            X, y, aux = batch
            return X.to(device), y.to(device), aux.to(device)
        X, y = batch
        X_dev = X.to(device)
        return X_dev, y.to(device), X_dev

    def train_step(self, dataloader, model, optimizer, loss_fn) -> EpochMetrics:
        model.train()
        self._reset_metrics()
        total_loss = 0.0
        n_seen = 0

        for batch in dataloader:
            X, y, aux = self._unpack_batch(batch, self.device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y, aux)
            loss.backward()
            optimizer.step()

            bs = len(y)
            total_loss += loss.item() * bs
            n_seen += bs
            self._update_metrics(pred.detach(), y, aux)

        n_samples = max(n_seen, 1)
        return self._build_epoch_metrics(avg_loss=total_loss / n_samples, n_samples=n_samples)

    def test_step(self, dataloader, model, loss_fn) -> EpochMetrics:
        model.eval()
        self._reset_metrics()
        total_loss = 0.0
        n_seen = 0
        preds_all = []

        with torch.no_grad():
            for batch in dataloader:
                X, y, aux = self._unpack_batch(batch, self.device)
                pred = model(X)

                bs = len(y)
                pred_inv = self._inverse(pred)
                y_inv = self._inverse(y)
                total_loss += loss_fn(pred_inv, y_inv, aux).item() * bs
                n_seen += bs
                self._update_metrics(pred, y, aux)
                preds_all.append(pred.cpu())

        n_samples = max(n_seen, 1)
        return self._build_epoch_metrics(
            avg_loss=total_loss / n_samples,
            n_samples=n_samples,
            predictions=torch.cat(preds_all),
        )

    def evaluate_dataloader(self, dataloader, model, loss_fn) -> dict[str, float]:
        return self.test_step(dataloader, model, loss_fn).to_dict()

    def predict_dataloader(self, dataloader, model, clip_preds: bool = True) -> np.ndarray:
        model.eval()
        preds_all = []
        with torch.no_grad():
            for batch in dataloader:
                X = batch[0].to(self.device)
                pred = model(X)
                preds_all.append(pred.cpu())
        preds_tensor = torch.cat(preds_all)
        preds_inv = self._inverse(preds_tensor).detach().cpu().numpy().flatten()
        if clip_preds:
            return np.clip(preds_inv, 0.0, 15.0)
        return preds_inv

    @staticmethod
    def _record_epoch_history(
        history: dict[str, list[float]],
        prefix: str,
        metrics: EpochMetrics,
    ) -> None:
        for key, value in metrics.to_dict().items():
            history[f"{prefix}_{key}"].append(value)

    @staticmethod
    def _print_epoch_summary(
        epoch: int,
        current_lr: float,
        train_metrics: EpochMetrics,
        valid_metrics: EpochMetrics,
    ) -> None:
        print(
            f"Epoch {epoch:4d} | Current LR: {current_lr:.6f}\n"
            f"   Train  Loss={train_metrics.loss:.4f} | "
            f"MSE@B={train_metrics.mse_budget:.4f} MSE@M={train_metrics.mse_mid:.4f} MSE@P={train_metrics.mse_premium:.4f} | "
            f"MAE@B={train_metrics.mae_budget:.4f} MAE@M={train_metrics.mae_mid:.4f} MAE@P={train_metrics.mae_premium:.4f} | "
            f"AdjR2={train_metrics.adj_r2:.4f}\n"
            f"   Valid  Loss={valid_metrics.loss:.4f} | "
            f"MSE@B={valid_metrics.mse_budget:.4f} MSE@M={valid_metrics.mse_mid:.4f} MSE@P={valid_metrics.mse_premium:.4f} | "
            f"MAE@B={valid_metrics.mae_budget:.4f} MAE@M={valid_metrics.mae_mid:.4f} MAE@P={valid_metrics.mae_premium:.4f} | "
            f"AdjR2={valid_metrics.adj_r2:.4f}"
        )

    def model_eval(
        self,
        train_dataloader,
        test_dataloader,
        model,
        optimizer,
        loss_fn,
        scheduler=None,
        num_epochs: int = 100,
        patience: int = 10,
        tolerance: float = 1e-6,
        monitor: str = "mse",
        verbose: bool = True,
    ):
        best_state = copy.deepcopy(model.state_dict())
        mode = "max" if monitor == "adj_r2" else "min"
        best_metric = -float("inf") if mode == "max" else float("inf")
        counter = 0
        results: dict[str, list[float]] = defaultdict(list)

        for epoch in tqdm(range(num_epochs), desc=model.__class__.__name__, disable=not verbose):
            train_metrics = self.train_step(train_dataloader, model, optimizer, loss_fn)
            valid_metrics = self.test_step(test_dataloader, model, loss_fn)

            current_metric = valid_metrics.get_monitored_value(monitor)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_metric)
                else:
                    scheduler.step()

            self._record_epoch_history(results, "train", train_metrics)
            self._record_epoch_history(results, "test", valid_metrics)

            if verbose and epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                self._print_epoch_summary(epoch, current_lr, train_metrics, valid_metrics)

            improved = (
                current_metric > best_metric + tolerance
                if mode == "max"
                else current_metric < best_metric - tolerance
            )
            if improved:
                best_metric = current_metric
                counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        return dict(results), best_state
