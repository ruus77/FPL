import copy
import random
from collections import defaultdict
import numpy as np
import sklearn
import torch
from torchmetrics import R2Score
from tqdm.auto import tqdm

from .custom_metrics import CustomMAE, CustomMSE


def set_seed(seed: int = 77):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:

    def __init__(
        self,
        device: str,
        num_features: int,
        random_state: int = 77,
        p_low: float = 0.3,
        p_high: float = 0.8,
        value_idx: int = 0,
        model_type: str = "mlp",
        scaler: sklearn.preprocessing.StandardScaler = None,
    ):

        set_seed(random_state)

        self.device = device
        self.num_features = num_features

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
        self.scaler = scaler

    def _adjusted_r2(self, r2, n_samples):
        if n_samples <= self.num_features + 1:
            return 0.0

        return 1 - (
            (1 - r2) * (n_samples - 1) / (n_samples - self.num_features - 1)
        )

    def _inverse(self, x: torch.Tensor):
        if self.scaler is None:
            return x

        x = x.detach().cpu().numpy()

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        x = self.scaler.inverse_transform(x)
        return torch.tensor(x, device=self.device)

    def set_model_config(self, value_idx: int, model_type: str):
        self.mse_at.set_config(value_idx=value_idx, model_type=model_type)
        self.mae_at.set_config(value_idx=value_idx, model_type=model_type)

    def _reset_metrics(self):
        self.mse_at.reset()
        self.mae_at.reset()
        self.r2_metric.reset()

    def _update_metrics(self, y_pred, y_true, x):
        y_pred = self._inverse(y_pred)
        y_true = self._inverse(y_true)

        self.mse_at(preds=y_pred, target=y_true, x=x)
        self.mae_at(preds=y_pred, target=y_true, x=x)
        self.r2_metric(y_pred.flatten(), y_true.flatten())

    def _compute_metrics(self, n_samples):
        mse = self.mse_at.compute()
        mae = self.mae_at.compute()
        r2 = self.r2_metric.compute().item()

        total_n_mse = (self.mse_at.n_b + self.mse_at.n_m + self.mse_at.n_p).clamp(min=1)
        overall_mse = ((self.mse_at.err_sum_b + self.mse_at.err_sum_m + self.mse_at.err_sum_p) / total_n_mse).item()

        total_n_mae = (self.mae_at.n_b + self.mae_at.n_m + self.mae_at.n_p).clamp(min=1)
        overall_mae = ((self.mae_at.err_sum_b + self.mae_at.err_sum_m + self.mae_at.err_sum_p) / total_n_mae).item()

        return (
            overall_mse,
            overall_mae,
            mse["mse@budget"].item(),
            mse["mse@mid"].item(),
            mse["mse@premium"].item(),
            mae["mae@budget"].item(),
            mae["mae@mid"].item(),
            mae["mae@premium"].item(),
            self._adjusted_r2(r2, n_samples),
        )

    @staticmethod
    def _unpack_batch(batch, device: str):
        if len(batch) == 3:
            X, y, aux = batch
            return X.to(device), y.to(device), aux.to(device)
        X, y = batch
        X = X.to(device)
        return X, y.to(device), X

    def train_step(self, dataloader, model, optimizer, loss_fn):
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
        return total_loss / n_samples, *self._compute_metrics(n_samples)

    def test_step(self, dataloader, model, loss_fn):
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
        return (
            total_loss / n_samples,
            *self._compute_metrics(n_samples),
            torch.cat(preds_all),
        )

    def evaluate_dataloader(self, dataloader, model, loss_fn) -> dict[str, float]:
        (
            loss,
            mse,
            mae,
            mse_b,
            mse_m,
            mse_p,
            mae_b,
            mae_m,
            mae_p,
            adj_r2,
            _,
        ) = self.test_step(dataloader, model, loss_fn)
        return {
            "loss": loss,
            "mse": mse,
            "mae": mae,
            "adj_r2": adj_r2,
            "mse@budget": mse_b,
            "mse@mid": mse_m,
            "mse@premium": mse_p,
            "mae@budget": mae_b,
            "mae@mid": mae_m,
            "mae@premium": mae_p,
        }

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
        results = defaultdict(list)

        for epoch in tqdm(range(num_epochs), desc=model.__class__.__name__, disable=not verbose):

            (
                tr_loss,
                tr_mse,
                tr_mae,
                tr_mse_b,
                tr_mse_m,
                tr_mse_p,
                tr_mae_b,
                tr_mae_m,
                tr_mae_p,
                tr_adj_r2,
            ) = self.train_step(train_dataloader, model, optimizer, loss_fn)

            (
                ts_loss,
                ts_mse,
                ts_mae,
                ts_mse_b,
                ts_mse_m,
                ts_mse_p,
                ts_mae_b,
                ts_mae_m,
                ts_mae_p,
                ts_adj_r2,
                _,
            ) = self.test_step(test_dataloader, model, loss_fn)

            monitor_map = {
                "mse": ts_mse,
                "mae": ts_mae,
                "adj_r2": ts_adj_r2,
                "loss": ts_loss,
            }
            if monitor not in monitor_map:
                raise ValueError(f"Nieznana metryka monitorowania: {monitor}")
            current_metric = monitor_map[monitor]

            if scheduler is not None:
                if isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step(current_metric)
                else:
                    scheduler.step()

            results["train_loss"].append(tr_loss)
            results["test_loss"].append(ts_loss)
            results["train_mse"].append(tr_mse)
            results["test_mse"].append(ts_mse)
            results["train_mae"].append(tr_mae)
            results["test_mae"].append(ts_mae)

            results["train_mse@budget"].append(tr_mse_b)
            results["test_mse@budget"].append(ts_mse_b)
            results["train_mse@mid"].append(tr_mse_m)
            results["test_mse@mid"].append(ts_mse_m)
            results["train_mse@premium"].append(tr_mse_p)
            results["test_mse@premium"].append(ts_mse_p)

            results["train_mae@budget"].append(tr_mae_b)
            results["test_mae@budget"].append(ts_mae_b)
            results["train_mae@mid"].append(tr_mae_m)
            results["test_mae@mid"].append(ts_mae_m)
            results["train_mae@premium"].append(tr_mae_p)
            results["test_mae@premium"].append(ts_mae_p)

            results["train_adj_r2"].append(tr_adj_r2)
            results["test_adj_r2"].append(ts_adj_r2)

            if verbose and epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:4d} | Current LR: {current_lr:.6f}\n"
                    f"   Train  Loss={tr_loss:.4f} | "
                    f"MSE@B={tr_mse_b:.4f} MSE@M={tr_mse_m:.4f} MSE@P={tr_mse_p:.4f} | "
                    f"MAE@B={tr_mae_b:.4f} MAE@M={tr_mae_m:.4f} MAE@P={tr_mae_p:.4f} | "
                    f"AdjR2={tr_adj_r2:.4f}\n"
                    f"   Valid  Loss={ts_loss:.4f} | "
                    f"MSE@B={ts_mse_b:.4f} MSE@M={ts_mse_m:.4f} MSE@P={ts_mse_p:.4f} | "
                    f"MAE@B={ts_mae_b:.4f} MAE@M={ts_mae_m:.4f} MAE@P={ts_mae_p:.4f} | "
                    f"AdjR2={ts_adj_r2:.4f}"
                )

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
