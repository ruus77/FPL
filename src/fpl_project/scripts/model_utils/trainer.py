import torch
import torch.nn as nn
import numpy as np
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset, Dataset
from collections import defaultdict
import copy

def set_seed(seed: int = 77):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(77)


class Trainer:
    """
    Uniwersalny trainer dla modeli regresyjnych.

    Kompatybilny z dowolną architekturą nn.Module — MLP, Conv1D, LSTM, hybrydami.
    Model sam obsługuje kształt wejścia w metodzie forward().
    """

    def __init__(self, device: str, random_state: int = 77):
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self.device = device
        self.mse      = MeanSquaredError().to(device)
        self.mae      = MeanAbsoluteError().to(device)
        self.r2_metric = R2Score().to(device)

    def _reset_metrics(self):
        self.mse.reset()
        self.mae.reset()
        self.r2_metric.reset()

    def _update_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.mse(y_pred, y_true)
        self.mae(y_pred, y_true)
        self.r2_metric(y_pred, y_true)

    def _compute_metrics(self) -> tuple[float, float, float]:
        return (
            self.mse.compute().item(),
            self.mae.compute().item(),
            self.r2_metric.compute().item()
        )

    def train_step(self,
                   train_dataloader: DataLoader,
                   model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module) -> tuple[float, float, float, float]:

        model.to(self.device).train()
        self._reset_metrics()
        total_loss = 0.0

        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss   = loss_fn(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.detach().item()
            self._update_metrics(y_pred.detach(), y_batch)

        mse, mae, r2 = self._compute_metrics()
        return total_loss / len(train_dataloader), mse, mae, r2

    def test_step(self,
                  test_dataloader: DataLoader,
                  model: nn.Module,
                  loss_fn: nn.Module) -> tuple[float, float, float, float, torch.Tensor]:

        model.to(self.device).eval()
        self._reset_metrics()
        total_loss = 0.0
        all_preds  = []

        with torch.inference_mode():
            for X_batch, y_batch in test_dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = model(X_batch)
                loss   = loss_fn(y_pred, y_batch)

                total_loss += loss.detach().item()
                self._update_metrics(y_pred, y_batch)
                all_preds.append(y_pred.cpu())

        mse, mae, r2 = self._compute_metrics()
        return total_loss / len(test_dataloader), mse, mae, r2, torch.cat(all_preds)

    def model_eval(self,
                   train_dataloader: DataLoader,
                   test_dataloader: DataLoader,
                   model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module,
                   num_epochs: int = 100,
                   patience: int = 10,
                   tolerance: float = 1e-4) -> tuple[dict, dict]:

        best_state = copy.deepcopy(model.state_dict())
        best_loss  = float("inf")
        counter    = 0
        results    = defaultdict(list)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
        )

        for epoch in tqdm(range(num_epochs), desc=model.__class__.__name__):
            tr_loss, tr_mse, tr_mae, tr_r2 = self.train_step(
                train_dataloader, model, optimizer, loss_fn)
            ts_loss, ts_mse, ts_mae, ts_r2, _ = self.test_step(
                test_dataloader, model, loss_fn)

            scheduler.step(ts_loss)
            lr = optimizer.param_groups[0]["lr"]

            results["train_loss"].append(tr_loss)
            results["test_loss"].append(ts_loss)
            results["train_mse"].append(tr_mse);  results["test_mse"].append(ts_mse)
            results["train_mae"].append(tr_mae);  results["test_mae"].append(ts_mae)
            results["train_r2"].append(tr_r2);    results["test_r2"].append(ts_r2)
            results["lr"].append(lr)

            log_interval = max(1, num_epochs // 10)
            if epoch % log_interval == 0:
                print(
                    f"Epoch {epoch:4d} | LR: {lr:.2e}\n"
                    f"  Train  Loss={tr_loss:.4f}  MSE={tr_mse:.4f}  MAE={tr_mae:.4f}  R2={tr_r2:.4f}\n"
                    f"  Valid  Loss={ts_loss:.4f}  MSE={ts_mse:.4f}  MAE={ts_mae:.4f}  R2={ts_r2:.4f}\n"
                    f"{'─'*75}"
                )

            if ts_loss < best_loss - tolerance:
                best_loss  = ts_loss
                counter    = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping po {epoch} epokach (brak poprawy przez {patience} epok)")
                    break

        return dict(results), best_state
