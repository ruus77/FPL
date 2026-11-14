import numpy as np
import pandas as pd
import sklearn
import sklearn.base
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import torch
from torchmetrics import R2Score

from fpl_project.scripts.model_utils.custom_metrics import CustomMAE, CustomMSE


def initialize_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        try:
            import cuml.accel
            cuml.accel.activate()
        except ImportError:
            pass

    return device


def make_gw_time_series_splits(
    seasons: np.ndarray | pd.Series,
    gws: np.ndarray | pd.Series,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create TimeSeriesSplit fold indices aligned strictly on (season, gw) boundaries.

    Requires data to be sorted globally by (season, gw) so no fold splits inside a gameweek.
    """
    seasons_arr = np.asarray(seasons, dtype=np.int64)
    gws_arr = np.asarray(gws, dtype=np.int64)
    keys = seasons_arr * 100 + gws_arr

    if len(keys) > 1 and np.any(np.diff(keys) < 0):
        raise ValueError("Dane przekazywane do TimeSeriesSplit muszą być posortowane globalnie po (season, gw)!")

    unique_keys = np.unique(keys)
    tss = TimeSeriesSplit(n_splits=n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for tr_gw_idx, val_gw_idx in tss.split(unique_keys):
        tr_keys = unique_keys[tr_gw_idx]
        val_keys = unique_keys[val_gw_idx]
        tr_row_idx = np.flatnonzero(np.isin(keys, tr_keys))
        val_row_idx = np.flatnonzero(np.isin(keys, val_keys))
        # Verify strict boundary between gameweeks
        assert len(np.intersect1d(keys[tr_row_idx], keys[val_row_idx])) == 0
        assert tr_row_idx.max() < val_row_idx.min()
        splits.append((tr_row_idx, val_row_idx))

    return splits


class ModelSelector:

    def __init__(
        self,
        p_low: float,
        p_high: float,
        value_idx: int,
        random_state: int = 77,
        scoring: str = "neg_mean_squared_error",
    ):
        self.random_state = random_state
        self.scoring = scoring

        self.mse_at = CustomMSE(
            p_low=p_low,
            p_high=p_high,
            value_idx=value_idx,
        )
        self.mae_at = CustomMAE(
            p_low=p_low,
            p_high=p_high,
            value_idx=value_idx,
        )
        self.r2_metric = R2Score()

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.to_numpy()
        return torch.tensor(x, dtype=torch.float32)

    def metrics_report(self, y_pred: np.ndarray, y_true: np.ndarray, x: np.ndarray):
        y_pred_t = self._to_tensor(y_pred)
        y_true_t = self._to_tensor(y_true)

        x_safe = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)

        val_idx = self.mse_at.value_idx
        x_safe[:, val_idx] = x[:, val_idx].astype(np.float32)

        x_t = torch.tensor(x_safe, dtype=torch.float32)
        self.mse_at.reset()
        self.mae_at.reset()
        self.r2_metric.reset()

        self.mse_at(preds=y_pred_t, target=y_true_t, x=x_t)
        self.mae_at(preds=y_pred_t, target=y_true_t, x=x_t)
        self.r2_metric(y_pred_t.flatten(), y_true_t.flatten())

        mse_dict = self.mse_at.compute()
        mae_dict = self.mae_at.compute()
        r2_val = self.r2_metric.compute().item()

        return mse_dict, mae_dict, r2_val

    @staticmethod
    def score_fold(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Compute negative MSE (unweighted) or negative FPLLoss weighted MSE on a CV validation fold."""
        y_p = np.asarray(y_pred, dtype=np.float64)
        if not np.all(np.isfinite(y_p)):
            return -float("inf")
        y_p = np.clip(y_p, 0.0, 15.0)
        err_sq = (np.asarray(y_true, dtype=np.float64) - y_p) ** 2
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64)
            return -float(np.mean(w * err_sq))
        return -float(np.mean(err_sq))

    def params_search(
        self,
        models,
        models_names,
        params_grid,
        X_train,
        y_train,
        cv: int = 5,
        scoring: str | None = None,
        n_iter: int = 20,
        sample_weight: np.ndarray | None = None,
        cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
        price_unscaled: np.ndarray | None = None,
    ):
        sklearn.set_config(enable_metadata_routing=True)
        scoring_label = scoring if scoring else (
            "neg_weighted_mse" if sample_weight is not None else self.scoring
        )

        results_list = []
        best_models_map = {}

        X = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
        y = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
        w = np.asarray(sample_weight, dtype=np.float64) if sample_weight is not None else None

        cv_splitter = list(cv_splits) if cv_splits is not None else TimeSeriesSplit(n_splits=cv)
        scorer = make_scorer(self.score_fold, greater_is_better=True)
        if w is not None:
            scorer = scorer.set_score_request(sample_weight=True)

        for model, name, grid in zip(models, models_names, params_grid):
            est = sklearn.base.clone(model)

            if w is not None and hasattr(est, "set_fit_request"):
                est.set_fit_request(sample_weight=True)

            random_search = RandomizedSearchCV(
                cv=cv_splitter,
                n_iter=n_iter,
                estimator=est,
                param_distributions=grid,
                scoring=scorer,
                random_state=self.random_state,
                n_jobs=1,
                verbose=1,
                error_score=-float("inf"),
                refit=True,
            )

            if w is not None:
                random_search.fit(X, y, sample_weight=w)
            else:
                random_search.fit(X, y)

            best_estimator = random_search.best_estimator_
            best_models_map[name] = best_estimator
            best_score = float(random_search.best_score_)
            best_params = random_search.best_params_

            y_train_pred = best_estimator.predict(X)

            if price_unscaled is not None:
                x_for_metrics = np.zeros((X.shape[0], max(X.shape[1], self.mse_at.value_idx + 1)), dtype=np.float32)
                x_for_metrics[:, self.mse_at.value_idx] = np.asarray(price_unscaled, dtype=np.float32)
            else:
                x_for_metrics = X

            mse_dict, mae_dict, r2_val = self.metrics_report(
                y_pred=y_train_pred, y_true=y, x=x_for_metrics
            )

            row = {
                "data": "train",
                "name": name,
                f"cv_mean_{scoring_label}": best_score,
                "best_params": best_params,
                "mse@budget": mse_dict["mse@budget"].item(),
                "mse@mid": mse_dict["mse@mid"].item(),
                "mse@premium": mse_dict["mse@premium"].item(),
                "mae@budget": mae_dict["mae@budget"].item(),
                "mae@mid": mae_dict["mae@mid"].item(),
                "mae@premium": mae_dict["mae@premium"].item(),
                "r2": r2_val,
            }
            results_list.append(row)

        return (
            pd.DataFrame(results_list).sort_values(by=f"cv_mean_{scoring_label}", ascending=False),
            best_models_map,
        )
