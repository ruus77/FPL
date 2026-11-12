import numpy as np
import pandas as pd
import sklearn.base
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, TimeSeriesSplit
import torch
from torchmetrics import R2Score
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

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
    def _to_dmatrix(X, y) -> xgb.DMatrix:
        X = X.to_numpy() if hasattr(X, "to_numpy") else X
        y = y.to_numpy() if hasattr(y, "to_numpy") else y
        return xgb.DMatrix(data=X, label=y)

    @staticmethod
    def _to_pool(X, y, cat_feat) -> Pool:
        X = X.to_numpy() if hasattr(X, "to_numpy") else X
        y = y.to_numpy() if hasattr(y, "to_numpy") else y
        return Pool(data=X, label=y, cat_features=cat_feat)

    @staticmethod
    def score_fold(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Compute negative MSE (unweighted) or negative FPLLoss weighted MSE on a CV validation fold."""
        err_sq = (np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)) ** 2
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
        cat_features: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
        cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
        price_unscaled: np.ndarray | None = None,
    ):
        scoring_label = scoring if scoring else (
            "neg_weighted_mse" if sample_weight is not None else self.scoring
        )

        results_list = []
        best_models_map = {}

        X = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
        y = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
        w = np.asarray(sample_weight, dtype=np.float64) if sample_weight is not None else None

        if cv_splits is None:
            tss = TimeSeriesSplit(n_splits=cv)
            splits = list(tss.split(X))
        else:
            splits = list(cv_splits)

        for model, name, grid in zip(models, models_names, params_grid):
            if isinstance(model, CatBoostRegressor):
                model.set_params(allow_writing_files=False)

            candidates = list(ParameterSampler(grid, n_iter=n_iter, random_state=self.random_state))
            print(
                f"Fitting {len(splits)} folds for each of {len(candidates)} candidates, "
                f"totalling {len(splits) * len(candidates)} fits ({name}, scoring={scoring_label})",
                flush=True,
            )

            best_score = -float("inf")
            best_params = None

            for cand_params in candidates:
                fold_scores = []
                valid_candidate = True
                for tr_idx, val_idx in splits:
                    est = sklearn.base.clone(model)
                    est.set_params(**cand_params)

                    X_tr, y_tr = X[tr_idx], y[tr_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    try:
                        if w is not None:
                            est.fit(X_tr, y_tr, sample_weight=w[tr_idx])
                        else:
                            est.fit(X_tr, y_tr)
                        y_val_pred = np.asarray(est.predict(X_val), dtype=np.float64)
                        if not np.all(np.isfinite(y_val_pred)):
                            valid_candidate = False
                            break
                        y_val_pred = np.clip(y_val_pred, 0.0, 15.0)
                        w_val = w[val_idx] if w is not None else None
                        fold_scores.append(self.score_fold(y_val, y_val_pred, sample_weight=w_val))
                    except Exception:
                        valid_candidate = False
                        break

                if valid_candidate and fold_scores:
                    mean_score = float(np.mean(fold_scores))
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = cand_params

            if best_params is None:
                best_params = candidates[0]

            best_estimator = sklearn.base.clone(model)
            best_estimator.set_params(**best_params)
            if w is not None:
                best_estimator.fit(X, y, sample_weight=w)
            else:
                best_estimator.fit(X, y)

            best_models_map[name] = best_estimator
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

    def evaluate(self, trained_models_map, X_eval, y_eval):
        results_list = []
        y_preds = {}

        X_eval_np = X_eval.to_numpy() if hasattr(X_eval, "to_numpy") else X_eval
        y_eval_np = y_eval.to_numpy() if hasattr(y_eval, "to_numpy") else y_eval

        for name, model in trained_models_map.items():
            if isinstance(model, CatBoostRegressor):
                eval_pool = self._to_pool(X_eval_np, y_eval_np, cat_feat=model.get_cat_features())
                y_pred = model.predict(eval_pool)
            else:
                y_pred = model.predict(X_eval_np)

            y_preds[name] = y_pred

            mse_dict, mae_dict, r2_val = self.metrics_report(y_pred=y_pred, y_true=y_eval_np, x=X_eval_np)

            row = {
                "data": "test",
                "model_name": name,
                "mse@budget": mse_dict["mse@budget"].item(),
                "mse@mid": mse_dict["mse@mid"].item(),
                "mse@premium": mse_dict["mse@premium"].item(),
                "mae@budget": mae_dict["mae@budget"].item(),
                "mae@mid": mae_dict["mae@mid"].item(),
                "mae@premium": mae_dict["mae@premium"].item(),
                "r2": r2_val,
            }
            results_list.append(row)

        df_results = pd.DataFrame(results_list)
        sort_column = f"cv_mean_{self.scoring}" if f"cv_mean_{self.scoring}" in df_results.columns else "mse@premium"

        return df_results.sort_values(by=sort_column, ascending=False, errors="ignore"), y_preds
