import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import torch
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from fpl_project.scripts.model_utils.custom_metrics import CustomMAE, CustomMSE
from torchmetrics import R2Score



def initialize_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        try:
            import cuml.accel
            cuml.accel.activate()
        except ImportError:
            pass
            
    return device


class ModelSelector:

    def __init__(self, p_low: float, p_high: float, value_idx: int, random_state: int = 77, scoring: str = "neg_mean_squared_error"):
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

    def params_search(self, models, models_names, params_grid, X_train, y_train, cv: int = 5,
                      scoring: str | None = None, n_iter: int = 20, cat_features: list[str] | None = None):

        scoring = scoring if scoring else self.scoring

        results_list = []
        best_models_map = {}

        X = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
        y = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
        columns_list = list(X_train.columns) if hasattr(X_train, "columns") else []

        for model, name, grid in zip(models, models_names, params_grid):
            random_search = RandomizedSearchCV(cv=TimeSeriesSplit(n_splits=cv),
                                               n_iter=n_iter,
                                               estimator=model,
                                               scoring=scoring,
                                               param_distributions=grid,
                                               verbose=1,
                                               error_score="raise",
                                               random_state=self.random_state,
                                               refit=True)

            if isinstance(model, CatBoostRegressor):
                if cat_features:
                    cat_indices = [columns_list.index(col) for col in cat_features if col in columns_list]
                else:
                    cat_indices = None

                random_search.fit(
                    X,
                    y,
                    cat_features=cat_indices 
                )
            else:
                random_search.fit(X, y)

            best_models_map[name] = random_search.best_estimator_
            cv_score = random_search.best_score_

            if isinstance(model, CatBoostRegressor):
                eval_pool = self._to_pool(X, y, cat_feat=cat_indices)
                y_train_pred = random_search.best_estimator_.predict(eval_pool)
            else:
                y_train_pred = random_search.best_estimator_.predict(X)
                
            mse_dict, mae_dict, r2_val = self.metrics_report(y_pred=y_train_pred, y_true=y, x=X)
            
            row = {
                "data": "train",
                "name": name,
                f"cv_mean_{scoring}": cv_score,
                "best_params": random_search.best_params_,
                "mse@budget": mse_dict["mse@budget"].item(),
                "mse@mid": mse_dict["mse@mid"].item(),
                "mse@premium": mse_dict["mse@premium"].item(),
                "mae@budget": mae_dict["mae@budget"].item(),
                "mae@mid": mae_dict["mae@mid"].item(),
                "mae@premium": mae_dict["mae@premium"].item(),
                "r2": r2_val
            }

            results_list.append(row)

        return pd.DataFrame(results_list).sort_values(by=f"cv_mean_{scoring}", ascending=False), best_models_map

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
                "r2": r2_val
            }
            results_list.append(row)

        df_results = pd.DataFrame(results_list)
        sort_column = f"cv_mean_{self.scoring}" if f"cv_mean_{self.scoring}" in df_results.columns else "mse@premium"

        return df_results.sort_values(by=sort_column, ascending=False, errors='ignore'), y_preds