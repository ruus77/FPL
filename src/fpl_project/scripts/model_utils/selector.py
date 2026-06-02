import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from catboost import CatBoostRegressor, Pool


class ModelSelector:

    def __init__(self, random_state: int = 77, scoring: str = "mse"):
        self.random_state = random_state
        self.scoring = scoring

    @staticmethod
    def metrics_report(y_pred: np.ndarray, y_true: np.ndarray):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return mse, mae, r2

    def params_search(self, models, models_names, params_grid, X_train, y_train, cv: int = 5,
                      scoring: str | None = None, n_iter: int = 20, cat_features: list[str] | None = None):

        scoring = scoring if scoring else self.scoring

        results_list = []
        best_models_map = {}

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
                random_search.fit(
                    X_train,
                    y_train,
                    cat_features=cat_features
                )
            else:
                random_search.fit(X_train, y_train)

            best_models_map[name] = random_search.best_estimator_
            cv_score = random_search.best_score_

            y_train_pred = random_search.best_estimator_.predict(X_train)
            metrics = self.metrics_report(y_true=y_train,
                                          y_pred=y_train_pred)
            row = {
                "data": "train",
                "name": name,
                f"cv_mean_{scoring}": cv_score,
                "best_params": random_search.best_params_,
                "mse": metrics[0],
                "mae": metrics[1],
                "r2": metrics[2]}

            results_list.append(row)

        return pd.DataFrame(results_list).sort_values(by=f"cv_mean_{scoring}", ascending=False), best_models_map

    def evaluate(self, trained_models_map, X_eval, y_eval):
        results_list = []
        y_preds = {}

        for name, model in trained_models_map.items():
            y_pred = model.predict(X_eval)
            y_preds[name] = y_pred

            metrics = self.metrics_report(y_pred=y_pred, y_true=y_eval)

            row = {
                "data": "test",
                "model_name": name,
                "mse": metrics[0],
                "mae": metrics[1],
                "r2": metrics[2]}
            results_list.append(row)

        df_results = pd.DataFrame(results_list)

        sort_column = self.scoring if self.scoring in df_results.columns else "mse"

        return df_results.sort_values(by=sort_column, ascending=False), y_preds