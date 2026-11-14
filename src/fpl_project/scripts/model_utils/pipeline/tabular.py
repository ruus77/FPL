"""Stage 2a: tune and fit tabular base models (GLM, XGBoost) in MSE and FPLLoss variants."""

from glum import GeneralizedLinearRegressor
from glum._glm import get_family, get_link
import numpy as np
import xgboost as xgb

from fpl_project.scripts import config
from fpl_project.scripts.model_utils.pipeline.containers import (
    PredictionSplit,
    PreparedData,
    TabularTuningResult,
)
from fpl_project.scripts.model_utils.selector import ModelSelector, make_gw_time_series_splits
from fpl_project.scripts.model_utils.trainer import set_seed


def _print_glum_defaults() -> None:
    nb_default = get_family("negative.binomial")
    tw_default = get_family("tweedie")
    auto_links = {
        fam_name: type(get_link("auto", get_family(fam_name))).__name__
        for fam_name in ["poisson", "negative.binomial", "tweedie", "normal"]
    }
    print("=" * 90)
    print("DOMYŚLNE PARAMETRY W BIBLIOTECE GLUM (glum._glm.get_family / get_link):")
    print(f'  family="negative.binomial" (bez parametru) -> theta = {float(nb_default.theta):.1f} ({type(nb_default).__name__})')
    print(f'  family="tweedie"           (bez parametru) -> power = {float(tw_default.power):.1f} ({type(tw_default).__name__})')
    print(
        '  link="auto" dla rodzin                     -> '
        + ", ".join(f"{k}: {v}" for k, v in auto_links.items())
    )
    print("=" * 90)


def tune_tabular_models(data: PreparedData, device: str) -> TabularTuningResult:
    """Tune and fit GLM and XGBoost in MSE and FPLLoss variants using TimeSeriesSplit CV."""
    _print_glum_defaults()

    cv_splits = make_gw_time_series_splits(
        seasons=data.raw.X_train["season"],
        gws=data.raw.X_train["gw"],
        n_splits=config.CV_SPLITS,
    )

    val_idx_tab = list(data.tabular.X_train.columns).index("num__value_ema_3")
    selector = ModelSelector(
        p_low=data.weights.p_low,
        p_high=data.weights.p_high,
        value_idx=val_idx_tab,
        random_state=config.SEED,
    )
    price_train_raw = data.raw.X_train["value_ema_3"].to_numpy(dtype=np.float64)

    preds_valid: dict[str, np.ndarray] = {}
    preds_test: dict[str, np.ndarray] = {}
    best_params_log: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}

    for variant, sw_train in [("mse", None), ("fplloss", data.weights.w_train)]:
        print("\n" + "=" * 90)
        print(f"STROJENIE I TRENING MODELI TABELARYCZNYCH (GLM, XGBOOST) — WARIANT: {variant.upper()}")
        print("=" * 90)
        set_seed(config.SEED)

        glm_est = GeneralizedLinearRegressor(random_state=config.SEED, max_iter=500)
        xgb_est = xgb.XGBRegressor(
            random_state=config.SEED,
            objective="reg:squarederror",
            eval_metric="rmse",
            device=device,
            n_jobs=-1,
        )

        cv_df, best_map = selector.params_search(
            models=[glm_est, xgb_est],
            models_names=["glm", "xgboost"],
            params_grid=[config.GLM_PARAM_GRID, config.XGB_PARAM_GRID],
            X_train=data.tabular.X_train,
            y_train=data.raw.y_train,
            cv=config.CV_SPLITS,
            n_iter=config.N_ITER_SEARCH,
            sample_weight=sw_train,
            cv_splits=cv_splits,
            price_unscaled=price_train_raw,
        )

        for m_name in ["glm", "xgboost"]:
            fitted_model = best_map[m_name]
            row_cv = cv_df[cv_df["name"] == m_name].iloc[0]
            best_params_log[f"{m_name}_{variant}"] = row_cv["best_params"]
            fitted_models[f"{m_name}_{variant}"] = fitted_model
            print(f"  [{m_name.upper()} - {variant}] Najlepsze parametry: {row_cv['best_params']}")

            preds_valid[f"{m_name}_{variant}"] = np.clip(
                fitted_model.predict(data.tabular.X_valid.to_numpy()),
                config.PRED_CLIP_MIN,
                config.PRED_CLIP_MAX,
            )
            preds_test[f"{m_name}_{variant}"] = np.clip(
                fitted_model.predict(data.tabular.X_test.to_numpy()),
                config.PRED_CLIP_MIN,
                config.PRED_CLIP_MAX,
            )

    return TabularTuningResult(
        predictions=PredictionSplit(valid=preds_valid, test=preds_test),
        best_params=best_params_log,
        models=fitted_models,
    )
