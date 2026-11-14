"""Stage 3: merge base predictions and fit/evaluate meta-models (NNLS, context Ridge, mean, best single)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from fpl_project.scripts import config
from fpl_project.scripts.model_utils.evaluation import evaluate
from fpl_project.scripts.model_utils.pipeline.containers import (
    MergedPredictions,
    MetaEvaluationResult,
    PredictionSplit,
    PreparedData,
)


def merge_base_predictions(
    data: PreparedData,
    predictions: PredictionSplit,
) -> MergedPredictions:
    """Inner-join predictions of all base models by (code, season, gw)."""
    id_cols = [
        "code", "name", "position", "team", "opponent_team",
        "was_home", "season", "gw", "value_ema_3", "minutes_ema_3",
    ]
    valid_df = data.raw.X_valid[id_cols].copy()
    valid_df["y"] = data.raw.y_valid.to_numpy(dtype=np.float64)

    test_df = data.raw.X_test[id_cols].copy()
    test_df["y"] = data.raw.y_test.to_numpy(dtype=np.float64)

    for col_name, p_arr in predictions.valid.items():
        part_df = pd.DataFrame({
            "code": data.raw.X_valid["code"].values,
            "season": data.raw.X_valid["season"].values,
            "gw": data.raw.X_valid["gw"].values,
            col_name: p_arr,
        })
        valid_df = pd.merge(valid_df, part_df, on=["code", "season", "gw"], how="inner")

    for col_name, p_arr in predictions.test.items():
        part_df = pd.DataFrame({
            "code": data.raw.X_test["code"].values,
            "season": data.raw.X_test["season"].values,
            "gw": data.raw.X_test["gw"].values,
            col_name: p_arr,
        })
        test_df = pd.merge(test_df, part_df, on=["code", "season", "gw"], how="inner")

    print(
        f"Valid po inner join 4 modeli: {len(valid_df)} wierszy (wypadło: {len(data.raw.X_valid) - len(valid_df)}) | "
        f"Test po inner join 4 modeli: {len(test_df)} wierszy (wypadło: {len(data.raw.X_test) - len(test_df)})"
    )
    return MergedPredictions(valid_df=valid_df, test_df=test_df)


def build_context_matrix(
    preds_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    model_cols: list[str],
    p_low: float,
    p_high: float,
) -> pd.DataFrame:
    """Build feature matrix for meta_context with base predictions, context, and interactions."""
    out = {f"pred_{m_col}": preds_df[m_col].to_numpy(dtype=np.float64) for m_col in model_cols}

    pos = meta_df["position"].astype(str).to_numpy()
    price = meta_df["value_ema_3"].to_numpy(dtype=np.float64)
    minutes_norm = np.clip(meta_df["minutes_ema_3"].to_numpy(dtype=np.float64) / 90.0, 0.0, 1.0)

    context_vars: dict[str, np.ndarray] = {
        f"pos_{pos_label}": (pos == pos_label).astype(np.float64)
        for pos_label in ["GK", "DEF", "MID", "FWD"]
    }
    context_vars["tier_budget"] = (price < p_low).astype(np.float64)
    context_vars["tier_mid"] = ((price >= p_low) & (price < p_high)).astype(np.float64)
    context_vars["tier_premium"] = (price >= p_high).astype(np.float64)
    context_vars["minutes_ema_3_norm"] = minutes_norm

    out.update(context_vars)
    for m_col in model_cols:
        p_vec = preds_df[m_col].to_numpy(dtype=np.float64)
        for c_name, c_vec in context_vars.items():
            out[f"pred_{m_col}__x__{c_name}"] = p_vec * c_vec

    return pd.DataFrame(out, index=preds_df.index)


def _append_meta_weights(
    rows: list[dict],
    meta_model_name: str,
    variant: str,
    intercept: float,
    feature_names: list[str],
    coefs: np.ndarray,
) -> None:
    rows.append({
        "meta_model": meta_model_name,
        "variant": variant,
        "feature": "intercept",
        "weight": float(intercept),
    })
    for feat_name, coef_val in zip(feature_names, coefs):
        rows.append({
            "meta_model": meta_model_name,
            "variant": variant,
            "feature": feat_name,
            "weight": float(coef_val),
        })


def fit_meta_models(
    data: PreparedData,
    merged: MergedPredictions,
) -> MetaEvaluationResult:
    """Fit meta-models on validation predictions and evaluate all models on valid and test sets."""
    valid_df = merged.valid_df
    test_df = merged.test_df

    base_models_order = ["glm", "xgboost", "mlp", "conv1d"]
    p_map_base = {
        "glm": data.tabular.n_features,
        "xgboost": data.tabular.n_features,
        "mlp": data.mlp_bundle.n_features,
        "conv1d": data.conv_bundle.n_features,
    }

    metrics_valid_rows: list[dict] = []
    metrics_test_rows: list[dict] = []
    meta_weights_rows: list[dict] = []

    y_val = valid_df["y"].to_numpy(dtype=np.float64)
    y_tst = test_df["y"].to_numpy(dtype=np.float64)
    price_val = valid_df["value_ema_3"].to_numpy(dtype=np.float64)
    price_tst = test_df["value_ema_3"].to_numpy(dtype=np.float64)
    p_low, p_high = data.weights.p_low, data.weights.p_high

    for variant in ["mse", "fplloss"]:
        variant_cols = [f"{m}_{variant}" for m in base_models_order]
        best_val_criterion = float("inf")
        best_single_name = base_models_order[0]
        best_single_col = variant_cols[0]

        for m_name in base_models_order:
            col = f"{m_name}_{variant}"
            p_feat = p_map_base[m_name]

            m_val = evaluate(y_val, valid_df[col].to_numpy(), price_val, data.weights.w_valid, p_low, p_high, p_feat)
            m_tst = evaluate(y_tst, test_df[col].to_numpy(), price_tst, data.weights.w_test, p_low, p_high, p_feat)

            metrics_valid_rows.append({"model": m_name, "variant": variant, "level": "base", **m_val})
            metrics_test_rows.append({"model": m_name, "variant": variant, "level": "base", **m_tst})

            crit = m_val["mse"] if variant == "mse" else m_val["fplloss"]
            if crit < best_val_criterion:
                best_val_criterion = crit
                best_single_name = m_name
                best_single_col = col

        X_meta_val = valid_df[variant_cols].copy()
        X_meta_val.columns = base_models_order
        X_meta_tst = test_df[variant_cols].copy()
        X_meta_tst.columns = base_models_order

        sw_val = data.weights.w_valid if variant == "fplloss" else None

        meta_nnls = LinearRegression(positive=True, fit_intercept=True)
        meta_nnls.fit(X_meta_val, y_val, sample_weight=sw_val)
        p_val_nnls = meta_nnls.predict(X_meta_val)
        p_tst_nnls = meta_nnls.predict(X_meta_tst)
        valid_df[f"meta_nnls_{variant}"] = p_val_nnls
        test_df[f"meta_nnls_{variant}"] = p_tst_nnls
        _append_meta_weights(
            meta_weights_rows, "meta_nnls", variant, meta_nnls.intercept_, base_models_order, meta_nnls.coef_
        )

        X_ctx_val = build_context_matrix(X_meta_val, valid_df, base_models_order, p_low, p_high)
        X_ctx_tst = build_context_matrix(X_meta_tst, test_df, base_models_order, p_low, p_high)
        p_ctx = int(X_ctx_val.shape[1])

        meta_ctx = Ridge(alpha=config.META_RIDGE_ALPHA, positive=True, fit_intercept=True, random_state=config.SEED)
        meta_ctx.fit(X_ctx_val, y_val, sample_weight=sw_val)
        p_val_ctx = meta_ctx.predict(X_ctx_val)
        p_tst_ctx = meta_ctx.predict(X_ctx_tst)
        valid_df[f"meta_context_{variant}"] = p_val_ctx
        test_df[f"meta_context_{variant}"] = p_tst_ctx
        _append_meta_weights(
            meta_weights_rows, "meta_context", variant, meta_ctx.intercept_, list(X_ctx_val.columns), meta_ctx.coef_
        )

        p_val_avg = X_meta_val.mean(axis=1).to_numpy(dtype=np.float64)
        p_tst_avg = X_meta_tst.mean(axis=1).to_numpy(dtype=np.float64)
        valid_df[f"mean_ensemble_{variant}"] = p_val_avg
        test_df[f"mean_ensemble_{variant}"] = p_tst_avg

        p_val_best = valid_df[best_single_col].to_numpy(dtype=np.float64)
        p_tst_best = test_df[best_single_col].to_numpy(dtype=np.float64)
        valid_df[f"best_single_{variant}"] = p_val_best
        test_df[f"best_single_{variant}"] = p_tst_best

        meta_eval_items = [
            ("meta_nnls", p_val_nnls, p_tst_nnls, 4),
            ("meta_context", p_val_ctx, p_tst_ctx, p_ctx),
            ("mean_ensemble", p_val_avg, p_tst_avg, 4),
            (f"best_single({best_single_name})", p_val_best, p_tst_best, p_map_base[best_single_name]),
        ]

        for m_label, pv, pt, p_cnt in meta_eval_items:
            mv = evaluate(y_val, pv, price_val, data.weights.w_valid, p_low, p_high, p_cnt)
            mt = evaluate(y_tst, pt, price_tst, data.weights.w_test, p_low, p_high, p_cnt)
            metrics_valid_rows.append({"model": m_label, "variant": variant, "level": "meta", **mv})
            metrics_test_rows.append({"model": m_label, "variant": variant, "level": "meta", **mt})

    return MetaEvaluationResult(
        metrics_valid=pd.DataFrame(metrics_valid_rows),
        metrics_test=pd.DataFrame(metrics_test_rows),
        meta_weights=pd.DataFrame(meta_weights_rows),
    )
