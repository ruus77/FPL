"""Correctness checks executed before the experiment (FPLLoss weights, sample_weight routing in CV)."""

from glum import GeneralizedLinearRegressor
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from fpl_project.scripts import config
from fpl_project.scripts.model_utils.losses import FPLLoss, compute_fpl_weights
from fpl_project.scripts.model_utils.selector import ModelSelector, make_gw_time_series_splits
from fpl_project.scripts.model_utils.trainer import set_seed


def _verify_fpl_loss_equivalence() -> None:
    toy_df = pd.DataFrame({
        "value_ema_3": [4.0, 4.5, 5.5, 7.0, 11.5],
        "minutes_ema_3": [0.0, 30.0, 60.0, 85.0, 90.0],
    })
    p_low_t, p_high_t = 4.5, 7.0
    w_np = compute_fpl_weights(
        toy_df,
        p_low=p_low_t,
        p_high=p_high_t,
        w_mid=config.W_MID,
        w_premium=config.W_PREMIUM,
        eps_min=config.EPS_MIN,
    )
    loss_mod = FPLLoss(
        p_low=p_low_t,
        p_high=p_high_t,
        value_idx=0,
        minutes_idx=1,
        model_type="mlp",
        w_mid=config.W_MID,
        w_premium=config.W_PREMIUM,
        eps_min=config.EPS_MIN,
    )
    y_true_t = np.array([1.0, 2.0, 5.0, 8.0, 12.0], dtype=np.float32)
    y_pred_t = np.array([1.5, 1.0, 4.0, 6.0, 10.0], dtype=np.float32)
    aux_t = np.column_stack([
        toy_df["value_ema_3"].to_numpy(dtype=np.float32),
        np.clip(toy_df["minutes_ema_3"].to_numpy(dtype=np.float32) / 90.0, 0.0, 1.0),
    ])
    expected_loss = float(np.mean(w_np * (y_true_t - y_pred_t) ** 2))
    torch_loss_aux = float(
        loss_mod(torch.tensor(y_pred_t), torch.tensor(y_true_t), torch.tensor(aux_t)).item()
    )
    torch_loss_w = float(
        loss_mod(torch.tensor(y_pred_t), torch.tensor(y_true_t), torch.tensor(w_np, dtype=torch.float32)).item()
    )
    assert np.isclose(expected_loss, torch_loss_aux, atol=1e-6)
    assert np.isclose(expected_loss, torch_loss_w, atol=1e-6)
    print(f"[PASS] Test 1: compute_fpl_weights == FPLLoss (loss={expected_loss:.6f})")


def _verify_cv_sample_weight_routing() -> None:
    rng = np.random.default_rng(config.SEED)
    X_toy = rng.normal(size=(120, 4))
    seasons_toy = np.repeat([2223, 2324, 2425], 40)
    gws_toy = np.tile(np.repeat(np.arange(1, 11), 4), 3)
    mask_high = X_toy[:, 0] > 0.0
    y_toy = np.where(
        mask_high,
        3.0 * np.maximum(X_toy[:, 0], 0.0) + 1.0,
        0.5 * np.maximum(X_toy[:, 1], 0.0) + 0.5,
    )
    w_uniform = np.ones_like(y_toy)
    w_skewed = np.where(mask_high, 20.0, 0.1)

    glm_u = GeneralizedLinearRegressor(family="poisson", alpha=0.05)
    glm_w = GeneralizedLinearRegressor(family="poisson", alpha=0.05)
    glm_u.fit(X_toy, y_toy, sample_weight=w_uniform)
    glm_w.fit(X_toy, y_toy, sample_weight=w_skewed)
    assert not np.allclose(glm_u.coef_, glm_w.coef_)

    xgb_u = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=config.SEED, objective="reg:squarederror")
    xgb_w = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=config.SEED, objective="reg:squarederror")
    xgb_u.fit(X_toy, y_toy, sample_weight=w_uniform)
    xgb_w.fit(X_toy, y_toy, sample_weight=w_skewed)
    assert not np.allclose(xgb_u.predict(X_toy), xgb_w.predict(X_toy))

    splits_toy = make_gw_time_series_splits(seasons_toy, gws_toy, n_splits=3)
    sel = ModelSelector(p_low=0.0, p_high=1.0, value_idx=0, random_state=config.SEED)
    score_unw = sel.score_fold(y_toy[:20], xgb_u.predict(X_toy[:20]), sample_weight=None)
    score_wei = sel.score_fold(y_toy[:20], xgb_u.predict(X_toy[:20]), sample_weight=w_skewed[:20])
    assert not np.isclose(score_unw, score_wei)

    res_u, map_u = sel.params_search(
        models=[GeneralizedLinearRegressor(family="poisson")],
        models_names=["glm_test"],
        params_grid=[{"alpha": [0.001, 0.5], "l1_ratio": [0.0, 0.8]}],
        X_train=X_toy,
        y_train=y_toy,
        cv_splits=splits_toy,
        n_iter=4,
        sample_weight=None,
    )
    res_w, map_w = sel.params_search(
        models=[GeneralizedLinearRegressor(family="poisson")],
        models_names=["glm_test"],
        params_grid=[{"alpha": [0.001, 0.5], "l1_ratio": [0.0, 0.8]}],
        X_train=X_toy,
        y_train=y_toy,
        cv_splits=splits_toy,
        n_iter=4,
        sample_weight=w_skewed,
    )
    assert not np.isclose(
        res_u["cv_mean_neg_mean_squared_error"].iloc[0],
        res_w["cv_mean_neg_weighted_mse"].iloc[0],
    )
    assert not np.allclose(map_u["glm_test"].predict(X_toy), map_w["glm_test"].predict(X_toy))
    print("[PASS] Test 2: sample_weight trafia do fit() oraz oceny foldu w TimeSeriesSplit CV")


def run_verification_tests() -> None:
    """Verify FPLLoss weight equivalence and sample_weight routing in CV."""
    print("=" * 90)
    print("WERYFIKACJA POPRAWNOŚCI (TESTY JEDNOSTKOWE PRZED EKSPERYMENTEM)")
    print("=" * 90)
    set_seed(config.SEED)
    _verify_fpl_loss_equivalence()
    _verify_cv_sample_weight_routing()
    print("=" * 90)
