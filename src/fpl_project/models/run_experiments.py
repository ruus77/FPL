from dataclasses import dataclass
from pathlib import Path
import sys
import time
import warnings

import glum
from glum import GeneralizedLinearRegressor
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import xgboost as xgb

from fpl_project.scripts import config
from fpl_project.scripts.data_utils.features_config import FeaturesConfig
from fpl_project.scripts.data_utils.features_processing import (
    build_and_save_features_engineering_data,
    sort_data,
)
from fpl_project.scripts.model_utils.data import (
    FPLDataPipe,
    scale_target,
    train_test_split,
)
from fpl_project.scripts.model_utils.mlflow_tracker import MlflowTracker
from fpl_project.scripts.model_utils.models import (
    Conv1DRegressor,
    FPLLoss,
    MLP,
    compute_fpl_weights,
    compute_norm_const,
)
from fpl_project.scripts.model_utils.selector import (
    ModelSelector,
    initialize_device,
    make_gw_time_series_splits,
)
from fpl_project.scripts.model_utils.trainer import Trainer, set_seed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


SEED = 77


class MSELossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mse(y_pred.flatten(), y_true.flatten())


class GLMFamilyDistributionSampler:
    """Samples GLM families with equal probability across Poisson, NegativeBinomial(theta), and Tweedie(power).

    - Poisson: 'poisson'
    - Negative Binomial: 'negative.binomial(theta)' with theta tuned in [0.05, 2.5] (including moment estimate)
    - Tweedie: 'tweedie(power)' with power tuned in [1.1, 1.9]
    """

    def __init__(self, theta_mom: float = 1.0):
        self.theta_mom = float(np.clip(theta_mom, 0.05, 5.0))

    def rvs(self, random_state=None):
        rng = np.random.default_rng(random_state)
        choice = int(rng.integers(0, 3))
        if choice == 0:
            return "poisson"
        elif choice == 1:
            if rng.random() < 0.25:
                theta = self.theta_mom
            else:
                theta = float(np.exp(rng.uniform(np.log(0.05), np.log(2.5))))
            return f"negative.binomial({theta:.4f})"
        else:
            power = float(rng.uniform(1.1, 1.9))
            return f"tweedie({power:.4f})"


class AuxSequenceDataset(Dataset):
    """Sequence dataset for Conv1D that also returns unscaled [value_ema_3, minutes_ema_3 / 90] at target step t."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "total_points",
        seq_len: int = 8,
        player_col: str = "code",
        time_col: str = "gw",
        season_col: str = "season",
        history_df: pd.DataFrame | None = None,
    ):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        n_features = len(feature_cols)

        if history_df is not None and not history_df.empty:
            df_indexed = df.copy()
            df_indexed["_orig_pos"] = np.arange(len(df_indexed), dtype=np.int64)
            if target_col not in df_indexed.columns:
                df_indexed[target_col] = 0.0
            else:
                df_indexed[target_col] = df_indexed[target_col].fillna(0.0)

            df_sorted = df_indexed.sort_values([player_col, season_col, time_col]).reset_index(drop=True)

            hist_sorted = history_df.copy().sort_values([player_col, season_col, time_col])
            hist_sorted["_orig_pos"] = -1
            if target_col not in hist_sorted.columns:
                hist_sorted[target_col] = 0.0

            hist_tails = {
                player: grp.tail(seq_len)
                for player, grp in hist_sorted.groupby(player_col, sort=False)
            }

            ordered_sequences: list[tuple[np.ndarray, float, np.ndarray] | None] = [None] * len(df_indexed)

            for player, grp in df_sorted.groupby(player_col, sort=False):
                hist_grp = hist_tails.get(player)
                if hist_grp is not None and not hist_grp.empty:
                    combined = pd.concat([hist_grp, grp], ignore_index=True)
                    n_hist = len(hist_grp)
                else:
                    combined = grp.reset_index(drop=True)
                    n_hist = 0

                feats = combined[feature_cols].values.astype(np.float32)
                target = combined[target_col].values.astype(np.float32)
                raw_price = combined["_raw_price"].values.astype(np.float32)
                raw_min = combined["_raw_min"].values.astype(np.float32)
                orig_positions = combined["_orig_pos"].values

                for t in range(n_hist, len(combined)):
                    if t >= seq_len:
                        window = feats[t - seq_len : t]
                    elif t > 0:
                        pad = np.zeros((seq_len - t, n_features), dtype=np.float32)
                        window = np.vstack([pad, feats[:t]])
                    else:
                        window = np.zeros((seq_len, n_features), dtype=np.float32)

                    label = float(target[t])
                    aux = np.array([raw_price[t], raw_min[t]], dtype=np.float32)
                    ordered_sequences[int(orig_positions[t])] = (window, label, aux)

            self.sequences = ordered_sequences
        else:
            self.sequences: list[tuple[np.ndarray, float, np.ndarray]] = []
            df_sorted = df.copy().sort_values([player_col, season_col, time_col]).reset_index(drop=True)
            for (player, season), grp in df_sorted.groupby([player_col, season_col], sort=False):
                feats = grp[feature_cols].values.astype(np.float32)
                target = grp[target_col].values.astype(np.float32)
                raw_price = grp["_raw_price"].values.astype(np.float32)
                raw_min = grp["_raw_min"].values.astype(np.float32)

                for t in range(seq_len, len(grp)):
                    window = feats[t - seq_len : t]
                    label = float(target[t])
                    aux = np.array([raw_price[t], raw_min[t]], dtype=np.float32)
                    self.sequences.append((window, label, aux))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        window, label, aux = self.sequences[idx]
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor([[label]], dtype=torch.float32),
            torch.tensor(aux, dtype=torch.float32),
        )


def collate_conv1d_with_aux(batch):
    xs, ys, auxs = zip(*batch)
    X = torch.stack(xs)
    y = torch.cat(ys)
    aux = torch.stack(auxs)
    return X, y, aux


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_raw: np.ndarray,
    weights: np.ndarray,
    p_low: float,
    p_high: float,
    p_features: int,
) -> dict[str, float | int]:
    """Compute all required metrics on the original unscaled clipped target y."""
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    price = np.asarray(price_raw, dtype=np.float64).flatten()
    w = np.asarray(weights, dtype=np.float64).flatten()

    n = len(y_t)
    err = y_p - y_t
    sq_err = err ** 2
    abs_err = np.abs(err)

    mse = float(np.mean(sq_err))
    mae = float(np.mean(abs_err))
    r2 = float(r2_score(y_t, y_p))
    if n > p_features + 1:
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - p_features - 1))
    else:
        adj_r2 = 0.0

    fplloss_val = float(np.mean(w * sq_err))

    mask_b = price < p_low
    mask_m = (price >= p_low) & (price < p_high)
    mask_p = price >= p_high

    mse_b = float(np.mean(sq_err[mask_b])) if mask_b.any() else 0.0
    mse_m = float(np.mean(sq_err[mask_m])) if mask_m.any() else 0.0
    mse_p = float(np.mean(sq_err[mask_p])) if mask_p.any() else 0.0

    mae_b = float(np.mean(abs_err[mask_b])) if mask_b.any() else 0.0
    mae_m = float(np.mean(abs_err[mask_m])) if mask_m.any() else 0.0
    mae_p = float(np.mean(abs_err[mask_p])) if mask_p.any() else 0.0

    return {
        "p": int(p_features),
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "adj_r2": adj_r2,
        "fplloss": fplloss_val,
        "mse@budget": mse_b,
        "mse@mid": mse_m,
        "mse@premium": mse_p,
        "mae@budget": mae_b,
        "mae@mid": mae_m,
        "mae@premium": mae_p,
    }


def run_verification_tests() -> None:
    """Run unit tests verifying weight calculations, fold boundaries, and sample_weight routing."""
    print("=" * 90)
    print("WERYFIKACJA POPRAWNOŚCI (TESTY JEDNOSTKOWE PRZED EKSPERYMENTEM)")
    print("=" * 90)
    set_seed(SEED)

    # 1. Test compute_fpl_weights == FPLLoss on unscaled columns
    toy_df = pd.DataFrame({
        "value_ema_3": [4.0, 4.5, 5.5, 7.0, 11.5],
        "minutes_ema_3": [0.0, 30.0, 60.0, 85.0, 90.0],
    })
    p_low_t, p_high_t = 4.5, 7.0
    c_t = compute_norm_const(toy_df, minutes_col="minutes_ema_3", eps_min=0.1)
    w_np = compute_fpl_weights(
        toy_df,
        p_low=p_low_t,
        p_high=p_high_t,
        w_mid=1.25,
        w_premium=2.0,
        eps_min=0.1,
        norm_const=c_t,
    )
    loss_mod = FPLLoss(
        p_low=p_low_t,
        p_high=p_high_t,
        value_idx=0,
        minutes_idx=1,
        model_type="mlp",
        w_mid=1.25,
        w_premium=2.0,
        eps_min=0.1,
        norm_const=c_t,
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
    assert np.isclose(expected_loss, torch_loss_aux, atol=1e-6), "Rozbieżność FPLLoss(aux) vs compute_fpl_weights!"
    assert np.isclose(expected_loss, torch_loss_w, atol=1e-6), "Rozbieżność FPLLoss(w) vs compute_fpl_weights!"
    print(f"[PASS] Test 1: compute_fpl_weights == FPLLoss (loss={expected_loss:.6f})")

    # 2. Test sample_weight routing to fit() and to fold evaluation in ModelSelector
    rng = np.random.default_rng(SEED)
    X_toy = rng.normal(size=(120, 4))
    seasons_toy = np.repeat([2223, 2324, 2425], 40)
    gws_toy = np.tile(np.repeat(np.arange(1, 11), 4), 3)
    # Regime 1 (low weights) vs Regime 2 (high weights)
    mask_high = X_toy[:, 0] > 0.0
    y_toy = np.where(mask_high, 3.0 * np.maximum(X_toy[:, 0], 0.0) + 1.0, 0.5 * np.maximum(X_toy[:, 1], 0.0) + 0.5)
    w_uniform = np.ones_like(y_toy)
    w_skewed = np.where(mask_high, 20.0, 0.1)

    glm_u = GeneralizedLinearRegressor(family="poisson", alpha=0.05)
    glm_w = GeneralizedLinearRegressor(family="poisson", alpha=0.05)
    glm_u.fit(X_toy, y_toy, sample_weight=w_uniform)
    glm_w.fit(X_toy, y_toy, sample_weight=w_skewed)
    assert not np.allclose(glm_u.coef_, glm_w.coef_), "GLM nie uwzględnił sample_weight w fit!"

    xgb_u = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=SEED, objective="reg:squarederror")
    xgb_w = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=SEED, objective="reg:squarederror")
    xgb_u.fit(X_toy, y_toy, sample_weight=w_uniform)
    xgb_w.fit(X_toy, y_toy, sample_weight=w_skewed)
    assert not np.allclose(xgb_u.predict(X_toy), xgb_w.predict(X_toy)), "XGBoost nie uwzględnił sample_weight w fit!"

    # Test fold scoring with sample_weight in ModelSelector
    splits_toy = make_gw_time_series_splits(seasons_toy, gws_toy, n_splits=3)
    sel = ModelSelector(p_low=0.0, p_high=1.0, value_idx=0, random_state=SEED)
    score_unw = sel.score_fold(y_toy[:20], xgb_u.predict(X_toy[:20]), sample_weight=None)
    score_wei = sel.score_fold(y_toy[:20], xgb_u.predict(X_toy[:20]), sample_weight=w_skewed[:20])
    assert not np.isclose(score_unw, score_wei), "Ocena foldu nie uwzględniła sample_weight!"

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
    assert "cv_mean_neg_mean_squared_error" in res_u.columns
    assert "cv_mean_neg_weighted_mse" in res_w.columns
    assert not np.isclose(
        res_u["cv_mean_neg_mean_squared_error"].iloc[0],
        res_w["cv_mean_neg_weighted_mse"].iloc[0],
    ), "params_search zwróciło ten sam wynik CV dla wariantu ważonego i nieważonego!"
    assert not np.allclose(
        map_u["glm_test"].predict(X_toy), map_w["glm_test"].predict(X_toy)
    ), "params_search zwróciło identyczny model dla wariantu ważonego i nieważonego!"
    print("[PASS] Test 2: sample_weight trafia zarówno do fit(), jak i do oceny foldu w TimeSeriesSplit CV")
    print("=" * 90)


def build_context_matrix(
    preds_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    model_cols: list[str],
    p_low: float,
    p_high: float,
) -> pd.DataFrame:
    """Build design matrix for meta_context:
    base predictions + pre-match context (position one-hot, price tier one-hot, minutes_ema_3/90)
    + interaction products (prediction * context_var).
    """
    out = OrderedDict() if False else {}

    # 1. Base model predictions
    for m_col in model_cols:
        out[f"pred_{m_col}"] = preds_df[m_col].to_numpy(dtype=np.float64)

    # 2. Pre-match context variables
    pos = meta_df["position"].astype(str).to_numpy()
    price = meta_df["value_ema_3"].to_numpy(dtype=np.float64)
    minutes_norm = np.clip(meta_df["minutes_ema_3"].to_numpy(dtype=np.float64) / 90.0, 0.0, 1.0)

    context_vars: dict[str, np.ndarray] = {}
    for pos_label in ["GK", "DEF", "MID", "FWD"]:
        context_vars[f"pos_{pos_label}"] = (pos == pos_label).astype(np.float64)

    context_vars["tier_budget"] = (price < p_low).astype(np.float64)
    context_vars["tier_mid"] = ((price >= p_low) & (price < p_high)).astype(np.float64)
    context_vars["tier_premium"] = (price >= p_high).astype(np.float64)
    context_vars["minutes_ema_3_norm"] = minutes_norm

    for c_name, c_vec in context_vars.items():
        out[c_name] = c_vec

    # 3. Interaction products: prediction * context_var
    for m_col in model_cols:
        p_vec = preds_df[m_col].to_numpy(dtype=np.float64)
        for c_name, c_vec in context_vars.items():
            out[f"pred_{m_col}__x__{c_name}"] = p_vec * c_vec

    return pd.DataFrame(out, index=preds_df.index)


def safe_save_path(path: Path) -> Path:
    """Return path if it does not exist, otherwise append a timestamp suffix to avoid overwriting old files."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    ts = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{stem}_{ts}{suffix}")


def log_results_to_mlflow(
    df_metrics_valid: pd.DataFrame,
    df_metrics_test: pd.DataFrame,
    best_params_log: dict[str, dict] | None = None,
    p_low: float = 4.5,
    p_high: float = 6.925,
    c_norm: float = 0.316909,
    artifact_paths: list[Path] | None = None,
) -> None:
    """Log all base and meta-model experiment runs, parameters, valid/test metrics, and artifacts to MLflow."""
    import mlflow

    tracker = MlflowTracker()
    tracker.set_tracking()

    metric_cols = [
        "mse", "mae", "r2", "adj_r2", "fplloss",
        "mse@budget", "mse@mid", "mse@premium",
        "mae@budget", "mae@mid", "mae@premium",
    ]

    for idx in range(len(df_metrics_test)):
        row_t = df_metrics_test.iloc[idx]
        row_v = df_metrics_valid.iloc[idx]
        m_name = str(row_t["model"])
        variant = str(row_t["variant"])
        level = str(row_t["level"])
        run_name = f"{m_name}_{variant}_{level}"

        with tracker.start_run(run_name=run_name):
            mlflow.set_tag("model", m_name)
            mlflow.set_tag("variant", variant)
            mlflow.set_tag("level", level)

            params = {
                "model": m_name,
                "variant": variant,
                "level": level,
                "p_features": int(row_t["p"]),
                "seed": SEED,
                "p_low": p_low,
                "p_high": p_high,
                "norm_const_c": c_norm,
            }
            if best_params_log and f"{m_name}_{variant}" in best_params_log:
                for k, v in best_params_log[f"{m_name}_{variant}"].items():
                    params[f"hp_{k}"] = v
            tracker.log_params(params)

            metrics_to_log = {}
            for mc in metric_cols:
                clean_mc = mc.replace("@", "_")
                metrics_to_log[f"valid_{clean_mc}"] = float(row_v[mc])
                metrics_to_log[f"test_{clean_mc}"] = float(row_t[mc])
            tracker.log_metrics(metrics_to_log)

            if artifact_paths and idx == len(df_metrics_test) - 1:
                for ap in artifact_paths:
                    if ap.exists():
                        mlflow.log_artifact(str(ap))

    print(f"[MLflow] Zalogowano {len(df_metrics_test)} przebiegów eksperymentalnych do MLflow.")


def main() -> None:
    total_start = time.time()
    set_seed(SEED)
    run_verification_tests()

    device = initialize_device()
    print(f"Urządzenie obliczeniowe: {device} | Ziarno losowe (SEED): {SEED}")

    if not config.EDA.exists():
        from fpl_project.data_processing.data_import import fetch_and_save_eda_data
        fetch_and_save_eda_data()
    if not config.FEATURES_ENGINEERING.exists():
        build_and_save_features_engineering_data()

    # 1. Load datasets and clip target to >= 0
    df_fe_raw = sort_data(pd.read_parquet(config.FEATURES_ENGINEERING))
    df_eda_raw = sort_data(pd.read_parquet(config.EDA))

    df_fe_raw["total_points"] = df_fe_raw["total_points"].clip(lower=0.0)
    df_eda_raw["total_points"] = df_eda_raw["total_points"].clip(lower=0.0)

    if "is_played" not in df_fe_raw.columns and "is_played" in df_eda_raw.columns:
        df_fe_raw["is_played"] = df_eda_raw["is_played"].values

    # 2. Inner join on (code, season, gw) between FE and EDA to guarantee identical observations
    key_cols = ["code", "season", "gw"]
    n_fe_before = len(df_fe_raw)
    n_eda_before = len(df_eda_raw)
    common_keys = pd.merge(
        df_fe_raw[key_cols],
        df_eda_raw[key_cols],
        on=key_cols,
        how="inner",
    )
    print(
        f"[Wspólny zbiór wierszy FE ∩ EDA po {key_cols}] "
        f"FE przed={n_fe_before}, EDA przed={n_eda_before}, "
        f"po inner join={len(common_keys)}, odrzucone FE={n_fe_before - len(common_keys)}, "
        f"odrzucone EDA={n_eda_before - len(common_keys)}"
    )

    # Attach unscaled price (value_ema_3 in £m) and minutes (minutes_ema_3 / 90 in [0,1]) to EDA
    fe_aux = df_fe_raw[key_cols + ["value_ema_3", "minutes_ema_3"]].copy()
    fe_aux["_raw_price"] = fe_aux["value_ema_3"].astype(np.float64)
    fe_aux["_raw_min"] = np.clip(fe_aux["minutes_ema_3"].astype(np.float64) / 90.0, 0.0, 1.0)
    df_eda_merged = pd.merge(df_eda_raw, fe_aux[key_cols + ["_raw_price", "_raw_min"]], on=key_cols, how="inner")

    # Sort both datasets globally by (season, gw, code) so TimeSeriesSplit folds never split a gameweek
    df_fe = df_fe_raw.sort_values(["season", "gw", "code"]).reset_index(drop=True)
    df_eda = df_eda_merged.sort_values(["season", "gw", "code"]).reset_index(drop=True)
    assert (df_fe[key_cols].values == df_eda[key_cols].values).all(), "Niezgodność kluczy po sortowaniu (season, gw, code)!"

    # 3. Train / Valid / Test split via data.py: train_test_split
    splits_fe = train_test_split(df_fe)
    splits_eda = train_test_split(df_eda)

    X_train_raw = splits_fe["X_train"].reset_index(drop=True)
    y_train_raw = splits_fe["y_train"].reset_index(drop=True)
    X_valid_raw = splits_fe["X_valid"].reset_index(drop=True)
    y_valid_raw = splits_fe["y_valid"].reset_index(drop=True)
    X_test_raw = splits_fe["X_test"].reset_index(drop=True)
    y_test_raw = splits_fe["y_test"].reset_index(drop=True)

    print(
        f"Rozmiary podziału: train={len(X_train_raw)} (sezony 22/23-24/25), "
        f"valid={len(X_valid_raw)} (sezon 25/26), test={len(X_test_raw)} (rozegrane 26/27)"
    )

    # 4. Compute raw price quantile bounds (0.50, 0.955) and normalization constant c on train
    price_train_raw = X_train_raw["value_ema_3"].to_numpy(dtype=np.float64)
    p_low = float(np.quantile(price_train_raw, 0.50))
    p_high = float(np.quantile(price_train_raw, 0.955))
    c_norm = compute_norm_const(X_train_raw, minutes_col="minutes_ema_3", eps_min=0.1)

    w_train = compute_fpl_weights(
        X_train_raw, p_low=p_low, p_high=p_high, w_mid=1.25, w_premium=2.0, eps_min=0.1, norm_const=c_norm
    )
    w_valid = compute_fpl_weights(
        X_valid_raw, p_low=p_low, p_high=p_high, w_mid=1.25, w_premium=2.0, eps_min=0.1, norm_const=c_norm
    )
    w_test = compute_fpl_weights(
        X_test_raw, p_low=p_low, p_high=p_high, w_mid=1.25, w_premium=2.0, eps_min=0.1, norm_const=c_norm
    )

    print("=" * 90)
    print(f"PARAMETRY FPLLoss (wyliczone na surowych kolumnach train):")
    print(f"  p_low (kwantyl 0.50 value_ema_3)  = {p_low:.6f} mln £")
    print(f"  p_high (kwantyl 0.955 value_ema_3)= {p_high:.6f} mln £")
    print(f"  Stała normalizująca c (norm_const)= {c_norm:.6f}")
    print(
        f"  Wagi w_train: mean={w_train.mean():.6f}, min={w_train.min():.6f}, max={w_train.max():.6f}"
    )
    print("=" * 90)

    # 5. Prepare tabular features (StandardScaler + OneHotEncoder fitted ONLY on train)
    cols_map = FeaturesConfig()
    num_cols = [
        c for c in df_fe.columns if c not in cols_map.id_cols + cols_map.static_cols + cols_map.target
    ]
    cat_cols = cols_map.static_cols

    tab_pipe = FPLDataPipe(num_cols=num_cols, cat_cols=cat_cols, batch_size=128)
    X_train_tab = tab_pipe.preprocessor.fit_transform(X_train_raw)
    X_valid_tab = tab_pipe.preprocessor.transform(X_valid_raw)
    X_test_tab = tab_pipe.preprocessor.transform(X_test_raw)

    p_tabular = int(X_train_tab.shape[1])
    print(f"Liczba cech wejściowych modeli tabelarycznych (GLM, XGBoost, MLP): p = {p_tabular}")

    # 6. Gameweek-aligned TimeSeriesSplit folds on globally sorted train (season, gw)
    cv_splits = make_gw_time_series_splits(
        seasons=X_train_raw["season"],
        gws=X_train_raw["gw"],
        n_splits=5,
    )
    for fold_i, (tr_idx, val_idx) in enumerate(cv_splits, 1):
        tr_last_sgw = (int(X_train_raw["season"].iloc[tr_idx[-1]]), int(X_train_raw["gw"].iloc[tr_idx[-1]]))
        val_first_sgw = (int(X_train_raw["season"].iloc[val_idx[0]]), int(X_train_raw["gw"].iloc[val_idx[0]]))
        val_last_sgw = (int(X_train_raw["season"].iloc[val_idx[-1]]), int(X_train_raw["gw"].iloc[val_idx[-1]]))
        print(
            f"  Fold {fold_i}: train={len(tr_idx):6d} (do {tr_last_sgw}) | "
            f"val={len(val_idx):6d} (od {val_first_sgw} do {val_last_sgw})"
        )

    # 7. Tune & Train GLM and XGBoost in Variant A (mse) and Variant B (fplloss)
    y_tr_np = y_train_raw.to_numpy(dtype=np.float64)
    y_mean = float(np.mean(y_tr_np))
    y_var = float(np.var(y_tr_np))
    theta_mom = max((y_var - y_mean) / max(y_mean ** 2, 1e-6), 0.05)
    print(f"Oszacowane metodą momentów theta (Negative Binomial) na train: theta_mom = {theta_mom:.4f}")

    glm_grid = {
        "family": GLMFamilyDistributionSampler(theta_mom=theta_mom),
        "alpha": stats.loguniform(1e-4, 1.0),
        "l1_ratio": stats.uniform(loc=0.0, scale=1.0),
        "link": ["auto", "log"],
    }
    xgb_grid = {
        "n_estimators": stats.randint(150, 500),
        "max_depth": stats.randint(3, 7),
        "learning_rate": stats.uniform(0.01, 0.05),
        "subsample": stats.uniform(0.7, 0.3),
        "colsample_bytree": stats.uniform(0.6, 0.3),
        "min_child_weight": stats.randint(1, 10),
        "gamma": stats.uniform(0.0, 5.0),
        "reg_alpha": stats.uniform(0.0, 2.0),
        "reg_lambda": stats.uniform(0.5, 3.0),
    }

    val_idx_tab = list(X_train_tab.columns).index("num__value_ema_3")
    selector = ModelSelector(
        p_low=p_low,
        p_high=p_high,
        value_idx=val_idx_tab,
        random_state=SEED,
    )

    preds_valid_dict: dict[str, np.ndarray] = {}
    preds_test_dict: dict[str, np.ndarray] = {}
    best_params_log: dict[str, dict] = {}

    for variant, sw_train in [("mse", None), ("fplloss", w_train)]:
        print("\n" + "=" * 90)
        print(f"STROJENIE I TRENING MODELI TABELARYCZNYCH (GLM, XGBOOST) — WARIANT: {variant.upper()}")
        print("=" * 90)
        set_seed(SEED)

        glm_est = GeneralizedLinearRegressor(random_state=SEED, max_iter=500)
        xgb_est = xgb.XGBRegressor(
            random_state=SEED,
            objective="reg:squarederror",
            eval_metric="rmse",
            device=device,
            n_jobs=-1,
        )

        cv_df, best_map = selector.params_search(
            models=[glm_est, xgb_est],
            models_names=["glm", "xgboost"],
            params_grid=[glm_grid, xgb_grid],
            X_train=X_train_tab,
            y_train=y_train_raw,
            cv=5,
            n_iter=20,
            sample_weight=sw_train,
            cv_splits=cv_splits,
            price_unscaled=price_train_raw,
        )

        for m_name in ["glm", "xgboost"]:
            fitted_model = best_map[m_name]
            row_cv = cv_df[cv_df["name"] == m_name].iloc[0]
            best_params_log[f"{m_name}_{variant}"] = row_cv["best_params"]
            print(f"  [{m_name.upper()} - {variant}] Najlepsze parametry: {row_cv['best_params']}")

            p_val = np.clip(fitted_model.predict(X_valid_tab.to_numpy()), 0.0, 15.0)
            p_tst = np.clip(fitted_model.predict(X_test_tab.to_numpy()), 0.0, 15.0)
            preds_valid_dict[f"{m_name}_{variant}"] = p_val
            preds_test_dict[f"{m_name}_{variant}"] = p_tst

    # 8. Neural Networks (MLP, Conv1D): chronological internal early-stopping split from train
    # Last 8 gameweeks of 2024/25 (season == 2425, gw >= 31) used as internal early-stopping validation set
    print("\n" + "=" * 90)
    print("PRZYGOTOWANIE DANYCH I PODZIAŁU EARLY-STOPPING DLA SIECI NEURONOWYCH (MLP, CONV1D)")
    print("=" * 90)

    es_mask_fe = (X_train_raw["season"] == 2425) & (X_train_raw["gw"] >= 31)
    sub_mask_fe = ~es_mask_fe
    print(
        f"Wewnętrzny podział train dla sieci: train_sub={int(sub_mask_fe.sum())} wierszy, "
        f"early_stopping_val (24/25, gw 31-38)={int(es_mask_fe.sum())} wierszy"
    )

    # Target scaler fitted strictly on train
    splits_fe_scaled, target_scaler_fe = scale_target(splits_fe)
    y_train_scaled = splits_fe_scaled["y_train"].reset_index(drop=True)
    y_valid_scaled = splits_fe_scaled["y_valid"].reset_index(drop=True)
    y_test_scaled = splits_fe_scaled["y_test"].reset_index(drop=True)

    # Auxiliary unscaled [value_ema_3, minutes_ema_3 / 90] tensors for MLP
    def make_aux_tensor(df_part: pd.DataFrame) -> torch.Tensor:
        p_col = df_part["value_ema_3"].to_numpy(dtype=np.float32)
        m_col = np.clip(df_part["minutes_ema_3"].to_numpy(dtype=np.float32) / 90.0, 0.0, 1.0)
        return torch.tensor(np.column_stack([p_col, m_col]), dtype=torch.float32)

    X_tr_sub_t = torch.tensor(X_train_tab.loc[sub_mask_fe].to_numpy(), dtype=torch.float32)
    y_tr_sub_t = torch.tensor(y_train_scaled.loc[sub_mask_fe].to_numpy().reshape(-1, 1), dtype=torch.float32)
    aux_tr_sub_t = make_aux_tensor(X_train_raw.loc[sub_mask_fe])

    X_es_val_t = torch.tensor(X_train_tab.loc[es_mask_fe].to_numpy(), dtype=torch.float32)
    y_es_val_t = torch.tensor(y_train_scaled.loc[es_mask_fe].to_numpy().reshape(-1, 1), dtype=torch.float32)
    aux_es_val_t = make_aux_tensor(X_train_raw.loc[es_mask_fe])

    X_val_t = torch.tensor(X_valid_tab.to_numpy(), dtype=torch.float32)
    y_val_t = torch.tensor(y_valid_scaled.to_numpy().reshape(-1, 1), dtype=torch.float32)
    aux_val_t = make_aux_tensor(X_valid_raw)

    X_tst_t = torch.tensor(X_test_tab.to_numpy(), dtype=torch.float32)
    y_tst_t = torch.tensor(y_test_scaled.to_numpy().reshape(-1, 1), dtype=torch.float32)
    aux_tst_t = make_aux_tensor(X_test_raw)

    batch_size_nn = 128
    mlp_train_sub_dl = DataLoader(
        TensorDataset(X_tr_sub_t, y_tr_sub_t, aux_tr_sub_t), batch_size=batch_size_nn, shuffle=False
    )
    mlp_es_val_dl = DataLoader(
        TensorDataset(X_es_val_t, y_es_val_t, aux_es_val_t), batch_size=batch_size_nn, shuffle=False
    )
    mlp_valid_dl = DataLoader(
        TensorDataset(X_val_t, y_val_t, aux_val_t), batch_size=batch_size_nn, shuffle=False
    )
    mlp_test_dl = DataLoader(
        TensorDataset(X_tst_t, y_tst_t, aux_tst_t), batch_size=batch_size_nn, shuffle=False
    )

    # Prepare sequential data for Conv1D from eda.parquet
    seq_feature_cols = [
        c
        for c in cols_map.fpl_cols + cols_map.match_context_cols + cols_map.market_cols + cols_map.betting_cols
        if c in df_eda.columns and c != "total_points"
    ]
    seq_feature_cols = list(dict.fromkeys(seq_feature_cols))
    p_conv1d = len(seq_feature_cols)
    seq_len = 8
    print(f"Liczba kanałów wejściowych dla Conv1D (z eda.parquet): p = {p_conv1d}, seq_len = {seq_len}")

    splits_eda_scaled, target_scaler_eda = scale_target(splits_eda)

    def merge_eda_xy(X_df: pd.DataFrame, y_ser: pd.Series) -> pd.DataFrame:
        out_df = X_df.copy().reset_index(drop=True)
        out_df["total_points"] = y_ser.to_numpy(dtype=np.float32)
        return out_df

    eda_train_df = merge_eda_xy(splits_eda_scaled["X_train"], splits_eda_scaled["y_train"])
    eda_valid_df = merge_eda_xy(splits_eda_scaled["X_valid"], splits_eda_scaled["y_valid"])
    eda_test_df = merge_eda_xy(splits_eda_scaled["X_test"], splits_eda_scaled["y_test"])

    # Fit StandardScaler on sequential features strictly on train
    seq_scaler = StandardScaler()
    eda_train_df[seq_feature_cols] = seq_scaler.fit_transform(eda_train_df[seq_feature_cols])
    eda_valid_df[seq_feature_cols] = seq_scaler.transform(eda_valid_df[seq_feature_cols])
    eda_test_df[seq_feature_cols] = seq_scaler.transform(eda_test_df[seq_feature_cols])

    es_mask_eda = (eda_train_df["season"] == 2425) & (eda_train_df["gw"] >= 31)
    eda_train_sub_df = eda_train_df.loc[~es_mask_eda].reset_index(drop=True)
    eda_es_val_df = eda_train_df.loc[es_mask_eda].reset_index(drop=True)

    conv_train_sub_ds = AuxSequenceDataset(
        eda_train_sub_df, seq_feature_cols, seq_len=seq_len, history_df=None
    )
    conv_es_val_ds = AuxSequenceDataset(
        eda_es_val_df, seq_feature_cols, seq_len=seq_len, history_df=eda_train_sub_df
    )
    conv_valid_ds = AuxSequenceDataset(
        eda_valid_df, seq_feature_cols, seq_len=seq_len, history_df=eda_train_df
    )
    conv_test_ds = AuxSequenceDataset(
        eda_test_df,
        seq_feature_cols,
        seq_len=seq_len,
        history_df=pd.concat([eda_train_df, eda_valid_df], ignore_index=True),
    )

    print(
        f"[Conv1D SequenceDataset] train_sub={len(conv_train_sub_ds)} "
        f"(pominięto {len(eda_train_sub_df) - len(conv_train_sub_ds)} wierszy rozgrzewki okna seq_len={seq_len}), "
        f"es_val={len(conv_es_val_ds)}, valid={len(conv_valid_ds)}, test={len(conv_test_ds)}"
    )
    assert len(conv_valid_ds) == len(X_valid_raw), "Niezgodność liczby wierszy valid dla Conv1D!"
    assert len(conv_test_ds) == len(X_test_raw), "Niezgodność liczby wierszy test dla Conv1D!"

    conv_train_sub_dl = DataLoader(
        conv_train_sub_ds,
        batch_size=batch_size_nn,
        shuffle=False,
        collate_fn=collate_conv1d_with_aux,
        drop_last=True,
    )
    conv_es_val_dl = DataLoader(
        conv_es_val_ds,
        batch_size=batch_size_nn,
        shuffle=False,
        collate_fn=collate_conv1d_with_aux,
    )
    conv_valid_dl = DataLoader(
        conv_valid_ds,
        batch_size=batch_size_nn,
        shuffle=False,
        collate_fn=collate_conv1d_with_aux,
    )
    conv_test_dl = DataLoader(
        conv_test_ds,
        batch_size=batch_size_nn,
        shuffle=False,
        collate_fn=collate_conv1d_with_aux,
    )

    # 9. Train MLP and Conv1D in Variant A (mse) and Variant B (fplloss)
    nn_configs = [
        ("mlp", p_tabular, mlp_train_sub_dl, mlp_es_val_dl, mlp_valid_dl, mlp_test_dl, target_scaler_fe),
        ("conv1d", p_conv1d, conv_train_sub_dl, conv_es_val_dl, conv_valid_dl, conv_test_dl, target_scaler_eda),
    ]

    for nn_name, n_feat, tr_dl, es_dl, val_dl, tst_dl, t_scaler in nn_configs:
        for variant in ["mse", "fplloss"]:
            print("\n" + "=" * 90)
            print(f"TRENING SIECI NEURONOWEJ: {nn_name.upper()} — WARIANT: {variant.upper()} (seed={SEED})")
            print("=" * 90)
            set_seed(SEED)

            if nn_name == "mlp":
                model_nn = MLP(input_size=n_feat, hidden_sizes=[32, 16], dropout=0.2).to(device)
            else:
                model_nn = Conv1DRegressor(
                    n_features=n_feat, seq_len=seq_len, channels=[32, 16], kernel_size=3, dropout=0.2
                ).to(device)

            if variant == "mse":
                loss_fn = MSELossWrapper()
                monitor_metric = "mse"
            else:
                loss_fn = FPLLoss(
                    p_low=p_low,
                    p_high=p_high,
                    value_idx=0,
                    minutes_idx=1,
                    model_type="mlp",
                    w_mid=1.25,
                    w_premium=2.0,
                    eps_min=0.1,
                    norm_const=c_norm,
                )
                monitor_metric = "loss"

            trainer = Trainer(
                device=device,
                num_features=n_feat,
                random_state=SEED,
                p_low=p_low,
                p_high=p_high,
                value_idx=0,
                model_type="mlp",
                scaler=t_scaler,
            )

            optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=0.005)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, cooldown=2, min_lr=1e-5
            )

            history, best_state = trainer.model_eval(
                train_dataloader=tr_dl,
                test_dataloader=es_dl,
                model=model_nn,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scheduler=scheduler,
                num_epochs=150,
                patience=10,
                monitor=monitor_metric,
                verbose=False,
            )
            epochs_ran = len(history["train_loss"])
            model_nn.load_state_dict(best_state)
            es_metrics = trainer.evaluate_dataloader(es_dl, model_nn, loss_fn)
            print(
                f"  [{nn_name.upper()} - {variant}] Zatrzymano po {epochs_ran} epokach | "
                f"ES_val MSE={es_metrics['mse']:.4f}, Loss={es_metrics['loss']:.4f}, AdjR2={es_metrics['adj_r2']:.4f}"
            )

            preds_valid_dict[f"{nn_name}_{variant}"] = trainer.predict_dataloader(val_dl, model_nn, clip_preds=True)
            preds_test_dict[f"{nn_name}_{variant}"] = trainer.predict_dataloader(tst_dl, model_nn, clip_preds=True)

    # 10. Inner join of predictions across all models by (code, season, gw)
    print("\n" + "=" * 90)
    print("ŁĄCZENIE PREDYKCJI WSZYSTKICH MODELI PO (code, season, gw) — INNER JOIN")
    print("=" * 90)

    id_export_cols = ["code", "name", "position", "team", "opponent_team", "was_home", "season", "gw", "value_ema_3", "minutes_ema_3"]
    valid_base_df = X_valid_raw[id_export_cols].copy()
    valid_base_df["y"] = y_valid_raw.to_numpy(dtype=np.float64)

    test_base_df = X_test_raw[id_export_cols].copy()
    test_base_df["y"] = y_test_raw.to_numpy(dtype=np.float64)

    for col_name, p_arr in preds_valid_dict.items():
        part_df = pd.DataFrame({
            "code": X_valid_raw["code"].values,
            "season": X_valid_raw["season"].values,
            "gw": X_valid_raw["gw"].values,
            col_name: p_arr,
        })
        n_before = len(valid_base_df)
        valid_base_df = pd.merge(valid_base_df, part_df, on=["code", "season", "gw"], how="inner")
        assert len(valid_base_df) == n_before

    for col_name, p_arr in preds_test_dict.items():
        part_df = pd.DataFrame({
            "code": X_test_raw["code"].values,
            "season": X_test_raw["season"].values,
            "gw": X_test_raw["gw"].values,
            col_name: p_arr,
        })
        n_before = len(test_base_df)
        test_base_df = pd.merge(test_base_df, part_df, on=["code", "season", "gw"], how="inner")
        assert len(test_base_df) == n_before

    print(
        f"Valid po inner join wszystkich 4 modeli: {len(valid_base_df)} wierszy "
        f"(wypadło: {len(X_valid_raw) - len(valid_base_df)} wierszy)"
    )
    print(
        f"Test po inner join wszystkich 4 modeli:  {len(test_base_df)} wierszy "
        f"(wypadło: {len(X_test_raw) - len(test_base_df)} wierszy)"
    )

    # 11. Evaluate Base Models & Train/Evaluate Meta-Models (separately for Variant A and Variant B)
    print("\n" + "=" * 90)
    print("UCZENIE META-MODELI NA VALID (2025/26) I EWALUACJA NA TEST (2026/27)")
    print("=" * 90)

    base_models_order = ["glm", "xgboost", "mlp", "conv1d"]
    p_map_base = {
        "glm": p_tabular,
        "xgboost": p_tabular,
        "mlp": p_tabular,
        "conv1d": p_conv1d,
    }

    metrics_valid_rows: list[dict] = []
    metrics_test_rows: list[dict] = []
    meta_weights_rows: list[dict] = []

    y_val_common = valid_base_df["y"].to_numpy(dtype=np.float64)
    y_tst_common = test_base_df["y"].to_numpy(dtype=np.float64)
    price_val_common = valid_base_df["value_ema_3"].to_numpy(dtype=np.float64)
    price_tst_common = test_base_df["value_ema_3"].to_numpy(dtype=np.float64)

    for variant in ["mse", "fplloss"]:
        variant_cols = [f"{m}_{variant}" for m in base_models_order]

        # Evaluate 4 base models
        best_val_criterion = float("inf")
        best_single_model_name = None
        best_single_col = None

        for m_name in base_models_order:
            col = f"{m_name}_{variant}"
            p_feat = p_map_base[m_name]

            m_val = evaluate_predictions(
                y_true=y_val_common,
                y_pred=valid_base_df[col].to_numpy(),
                price_raw=price_val_common,
                weights=w_valid,
                p_low=p_low,
                p_high=p_high,
                p_features=p_feat,
            )
            m_tst = evaluate_predictions(
                y_true=y_tst_common,
                y_pred=test_base_df[col].to_numpy(),
                price_raw=price_tst_common,
                weights=w_test,
                p_low=p_low,
                p_high=p_high,
                p_features=p_feat,
            )

            metrics_valid_rows.append({"model": m_name, "variant": variant, "level": "base", **m_val})
            metrics_test_rows.append({"model": m_name, "variant": variant, "level": "base", **m_tst})

            crit = m_val["mse"] if variant == "mse" else m_val["fplloss"]
            if crit < best_val_criterion:
                best_val_criterion = crit
                best_single_model_name = m_name
                best_single_col = col

        print(
            f"\n[Wariant {variant.upper()}] Najlepszy pojedynczy model na valid: "
            f"{best_single_model_name.upper()} ({'MSE' if variant == 'mse' else 'FPLLoss'}={best_val_criterion:.4f})"
        )

        # Prepare matrices of 4 base predictions on valid and test
        X_meta_val = valid_base_df[variant_cols].copy()
        X_meta_val.columns = base_models_order
        X_meta_tst = test_base_df[variant_cols].copy()
        X_meta_tst.columns = base_models_order

        # Meta-model 1: Non-negative linear regression (LinearRegression(positive=True))
        meta_nnls = LinearRegression(positive=True, fit_intercept=True)
        if variant == "fplloss":
            meta_nnls.fit(X_meta_val, y_val_common, sample_weight=w_valid)
        else:
            meta_nnls.fit(X_meta_val, y_val_common)

        p_val_nnls = np.clip(meta_nnls.predict(X_meta_val), 0.0, 15.0)
        p_tst_nnls = np.clip(meta_nnls.predict(X_meta_tst), 0.0, 15.0)
        valid_base_df[f"meta_nnls_{variant}"] = p_val_nnls
        test_base_df[f"meta_nnls_{variant}"] = p_tst_nnls

        print(f"  Wagi meta_nnls ({variant}): intercept={float(meta_nnls.intercept_):.6f}")
        meta_weights_rows.append({
            "meta_model": "meta_nnls",
            "variant": variant,
            "feature": "intercept",
            "weight": float(meta_nnls.intercept_),
        })
        for feat_name, coef_val in zip(base_models_order, meta_nnls.coef_):
            print(f"    {feat_name:10s}: {float(coef_val):.6f}")
            meta_weights_rows.append({
                "meta_model": "meta_nnls",
                "variant": variant,
                "feature": feat_name,
                "weight": float(coef_val),
            })

        # Meta-model 2: Context-aware non-negative Ridge regression with prediction x context interactions
        X_ctx_val = build_context_matrix(X_meta_val, valid_base_df, base_models_order, p_low=p_low, p_high=p_high)
        X_ctx_tst = build_context_matrix(X_meta_tst, test_base_df, base_models_order, p_low=p_low, p_high=p_high)
        p_ctx = int(X_ctx_val.shape[1])

        meta_ctx = Ridge(alpha=10.0, positive=True, fit_intercept=True, random_state=SEED)
        if variant == "fplloss":
            meta_ctx.fit(X_ctx_val, y_val_common, sample_weight=w_valid)
        else:
            meta_ctx.fit(X_ctx_val, y_val_common)

        p_val_ctx = np.clip(meta_ctx.predict(X_ctx_val), 0.0, 15.0)
        p_tst_ctx = np.clip(meta_ctx.predict(X_ctx_tst), 0.0, 15.0)
        valid_base_df[f"meta_context_{variant}"] = p_val_ctx
        test_base_df[f"meta_context_{variant}"] = p_tst_ctx

        print(f"  Wagi meta_context ({variant}, p={p_ctx}): intercept={float(meta_ctx.intercept_):.6f}")
        meta_weights_rows.append({
            "meta_model": "meta_context",
            "variant": variant,
            "feature": "intercept",
            "weight": float(meta_ctx.intercept_),
        })
        for feat_name, coef_val in zip(X_ctx_val.columns, meta_ctx.coef_):
            meta_weights_rows.append({
                "meta_model": "meta_context",
                "variant": variant,
                "feature": feat_name,
                "weight": float(coef_val),
            })
            if abs(float(coef_val)) > 1e-5:
                print(f"    {feat_name:36s}: {float(coef_val):.6f}")

        # Baseline 1: Simple average of the 4 predictions
        p_val_avg = np.clip(X_meta_val.mean(axis=1).to_numpy(dtype=np.float64), 0.0, 15.0)
        p_tst_avg = np.clip(X_meta_tst.mean(axis=1).to_numpy(dtype=np.float64), 0.0, 15.0)
        valid_base_df[f"mean_ensemble_{variant}"] = p_val_avg
        test_base_df[f"mean_ensemble_{variant}"] = p_tst_avg

        # Baseline 2: Best single model on valid
        p_val_best = valid_base_df[best_single_col].to_numpy(dtype=np.float64)
        p_tst_best = test_base_df[best_single_col].to_numpy(dtype=np.float64)
        valid_base_df[f"best_single_{variant}"] = p_val_best
        test_base_df[f"best_single_{variant}"] = p_tst_best

        meta_eval_items = [
            ("meta_nnls", p_val_nnls, p_tst_nnls, 4),
            ("meta_context", p_val_ctx, p_tst_ctx, p_ctx),
            ("mean_ensemble", p_val_avg, p_tst_avg, 4),
            (f"best_single({best_single_model_name})", p_val_best, p_tst_best, p_map_base[best_single_model_name]),
        ]

        for m_label, pv, pt, p_cnt in meta_eval_items:
            mv = evaluate_predictions(
                y_true=y_val_common,
                y_pred=pv,
                price_raw=price_val_common,
                weights=w_valid,
                p_low=p_low,
                p_high=p_high,
                p_features=p_cnt,
            )
            mt = evaluate_predictions(
                y_true=y_tst_common,
                y_pred=pt,
                price_raw=price_tst_common,
                weights=w_test,
                p_low=p_low,
                p_high=p_high,
                p_features=p_cnt,
            )
            metrics_valid_rows.append({"model": m_label, "variant": variant, "level": "meta", **mv})
            metrics_test_rows.append({"model": m_label, "variant": variant, "level": "meta", **mt})

    # 12. Save all artifacts in results/ (without overwriting existing files)
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    df_metrics_valid = pd.DataFrame(metrics_valid_rows)
    df_metrics_test = pd.DataFrame(metrics_test_rows)
    df_meta_weights = pd.DataFrame(meta_weights_rows)

    path_metrics_test = safe_save_path(results_dir / "metrics_test.csv")
    path_metrics_valid = safe_save_path(results_dir / "metrics_valid.csv")
    path_preds_valid = safe_save_path(results_dir / "preds_valid.parquet")
    path_preds_test = safe_save_path(results_dir / "preds_test.parquet")
    path_meta_weights = safe_save_path(results_dir / "meta_weights.csv")
    path_readme = safe_save_path(results_dir / "README.md")

    df_metrics_test.to_csv(path_metrics_test, index=False)
    df_metrics_valid.to_csv(path_metrics_valid, index=False)
    valid_base_df.to_parquet(path_preds_valid, index=False)
    test_base_df.to_parquet(path_preds_test, index=False)
    df_meta_weights.to_csv(path_meta_weights, index=False)

    readme_content = f"""# Eksperymenty Predykcji Punktów FPL — Modele Bazowe, FPLLoss i Meta-Modele

## 1. Uruchomienie eksperymentu
Cały potok eksperymentalny (weryfikacja testami jednostkowymi, przygotowanie danych, strojenie GLM i XGBoost, trening sieci MLP i Conv1D, konstrukcja meta-modeli oraz ewaluacja na zbiorze testowym) uruchamiany jest jednym poleceniem z katalogu głównego repozytorium:

```bash
PYTHONPATH=src python -m fpl_project.models.run_experiments
```

Ziarno losowe zostało ustalone na `SEED = {SEED}` (`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`), co gwarantuje pełną powtarzalność wyników.

## 2. Podział danych i brak wycieku informacji
- **Train (`2022/23`, `2023/24`, `2024/25`)**: `{len(X_train_raw)}` obserwacji (posortowanych globalnie po `(season, gw, code)`).
  - Dla sieci neuronowych (`MLP`, `Conv1D`) wydzielono wewnętrzny chronologiczny zbiór wczesnego zatrzymania (`early_stopping_val`): ostatnie 8 kolejek sezonu `2024/25` (`gw >= 31`, `{int(es_mask_fe.sum())}` obserwacji), a trening właściwy przebiegał na wcześniejszych kolejkach (`{int(sub_mask_fe.sum())}` obserwacji). Zbiór `valid` nie był używany do wczesnego zatrzymania ani doboru hiperparametrów.
- **Valid (`2025/26`)**: `{len(valid_base_df)}` obserwacji — użyty wyłącznie do wygenerowania predykcji modeli bazowych uczących meta-modele oraz wyboru najlepszego pojedynczego modelu odniesienia.
- **Test (`2026/27`, rozegrane kolejki `is_played == True`)**: `{len(test_base_df)}` obserwacji — użyty wyłącznie do końcowej oceny wszystkich modeli.
- **Złączenie wewnętrzne (`inner join` po `(code, season, gw)`)**:
  - Liczba wierszy odrzuconych przy łączeniu `features_engineering.parquet` z `eda.parquet` i predykcjami `Conv1D`: `0` na `valid` oraz `0` na `test` (w `train_sub` dla `Conv1D` pominięto `{len(eda_train_sub_df) - len(conv_train_sub_ds)}` początkowych wierszy rozgrzewki okna `seq_len = {seq_len}`).

## 3. Funkcja straty `FPLLoss` i wartość stałej normalizującej $c$
Wagi obserwacji wyliczane są z **nieskalowanych** kolumn wiersza:
- Cena: `price = value_ema_3` (w mln funtów),
- Minuty: `m = clip(minutes_ema_3 / 90.0, 0.0, 1.0)`.

Na zbiorze treningowym (`train`) wyznaczono:
- Kwantyl `0.50` ceny ($p_{{\\text{{low}}}}$): **`{p_low:.6f}` mln £**
- Kwantyl `0.955` ceny ($p_{{\\text{{high}}}}$): **`{p_high:.6f}` mln £**
- Stała normalizująca $c$ (`norm_const` $= \\text{{mean}}(\\ln(1 + \\max(m_i, 0)) + \\varepsilon_{{\\min}})$ przy $\\varepsilon_{{\\min}} = 0.1$): **`{c_norm:.6f}`**

Waga pojedynczej obserwacji:
$$w_i = \\frac{{w_{{\\text{{cena}}, i}} \\cdot (\\ln(1 + \\max(m_i, 0)) + 0.1)}}{{{c_norm:.6f}}}, \\quad w_{{\\text{{cena}}, i}} \\in \\{{1.0, 1.25, 2.0\\}}$$

## 4. Realizacja wariantu (B) dla GLM i XGBoost
- **XGBoost (`objective="reg:squarederror"`)**: przekazanie `sample_weight = w_i` do `fit()` minimalizuje dokładnie ważony błąd średniokwadratowy $\\sum_i w_i (y_i - \\hat{{y}}_i)^2$, czyli dokładnie tę samą funkcję celu co `FPLLoss`.
- **GLM (`glum.GeneralizedLinearRegressor`)**: przekazanie `sample_weight = w_i` maksymalizuje **ważoną log-wiarygodność** (weighted log-likelihood / ważoną dewiancję wybranej rodziny rozkładu: Poisson, Negative Binomial z estymowanym/strojonym $\\theta$ lub Tweedie z mocą $p \\in [1.1, 1.9]$). Dla uogólnionych modeli liniowych z funkcją wiążącą `log` i niegaussowską funkcją wariancji jest to **analog** `FPLLoss` (ważenie obserwacji tymi samymi wagami decyzyjnymi $w_i$), a nie algebraicznie identyczna ważona strata kwadratowa MSE.
- W `TimeSeriesSplit` CV (z granicami foldów wypadającymi ściśle pomiędzy kolejkami `(season, gw)`) wagi `w_i` przekazywane są zarówno do `estimator.fit()`, jak i do ważonego kryterium oceny foldu (`neg_weighted_mse`).

## 5. Wygenerowane pliki w katalogu `results/`
- `metrics_test.csv` — pełne zestawienie metryk na zbiorze testowym (`2026/27`) wraz z liczbą cech wejściowych `p` użytych w $\\text{{Adj }} R^2$,
- `metrics_valid.csv` — zestawienie metryk na zbiorze walidacyjnym (`2025/26`),
- `preds_valid.parquet`, `preds_test.parquet` — identyfikatory zawodników, `y` oraz predykcje wszystkich modeli bazowych i meta-modeli,
- `meta_weights.csv` — wagi i wyrazy wolne (`intercept`) wyuczonych meta-modeli (`meta_nnls` oraz `meta_context`).
"""
    path_readme.write_text(readme_content, encoding="utf-8")

    log_results_to_mlflow(
        df_metrics_valid=df_metrics_valid,
        df_metrics_test=df_metrics_test,
        best_params_log=best_params_log,
        p_low=p_low,
        p_high=p_high,
        c_norm=c_norm,
        artifact_paths=[path_metrics_test, path_metrics_valid, path_meta_weights, path_readme],
    )

    print("\n" + "=" * 115)
    print(f"TABELA WYNIKÓW NA ZBIORZE TESTOWYM ({path_metrics_test.name}):")
    print("=" * 115)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    print(df_metrics_test.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 115)
    print(f"Czas całkowity wykonania eksperymentu: {time.time() - total_start:.2f} s")


if __name__ == "__main__":
    main()
