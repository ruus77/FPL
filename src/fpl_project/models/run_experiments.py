from dataclasses import dataclass
from pathlib import Path
import sys
import time
import warnings

from glum import GeneralizedLinearRegressor
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
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


@dataclass
class PreparedData:
    """Container for split datasets, weights, scalers, and dataloaders."""

    X_train_raw: pd.DataFrame
    y_train_raw: pd.Series
    X_valid_raw: pd.DataFrame
    y_valid_raw: pd.Series
    X_test_raw: pd.DataFrame
    y_test_raw: pd.Series
    X_train_tab: pd.DataFrame
    X_valid_tab: pd.DataFrame
    X_test_tab: pd.DataFrame
    p_low: float
    p_high: float
    c_norm: float
    w_train: np.ndarray
    w_valid: np.ndarray
    w_test: np.ndarray
    p_tabular: int
    p_conv1d: int
    n_train_sub: int
    n_es_val: int
    n_conv_warmup_dropped: int
    mlp_dataloaders: tuple[DataLoader, DataLoader, DataLoader, DataLoader]
    conv_dataloaders: tuple[DataLoader, DataLoader, DataLoader, DataLoader]
    target_scaler_fe: StandardScaler
    target_scaler_eda: StandardScaler


class MSELossWrapper(nn.Module):
    """Standard MSE loss wrapper accepting optional auxiliary inputs."""

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
    """Samples GLM families: Poisson, NegativeBinomial(theta), and Tweedie(power)."""

    def __init__(self, theta_mom: float = 1.0):
        self.theta_mom = float(np.clip(theta_mom, 0.05, 5.0))

    def rvs(self, random_state=None):
        rng = np.random.default_rng(random_state)
        choice = int(rng.integers(0, 3))
        if choice == 0:
            return "poisson"
        if choice == 1:
            if rng.random() < 0.25:
                theta = self.theta_mom
            else:
                theta = float(np.exp(rng.uniform(np.log(0.05), np.log(2.5))))
            return f"negative.binomial({theta:.4f})"
        power = float(rng.uniform(1.1, 1.9))
        return f"tweedie({power:.4f})"


class AuxSequenceDataset(Dataset):
    """Sequence dataset for Conv1D returning (window, target, [value_ema_3, minutes_ema_3 / 90])."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "total_points",
        seq_len: int = config.NN_SEQ_LEN,
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
            df_indexed[target_col] = (
                df_indexed[target_col].fillna(0.0) if target_col in df_indexed.columns else 0.0
            )
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

                    aux = np.array([raw_price[t], raw_min[t]], dtype=np.float32)
                    ordered_sequences[int(orig_positions[t])] = (window, float(target[t]), aux)

            self.sequences = ordered_sequences
        else:
            self.sequences = []
            df_sorted = df.copy().sort_values([player_col, season_col, time_col]).reset_index(drop=True)
            for _, grp in df_sorted.groupby([player_col, season_col], sort=False):
                feats = grp[feature_cols].values.astype(np.float32)
                target = grp[target_col].values.astype(np.float32)
                raw_price = grp["_raw_price"].values.astype(np.float32)
                raw_min = grp["_raw_min"].values.astype(np.float32)

                for t in range(seq_len, len(grp)):
                    window = feats[t - seq_len : t]
                    aux = np.array([raw_price[t], raw_min[t]], dtype=np.float32)
                    self.sequences.append((window, float(target[t]), aux))

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
    """Collate function stacking sequence windows, targets, and auxiliary vectors."""
    xs, ys, auxs = zip(*batch)
    return torch.stack(xs), torch.cat(ys), torch.stack(auxs)


def run_verification_tests() -> None:
    """Verify FPLLoss weight equivalence and sample_weight routing in CV."""
    print("=" * 90)
    print("WERYFIKACJA POPRAWNOŚCI (TESTY JEDNOSTKOWE PRZED EKSPERYMENTEM)")
    print("=" * 90)
    set_seed(config.SEED)

    toy_df = pd.DataFrame({
        "value_ema_3": [4.0, 4.5, 5.5, 7.0, 11.5],
        "minutes_ema_3": [0.0, 30.0, 60.0, 85.0, 90.0],
    })
    p_low_t, p_high_t = 4.5, 7.0
    c_t = compute_norm_const(toy_df, minutes_col="minutes_ema_3", eps_min=config.EPS_MIN)
    w_np = compute_fpl_weights(
        toy_df,
        p_low=p_low_t,
        p_high=p_high_t,
        w_mid=config.W_MID,
        w_premium=config.W_PREMIUM,
        eps_min=config.EPS_MIN,
        norm_const=c_t,
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
    assert np.isclose(expected_loss, torch_loss_aux, atol=1e-6)
    assert np.isclose(expected_loss, torch_loss_w, atol=1e-6)
    print(f"[PASS] Test 1: compute_fpl_weights == FPLLoss (loss={expected_loss:.6f})")

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
    print("=" * 90)


def load_data() -> PreparedData:
    """Load, align, split, and scale tabular and sequential datasets."""
    if not config.EDA.exists():
        from fpl_project.data_processing.data_import import fetch_and_save_eda_data
        fetch_and_save_eda_data()
    if not config.FEATURES_ENGINEERING.exists():
        build_and_save_features_engineering_data()

    df_fe_raw = sort_data(pd.read_parquet(config.FEATURES_ENGINEERING))
    df_eda_raw = sort_data(pd.read_parquet(config.EDA))

    df_fe_raw["total_points"] = df_fe_raw["total_points"].clip(lower=0.0)
    df_eda_raw["total_points"] = df_eda_raw["total_points"].clip(lower=0.0)

    if "is_played" not in df_fe_raw.columns and "is_played" in df_eda_raw.columns:
        df_fe_raw["is_played"] = df_eda_raw["is_played"].values

    key_cols = ["code", "season", "gw"]
    common_keys = pd.merge(df_fe_raw[key_cols], df_eda_raw[key_cols], on=key_cols, how="inner")
    print(
        f"[Wspólny zbiór wierszy FE ∩ EDA po {key_cols}] "
        f"FE={len(df_fe_raw)}, EDA={len(df_eda_raw)}, wspólne={len(common_keys)}, "
        f"odrzucone FE={len(df_fe_raw) - len(common_keys)}, odrzucone EDA={len(df_eda_raw) - len(common_keys)}"
    )

    fe_aux = df_fe_raw[key_cols + ["value_ema_3", "minutes_ema_3"]].copy()
    fe_aux["_raw_price"] = fe_aux["value_ema_3"].astype(np.float64)
    fe_aux["_raw_min"] = np.clip(fe_aux["minutes_ema_3"].astype(np.float64) / 90.0, 0.0, 1.0)
    df_eda_merged = pd.merge(
        df_eda_raw, fe_aux[key_cols + ["_raw_price", "_raw_min"]], on=key_cols, how="inner"
    )

    df_fe = df_fe_raw.sort_values(["season", "gw", "code"]).reset_index(drop=True)
    df_eda = df_eda_merged.sort_values(["season", "gw", "code"]).reset_index(drop=True)
    assert (df_fe[key_cols].values == df_eda[key_cols].values).all()

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

    price_train_raw = X_train_raw["value_ema_3"].to_numpy(dtype=np.float64)
    p_low = float(np.quantile(price_train_raw, config.PRICE_QUANTILE_LOW))
    p_high = float(np.quantile(price_train_raw, config.PRICE_QUANTILE_HIGH))
    c_norm = compute_norm_const(X_train_raw, minutes_col="minutes_ema_3", eps_min=config.EPS_MIN)

    w_train = compute_fpl_weights(X_train_raw, p_low=p_low, p_high=p_high, norm_const=c_norm)
    w_valid = compute_fpl_weights(X_valid_raw, p_low=p_low, p_high=p_high, norm_const=c_norm)
    w_test = compute_fpl_weights(X_test_raw, p_low=p_low, p_high=p_high, norm_const=c_norm)

    print("=" * 90)
    print("PARAMETRY FPLLoss (wyliczone na surowych kolumnach train):")
    print(f"  eps_min = {config.EPS_MIN}, w_mid = {config.W_MID}, w_premium = {config.W_PREMIUM}")
    print(f"  p_low (kwantyl {config.PRICE_QUANTILE_LOW:.2f} value_ema_3)  = {p_low:.4f} mln £")
    print(f"  p_high (kwantyl {config.PRICE_QUANTILE_HIGH:.3f} value_ema_3)= {p_high:.4f} mln £")
    print(f"  Stała normalizująca c (norm_const)= {c_norm:.4f}")
    print(f"  Wagi w_train: mean={w_train.mean():.4f}, min={w_train.min():.4f}, max={w_train.max():.4f}")
    print("=" * 90)

    cols_map = FeaturesConfig()
    num_cols = [
        c for c in df_fe.columns if c not in cols_map.id_cols + cols_map.static_cols + cols_map.target
    ]
    cat_cols = cols_map.static_cols

    tab_pipe = FPLDataPipe(num_cols=num_cols, cat_cols=cat_cols, batch_size=config.NN_BATCH_SIZE)
    X_train_tab = tab_pipe.preprocessor.fit_transform(X_train_raw)
    X_valid_tab = tab_pipe.preprocessor.transform(X_valid_raw)
    X_test_tab = tab_pipe.preprocessor.transform(X_test_raw)
    p_tabular = int(X_train_tab.shape[1])

    es_mask_fe = (X_train_raw["season"] == config.NN_ES_SEASON) & (X_train_raw["gw"] >= config.NN_ES_MIN_GW)
    sub_mask_fe = ~es_mask_fe

    splits_fe_scaled, target_scaler_fe = scale_target(splits_fe)
    y_train_scaled = splits_fe_scaled["y_train"].reset_index(drop=True)
    y_valid_scaled = splits_fe_scaled["y_valid"].reset_index(drop=True)
    y_test_scaled = splits_fe_scaled["y_test"].reset_index(drop=True)

    def make_aux_tensor(df_part: pd.DataFrame) -> torch.Tensor:
        p_col = df_part["value_ema_3"].to_numpy(dtype=np.float32)
        m_col = np.clip(df_part["minutes_ema_3"].to_numpy(dtype=np.float32) / 90.0, 0.0, 1.0)
        return torch.tensor(np.column_stack([p_col, m_col]), dtype=torch.float32)

    mlp_dataloaders = (
        DataLoader(
            TensorDataset(
                torch.tensor(X_train_tab.loc[sub_mask_fe].to_numpy(), dtype=torch.float32),
                torch.tensor(y_train_scaled.loc[sub_mask_fe].to_numpy().reshape(-1, 1), dtype=torch.float32),
                make_aux_tensor(X_train_raw.loc[sub_mask_fe]),
            ),
            batch_size=config.NN_BATCH_SIZE,
            shuffle=False,
        ),
        DataLoader(
            TensorDataset(
                torch.tensor(X_train_tab.loc[es_mask_fe].to_numpy(), dtype=torch.float32),
                torch.tensor(y_train_scaled.loc[es_mask_fe].to_numpy().reshape(-1, 1), dtype=torch.float32),
                make_aux_tensor(X_train_raw.loc[es_mask_fe]),
            ),
            batch_size=config.NN_BATCH_SIZE,
            shuffle=False,
        ),
        DataLoader(
            TensorDataset(
                torch.tensor(X_valid_tab.to_numpy(), dtype=torch.float32),
                torch.tensor(y_valid_scaled.to_numpy().reshape(-1, 1), dtype=torch.float32),
                make_aux_tensor(X_valid_raw),
            ),
            batch_size=config.NN_BATCH_SIZE,
            shuffle=False,
        ),
        DataLoader(
            TensorDataset(
                torch.tensor(X_test_tab.to_numpy(), dtype=torch.float32),
                torch.tensor(y_test_scaled.to_numpy().reshape(-1, 1), dtype=torch.float32),
                make_aux_tensor(X_test_raw),
            ),
            batch_size=config.NN_BATCH_SIZE,
            shuffle=False,
        ),
    )

    seq_feature_cols = list(dict.fromkeys([
        c
        for c in cols_map.fpl_cols + cols_map.match_context_cols + cols_map.market_cols + cols_map.betting_cols
        if c in df_eda.columns and c != "total_points"
    ]))
    p_conv1d = len(seq_feature_cols)

    splits_eda_scaled, target_scaler_eda = scale_target(splits_eda)

    def merge_eda_xy(X_df: pd.DataFrame, y_ser: pd.Series) -> pd.DataFrame:
        out_df = X_df.copy().reset_index(drop=True)
        out_df["total_points"] = y_ser.to_numpy(dtype=np.float32)
        return out_df

    eda_train_df = merge_eda_xy(splits_eda_scaled["X_train"], splits_eda_scaled["y_train"])
    eda_valid_df = merge_eda_xy(splits_eda_scaled["X_valid"], splits_eda_scaled["y_valid"])
    eda_test_df = merge_eda_xy(splits_eda_scaled["X_test"], splits_eda_scaled["y_test"])

    seq_scaler = StandardScaler()
    eda_train_df[seq_feature_cols] = seq_scaler.fit_transform(eda_train_df[seq_feature_cols])
    eda_valid_df[seq_feature_cols] = seq_scaler.transform(eda_valid_df[seq_feature_cols])
    eda_test_df[seq_feature_cols] = seq_scaler.transform(eda_test_df[seq_feature_cols])

    es_mask_eda = (eda_train_df["season"] == config.NN_ES_SEASON) & (eda_train_df["gw"] >= config.NN_ES_MIN_GW)
    eda_train_sub_df = eda_train_df.loc[~es_mask_eda].reset_index(drop=True)
    eda_es_val_df = eda_train_df.loc[es_mask_eda].reset_index(drop=True)

    conv_train_sub_ds = AuxSequenceDataset(eda_train_sub_df, seq_feature_cols, seq_len=config.NN_SEQ_LEN)
    conv_es_val_ds = AuxSequenceDataset(
        eda_es_val_df, seq_feature_cols, seq_len=config.NN_SEQ_LEN, history_df=eda_train_sub_df
    )
    conv_valid_ds = AuxSequenceDataset(
        eda_valid_df, seq_feature_cols, seq_len=config.NN_SEQ_LEN, history_df=eda_train_df
    )
    conv_test_ds = AuxSequenceDataset(
        eda_test_df,
        seq_feature_cols,
        seq_len=config.NN_SEQ_LEN,
        history_df=pd.concat([eda_train_df, eda_valid_df], ignore_index=True),
    )

    conv_dataloaders = (
        DataLoader(
            conv_train_sub_ds,
            batch_size=config.NN_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_conv1d_with_aux,
            drop_last=True,
        ),
        DataLoader(
            conv_es_val_ds,
            batch_size=config.NN_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_conv1d_with_aux,
        ),
        DataLoader(
            conv_valid_ds,
            batch_size=config.NN_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_conv1d_with_aux,
        ),
        DataLoader(
            conv_test_ds,
            batch_size=config.NN_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_conv1d_with_aux,
        ),
    )

    return PreparedData(
        X_train_raw=X_train_raw,
        y_train_raw=y_train_raw,
        X_valid_raw=X_valid_raw,
        y_valid_raw=y_valid_raw,
        X_test_raw=X_test_raw,
        y_test_raw=y_test_raw,
        X_train_tab=X_train_tab,
        X_valid_tab=X_valid_tab,
        X_test_tab=X_test_tab,
        p_low=p_low,
        p_high=p_high,
        c_norm=c_norm,
        w_train=w_train,
        w_valid=w_valid,
        w_test=w_test,
        p_tabular=p_tabular,
        p_conv1d=p_conv1d,
        n_train_sub=int(sub_mask_fe.sum()),
        n_es_val=int(es_mask_fe.sum()),
        n_conv_warmup_dropped=len(eda_train_sub_df) - len(conv_train_sub_ds),
        mlp_dataloaders=mlp_dataloaders,
        conv_dataloaders=conv_dataloaders,
        target_scaler_fe=target_scaler_fe,
        target_scaler_eda=target_scaler_eda,
    )


def tune_tabular_models(
    data: PreparedData,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict]]:
    """Tune and fit GLM and XGBoost in MSE and FPLLoss variants using TimeSeriesSplit CV."""
    cv_splits = make_gw_time_series_splits(
        seasons=data.X_train_raw["season"],
        gws=data.X_train_raw["gw"],
        n_splits=config.CV_SPLITS,
    )

    y_tr_np = data.y_train_raw.to_numpy(dtype=np.float64)
    y_mean = float(np.mean(y_tr_np))
    y_var = float(np.var(y_tr_np))
    theta_mom = max((y_var - y_mean) / max(y_mean ** 2, 1e-6), 0.05)

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

    val_idx_tab = list(data.X_train_tab.columns).index("num__value_ema_3")
    selector = ModelSelector(
        p_low=data.p_low,
        p_high=data.p_high,
        value_idx=val_idx_tab,
        random_state=config.SEED,
    )
    price_train_raw = data.X_train_raw["value_ema_3"].to_numpy(dtype=np.float64)

    preds_valid: dict[str, np.ndarray] = {}
    preds_test: dict[str, np.ndarray] = {}
    best_params_log: dict[str, dict] = {}

    for variant, sw_train in [("mse", None), ("fplloss", data.w_train)]:
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
            params_grid=[glm_grid, xgb_grid],
            X_train=data.X_train_tab,
            y_train=data.y_train_raw,
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
            print(f"  [{m_name.upper()} - {variant}] Najlepsze parametry: {row_cv['best_params']}")

            preds_valid[f"{m_name}_{variant}"] = np.clip(
                fitted_model.predict(data.X_valid_tab.to_numpy()),
                config.PRED_CLIP_MIN,
                config.PRED_CLIP_MAX,
            )
            preds_test[f"{m_name}_{variant}"] = np.clip(
                fitted_model.predict(data.X_test_tab.to_numpy()),
                config.PRED_CLIP_MIN,
                config.PRED_CLIP_MAX,
            )

    return preds_valid, preds_test, best_params_log


def train_neural_networks(
    data: PreparedData,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Train MLP and Conv1D in MSE and FPLLoss variants with internal chronological early stopping."""
    preds_valid: dict[str, np.ndarray] = {}
    preds_test: dict[str, np.ndarray] = {}

    nn_configs = [
        ("mlp", data.p_tabular, *data.mlp_dataloaders, data.target_scaler_fe),
        ("conv1d", data.p_conv1d, *data.conv_dataloaders, data.target_scaler_eda),
    ]

    for nn_name, n_feat, tr_dl, es_dl, val_dl, tst_dl, t_scaler in nn_configs:
        for variant in ["mse", "fplloss"]:
            print("\n" + "=" * 90)
            print(f"TRENING SIECI NEURONOWEJ: {nn_name.upper()} — WARIANT: {variant.upper()} (seed={config.SEED})")
            print("=" * 90)
            set_seed(config.SEED)

            if nn_name == "mlp":
                model_nn = MLP(
                    input_size=n_feat,
                    hidden_sizes=config.NN_HIDDEN_SIZES,
                    dropout=config.NN_DROPOUT,
                ).to(device)
            else:
                model_nn = Conv1DRegressor(
                    n_features=n_feat,
                    seq_len=config.NN_SEQ_LEN,
                    channels=config.NN_CONV_CHANNELS,
                    kernel_size=config.NN_KERNEL_SIZE,
                    dropout=config.NN_DROPOUT,
                ).to(device)

            if variant == "mse":
                loss_fn = MSELossWrapper()
                monitor_metric = "mse"
            else:
                loss_fn = FPLLoss(
                    p_low=data.p_low,
                    p_high=data.p_high,
                    value_idx=0,
                    minutes_idx=1,
                    model_type="mlp",
                    w_mid=config.W_MID,
                    w_premium=config.W_PREMIUM,
                    eps_min=config.EPS_MIN,
                    norm_const=data.c_norm,
                )
                monitor_metric = "loss"

            trainer = Trainer(
                device=device,
                num_features=n_feat,
                random_state=config.SEED,
                p_low=data.p_low,
                p_high=data.p_high,
                value_idx=0,
                model_type="mlp",
                scaler=t_scaler,
            )

            optimizer = torch.optim.AdamW(
                model_nn.parameters(), lr=config.NN_LR, weight_decay=config.NN_WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.NN_SCHEDULER_FACTOR,
                patience=config.NN_SCHEDULER_PATIENCE,
                cooldown=config.NN_SCHEDULER_COOLDOWN,
                min_lr=config.NN_MIN_LR,
            )

            history, best_state = trainer.model_eval(
                train_dataloader=tr_dl,
                test_dataloader=es_dl,
                model=model_nn,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scheduler=scheduler,
                num_epochs=config.NN_EPOCHS,
                patience=config.NN_PATIENCE,
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

            preds_valid[f"{nn_name}_{variant}"] = trainer.predict_dataloader(val_dl, model_nn, clip_preds=True)
            preds_test[f"{nn_name}_{variant}"] = trainer.predict_dataloader(tst_dl, model_nn, clip_preds=True)

    return preds_valid, preds_test


def merge_base_predictions(
    data: PreparedData,
    preds_valid: dict[str, np.ndarray],
    preds_test: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join predictions of all base models by (code, season, gw)."""
    id_cols = [
        "code", "name", "position", "team", "opponent_team",
        "was_home", "season", "gw", "value_ema_3", "minutes_ema_3",
    ]
    valid_df = data.X_valid_raw[id_cols].copy()
    valid_df["y"] = data.y_valid_raw.to_numpy(dtype=np.float64)

    test_df = data.X_test_raw[id_cols].copy()
    test_df["y"] = data.y_test_raw.to_numpy(dtype=np.float64)

    for col_name, p_arr in preds_valid.items():
        part_df = pd.DataFrame({
            "code": data.X_valid_raw["code"].values,
            "season": data.X_valid_raw["season"].values,
            "gw": data.X_valid_raw["gw"].values,
            col_name: p_arr,
        })
        valid_df = pd.merge(valid_df, part_df, on=["code", "season", "gw"], how="inner")

    for col_name, p_arr in preds_test.items():
        part_df = pd.DataFrame({
            "code": data.X_test_raw["code"].values,
            "season": data.X_test_raw["season"].values,
            "gw": data.X_test_raw["gw"].values,
            col_name: p_arr,
        })
        test_df = pd.merge(test_df, part_df, on=["code", "season", "gw"], how="inner")

    print(
        f"Valid po inner join 4 modeli: {len(valid_df)} wierszy (wypadło: {len(data.X_valid_raw) - len(valid_df)}) | "
        f"Test po inner join 4 modeli: {len(test_df)} wierszy (wypadło: {len(data.X_test_raw) - len(test_df)})"
    )
    return valid_df, test_df


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_raw: np.ndarray,
    weights: np.ndarray,
    p_low: float,
    p_high: float,
    p_features: int,
) -> dict[str, float | int]:
    """Compute overall and price-tier metrics on original clipped target scale."""
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
    adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - p_features - 1)) if n > p_features + 1 else 0.0
    fplloss_val = float(np.mean(w * sq_err))

    mask_b = price < p_low
    mask_m = (price >= p_low) & (price < p_high)
    mask_p = price >= p_high

    return {
        "p": int(p_features),
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "adj_r2": adj_r2,
        "fplloss": fplloss_val,
        "mse@budget": float(np.mean(sq_err[mask_b])) if mask_b.any() else 0.0,
        "mse@mid": float(np.mean(sq_err[mask_m])) if mask_m.any() else 0.0,
        "mse@premium": float(np.mean(sq_err[mask_p])) if mask_p.any() else 0.0,
        "mae@budget": float(np.mean(abs_err[mask_b])) if mask_b.any() else 0.0,
        "mae@mid": float(np.mean(abs_err[mask_m])) if mask_m.any() else 0.0,
        "mae@premium": float(np.mean(abs_err[mask_p])) if mask_p.any() else 0.0,
    }


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


def fit_meta_models(
    data: PreparedData,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit meta-models on validation predictions and evaluate all models on valid and test sets."""
    base_models_order = ["glm", "xgboost", "mlp", "conv1d"]
    p_map_base = {
        "glm": data.p_tabular,
        "xgboost": data.p_tabular,
        "mlp": data.p_tabular,
        "conv1d": data.p_conv1d,
    }

    metrics_valid_rows: list[dict] = []
    metrics_test_rows: list[dict] = []
    meta_weights_rows: list[dict] = []

    y_val = valid_df["y"].to_numpy(dtype=np.float64)
    y_tst = test_df["y"].to_numpy(dtype=np.float64)
    price_val = valid_df["value_ema_3"].to_numpy(dtype=np.float64)
    price_tst = test_df["value_ema_3"].to_numpy(dtype=np.float64)

    for variant in ["mse", "fplloss"]:
        variant_cols = [f"{m}_{variant}" for m in base_models_order]
        best_val_criterion = float("inf")
        best_single_name = None
        best_single_col = None

        for m_name in base_models_order:
            col = f"{m_name}_{variant}"
            p_feat = p_map_base[m_name]

            m_val = evaluate(y_val, valid_df[col].to_numpy(), price_val, data.w_valid, data.p_low, data.p_high, p_feat)
            m_tst = evaluate(y_tst, test_df[col].to_numpy(), price_tst, data.w_test, data.p_low, data.p_high, p_feat)

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

        meta_nnls = LinearRegression(positive=True, fit_intercept=True)
        if variant == "fplloss":
            meta_nnls.fit(X_meta_val, y_val, sample_weight=data.w_valid)
        else:
            meta_nnls.fit(X_meta_val, y_val)

        p_val_nnls = np.clip(meta_nnls.predict(X_meta_val), config.PRED_CLIP_MIN, config.PRED_CLIP_MAX)
        p_tst_nnls = np.clip(meta_nnls.predict(X_meta_tst), config.PRED_CLIP_MIN, config.PRED_CLIP_MAX)
        valid_df[f"meta_nnls_{variant}"] = p_val_nnls
        test_df[f"meta_nnls_{variant}"] = p_tst_nnls

        meta_weights_rows.append({
            "meta_model": "meta_nnls",
            "variant": variant,
            "feature": "intercept",
            "weight": float(meta_nnls.intercept_),
        })
        for feat_name, coef_val in zip(base_models_order, meta_nnls.coef_):
            meta_weights_rows.append({
                "meta_model": "meta_nnls",
                "variant": variant,
                "feature": feat_name,
                "weight": float(coef_val),
            })

        X_ctx_val = build_context_matrix(X_meta_val, valid_df, base_models_order, data.p_low, data.p_high)
        X_ctx_tst = build_context_matrix(X_meta_tst, test_df, base_models_order, data.p_low, data.p_high)
        p_ctx = int(X_ctx_val.shape[1])

        meta_ctx = Ridge(alpha=config.META_RIDGE_ALPHA, positive=True, fit_intercept=True, random_state=config.SEED)
        if variant == "fplloss":
            meta_ctx.fit(X_ctx_val, y_val, sample_weight=data.w_valid)
        else:
            meta_ctx.fit(X_ctx_val, y_val)

        p_val_ctx = np.clip(meta_ctx.predict(X_ctx_val), config.PRED_CLIP_MIN, config.PRED_CLIP_MAX)
        p_tst_ctx = np.clip(meta_ctx.predict(X_ctx_tst), config.PRED_CLIP_MIN, config.PRED_CLIP_MAX)
        valid_df[f"meta_context_{variant}"] = p_val_ctx
        test_df[f"meta_context_{variant}"] = p_tst_ctx

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

        p_val_avg = np.clip(X_meta_val.mean(axis=1).to_numpy(dtype=np.float64), config.PRED_CLIP_MIN, config.PRED_CLIP_MAX)
        p_tst_avg = np.clip(X_meta_tst.mean(axis=1).to_numpy(dtype=np.float64), config.PRED_CLIP_MIN, config.PRED_CLIP_MAX)
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
            mv = evaluate(y_val, pv, price_val, data.w_valid, data.p_low, data.p_high, p_cnt)
            mt = evaluate(y_tst, pt, price_tst, data.w_test, data.p_low, data.p_high, p_cnt)
            metrics_valid_rows.append({"model": m_label, "variant": variant, "level": "meta", **mv})
            metrics_test_rows.append({"model": m_label, "variant": variant, "level": "meta", **mt})

    return (
        pd.DataFrame(metrics_valid_rows),
        pd.DataFrame(metrics_test_rows),
        pd.DataFrame(meta_weights_rows),
    )


def compute_tier_counts_table(
    splits_dict: dict[str, pd.DataFrame],
    p_low: float,
    p_high: float,
) -> pd.DataFrame:
    """Count observations in budget, mid, and premium tiers across splits."""
    rows = []
    for split_name, df_part in splits_dict.items():
        price = df_part["value_ema_3"].to_numpy(dtype=np.float64)
        rows.append({
            "split": split_name,
            "budget": int(np.sum(price < p_low)),
            "mid": int(np.sum((price >= p_low) & (price < p_high))),
            "premium": int(np.sum(price >= p_high)),
            "total": int(len(df_part)),
        })
    return pd.DataFrame(rows)


def build_table_tiers(df_metrics_test: pd.DataFrame) -> pd.DataFrame:
    """Build publication-ready table_tiers.csv rounded to 2 decimal places."""
    models_order = ["glm", "xgboost", "mlp", "conv1d", "meta_nnls", "meta_context", "mean_ensemble"]
    rows = []
    for variant, variant_label in [("mse", "MSE"), ("fplloss", "FPLLoss")]:
        for m_name in models_order:
            r = df_metrics_test[
                (df_metrics_test["model"] == m_name) & (df_metrics_test["variant"] == variant)
            ].iloc[0]
            rows.append({
                "Model": m_name,
                "Wariant": variant_label,
                "MSE": round(float(r["mse"]), 2),
                "MSE budget": round(float(r["mse@budget"]), 2),
                "MSE mid": round(float(r["mse@mid"]), 2),
                "MSE premium": round(float(r["mse@premium"]), 2),
                "MAE": round(float(r["mae"]), 2),
                "MAE budget": round(float(r["mae@budget"]), 2),
                "MAE mid": round(float(r["mae@mid"]), 2),
                "MAE premium": round(float(r["mae@premium"]), 2),
            })
    return pd.DataFrame(rows)


def log_results_to_mlflow(
    df_metrics_valid: pd.DataFrame,
    df_metrics_test: pd.DataFrame,
    best_params_log: dict[str, dict],
    data: PreparedData,
    artifact_paths: list[Path],
) -> None:
    """Log experiment runs, parameters, valid/test metrics, and result artifacts to MLflow."""
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
                "seed": config.SEED,
                "p_low": data.p_low,
                "p_high": data.p_high,
                "norm_const_c": round(data.c_norm, 4),
                "eps_min": config.EPS_MIN,
                "w_mid": config.W_MID,
                "w_premium": config.W_PREMIUM,
            }
            if f"{m_name}_{variant}" in best_params_log:
                for k, v in best_params_log[f"{m_name}_{variant}"].items():
                    params[f"hp_{k}"] = v
            tracker.log_params(params)

            metrics_to_log = {}
            for mc in metric_cols:
                clean_mc = mc.replace("@", "_")
                metrics_to_log[f"valid_{clean_mc}"] = float(row_v[mc])
                metrics_to_log[f"test_{clean_mc}"] = float(row_t[mc])
            tracker.log_metrics(metrics_to_log)

            if idx == len(df_metrics_test) - 1:
                for ap in artifact_paths:
                    if ap.exists():
                        mlflow.log_artifact(str(ap))

    print(f"[MLflow] Zalogowano {len(df_metrics_test)} przebiegów eksperymentalnych do MLflow.")


def save_results(
    data: PreparedData,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    df_metrics_valid: pd.DataFrame,
    df_metrics_test: pd.DataFrame,
    df_meta_weights: pd.DataFrame,
    best_params_log: dict[str, dict],
) -> None:
    """Save all result CSV/Parquet files, update results/README.md, and log to MLflow."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_tier_counts = compute_tier_counts_table(
        splits_dict={"train": data.X_train_raw, "valid": valid_df, "test": test_df},
        p_low=data.p_low,
        p_high=data.p_high,
    )
    df_table_tiers = build_table_tiers(df_metrics_test)

    df_metrics_test.to_csv(config.METRICS_TEST_PATH, index=False)
    df_metrics_valid.to_csv(config.METRICS_VALID_PATH, index=False)
    valid_df.to_parquet(config.PREDS_VALID_PATH, index=False)
    test_df.to_parquet(config.PREDS_TEST_PATH, index=False)
    df_meta_weights.to_csv(config.META_WEIGHTS_PATH, index=False)
    df_tier_counts.to_csv(config.TIER_COUNTS_PATH, index=False)
    df_table_tiers.to_csv(config.TABLE_TIERS_PATH, index=False, float_format="%.2f")

    tc_train = df_tier_counts[df_tier_counts["split"] == "train"].iloc[0]
    tc_valid = df_tier_counts[df_tier_counts["split"] == "valid"].iloc[0]
    tc_test = df_tier_counts[df_tier_counts["split"] == "test"].iloc[0]
    n_test_gws = int(test_df["gw"].nunique())

    readme_content = f"""# Eksperymenty Predykcji Punktów FPL — Modele Bazowe, FPLLoss i Meta-Modele

## 1. Uruchomienie eksperymentu
Cały potok eksperymentalny uruchamiany jest jednym poleceniem z katalogu głównego repozytorium:

```bash
PYTHONPATH=src python -m fpl_project.models.run_experiments
```

Ziarno losowe zostało ustalone na `SEED = {config.SEED}` (`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`).

## 2. Podział danych i liczebności grup cenowych (`tier_counts.csv`)
- **Train (`2022/23`, `2023/24`, `2024/25`)**: `{len(data.X_train_raw)}` obserwacji (posortowanych globalnie po `(season, gw, code)`).
  - Wewnętrzny chronologiczny zbiór wczesnego zatrzymania dla sieci (`MLP`, `Conv1D`): ostatnie 8 kolejek sezonu `2024/25` (`gw >= {config.NN_ES_MIN_GW}`, `{data.n_es_val}` obserwacji); trening właściwy sieci na wcześniejszych kolejkach (`{data.n_train_sub}` obserwacji).
- **Valid (`2025/26`)**: `{len(valid_df)}` obserwacji.
- **Test (`2026/27`, `is_played == True`)**: `{len(test_df)}` obserwacji (`{n_test_gws}` kolejek).
- **Złączenie wewnętrzne (`inner join` po `(code, season, gw)`)**: `0` odrzuconych wierszy na `valid` oraz `0` na `test` (w `train_sub` dla `Conv1D` pominięto `{data.n_conv_warmup_dropped}` początkowych wierszy rozgrzewki okna `seq_len = {config.NN_SEQ_LEN}`).

Progi cenowe na zbiorze `train`: $p_{{\\text{{low}}}} = {data.p_low:.4f}$ mln £ (kwantyl `{config.PRICE_QUANTILE_LOW:.2f}`), $p_{{\\text{{high}}}} = {data.p_high:.4f}$ mln £ (kwantyl `{config.PRICE_QUANTILE_HIGH:.3f}`).

| Zbiór (`split`) | `budget` ($< {data.p_low:.2f}$) | `mid` ($[{data.p_low:.2f}, {data.p_high:.3f})$) | `premium` ($\\ge {data.p_high:.3f}$) | Razem (`total`) |
| :--- | ---: | ---: | ---: | ---: |
| `train` | {int(tc_train['budget'])} | {int(tc_train['mid'])} | {int(tc_train['premium'])} | {int(tc_train['total'])} |
| `valid` | {int(tc_valid['budget'])} | {int(tc_valid['mid'])} | {int(tc_valid['premium'])} | {int(tc_valid['total'])} |
| `test` | {int(tc_test['budget'])} | {int(tc_test['mid'])} | {int(tc_test['premium'])} | {int(tc_test['total'])} |

## 3. Funkcja straty `FPLLoss` i stałe konfiguracyjne
Stałe funkcji straty zdefiniowane w `src/fpl_project/scripts/config.py`:
- `EPS_MIN` ($\\varepsilon_{{\\min}}$) = `{config.EPS_MIN}`
- `W_MID` ($w_{{\\text{{mid}}}}$) = `{config.W_MID}`
- `W_PREMIUM` ($w_{{\\text{{premium}}}}$) = `{config.W_PREMIUM}`
- Stała normalizująca $c$ (`norm_const` $= \\text{{mean}}(\\ln(1 + m_i) + \\varepsilon_{{\\min}})$ na `train`, gdzie $m_i = \\text{{clip}}(\\text{{minutes\\_ema\\_3}} / 90, 0, 1)$): **`{data.c_norm:.4f}`**

Wzór wagi obserwacji:
$$w_i = \\frac{{w_{{\\text{{cena}}, i}} \\cdot (\\ln(1 + m_i) + \\varepsilon_{{\\min}})}}{{c}}, \\quad w_{{\\text{{cena}}, i}} \\in \\{{1.0, {config.W_MID}, {config.W_PREMIUM}\\}}, \\quad c = {data.c_norm:.4f}$$

## 4. Realizacja wariantu (B) dla GLM i XGBoost
- **XGBoost (`objective="reg:squarederror"`)**: przekazanie `sample_weight = w_i` do `fit()` minimalizuje ważony błąd średniokwadratowy $\\sum_i w_i (y_i - \\hat{{y}}_i)^2$.
- **GLM (`glum.GeneralizedLinearRegressor`)**: przekazanie `sample_weight = w_i` maksymalizuje ważoną log-wiarygodność wybranej rodziny rozkładu (Poisson, Negative Binomial ze strojonym $\\theta$, Tweedie z mocą $p \\in [1.1, 1.9]$) — analog `FPLLoss`.
- W `TimeSeriesSplit` CV (granice foldów pomiędzy kolejkami `(season, gw)`) wagi `w_i` przekazywane są do `estimator.fit()` oraz do ważonego kryterium oceny foldu (`neg_weighted_mse`).

## 5. Pliki wynikowe w katalogu `results/`
- `metrics_test.csv`, `metrics_valid.csv` — metryki na zbiorze testowym i walidacyjnym,
- `tier_counts.csv` — liczebności grup `budget`, `mid`, `premium` dla `train`, `valid`, `test`,
- `table_tiers.csv` — tabela MSE i MAE w podziale na grupy cenowe (zaokrąglona do 2 miejsc po przecinku),
- `meta_weights.csv` — wagi i wyrazy wolne wyuczonych meta-modeli,
- `preds_valid.parquet`, `preds_test.parquet` — predykcje wszystkich modeli na zbiorze walidacyjnym i testowym,
- `experiment_run.log` — pełny zapis przebiegu eksperymentu.
"""
    config.RESULTS_README_PATH.write_text(readme_content, encoding="utf-8")

    log_results_to_mlflow(
        df_metrics_valid=df_metrics_valid,
        df_metrics_test=df_metrics_test,
        best_params_log=best_params_log,
        data=data,
        artifact_paths=[
            config.METRICS_TEST_PATH,
            config.METRICS_VALID_PATH,
            config.META_WEIGHTS_PATH,
            config.TIER_COUNTS_PATH,
            config.TABLE_TIERS_PATH,
            config.RESULTS_README_PATH,
        ],
    )

    print("\n" + "=" * 115)
    print(f"LICZEBNOŚCI GRUP CENOWYCH ({config.TIER_COUNTS_PATH.name}):")
    print(df_tier_counts.to_string(index=False))
    print("\n" + "=" * 115)
    print(f"TABELA GOTOWA DO PRACY ({config.TABLE_TIERS_PATH.name}):")
    print(df_table_tiers.to_string(index=False))
    print("\n" + "=" * 115)
    print(f"TABELA WYNIKÓW NA ZBIORZE TESTOWYM ({config.METRICS_TEST_PATH.name}):")
    print("=" * 115)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    print(df_metrics_test.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 115)


def main() -> None:
    """Run the complete FPL experiment pipeline."""
    total_start = time.time()
    set_seed(config.SEED)
    run_verification_tests()

    device = initialize_device()
    print(f"Urządzenie obliczeniowe: {device} | Ziarno losowe (SEED): {config.SEED}")

    data = load_data()
    tab_val_preds, tab_test_preds, best_params_log = tune_tabular_models(data, device)
    nn_val_preds, nn_test_preds = train_neural_networks(data, device)

    preds_valid = {**tab_val_preds, **nn_val_preds}
    preds_test = {**tab_test_preds, **nn_test_preds}

    valid_df, test_df = merge_base_predictions(data, preds_valid, preds_test)
    df_metrics_valid, df_metrics_test, df_meta_weights = fit_meta_models(data, valid_df, test_df)
    save_results(
        data=data,
        valid_df=valid_df,
        test_df=test_df,
        df_metrics_valid=df_metrics_valid,
        df_metrics_test=df_metrics_test,
        df_meta_weights=df_meta_weights,
        best_params_log=best_params_log,
    )
    print(f"Czas całkowity wykonania eksperymentu: {time.time() - total_start:.2f} s")


if __name__ == "__main__":
    main()
