from dataclasses import dataclass
from pathlib import Path
import sys
import time
import warnings

from glum import GeneralizedLinearRegressor
from glum._glm import get_family, get_link
import numpy as np
import pandas as pd
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
    prepare_sequence_dataframe,
    scale_target,
    train_test_split,
)
from fpl_project.scripts.model_utils.mlflow_tracker import MlflowTracker
from fpl_project.scripts.model_utils.models import (
    Conv1DRegressor,
    FPLLoss,
    MLP,
    compute_fpl_weights,
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


@dataclass(frozen=True)
class RawSplits:
    """Raw feature and target splits aligned across train, validation, and test sets."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


@dataclass(frozen=True)
class FPLWeightConfig:
    """Price thresholds and sample weights for FPLLoss across splits."""

    p_low: float
    p_high: float
    w_train: np.ndarray
    w_valid: np.ndarray
    w_test: np.ndarray


@dataclass(frozen=True)
class TabularDataBundle:
    """Preprocessed tabular feature matrices for GLM and XGBoost."""

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    n_features: int


@dataclass(frozen=True)
class NNDataBundle:
    """Dataloaders, target scaler, and metadata for a single neural network architecture."""

    name: str
    n_features: int
    train_dl: DataLoader
    es_val_dl: DataLoader
    valid_dl: DataLoader
    test_dl: DataLoader
    target_scaler: StandardScaler
    n_train_sub: int
    n_es_val: int
    n_warmup_dropped: int = 0


@dataclass(frozen=True)
class PreparedData:
    """Complete container of datasets, weights, and neural network bundles."""

    raw: RawSplits
    weights: FPLWeightConfig
    tabular: TabularDataBundle
    mlp_bundle: NNDataBundle
    conv_bundle: NNDataBundle


@dataclass(frozen=True)
class PredictionSplit:
    """Predictions dictionary for validation and test splits."""

    valid: dict[str, np.ndarray]
    test: dict[str, np.ndarray]


@dataclass(frozen=True)
class TabularTuningResult:
    """Predictions and best hyperparameters from tabular RandomizedSearchCV."""

    predictions: PredictionSplit
    best_params: dict[str, dict]


@dataclass(frozen=True)
class NNTrainingResult:
    """Predictions, training histories, and hyperparameters from neural network training."""

    predictions: PredictionSplit
    histories: dict[str, dict[str, list[float]]]
    nn_params: dict[str, dict]


@dataclass(frozen=True)
class MergedPredictions:
    """Aligned validation and test DataFrames containing all base model predictions."""

    valid_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass(frozen=True)
class MetaEvaluationResult:
    """Evaluation tables for validation, test, and learned meta-model weights."""

    metrics_valid: pd.DataFrame
    metrics_test: pd.DataFrame
    meta_weights: pd.DataFrame


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


class AuxSequenceDataset(Dataset):
    """Sequence dataset for Conv1D returning (window, target, [value_ema_3, minutes_ema_3 / 90])."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        context_cols: list[str] | None = None,
        target_col: str = "total_points",
        seq_len: int = config.NN_SEQ_LEN,
        player_col: str = "code",
        time_col: str = "gw",
        season_col: str = "season",
        history_df: pd.DataFrame | None = None,
    ):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.context_cols = context_cols or []
        n_hist = len(feature_cols)

        if history_df is not None and not history_df.empty:
            self.sequences = self._build_with_history(
                df=df,
                history_df=history_df,
                target_col=target_col,
                seq_len=seq_len,
                n_hist=n_hist,
                player_col=player_col,
                season_col=season_col,
                time_col=time_col,
            )
        else:
            self.sequences = self._build_without_history(
                df=df,
                target_col=target_col,
                seq_len=seq_len,
                player_col=player_col,
                season_col=season_col,
                time_col=time_col,
            )

    def _build_with_history(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame,
        target_col: str,
        seq_len: int,
        n_hist: int,
        player_col: str,
        season_col: str,
        time_col: str,
    ) -> list[tuple[np.ndarray, float, np.ndarray]]:
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
                n_prefix = len(hist_grp)
            else:
                combined = grp.reset_index(drop=True)
                n_prefix = 0

            feats = combined[self.feature_cols].values.astype(np.float32)
            ctx_feats = (
                combined[self.context_cols].values.astype(np.float32)
                if self.context_cols
                else None
            )
            target = combined[target_col].values.astype(np.float32)
            raw_price = combined["_raw_price"].values.astype(np.float32)
            raw_min = combined["_raw_min"].values.astype(np.float32)
            orig_positions = combined["_orig_pos"].values

            for t in range(n_prefix, len(combined)):
                hist_window = self._slice_or_pad(feats, t, seq_len, n_hist)
                window = self._attach_context(hist_window, ctx_feats, t, seq_len)
                aux = np.array([raw_price[t], raw_min[t]], dtype=np.float32)
                ordered_sequences[int(orig_positions[t])] = (window, float(target[t]), aux)

        return ordered_sequences  # type: ignore[return-value]

    def _build_without_history(
        self,
        df: pd.DataFrame,
        target_col: str,
        seq_len: int,
        player_col: str,
        season_col: str,
        time_col: str,
    ) -> list[tuple[np.ndarray, float, np.ndarray]]:
        sequences: list[tuple[np.ndarray, float, np.ndarray]] = []
        df_sorted = df.copy().sort_values([player_col, season_col, time_col]).reset_index(drop=True)
        for _, grp in df_sorted.groupby([player_col, season_col], sort=False):
            feats = grp[self.feature_cols].values.astype(np.float32)
            ctx_feats = (
                grp[self.context_cols].values.astype(np.float32)
                if self.context_cols
                else None
            )
            target = grp[target_col].values.astype(np.float32)
            raw_price = grp["_raw_price"].values.astype(np.float32)
            raw_min = grp["_raw_min"].values.astype(np.float32)

            for t in range(seq_len, len(grp)):
                hist_window = feats[t - seq_len : t]
                window = self._attach_context(hist_window, ctx_feats, t, seq_len)
                aux = np.array([raw_price[t], raw_min[t]], dtype=np.float32)
                sequences.append((window, float(target[t]), aux))
        return sequences

    @staticmethod
    def _slice_or_pad(feats: np.ndarray, t: int, seq_len: int, n_hist: int) -> np.ndarray:
        if t >= seq_len:
            return feats[t - seq_len : t]
        if t > 0:
            pad = np.zeros((seq_len - t, n_hist), dtype=np.float32)
            return np.vstack([pad, feats[:t]])
        return np.zeros((seq_len, n_hist), dtype=np.float32)

    @staticmethod
    def _attach_context(
        hist_window: np.ndarray,
        ctx_feats: np.ndarray | None,
        t: int,
        seq_len: int,
    ) -> np.ndarray:
        if ctx_feats is None:
            return hist_window
        ctx_step = np.broadcast_to(ctx_feats[t], (seq_len, ctx_feats.shape[1])).astype(np.float32)
        return np.hstack([hist_window, ctx_step])

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


def _load_and_align_raw_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
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
        df_eda_raw,
        fe_aux[key_cols + ["_raw_price", "_raw_min"]],
        on=key_cols,
        how="inner",
    )

    df_fe = df_fe_raw.sort_values(["season", "gw", "code"]).reset_index(drop=True)
    df_eda = df_eda_merged.sort_values(["season", "gw", "code"]).reset_index(drop=True)
    assert (df_fe[key_cols].values == df_eda[key_cols].values).all()
    return df_fe, df_eda


def _compute_fpl_weight_config(raw_splits: RawSplits) -> FPLWeightConfig:
    price_train_raw = raw_splits.X_train["value_ema_3"].to_numpy(dtype=np.float64)
    p_low = float(np.quantile(price_train_raw, config.PRICE_QUANTILE_LOW))
    p_high = float(np.quantile(price_train_raw, config.PRICE_QUANTILE_HIGH))

    w_train = compute_fpl_weights(raw_splits.X_train, p_low=p_low, p_high=p_high)
    w_valid = compute_fpl_weights(raw_splits.X_valid, p_low=p_low, p_high=p_high)
    w_test = compute_fpl_weights(raw_splits.X_test, p_low=p_low, p_high=p_high)

    print("=" * 90)
    print("PARAMETRY FPLLoss (wyliczone na surowych kolumnach train):")
    print(f"  eps_min = {config.EPS_MIN}, w_mid = {config.W_MID}, w_premium = {config.W_PREMIUM}")
    print(f"  p_low (kwantyl {config.PRICE_QUANTILE_LOW:.2f} value_ema_3)  = {p_low:.4f} mln £")
    print(f"  p_high (kwantyl {config.PRICE_QUANTILE_HIGH:.3f} value_ema_3)= {p_high:.4f} mln £")
    print(f"  Wagi w_train: mean={w_train.mean():.4f}, min={w_train.min():.4f}, max={w_train.max():.4f}")
    print("=" * 90)

    return FPLWeightConfig(
        p_low=p_low,
        p_high=p_high,
        w_train=w_train,
        w_valid=w_valid,
        w_test=w_test,
    )


def _build_tabular_bundle(
    df_fe: pd.DataFrame,
    raw_splits: RawSplits,
    cols_map: FeaturesConfig,
) -> TabularDataBundle:
    num_cols = [
        c for c in df_fe.columns if c not in cols_map.id_cols + cols_map.static_cols + cols_map.target
    ]
    cat_cols = cols_map.static_cols

    tab_pipe = FPLDataPipe(num_cols=num_cols, cat_cols=cat_cols, batch_size=config.NN_BATCH_SIZE)
    X_train_tab = tab_pipe.preprocessor.fit_transform(raw_splits.X_train)
    X_valid_tab = tab_pipe.preprocessor.transform(raw_splits.X_valid)
    X_test_tab = tab_pipe.preprocessor.transform(raw_splits.X_test)

    return TabularDataBundle(
        X_train=X_train_tab,
        X_valid=X_valid_tab,
        X_test=X_test_tab,
        n_features=int(X_train_tab.shape[1]),
    )


def _make_aux_tensor(df_part: pd.DataFrame) -> torch.Tensor:
    p_col = df_part["value_ema_3"].to_numpy(dtype=np.float32)
    m_col = np.clip(df_part["minutes_ema_3"].to_numpy(dtype=np.float32) / 90.0, 0.0, 1.0)
    return torch.tensor(np.column_stack([p_col, m_col]), dtype=torch.float32)


def _build_mlp_bundle(
    splits_fe: dict[str, pd.DataFrame | pd.Series],
    raw_splits: RawSplits,
    tabular: TabularDataBundle,
) -> NNDataBundle:
    es_mask = (
        (raw_splits.X_train["season"] == config.NN_ES_SEASON)
        & (raw_splits.X_train["gw"] >= config.NN_ES_MIN_GW)
    )
    sub_mask = ~es_mask

    splits_fe_scaled, target_scaler_fe = scale_target(splits_fe)
    y_train_scaled = splits_fe_scaled["y_train"].reset_index(drop=True)
    y_valid_scaled = splits_fe_scaled["y_valid"].reset_index(drop=True)
    y_test_scaled = splits_fe_scaled["y_test"].reset_index(drop=True)

    train_generator = torch.Generator().manual_seed(config.SEED)
    train_dl = DataLoader(
        TensorDataset(
            torch.tensor(tabular.X_train.loc[sub_mask].to_numpy(), dtype=torch.float32),
            torch.tensor(y_train_scaled.loc[sub_mask].to_numpy().reshape(-1, 1), dtype=torch.float32),
            _make_aux_tensor(raw_splits.X_train.loc[sub_mask]),
        ),
        batch_size=config.NN_BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
    )
    es_val_dl = DataLoader(
        TensorDataset(
            torch.tensor(tabular.X_train.loc[es_mask].to_numpy(), dtype=torch.float32),
            torch.tensor(y_train_scaled.loc[es_mask].to_numpy().reshape(-1, 1), dtype=torch.float32),
            _make_aux_tensor(raw_splits.X_train.loc[es_mask]),
        ),
        batch_size=config.NN_BATCH_SIZE,
        shuffle=False,
    )
    valid_dl = DataLoader(
        TensorDataset(
            torch.tensor(tabular.X_valid.to_numpy(), dtype=torch.float32),
            torch.tensor(y_valid_scaled.to_numpy().reshape(-1, 1), dtype=torch.float32),
            _make_aux_tensor(raw_splits.X_valid),
        ),
        batch_size=config.NN_BATCH_SIZE,
        shuffle=False,
    )
    test_dl = DataLoader(
        TensorDataset(
            torch.tensor(tabular.X_test.to_numpy(), dtype=torch.float32),
            torch.tensor(y_test_scaled.to_numpy().reshape(-1, 1), dtype=torch.float32),
            _make_aux_tensor(raw_splits.X_test),
        ),
        batch_size=config.NN_BATCH_SIZE,
        shuffle=False,
    )

    return NNDataBundle(
        name="mlp",
        n_features=tabular.n_features,
        train_dl=train_dl,
        es_val_dl=es_val_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,
        target_scaler=target_scaler_fe,
        n_train_sub=int(sub_mask.sum()),
        n_es_val=int(es_mask.sum()),
    )


def _merge_xy_dataframe(X_df: pd.DataFrame, y_ser: pd.Series) -> pd.DataFrame:
    out_df = X_df.copy().reset_index(drop=True)
    out_df["total_points"] = y_ser.to_numpy(dtype=np.float32)
    return out_df


def _build_conv1d_bundle(
    df_eda: pd.DataFrame,
    cols_map: FeaturesConfig,
) -> NNDataBundle:
    df_eda_seq, seq_feature_cols, context_cols = prepare_sequence_dataframe(df_eda, cols_map)
    p_conv1d = len(seq_feature_cols) + len(context_cols)

    splits_eda = train_test_split(df_eda_seq)
    splits_eda_scaled, target_scaler_eda = scale_target(splits_eda)

    eda_train_df = _merge_xy_dataframe(splits_eda_scaled["X_train"], splits_eda_scaled["y_train"])
    eda_valid_df = _merge_xy_dataframe(splits_eda_scaled["X_valid"], splits_eda_scaled["y_valid"])
    eda_test_df = _merge_xy_dataframe(splits_eda_scaled["X_test"], splits_eda_scaled["y_test"])

    seq_scaler = StandardScaler()
    eda_train_df[seq_feature_cols] = seq_scaler.fit_transform(eda_train_df[seq_feature_cols])
    eda_valid_df[seq_feature_cols] = seq_scaler.transform(eda_valid_df[seq_feature_cols])
    eda_test_df[seq_feature_cols] = seq_scaler.transform(eda_test_df[seq_feature_cols])

    es_mask_eda = (
        (eda_train_df["season"] == config.NN_ES_SEASON)
        & (eda_train_df["gw"] >= config.NN_ES_MIN_GW)
    )
    eda_train_sub_df = eda_train_df.loc[~es_mask_eda].reset_index(drop=True)
    eda_es_val_df = eda_train_df.loc[es_mask_eda].reset_index(drop=True)

    conv_train_sub_ds = AuxSequenceDataset(
        eda_train_sub_df,
        feature_cols=seq_feature_cols,
        context_cols=context_cols,
        seq_len=config.NN_SEQ_LEN,
    )
    conv_es_val_ds = AuxSequenceDataset(
        eda_es_val_df,
        feature_cols=seq_feature_cols,
        context_cols=context_cols,
        seq_len=config.NN_SEQ_LEN,
        history_df=eda_train_sub_df,
    )
    conv_valid_ds = AuxSequenceDataset(
        eda_valid_df,
        feature_cols=seq_feature_cols,
        context_cols=context_cols,
        seq_len=config.NN_SEQ_LEN,
        history_df=eda_train_df,
    )
    conv_test_ds = AuxSequenceDataset(
        eda_test_df,
        feature_cols=seq_feature_cols,
        context_cols=context_cols,
        seq_len=config.NN_SEQ_LEN,
        history_df=pd.concat([eda_train_df, eda_valid_df], ignore_index=True),
    )

    train_generator = torch.Generator().manual_seed(config.SEED)
    train_dl = DataLoader(
        conv_train_sub_ds,
        batch_size=config.NN_BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        collate_fn=collate_conv1d_with_aux,
        drop_last=True,
    )
    es_val_dl = DataLoader(
        conv_es_val_ds,
        batch_size=config.NN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_conv1d_with_aux,
    )
    valid_dl = DataLoader(
        conv_valid_ds,
        batch_size=config.NN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_conv1d_with_aux,
    )
    test_dl = DataLoader(
        conv_test_ds,
        batch_size=config.NN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_conv1d_with_aux,
    )

    return NNDataBundle(
        name="conv1d",
        n_features=p_conv1d,
        train_dl=train_dl,
        es_val_dl=es_val_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,
        target_scaler=target_scaler_eda,
        n_train_sub=len(eda_train_sub_df),
        n_es_val=len(eda_es_val_df),
        n_warmup_dropped=len(eda_train_sub_df) - len(conv_train_sub_ds),
    )


def load_data() -> PreparedData:
    """Load, align, split, and scale tabular and sequential datasets."""
    df_fe, df_eda = _load_and_align_raw_dataframes()
    splits_fe = train_test_split(df_fe)

    raw_splits = RawSplits(
        X_train=splits_fe["X_train"].reset_index(drop=True),
        y_train=splits_fe["y_train"].reset_index(drop=True),
        X_valid=splits_fe["X_valid"].reset_index(drop=True),
        y_valid=splits_fe["y_valid"].reset_index(drop=True),
        X_test=splits_fe["X_test"].reset_index(drop=True),
        y_test=splits_fe["y_test"].reset_index(drop=True),
    )

    print(
        f"Rozmiary podziału: train={len(raw_splits.X_train)} (sezony 22/23-24/25), "
        f"valid={len(raw_splits.X_valid)} (sezon 25/26), test={len(raw_splits.X_test)} (rozegrane 26/27)"
    )

    weights = _compute_fpl_weight_config(raw_splits)
    cols_map = FeaturesConfig()
    tabular = _build_tabular_bundle(df_fe, raw_splits, cols_map)
    mlp_bundle = _build_mlp_bundle(splits_fe, raw_splits, tabular)
    conv_bundle = _build_conv1d_bundle(df_eda, cols_map)

    return PreparedData(
        raw=raw_splits,
        weights=weights,
        tabular=tabular,
        mlp_bundle=mlp_bundle,
        conv_bundle=conv_bundle,
    )


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
    )


def _instantiate_neural_model(bundle: NNDataBundle, device: str) -> nn.Module:
    if bundle.name == "mlp":
        return MLP(
            input_size=bundle.n_features,
            hidden_sizes=config.NN_HIDDEN_SIZES,
            dropout=config.NN_DROPOUT,
        ).to(device)
    return Conv1DRegressor(
        n_features=bundle.n_features,
        seq_len=config.NN_SEQ_LEN,
        channels=config.NN_CONV_CHANNELS,
        kernel_size=config.NN_KERNEL_SIZE,
        dropout=config.NN_DROPOUT,
    ).to(device)


def _create_loss_and_monitor(variant: str, weights: FPLWeightConfig) -> tuple[nn.Module, str]:
    if variant == "mse":
        return MSELossWrapper(), "mse"
    loss_fn = FPLLoss(
        p_low=weights.p_low,
        p_high=weights.p_high,
        value_idx=0,
        minutes_idx=1,
        model_type="mlp",
        w_mid=config.W_MID,
        w_premium=config.W_PREMIUM,
        eps_min=config.EPS_MIN,
    )
    return loss_fn, "loss"


def train_neural_networks(data: PreparedData, device: str) -> NNTrainingResult:
    """Train MLP and Conv1D in MSE and FPLLoss variants with chronological early stopping."""
    preds_valid: dict[str, np.ndarray] = {}
    preds_test: dict[str, np.ndarray] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    nn_params: dict[str, dict] = {}

    for bundle in [data.mlp_bundle, data.conv_bundle]:
        for variant in ["mse", "fplloss"]:
            key = f"{bundle.name}_{variant}"
            print("\n" + "=" * 90)
            print(f"TRENING SIECI NEURONOWEJ: {bundle.name.upper()} — WARIANT: {variant.upper()} (seed={config.SEED})")
            print("=" * 90)
            set_seed(config.SEED)

            model_nn = _instantiate_neural_model(bundle, device)
            loss_fn, monitor_metric = _create_loss_and_monitor(variant, data.weights)

            trainer = Trainer(
                device=device,
                num_features=bundle.n_features,
                random_state=config.SEED,
                p_low=data.weights.p_low,
                p_high=data.weights.p_high,
                value_idx=0,
                model_type="mlp",
                scaler=bundle.target_scaler,
            )

            optimizer = torch.optim.AdamW(
                model_nn.parameters(),
                lr=config.NN_LR,
                weight_decay=config.NN_WEIGHT_DECAY,
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
                train_dataloader=bundle.train_dl,
                test_dataloader=bundle.es_val_dl,
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
            es_metrics = trainer.evaluate_dataloader(bundle.es_val_dl, model_nn, loss_fn)
            print(
                f"  [{bundle.name.upper()} - {variant}] Zatrzymano po {epochs_ran} epokach | "
                f"ES_val MSE={es_metrics['mse']:.4f}, Loss={es_metrics['loss']:.4f}, AdjR2={es_metrics['adj_r2']:.4f}"
            )

            preds_valid[key] = trainer.predict_dataloader(
                bundle.valid_dl, model_nn, clip_preds=True
            )
            preds_test[key] = trainer.predict_dataloader(
                bundle.test_dl, model_nn, clip_preds=True
            )
            histories[key] = history
            arch_info = (
                {"hidden_sizes": str(config.NN_HIDDEN_SIZES)}
                if bundle.name == "mlp"
                else {
                    "channels": str(config.NN_CONV_CHANNELS),
                    "kernel_size": config.NN_KERNEL_SIZE,
                    "seq_len": config.NN_SEQ_LEN,
                }
            )
            nn_params[key] = {
                **arch_info,
                "dropout": config.NN_DROPOUT,
                "lr": config.NN_LR,
                "weight_decay": config.NN_WEIGHT_DECAY,
                "batch_size": config.NN_BATCH_SIZE,
                "epochs_ran": epochs_ran,
            }

    return NNTrainingResult(
        predictions=PredictionSplit(valid=preds_valid, test=preds_test),
        histories=histories,
        nn_params=nn_params,
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
        p_val_nnls = np.clip(meta_nnls.predict(X_meta_val), config.PRED_CLIP_MIN, config.PRED_CLIP_MAX)
        p_tst_nnls = np.clip(meta_nnls.predict(X_meta_tst), config.PRED_CLIP_MIN, config.PRED_CLIP_MAX)
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
        p_val_ctx = np.clip(meta_ctx.predict(X_ctx_val), config.PRED_CLIP_MIN, config.PRED_CLIP_MAX)
        p_tst_ctx = np.clip(meta_ctx.predict(X_ctx_tst), config.PRED_CLIP_MIN, config.PRED_CLIP_MAX)
        valid_df[f"meta_context_{variant}"] = p_val_ctx
        test_df[f"meta_context_{variant}"] = p_tst_ctx
        _append_meta_weights(
            meta_weights_rows, "meta_context", variant, meta_ctx.intercept_, list(X_ctx_val.columns), meta_ctx.coef_
        )

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
            mv = evaluate(y_val, pv, price_val, data.weights.w_valid, p_low, p_high, p_cnt)
            mt = evaluate(y_tst, pt, price_tst, data.weights.w_test, p_low, p_high, p_cnt)
            metrics_valid_rows.append({"model": m_label, "variant": variant, "level": "meta", **mv})
            metrics_test_rows.append({"model": m_label, "variant": variant, "level": "meta", **mt})

    return MetaEvaluationResult(
        metrics_valid=pd.DataFrame(metrics_valid_rows),
        metrics_test=pd.DataFrame(metrics_test_rows),
        meta_weights=pd.DataFrame(meta_weights_rows),
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
    eval_result: MetaEvaluationResult,
    best_params_log: dict[str, dict],
    nn_result: NNTrainingResult,
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

    df_metrics_test = eval_result.metrics_test
    df_metrics_valid = eval_result.metrics_valid

    for idx in range(len(df_metrics_test)):
        row_t = df_metrics_test.iloc[idx]
        row_v = df_metrics_valid.iloc[idx]
        m_name = str(row_t["model"])
        variant = str(row_t["variant"])
        level = str(row_t["level"])
        run_name = f"{m_name}_{variant}_{level}"
        model_key = f"{m_name}_{variant}"

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
                "p_low": data.weights.p_low,
                "p_high": data.weights.p_high,
                "eps_min": config.EPS_MIN,
                "w_mid": config.W_MID,
                "w_premium": config.W_PREMIUM,
            }
            if model_key in best_params_log:
                for k, v in best_params_log[model_key].items():
                    params[f"hp_{k}"] = v
            if model_key in nn_result.nn_params:
                for k, v in nn_result.nn_params[model_key].items():
                    params[f"nn_{k}"] = v
            tracker.log_params(params)

            if model_key in nn_result.histories:
                hist = nn_result.histories[model_key]
                for ep in range(len(hist["train_loss"])):
                    tracker.log_metrics(
                        {
                            "train_loss": float(hist["train_loss"][ep]),
                            "es_val_loss": float(hist["test_loss"][ep]),
                            "train_mse": float(hist["train_mse"][ep]),
                            "es_val_mse": float(hist["test_mse"][ep]),
                            "train_mae": float(hist["train_mae"][ep]),
                            "es_val_mae": float(hist["test_mae"][ep]),
                        },
                        step=ep,
                    )

            metrics_to_log = {}
            for mc in metric_cols:
                clean_mc = mc.replace("@", "_")
                metrics_to_log[f"valid_{clean_mc}"] = float(row_v[mc])
                metrics_to_log[f"test_{clean_mc}"] = float(row_t[mc])
            if model_key in nn_result.nn_params:
                metrics_to_log["n_epochs_trained"] = float(nn_result.nn_params[model_key]["epochs_ran"])
            tracker.log_metrics(metrics_to_log)

            if idx == len(df_metrics_test) - 1:
                for ap in artifact_paths:
                    if ap.exists():
                        mlflow.log_artifact(str(ap))

    print(f"[MLflow] Zalogowano {len(df_metrics_test)} przebiegów eksperymentalnych do MLflow.")


def _render_results_readme(
    data: PreparedData,
    merged: MergedPredictions,
    eval_result: MetaEvaluationResult,
    df_tier_counts: pd.DataFrame,
    best_params_log: dict[str, dict],
) -> str:
    tc_train = df_tier_counts[df_tier_counts["split"] == "train"].iloc[0]
    tc_valid = df_tier_counts[df_tier_counts["split"] == "valid"].iloc[0]
    tc_test = df_tier_counts[df_tier_counts["split"] == "test"].iloc[0]
    n_test_gws = int(merged.test_df["gw"].nunique())

    nb_default = get_family("negative.binomial")
    tw_default = get_family("tweedie")
    auto_links = {
        fam_name: type(get_link("auto", get_family(fam_name))).__name__
        for fam_name in ["poisson", "negative.binomial", "tweedie", "normal"]
    }
    glm_params_mse = best_params_log.get("glm_mse", {})
    glm_params_fpl = best_params_log.get("glm_fplloss", {})

    metrics_md_lines = [
        "| Model | Wariant | Poziom | $p$ | MSE | MAE | $R^2$ | Adj $R^2$ | FPLLoss | MSE budget | MSE mid | MSE premium | MAE budget | MAE mid | MAE premium |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in eval_result.metrics_test.iterrows():
        metrics_md_lines.append(
            f"| `{r['model']}` | `{r['variant']}` | `{r['level']}` | {int(r['p'])} | "
            f"{r['mse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} | {r['adj_r2']:.4f} | {r['fplloss']:.4f} | "
            f"{r['mse@budget']:.4f} | {r['mse@mid']:.4f} | {r['mse@premium']:.4f} | "
            f"{r['mae@budget']:.4f} | {r['mae@mid']:.4f} | {r['mae@premium']:.4f} |"
        )
    metrics_md_table = "\n".join(metrics_md_lines)
    p_low, p_high = data.weights.p_low, data.weights.p_high

    return f"""# Eksperymenty Predykcji Punktów FPL — Modele Bazowe, FPLLoss i Meta-Modele

## 1. Uruchomienie eksperymentu
Cały potok eksperymentalny uruchamiany jest jednym poleceniem z katalogu głównego repozytorium:

```bash
PYTHONPATH=src python -m fpl_project.models.run_experiments
```

Ziarno losowe zostało ustalone na `SEED = {config.SEED}` (`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`).

## 2. Podział danych i liczebności grup cenowych (`tier_counts.csv`)
- **Train (`2022/23`, `2023/24`, `2024/25`)**: `{len(data.raw.X_train)}` obserwacji (posortowanych globalnie po `(season, gw, code)`).
  - Wewnętrzny chronologiczny zbiór wczesnego zatrzymania dla sieci (`MLP`, `Conv1D`): ostatnie 8 kolejek sezonu `2024/25` (`gw >= {config.NN_ES_MIN_GW}`, `{data.mlp_bundle.n_es_val}` obserwacji); trening właściwy sieci na wcześniejszych kolejkach (`{data.mlp_bundle.n_train_sub}` obserwacji).
- **Valid (`2025/26`)**: `{len(merged.valid_df)}` obserwacji.
- **Test (`2026/27`, `is_played == True`)**: `{len(merged.test_df)}` obserwacji (`{n_test_gws}` kolejek).
- **Złączenie wewnętrzne (`inner join` po `(code, season, gw)`)**: `0` odrzuconych wierszy na `valid` oraz `0` na `test` (w `train_sub` dla `Conv1D` pominięto `{data.conv_bundle.n_warmup_dropped}` początkowych wierszy rozgrzewki okna `seq_len = {config.NN_SEQ_LEN}`).

Progi cenowe na zbiorze `train`: $p_{{\\text{{low}}}} = {p_low:.4f}$ mln £ (kwantyl `{config.PRICE_QUANTILE_LOW:.2f}`), $p_{{\\text{{high}}}} = {p_high:.4f}$ mln £ (kwantyl `{config.PRICE_QUANTILE_HIGH:.3f}`).

| Zbiór (`split`) | `budget` ($< {p_low:.2f}$) | `mid` ($[{p_low:.2f}, {p_high:.3f})$) | `premium` ($\\ge {p_high:.3f}$) | Razem (`total`) |
| :--- | ---: | ---: | ---: | ---: |
| `train` | {int(tc_train['budget'])} | {int(tc_train['mid'])} | {int(tc_train['premium'])} | {int(tc_train['total'])} |
| `valid` | {int(tc_valid['budget'])} | {int(tc_valid['mid'])} | {int(tc_valid['premium'])} | {int(tc_valid['total'])} |
| `test` | {int(tc_test['budget'])} | {int(tc_test['mid'])} | {int(tc_test['premium'])} | {int(tc_test['total'])} |

## 3. Funkcja straty `FPLLoss` i stałe konfiguracyjne
Stałe funkcji straty zdefiniowane w `src/fpl_project/scripts/config.py`:
- `EPS_MIN` ($\\varepsilon_{{\\min}}$) = `{config.EPS_MIN}`
- `W_MID` ($w_{{\\text{{mid}}}}$) = `{config.W_MID}`
- `W_PREMIUM` ($w_{{\\text{{premium}}}}$) = `{config.W_PREMIUM}`

Wzór wagi obserwacji (gdzie $m_i = \\text{{clip}}(\\text{{minutes\\_ema\\_3}} / 90, 0, 1)$):
$$w_i = w_{{\\text{{cena}}, i}} \\cdot (\\ln(1 + m_i) + \\varepsilon_{{\\min}}), \\quad w_{{\\text{{cena}}, i}} \\in \\{{1.0, {config.W_MID}, {config.W_PREMIUM}\\}}$$

Ponieważ wagi $w_i$ nie są dzielone przez stałą normalizującą, wartości `FPLLoss` w tabelach są wyrażone w innej skali niż `MSE`.

## 4. Konfiguracja GLM (`glum`) i realizacja wariantu (B)
- **Przestrzeń przeszukiwania GLM (`GLM_PARAM_GRID`)**:
  - `alpha`: `stats.loguniform(1e-5, 0.01)`
  - `l1_ratio`: `stats.uniform(loc=0.0, scale=1.0)`
  - `family`: `["poisson", "negative.binomial", "tweedie", "normal"]`
  - `link`: `["auto", "log"]`
- **Domyślne parametry w bibliotece `glum` (`glum._glm.get_family` oraz `glum._glm.get_link`)**:
  - `family="negative.binomial"` (bez podanego parametru) przyjmuje $\\theta = {float(nb_default.theta):.1f}$ (`NegativeBinomialDistribution(1.0)`),
  - `family="tweedie"` (bez podanego parametru) przyjmuje moc $p = {float(tw_default.power):.1f}$ (`TweedieDistribution(1.5)`),
  - `link="auto"` zwraca dla poszczególnych rodzin:
    - `"poisson"` $\\rightarrow$ `{auto_links['poisson']}` (`log`),
    - `"negative.binomial"` $\\rightarrow$ `{auto_links['negative.binomial']}` (`log`),
    - `"tweedie"` $\\rightarrow$ `{auto_links['tweedie']}` (`log`),
    - `"normal"` $\\rightarrow$ `{auto_links['normal']}` (`identity`).
- **Wybrane hiperparametry GLM w `RandomizedSearchCV`**:
  - Wariant `mse`: `{glm_params_mse}`
  - Wariant `fplloss`: `{glm_params_fpl}`
- **Wariant (B) dla GLM i XGBoost**:
  - **XGBoost (`objective="reg:squarederror"`)**: przekazanie `sample_weight = w_i` do `fit()` minimalizuje ważony błąd średniokwadratowy $\\sum_i w_i (y_i - \\hat{{y}}_i)^2$.
  - **GLM (`glum.GeneralizedLinearRegressor`)**: przekazanie `sample_weight = w_i` maksymalizuje ważoną log-wiarygodność wybranej rodziny rozkładu.
  - W `TimeSeriesSplit` CV (granice foldów pomiędzy kolejkami `(season, gw)`) wagi `w_i` przekazywane są do `estimator.fit()` oraz do ważonego kryterium oceny foldu (`neg_weighted_mse`).

## 5. Wyniki na zbiorze testowym (`metrics_test.csv`)
{metrics_md_table}

## 6. Pliki wynikowe w katalogu `results/`
- `metrics_test.csv`, `metrics_valid.csv` — metryki na zbiorze testowym i walidacyjnym,
- `tier_counts.csv` — liczebności grup `budget`, `mid`, `premium` dla `train`, `valid`, `test`,
- `table_tiers.csv` — tabela MSE i MAE w podziale na grupy cenowe (zaokrąglona do 2 miejsc po przecinku),
- `meta_weights.csv` — wagi i wyrazy wolne wyuczonych meta-modeli,
- `preds_valid.parquet`, `preds_test.parquet` — predykcje wszystkich modeli na zbiorze walidacyjnym i testowym,
- `experiment_run.log` — pełny zapis przebiegu eksperymentu.
"""


def save_results(
    data: PreparedData,
    merged: MergedPredictions,
    eval_result: MetaEvaluationResult,
    best_params_log: dict[str, dict],
    nn_result: NNTrainingResult,
) -> None:
    """Save all result CSV/Parquet files, update results/README.md, and log to MLflow."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_tier_counts = compute_tier_counts_table(
        splits_dict={"train": data.raw.X_train, "valid": merged.valid_df, "test": merged.test_df},
        p_low=data.weights.p_low,
        p_high=data.weights.p_high,
    )
    df_table_tiers = build_table_tiers(eval_result.metrics_test)

    eval_result.metrics_test.to_csv(config.METRICS_TEST_PATH, index=False)
    eval_result.metrics_valid.to_csv(config.METRICS_VALID_PATH, index=False)
    merged.valid_df.to_parquet(config.PREDS_VALID_PATH, index=False)
    merged.test_df.to_parquet(config.PREDS_TEST_PATH, index=False)
    eval_result.meta_weights.to_csv(config.META_WEIGHTS_PATH, index=False)
    df_tier_counts.to_csv(config.TIER_COUNTS_PATH, index=False)
    df_table_tiers.to_csv(config.TABLE_TIERS_PATH, index=False, float_format="%.2f")

    readme_content = _render_results_readme(
        data=data,
        merged=merged,
        eval_result=eval_result,
        df_tier_counts=df_tier_counts,
        best_params_log=best_params_log,
    )
    config.RESULTS_README_PATH.write_text(readme_content, encoding="utf-8")

    log_results_to_mlflow(
        eval_result=eval_result,
        best_params_log=best_params_log,
        nn_result=nn_result,
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
    print(eval_result.metrics_test.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 115)


def main() -> None:
    """Run the complete FPL experiment pipeline."""
    total_start = time.time()
    set_seed(config.SEED)
    run_verification_tests()

    device = initialize_device()
    print(f"Urządzenie obliczeniowe: {device} | Ziarno losowe (SEED): {config.SEED}")

    data = load_data()
    tab_result = tune_tabular_models(data, device)
    nn_result = train_neural_networks(data, device)

    combined_preds = PredictionSplit(
        valid={**tab_result.predictions.valid, **nn_result.predictions.valid},
        test={**tab_result.predictions.test, **nn_result.predictions.test},
    )

    merged = merge_base_predictions(data, combined_preds)
    eval_result = fit_meta_models(data, merged)
    save_results(
        data=data,
        merged=merged,
        eval_result=eval_result,
        best_params_log=tab_result.best_params,
        nn_result=nn_result,
    )
    print(f"Czas całkowity wykonania eksperymentu: {time.time() - total_start:.2f} s")


if __name__ == "__main__":
    main()
