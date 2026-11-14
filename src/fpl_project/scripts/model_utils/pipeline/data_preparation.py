"""Stage 1: load, align, split, and scale tabular and sequential datasets."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

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
from fpl_project.scripts.model_utils.datasets import AuxSequenceDataset, collate_conv1d_with_aux
from fpl_project.scripts.model_utils.losses import compute_fpl_weights
from fpl_project.scripts.model_utils.pipeline.containers import (
    FPLWeightConfig,
    NNDataBundle,
    PreparedData,
    RawSplits,
    TabularDataBundle,
)


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
        preprocessor=tab_pipe.preprocessor,
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
        feature_scaler=seq_scaler,
        feature_cols=tuple(seq_feature_cols),
        context_cols=tuple(context_cols),
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
