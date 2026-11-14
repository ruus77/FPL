"""Stage 5: predict unplayed fixtures (is_played == False) with the already trained base models.

Runs after all evaluation stages and only reuses fitted models and preprocessing, so it does not
influence any metric. Inputs are prepared exactly as for the test split:
    - GLM / XGBoost / MLP: fitted tabular preprocessor (features are lagged, available before the match),
    - Conv1D: windows built from the history of played gameweeks (train + valid + test), scaled with the
      feature scaler fitted on train.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from fpl_project.scripts import config
from fpl_project.scripts.data_utils.features_config import FeaturesConfig
from fpl_project.scripts.model_utils.data import prepare_sequence_dataframe, train_test_split
from fpl_project.scripts.model_utils.datasets import AuxSequenceDataset, collate_conv1d_with_aux
from fpl_project.scripts.model_utils.pipeline.containers import (
    NNDataBundle,
    NNTrainingResult,
    PreparedData,
    TabularTuningResult,
)
from fpl_project.scripts.model_utils.pipeline.data_preparation import _load_and_align_raw_dataframes
from fpl_project.scripts.model_utils.trainer import Trainer

KEY_COLS = ["code", "season", "gw"]
FORECAST_ID_COLS = [
    "code", "name", "position", "team", "opponent_team",
    "was_home", "season", "gw", "kickoff_time", "value_ema_3", "minutes_ema_3",
]


def _predict_with_network(
    model: torch.nn.Module,
    dataloader: DataLoader,
    bundle: NNDataBundle,
    data: PreparedData,
    device: str,
) -> np.ndarray:
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
    return trainer.predict_dataloader(dataloader, model)


def _predict_tabular_and_mlp(
    X_predict: pd.DataFrame,
    data: PreparedData,
    tab_result: TabularTuningResult,
    nn_result: NNTrainingResult,
    device: str,
) -> dict[str, np.ndarray]:
    X_tab = data.tabular.preprocessor.transform(X_predict)
    preds = {
        key: model.predict(X_tab.to_numpy())
        for key, model in tab_result.models.items()
    }

    mlp_dl = DataLoader(
        TensorDataset(torch.tensor(X_tab.to_numpy(), dtype=torch.float32)),
        batch_size=config.NN_BATCH_SIZE,
        shuffle=False,
    )
    for key, model in nn_result.models.items():
        if key.startswith(data.mlp_bundle.name):
            preds[key] = _predict_with_network(model, mlp_dl, data.mlp_bundle, data, device)
    return preds


def _predict_conv1d(
    df_eda: pd.DataFrame,
    data: PreparedData,
    nn_result: NNTrainingResult,
    device: str,
) -> pd.DataFrame:
    bundle = data.conv_bundle
    df_seq, feature_cols, context_cols = prepare_sequence_dataframe(df_eda, FeaturesConfig())
    assert tuple(feature_cols) == bundle.feature_cols and tuple(context_cols) == bundle.context_cols

    splits = train_test_split(df_seq)
    history_df = pd.concat([splits["X_train"], splits["X_valid"], splits["X_test"]], ignore_index=True)
    target_df = splits["X_predict"].reset_index(drop=True)
    history_df[feature_cols] = bundle.feature_scaler.transform(history_df[feature_cols])
    target_df[feature_cols] = bundle.feature_scaler.transform(target_df[feature_cols])

    dataset = AuxSequenceDataset(
        target_df,
        feature_cols=feature_cols,
        context_cols=context_cols,
        seq_len=config.NN_SEQ_LEN,
        history_df=history_df,
    )
    conv_dl = DataLoader(
        dataset,
        batch_size=config.NN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_conv1d_with_aux,
    )

    out = target_df[KEY_COLS].copy()
    for key, model in nn_result.models.items():
        if key.startswith(bundle.name):
            out[key] = _predict_with_network(model, conv_dl, bundle, data, device)
    return out


def predict_unplayed_fixtures(
    data: PreparedData,
    tab_result: TabularTuningResult,
    nn_result: NNTrainingResult,
    device: str,
) -> pd.DataFrame:
    """Predict points of all base models for rows with is_played == False (empty frame if there are none)."""
    df_fe, df_eda = _load_and_align_raw_dataframes()
    X_predict = train_test_split(df_fe)["X_predict"].reset_index(drop=True)
    out = X_predict[FORECAST_ID_COLS].copy()
    model_keys = list(tab_result.models) + list(nn_result.models)

    if X_predict.empty:
        for key in model_keys:
            out[key] = np.array([], dtype=np.float64)
        return out

    for key, pred in _predict_tabular_and_mlp(X_predict, data, tab_result, nn_result, device).items():
        out[key] = pred
    conv_preds = _predict_conv1d(df_eda, data, nn_result, device)
    out = pd.merge(out, conv_preds, on=KEY_COLS, how="left")
    return out[FORECAST_ID_COLS + model_keys]


def save_unplayed_predictions(preds_predict: pd.DataFrame) -> None:
    """Save predictions for unplayed fixtures to results/preds_predict.parquet."""
    preds_predict.to_parquet(config.PREDS_PREDICT_PATH, index=False)
    print(
        f"[Predykcje] Nierozegrane mecze (is_played == False): {len(preds_predict)} wierszy "
        f"-> {config.PREDS_PREDICT_PATH}"
    )
