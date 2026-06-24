from torch.nn import MSELoss

from fpl_project.scripts import config
from fpl_project.scripts.model_utils.data import train_test_split, scale_target, FPLDataPipe, SequenceDataPipe
from fpl_project.scripts.model_utils.trainer import Trainer, set_seed
from fpl_project.scripts.data_utils.features_processing import sort_data
from fpl_project.scripts.data_utils.features_config import FeaturesConfig
from fpl_project.scripts.model_utils.models import MLP, Conv1DRegressor, LSTMRegressor, FPLLoss
from fpl_project.scripts.model_utils.mlflow_tracker import MlflowTracker

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import sklearn.preprocessing

from dataclasses import dataclass, field
from typing import NamedTuple


tracker = MlflowTracker()
tracker.set_tracking()
set_seed(77)

cols_map = FeaturesConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


@dataclass
class TrainingHyperparameters:
    seq_len: int = 8
    batch_size: int = 128
    lr: float = 0.0001
    weight_decay: float = 0.05
    num_epochs: int = 150
    patience: int = 50
    loss_fn: nn.Module = field(default_factory=MSELoss)

class ModelConfig(NamedTuple):
    model: nn.Module
    train_dl: torch.utils.data.DataLoader
    valid_dl: torch.utils.data.DataLoader
    value_idx: int
    minutes_idx: int
    model_type: str
    num_features: int
    scaler: sklearn.preprocessing.MinMaxScaler


hipers = TrainingHyperparameters()


data = sort_data(pd.read_parquet(config.FPL_DATA_PATH))
data_seq = sort_data(pd.read_parquet(config.TIDY_DATA_PATH))

data_splits = train_test_split(data)
data_splits_seq = train_test_split(data_seq)

data_splits, target_scaler = scale_target(data_splits)
data_splits_seq, target_scaler_seq = scale_target(data_splits_seq)

ignored_num = {"name", "position", "element", "opponent_team", "gw", "code", "season", "kickoff_time", "team"}
ignored_cat = {"web_name", "player_id", "gw"}
data_cols_set = set(data.columns)

ema_lagg_cols = [c for c in data.columns if "ema" in c or "lagg" in c]
pre_game_filtered = [c for c in getattr(cols_map, "pre_game_cols") if c in data_cols_set and c not in ignored_num]
num_cols_mlp = list(dict.fromkeys(ema_lagg_cols + pre_game_filtered))
cat_cols_mlp = [c for c in getattr(cols_map, "static_cols") if c in data_cols_set and c not in ignored_cat]

seq_feature_cols = [
    c for c in getattr(cols_map, "fpl_cols") + getattr(cols_map, "perf_cols")
    if c in data_seq.columns and c != "total_points"
]
seq_feature_cols = list(dict.fromkeys(seq_feature_cols))


fpl_pipe_mlp = FPLDataPipe(num_cols=num_cols_mlp, cat_cols=cat_cols_mlp, batch_size=hipers.batch_size)
fpl_pipe_mlp.prepare_data(**data_splits)
train_dl_mlp, valid_dl_mlp, _ = fpl_pipe_mlp.get_dataloaders()

seq_pipe = SequenceDataPipe(feature_cols=seq_feature_cols, seq_len=hipers.seq_len, batch_size=hipers.batch_size)
seq_pipe.prepare_data(**data_splits_seq)
train_dl_seq, valid_dl_seq, _ = seq_pipe.get_dataloaders()

INPUT_SIZE_MLP = next(iter(train_dl_mlp))[0].shape[1]
N_FEATURES_SEQ = len(seq_feature_cols)


mlp = MLP(input_size=INPUT_SIZE_MLP, hidden_sizes=[32, 16]).to(device)
conv1d_model = Conv1DRegressor(n_features=N_FEATURES_SEQ, seq_len=hipers.seq_len, channels=[32, 16], kernel_size=3,
                               dropout=0.2).to(device)
lstm_model = LSTMRegressor(n_features=N_FEATURES_SEQ, hidden_size=32, num_layers=2, dropout=0.2,
                           bidirectional=False).to(device)

value_idx_mlp = list(fpl_pipe_mlp.preprocessor.get_feature_names_out()).index("num__value")
value_idx_seq = seq_feature_cols.index("value")

minutes_idx_mlp = list(fpl_pipe_mlp.preprocessor.get_feature_names_out()).index("num__minutes_ema_3")
minutes_idx_seq = seq_feature_cols.index("minutes")


def calculate_scaled_bounds(dataloader: torch.utils.data.DataLoader, value_idx: int, model_type: str) -> tuple[
    float, float]:
    all_features = torch.cat([x for x, _ in dataloader])
    if model_type == "mlp":
        prices = all_features[:, value_idx].numpy()
    elif model_type in ["lstm", "conv1d"]:
        prices = all_features[:, -1, value_idx].numpy()
    else:
        raise ValueError(f"Nieznany model_type: {model_type}")
    return float(np.quantile(prices, 0.50)), float(np.quantile(prices, 0.955))


MODEL_REGISTRY = {
    mlp.__class__.__name__: ModelConfig(mlp, train_dl_mlp,
                                        valid_dl_mlp, value_idx_mlp,
                                        minutes_idx_mlp, "mlp",
                                        data.shape[1], target_scaler),

    #conv1d_model.__class__.__name__: ModelConfig(conv1d_model, train_dl_seq,
    #                                             valid_dl_seq, value_idx_seq,
    #                                             minutes_idx_seq, "conv1d",
    #                                             data_seq.shape[1], target_scaler_seq),

    #lstm_model.__class__.__name__: ModelConfig(lstm_model, train_dl_seq,
    #                                           valid_dl_seq, value_idx_seq,
    #                                           minutes_idx_seq, "lstm",
    #                                           data_seq.shape[1], target_scaler_seq)
}
# setup <- (model, train_dl, valid_dl, value_idx, minutes_idx, model_type, num_features, scaler)

for model_name, setup in MODEL_REGISTRY.items():
    p_low, p_high = calculate_scaled_bounds(dataloader=setup.train_dl, value_idx=setup.value_idx,
                                            model_type=setup.model_type)

    hipers.loss_fn = FPLLoss(p_low=p_low, p_high=p_high, value_idx=setup.value_idx, minutes_idx=setup.minutes_idx,
                             model_type=setup.model_type, w_premium=2, under_predict_penalty=3, device=device)

    trainer = Trainer(
        device=device,
        random_state=77,
        p_low=p_low,
        p_high=p_high,
        num_features=setup.num_features,
        value_idx=setup.value_idx,
        model_type=setup.model_type,
        scaler=setup.scaler
    )

    print(f"\n{'=' * 75}")
    print(f"{model_name} (p_low={p_low:.4f}, p_high={p_high:.4f})")

    optimizer = torch.optim.AdamW(setup.model.parameters(), lr=hipers.lr, weight_decay=hipers.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.7,
                                                           patience=12,
                                                           cooldown=10,
                                                           min_lr=1e-5)

    results, best_state = trainer.model_eval(
        train_dataloader=setup.train_dl,
        test_dataloader=setup.valid_dl,
        model=setup.model,
        optimizer=optimizer,
        loss_fn=hipers.loss_fn,
        num_epochs=hipers.num_epochs,
        patience=hipers.patience,
        scheduler=scheduler
    )

    print(f"{model_name} | epok: {len(results['train_loss'])}")


    with tracker.start_run(run_name=model_name):
        tracker.log_params({
            "model": model_name,
            "model_type": setup.model_type,
            "value_idx": setup.value_idx,
            "p_low": p_low,
            "p_high": p_high,
            "lr": hipers.lr,
            "weight_decay": hipers.weight_decay,
            "num_epochs": hipers.num_epochs,
            "patience": hipers.patience,
            "loss_fn": hipers.loss_fn.__class__.__name__,
            "device": device,
            "optimizer": optimizer.__class__.__name__,
            "seq_len": hipers.seq_len if setup.model_type != "mlp" else None,
            "batch_size": hipers.batch_size,
        })


        for epoch in range(len(results["train_loss"])):
            tracker.log_metrics({
                "train_loss": results["train_loss"][epoch],
                "valid_loss": results["test_loss"][epoch],
                "train_mse_budget": results["train_mse@budget"][epoch],
                "valid_mse_budget": results["test_mse@budget"][epoch],
                "train_mse_mid": results["train_mse@mid"][epoch],
                "valid_mse_mid": results["test_mse@mid"][epoch],
                "train_mse_premium": results["train_mse@premium"][epoch],
                "valid_mse_premium": results["test_mse@premium"][epoch],
                "train_mae_budget": results["train_mae@budget"][epoch],
                "valid_mae_budget": results["test_mae@budget"][epoch],
                "train_mae_mid": results["train_mae@mid"][epoch],
                "valid_mae_mid": results["test_mae@mid"][epoch],
                "train_mae_premium": results["train_mae@premium"][epoch],
                "valid_mae_premium": results["test_mae@premium"][epoch],
                "train_r2_adj": results["train_adj_r2"][epoch],
                "valid_r2_adj": results["test_adj_r2"][epoch],
            }, step=epoch)


        tracker.log_metrics({
            "final_valid_loss": results["test_loss"][-1],
            "final_valid_r2": results["test_adj_r2"][-1],
            "final_valid_mse_budget": results["test_mse@budget"][-1],
            "final_valid_mse_mid": results["test_mse@mid"][-1],
            "final_valid_mse_premium": results["test_mse@premium"][-1],
            "final_valid_mae_budget": results["test_mae@budget"][-1],
            "final_valid_mae_mid": results["test_mae@mid"][-1],
            "final_valid_mae_premium": results["test_mae@premium"][-1],
            "best_valid_loss": min(results["test_loss"]),
            "best_valid_r2_adj": max(results["test_adj_r2"]),
            "n_epochs_trained": len(results["train_loss"]),
        })


        #setup.model.load_state_dict(best_state)
        #tracker.log_model(setup.model, name=f"model_{model_name.lower()}")
        print(f"Zakończono synchronizację z MLflow dla: {model_name}")