from fpl_project.scripts import config
from fpl_project.scripts.model_utils.data import train_test_split, scale_target, FPLDataPipe, SequenceDataset, SequenceDataPipe
from fpl_project.scripts.model_utils.trainer import Trainer, set_seed
from fpl_project.scripts.data_utils.features_processing import sort_data
from fpl_project.scripts.data_utils.features_config import FEATURES_GROUP
from fpl_project.scripts.model_utils.models import MLP, Conv1DRegressor, LSTMRegressor
from fpl_project.scripts.model_utils.mlflow_tracker import MlflowTracker

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")

tracker = MlflowTracker()
tracker.set_tracking()

set_seed(77)

cols_map = FEATURES_GROUP
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

SEQ_LEN = 8
BATCH_SIZE = 64
LR = .0001
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 300
PATIENCE = 60
loss_fn = nn.MSELoss()

data = sort_data(pd.read_parquet(config.FPL_DATA_PATH))
data_seq = sort_data(pd.read_parquet(config.TIDY_DATA_PATH))

data_splits = train_test_split(data)
data_splits_seq = train_test_split(data_seq)

data_splits, target_scaler = scale_target(data_splits)
data_splits_seq, target_scaler_seq = scale_target(data_splits_seq)

ignored_num = {"name", "position", "element", "opponent_team", "gw", "code", "season", "kickoff_time"}
ignored_cat = {"web_name", "player_id", "gw"}
data_cols_set = set(data.columns)

ema_lagg_cols = [c for c in data.columns if "ema" in c or "lagg" in c]
pre_game_filtered = [c for c in cols_map["pre_game_cols"] if c in data_cols_set and c not in ignored_num]
num_cols_mlp = list(dict.fromkeys(ema_lagg_cols + pre_game_filtered))

cat_cols_mlp = [c for c in cols_map["static_cols"] if c in data_cols_set and c not in ignored_cat]

seq_feature_cols = [
    c for c in cols_map["fpl_cols"] + cols_map["perf_cols"]
    if c in data_seq.columns and c != "total_points"   
]
seq_feature_cols = list(dict.fromkeys(seq_feature_cols))

fpl_pipe_mlp = FPLDataPipe(num_cols=num_cols_mlp, cat_cols=cat_cols_mlp, batch_size=BATCH_SIZE)
fpl_pipe_mlp.prepare_data(**data_splits)
train_dl_mlp, valid_dl_mlp, test_dl_mlp = fpl_pipe_mlp.get_dataloaders()

seq_pipe = SequenceDataPipe(feature_cols=seq_feature_cols, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
seq_pipe.prepare_data(**data_splits_seq)
train_dl_seq, valid_dl_seq, test_dl_seq = seq_pipe.get_dataloaders() 

INPUT_SIZE_MLP = next(iter(train_dl_mlp))[0].shape[1]
N_FEATURES_SEQ = len(seq_feature_cols)

mlp = MLP(input_size=INPUT_SIZE_MLP, hidden_sizes=[32, 16]).to(device)

conv1d_model = Conv1DRegressor(
    n_features=N_FEATURES_SEQ, seq_len=SEQ_LEN, channels=[32, 16], kernel_size=3, dropout=0.2
).to(device)

lstm_model = LSTMRegressor(
    n_features=N_FEATURES_SEQ, hidden_size=32, num_layers=2, dropout=0.2, bidirectional=False
).to(device)

value_idx_mlp = list(fpl_pipe_mlp.preprocessor.get_feature_names_out()).index("num__value")
value_idx_seq = seq_feature_cols.index("value")


# ==========================================
# >>> MIEJSCE ZMIANY: DEFINICJA FUNKCJI <<<
# ==========================================
def calculate_scaled_bounds(dataloader: torch.utils.data.DataLoader, value_idx: int, model_type: str) -> tuple[float, float]:
    all_features = torch.cat([x for x, _ in dataloader])
    if model_type == "mlp":
        prices = all_features[:, value_idx].numpy()
    elif model_type == "lstm":
        prices = all_features[:, -1, value_idx].numpy()
    elif model_type == "conv1d":
        prices = all_features[:, value_idx, -1].numpy()
    else:
        raise ValueError(f"Nieznany model_type: {model_type}")
    return float(np.quantile(prices, 0.50)), float(np.quantile(prices, 0.955))
# ==========================================


MODEL_REGISTRY = {
    mlp.__class__.__name__: (
        mlp, train_dl_mlp, valid_dl_mlp, value_idx_mlp, "mlp", data.shape[1], target_scaler),
    conv1d_model.__class__.__name__: (
        conv1d_model, train_dl_seq, valid_dl_seq, value_idx_seq, "conv1d", data_seq.shape[1], target_scaler_seq),
    lstm_model.__class__.__name__: (
        lstm_model, train_dl_seq, valid_dl_seq, value_idx_seq, "lstm", data_seq.shape[1], target_scaler_seq)
}

ALL_RESULTS = {}
ALL_BEST_STATES = {}
DYNAMIC_BOUNDS = {}

for model_name, (model, tr_dl, va_dl, v_idx, m_type, num_features, scaler) in MODEL_REGISTRY.items():
    
    p_low, p_high = calculate_scaled_bounds(dataloader=tr_dl, value_idx=v_idx, model_type=m_type)
    DYNAMIC_BOUNDS[model_name] = {"p_low": p_low, "p_high": p_high}
    
    trainer = Trainer(
        device=device,
        random_state=77,
        p_low=p_low, 
        p_high=p_high,
        num_features=num_features,
        value_idx=v_idx, 
        model_type=m_type, 
        scaler=scaler
    )

    print(f"\n{'='*75}")
    print(f"TRAINING: {model_name} (p_low={p_low:.4f}, p_high={p_high:.4f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    results, best_state = trainer.model_eval(
        train_dataloader=tr_dl,
        test_dataloader=va_dl,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
    )

    ALL_RESULTS[model_name] = results
    ALL_BEST_STATES[model_name] = best_state

    best_state_cpu = {k: v.cpu() for k, v in best_state.items()}
    torch.save(best_state_cpu, f"best_{model_name.lower()}.pt")
    print(f"{model_name} | epok: {len(results['train_loss'])}")

print(f"\n{'='*20} MLFLOW LOGGING{'='*20}")

for model_name, (model, _, _, v_idx, m_type, _, _) in MODEL_REGISTRY.items():
    results = ALL_RESULTS[model_name]
    best_state = ALL_BEST_STATES[model_name]
    
    # Pobieramy progi dedykowane dla danego modelu wyznaczone przed treningiem
    p_low = DYNAMIC_BOUNDS[model_name]["p_low"]
    p_high = DYNAMIC_BOUNDS[model_name]["p_high"]

    with tracker.start_run(run_name=model_name):
        
        tracker.log_params({
            "model":         model_name,
            "model_type":    m_type,
            "value_idx":     v_idx,
            "p_low":         p_low,
            "p_high":        p_high,
            "lr":            LR,
            "weight_decay":  WEIGHT_DECAY,
            "num_epochs":    NUM_EPOCHS,
            "patience":      PATIENCE,
            "loss_fn":       loss_fn.__class__.__name__,
            "device":        device,
            "seq_len":       SEQ_LEN if m_type != "mlp" else None,
            "batch_size":    BATCH_SIZE,
        })

        for epoch in range(len(results["train_loss"])):
            tracker.log_metrics({
                "train_loss":        results["train_loss"][epoch],
                "valid_loss":        results["test_loss"][epoch],

                "train_mse_budget":  results["train_mse@budget"][epoch],
                "valid_mse_budget":  results["test_mse@budget"][epoch],
                "train_mse_mid":     results["train_mse@mid"][epoch],
                "valid_mse_mid":     results["test_mse@mid"][epoch],
                "train_mse_premium": results["train_mse@premium"][epoch],
                "valid_mse_premium": results["test_mse@premium"][epoch],

                "train_mae_budget":  results["train_mae@budget"][epoch],
                "valid_mae_budget":  results["test_mae@budget"][epoch],
                "train_mae_mid":     results["train_mae@mid"][epoch],
                "valid_mae_mid":     results["test_mae@mid"][epoch],
                "train_mae_premium": results["train_mae@premium"][epoch],
                "valid_mae_premium": results["test_mae@premium"][epoch],

                "train_r2_adj":      results["train_adj_r2"][epoch],
                "valid_r2_adj":      results["test_adj_r2"][epoch],
            }, step=epoch)

        tracker.log_metrics({
            "final_valid_loss":        results["test_loss"][-1],
            "final_valid_r2":          results["test_adj_r2"][-1],

            "final_valid_mse_budget":  results["test_mse@budget"][-1],
            "final_valid_mse_mid":     results["test_mse@mid"][-1],
            "final_valid_mse_premium": results["test_mse@premium"][-1],

            "final_valid_mae_budget":  results["test_mae@budget"][-1],
            "final_valid_mae_mid":     results["test_mae@mid"][-1],
            "final_valid_mae_premium": results["test_mae@premium"][-1],

            "best_valid_loss":         min(results["test_loss"]),
            "best_valid_r2_adj":       max(results["test_adj_r2"]),
            "n_epochs_trained":        len(results["train_loss"]),
        })

        model.load_state_dict(best_state)
        tracker.log_model(model, artifact_path=f"model_{model_name.lower()}")
        print(f" -> Zakończono synchronizację dla: {model_name}")