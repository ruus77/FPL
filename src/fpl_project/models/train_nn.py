from dataclasses import dataclass
from datetime import datetime
import sys
import time
from typing import NamedTuple

import mlflow
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.optim

from fpl_project.scripts import config
from fpl_project.scripts.data_utils.features_config import FeaturesConfig
from fpl_project.scripts.data_utils.features_processing import (
    build_and_save_features_engineering_data,
    sort_data,
)
from fpl_project.scripts.model_utils.data import (
    FPLDataPipe,
    SequenceDataPipe,
    prepare_sequence_dataframe,
    scale_target,
    train_test_split,
)
from fpl_project.scripts.model_utils.mlflow_tracker import MlflowTracker
from fpl_project.scripts.model_utils.models import (
    Conv1DRegressor,
    FPLLoss,
    MLP,
)
from fpl_project.scripts.model_utils.trainer import Trainer, set_seed

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

set_seed(77)

tracker = MlflowTracker()
tracker.set_tracking()

cols_map = FeaturesConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


@dataclass
class TrainingHyperparameters:
    seq_len: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 0.005
    dropout: float = 0.2
    num_epochs: int = 150
    patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_cooldown: int = 2
    seeds: int = 77

class MSELossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        return self.mse(y_pred.flatten(), y_true.flatten())


class ModelConfig(NamedTuple):
    train_dl: torch.utils.data.DataLoader
    valid_dl: torch.utils.data.DataLoader
    test_dl: torch.utils.data.DataLoader
    predict_dl: torch.utils.data.DataLoader | None
    value_idx: int
    minutes_idx: int
    m_threshold: float
    model_type: str
    num_features: int
    scaler: sklearn.preprocessing.StandardScaler


hipers = TrainingHyperparameters()

if not config.EDA.exists():
    from fpl_project.data_processing.data_import import fetch_and_save_eda_data
    fetch_and_save_eda_data()
if not config.FEATURES_ENGINEERING.exists():
    build_and_save_features_engineering_data()

data = sort_data(pd.read_parquet(config.FEATURES_ENGINEERING))
data.total_points = data.total_points.clip(lower=0)
data_seq = sort_data(pd.read_parquet(config.EDA))
data_seq.total_points = data_seq.total_points.clip(lower=0)
data_seq, seq_feature_cols, seq_context_cols = prepare_sequence_dataframe(data_seq, cols_map)

if "is_played" not in data.columns and "is_played" in data_seq.columns:
    data["is_played"] = data_seq["is_played"].values

data_splits = train_test_split(data)
data_splits_seq = train_test_split(data_seq)

data_splits, target_scaler = scale_target(data_splits)
data_splits_seq, target_scaler_seq = scale_target(data_splits_seq)

num_cols = [c for c in data.columns if c not in cols_map.id_cols + cols_map.static_cols + cols_map.target]
cat_cols = cols_map.static_cols

fpl_pipe_mlp = FPLDataPipe(num_cols=num_cols, cat_cols=cat_cols, batch_size=hipers.batch_size)
fpl_pipe_mlp.prepare_data(**data_splits)
train_dl_mlp, valid_dl_mlp, test_dl_mlp = fpl_pipe_mlp.get_dataloaders()
predict_dl_mlp = fpl_pipe_mlp.predict_dataloader

seq_pipe = SequenceDataPipe(
    feature_cols=seq_feature_cols,
    context_cols=seq_context_cols,
    seq_len=hipers.seq_len,
    batch_size=hipers.batch_size,
)
seq_pipe.prepare_data(**data_splits_seq)
train_dl_seq, valid_dl_seq, test_dl_seq = seq_pipe.get_dataloaders()
predict_dl_seq = seq_pipe.predict_dl

INPUT_SIZE_MLP = next(iter(train_dl_mlp))[0].shape[1]
N_FEATURES_SEQ = seq_pipe.total_input_features

print(
    f"[FPLDataPipe/mlp]  train={len(train_dl_mlp.dataset)}  "
    f"valid={len(valid_dl_mlp.dataset)}  test={len(test_dl_mlp.dataset)}  "
    f"num_features={INPUT_SIZE_MLP}"
)


def create_model_instance(model_name: str, dropout: float = 0.2) -> nn.Module:
    if model_name == "MLP":
        return MLP(
            input_size=INPUT_SIZE_MLP,
            hidden_sizes=config.NN_HIDDEN_SIZES,
            dropout=dropout,
        ).to(device)
    elif model_name == "Conv1DRegressor":
        return Conv1DRegressor(
            n_features=N_FEATURES_SEQ,
            seq_len=hipers.seq_len,
            channels=config.NN_CONV_CHANNELS,
            kernel_size=config.NN_KERNEL_SIZE,
            dropout=dropout,
        ).to(device)
    else:
        raise ValueError(f"Nieznany model: {model_name}")


feature_names_mlp = list(fpl_pipe_mlp.preprocessor.get_feature_names_out())
value_col_name_mlp = "num__value_ema_3"
minutes_col_name_mlp = "num__minutes_ema_3"
assert value_col_name_mlp in feature_names_mlp, f"Brak kolumny {value_col_name_mlp}"
assert minutes_col_name_mlp in feature_names_mlp, f"Brak kolumny {minutes_col_name_mlp}"

value_idx_mlp = feature_names_mlp.index(value_col_name_mlp)
value_idx_seq = seq_feature_cols.index("value")

minutes_idx_mlp = feature_names_mlp.index(minutes_col_name_mlp)
minutes_idx_seq = seq_feature_cols.index("minutes")


def calculate_scaled_minutes_threshold(
    scaler: sklearn.preprocessing.StandardScaler,
    col_idx: int,
    raw_minutes: float = 60.0,
) -> float:
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = float(scaler.mean_[col_idx])
        scale = float(scaler.scale_[col_idx])
        if scale <= 0:
            return 0.0
        return float((raw_minutes - mean) / scale)
    elif hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        data_min = float(scaler.data_min_[col_idx])
        data_max = float(scaler.data_max_[col_idx])
        if data_max <= data_min:
            return raw_minutes / 90.0
        return float(np.clip((raw_minutes - data_min) / (data_max - data_min), 0.0, 1.0))
    return raw_minutes / 90.0


mlp_num_scaler = fpl_pipe_mlp.preprocessor.named_transformers_["num"].named_steps["scaler"]
mlp_minutes_num_idx = num_cols.index(minutes_col_name_mlp.replace("num__", ""))
m_threshold_mlp = calculate_scaled_minutes_threshold(mlp_num_scaler, mlp_minutes_num_idx, raw_minutes=60.0)
m_threshold_seq = calculate_scaled_minutes_threshold(seq_pipe.scaler, minutes_idx_seq, raw_minutes=60.0)


def calculate_scaled_bounds(
    dataloader: torch.utils.data.DataLoader,
    value_idx: int,
    model_type: str,
) -> tuple[float, float]:
    all_features = torch.cat([x for x, _ in dataloader])
    if model_type == "mlp":
        prices = all_features[:, value_idx].numpy()
    elif model_type == "conv1d":
        prices = all_features[:, -1, value_idx].numpy()
    else:
        raise ValueError(f"Nieznany model_type: {model_type}")
    return float(np.quantile(prices, 0.50)), float(np.quantile(prices, 0.955))


MODEL_REGISTRY = {
    "MLP": ModelConfig(
        train_dl_mlp,
        valid_dl_mlp,
        test_dl_mlp,
        predict_dl_mlp,
        value_idx_mlp,
        minutes_idx_mlp,
        m_threshold_mlp,
        "mlp",
        INPUT_SIZE_MLP,
        target_scaler,
    ),
    "Conv1DRegressor": ModelConfig(
        train_dl_seq,
        valid_dl_seq,
        test_dl_seq,
        predict_dl_seq,
        value_idx_seq,
        minutes_idx_seq,
        m_threshold_seq,
        "conv1d",
        N_FEATURES_SEQ,
        target_scaler_seq,
    ),
}

METRIC_KEYS = [
    "mse",
    "mae",
    "adj_r2",
    "mse@budget",
    "mse@mid",
    "mse@premium",
    "mae@budget",
    "mae@mid",
    "mae@premium",
]


def aggregate_seed_metrics(seed_metrics: list[dict[str, float]], prefix: str) -> dict[str, float]:
    agg = {}
    for k in METRIC_KEYS:
        vals = np.array([m[k] for m in seed_metrics], dtype=np.float64)
        clean_k = k.replace("@", "_")
        agg[f"{prefix}_{clean_k}_mean"] = float(vals.mean())
    return agg


def run_multi_seed_experiment(
    model_name: str,
    setup: ModelConfig,
    loss_name: str,
    loss_fn: nn.Module,
    p_low: float,
    p_high: float,
    lr: float,
    weight_decay: float,
    dropout: float,
    seeds: int | tuple[int, ...] | list[int],
    num_epochs: int,
    tag_stage: str = "baseline",
    extra_params: dict | None = None,
    verbose_epochs: bool = True,
) -> dict:
    seeds_list = [seeds] if isinstance(seeds, int) else list(seeds)
    run_display_name = f"{model_name}_{loss_name}" if tag_stage == "baseline" else f"{model_name}_{loss_name}_{tag_stage}"
    print(f"\n{'=' * 85}")
    print(
        f"{run_display_name} | seeds={seeds_list} | "
        f"p_low={p_low:.4f}, p_high={p_high:.4f}, m_thresh={setup.m_threshold:.4f}"
    )

    valid_seed_metrics = []
    test_seed_metrics = []
    epochs_per_seed = []
    best_overall_valid_mse = float("inf")
    best_overall_model = None
    best_overall_trainer = None
    best_overall_results = None

    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    for seed in seeds_list:
        set_seed(seed)
        model = create_model_instance(model_name, dropout=dropout)

        trainer = Trainer(
            device=device,
            random_state=seed,
            p_low=p_low,
            p_high=p_high,
            num_features=setup.num_features,
            value_idx=setup.value_idx,
            model_type=setup.model_type,
            scaler=setup.scaler,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=hipers.scheduler_factor,
            patience=hipers.scheduler_patience,
            cooldown=hipers.scheduler_cooldown,
            min_lr=hipers.min_lr,
        )

        results, best_state = trainer.model_eval(
            train_dataloader=setup.train_dl,
            test_dataloader=setup.valid_dl,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epochs=num_epochs,
            patience=hipers.patience,
            scheduler=scheduler,
            monitor="mse",
            verbose=verbose_epochs,
        )

        model.load_state_dict(best_state)
        val_metrics = trainer.evaluate_dataloader(setup.valid_dl, model, loss_fn)
        tst_metrics = trainer.evaluate_dataloader(setup.test_dl, model, loss_fn)

        valid_seed_metrics.append(val_metrics)
        test_seed_metrics.append(tst_metrics)
        epochs_per_seed.append(len(results["train_loss"]))

        if val_metrics["mse"] < best_overall_valid_mse:
            best_overall_valid_mse = val_metrics["mse"]
            best_overall_model = model
            best_overall_trainer = trainer
            best_overall_results = results

        print(
            f"  [Seed {seed:4d}] Epok: {len(results['train_loss']):3d} | "
            f"Valid: MSE={val_metrics['mse']:.4f} MAE={val_metrics['mae']:.4f} AdjR2={val_metrics['adj_r2']:.4f} | "
            f"Test:  MSE={tst_metrics['mse']:.4f} MAE={tst_metrics['mae']:.4f} AdjR2={tst_metrics['adj_r2']:.4f}"
        )

    end_time = time.time()
    end_timestamp = datetime.now().isoformat()

    agg_valid = aggregate_seed_metrics(valid_seed_metrics, prefix="valid")
    agg_test = aggregate_seed_metrics(test_seed_metrics, prefix="test")

    preds_valid = np.array([], dtype=np.float32)
    preds_test = np.array([], dtype=np.float32)
    preds_predict = np.array([], dtype=np.float32)
    if best_overall_model is not None and best_overall_trainer is not None:
        preds_valid = best_overall_trainer.predict_dataloader(setup.valid_dl, best_overall_model)
        preds_test = best_overall_trainer.predict_dataloader(setup.test_dl, best_overall_model)
        if setup.predict_dl is not None:
            preds_predict = best_overall_trainer.predict_dataloader(setup.predict_dl, best_overall_model)

    with tracker.start_run(run_name=run_display_name):
        mlflow.set_tag("model_family", "neural_network")
        mlflow.set_tag("stage", tag_stage)

        model_arch = {}
        if model_name == "MLP":
            model_arch = {"hidden_sizes": config.NN_HIDDEN_SIZES, "dropout": dropout}
        elif model_name == "Conv1DRegressor":
            model_arch = {
                "channels": config.NN_CONV_CHANNELS,
                "kernel_size": config.NN_KERNEL_SIZE,
                "dropout": dropout,
            }

        params_to_log = {
            "model": model_name,
            "model_type": setup.model_type,
            "value_idx": setup.value_idx,
            "p_low": p_low,
            "p_high": p_high,
            "m_threshold_scaled": setup.m_threshold,
            "lr": lr,
            "min_lr": hipers.min_lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "patience": hipers.patience,
            "loss_fn": loss_name,
            "device": device,
            "optimizer": "AdamW",
            "seq_len": hipers.seq_len if setup.model_type != "mlp" else None,
            "batch_size": hipers.batch_size,
            "num_cols": len(num_cols) if setup.model_type == "mlp" else len(seq_feature_cols),
            "cat_cols": len(cat_cols) if setup.model_type == "mlp" else 0,
            "scheduler_factor": hipers.scheduler_factor,
            "scheduler_patience": hipers.scheduler_patience,
            "scheduler_cooldown": hipers.scheduler_cooldown,
            "scheduler_min_lr": hipers.min_lr,
            "seeds": str(seeds_list),
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            **model_arch,
        }
        if extra_params:
            params_to_log.update(extra_params)

        tracker.log_params(params_to_log)

        if best_overall_results is not None:
            for epoch in range(len(best_overall_results["train_loss"])):
                tracker.log_metrics({
                    "train_loss": best_overall_results["train_loss"][epoch],
                    "valid_loss": best_overall_results["test_loss"][epoch],
                    "train_mse": best_overall_results["train_mse"][epoch],
                    "valid_mse": best_overall_results["test_mse"][epoch],
                    "train_mae": best_overall_results["train_mae"][epoch],
                    "valid_mae": best_overall_results["test_mae"][epoch],
                    "train_mse_budget": best_overall_results["train_mse@budget"][epoch],
                    "valid_mse_budget": best_overall_results["test_mse@budget"][epoch],
                    "train_mse_mid": best_overall_results["train_mse@mid"][epoch],
                    "valid_mse_mid": best_overall_results["test_mse@mid"][epoch],
                    "train_mse_premium": best_overall_results["train_mse@premium"][epoch],
                    "valid_mse_premium": best_overall_results["test_mse@premium"][epoch],
                    "train_mae_budget": best_overall_results["train_mae@budget"][epoch],
                    "valid_mae_budget": best_overall_results["test_mae@budget"][epoch],
                    "train_mae_mid": best_overall_results["train_mae@mid"][epoch],
                    "valid_mae_mid": best_overall_results["test_mae@mid"][epoch],
                    "train_mae_premium": best_overall_results["train_mae@premium"][epoch],
                    "valid_mae_premium": best_overall_results["test_mae@premium"][epoch],
                    "train_r2_adj": best_overall_results["train_adj_r2"][epoch],
                    "valid_r2_adj": best_overall_results["test_adj_r2"][epoch],
                }, step=epoch)

        tracker.log_metrics({
            **agg_valid,
            **agg_test,
            "n_epochs_trained": float(np.mean(epochs_per_seed)),
            "training_time_seconds": end_time - start_time,
        })

        if best_overall_model is not None:
            tracker.log_model(best_overall_model, name=f"model_{run_display_name.lower()}")
            print(f"Zakończono synchronizację z MLflow dla: {run_display_name}")

    return {
        "model": model_name,
        "loss": loss_name,
        "stage": tag_stage,
        "preds_valid": preds_valid,
        "preds_test": preds_test,
        "preds_predict": preds_predict,
        **agg_valid,
        **agg_test,
    }


def save_nn_predictions(experiment_outputs: list[dict]) -> None:
    short_name_map = {
        "MLP": "mlp",
        "Conv1DRegressor": "conv1d",
    }

    valid_cols: dict[str, np.ndarray] = {}
    arch_runs: dict[str, list[dict]] = {}
    best_per_arch: dict[str, dict] = {}

    for item in experiment_outputs:
        run_col = f"{item['model'].lower()}_{item['loss'].lower()}"
        valid_cols[run_col] = item["preds_valid"]

        arch = item["model"]
        arch_runs.setdefault(arch, []).append(item)
        if arch not in best_per_arch or item["valid_mse_mean"] < best_per_arch[arch]["valid_mse_mean"]:
            best_per_arch[arch] = item

    for arch, best_item in best_per_arch.items():
        short_col = short_name_map.get(arch, arch.lower())
        valid_cols[short_col] = best_item["preds_valid"]

    preds_valid_nn = pd.DataFrame(valid_cols)
    if config.ML_PREDS_PATH.exists():
        existing_valid = pd.read_parquet(config.ML_PREDS_PATH)
        if len(existing_valid) == len(preds_valid_nn):
            for col in preds_valid_nn.columns:
                existing_valid[col] = preds_valid_nn[col].values
            preds_valid_nn = existing_valid
    preds_valid_nn.to_parquet(config.ML_PREDS_PATH)
    print(f"\nPreds valid shape: {preds_valid_nn.shape} -> {config.ML_PREDS_PATH}")

    meta_cols = [
        "name",
        "code",
        "team",
        "opponent_team",
        "position",
        "was_home",
        "season",
        "gw",
        "kickoff_time",
    ]
    X_test_raw = data_splits["X_test"]
    X_predict_raw = data_splits["X_predict"]

    test_meta = X_test_raw[meta_cols].copy().reset_index(drop=True)
    test_meta["is_played"] = True

    if not X_predict_raw.empty:
        pred_meta = X_predict_raw[meta_cols].copy().reset_index(drop=True)
        pred_meta["is_played"] = False
        season_2627_meta = pd.concat([test_meta, pred_meta], ignore_index=True)
    else:
        season_2627_meta = test_meta

    for arch, best_item in best_per_arch.items():
        short_col = short_name_map.get(arch, arch.lower())
        algo_df = season_2627_meta.copy()

        for item in arch_runs.get(arch, []):
            run_col = f"{item['model'].lower()}_{item['loss'].lower()}"
            if not X_predict_raw.empty and len(item["preds_predict"]) > 0:
                algo_df[run_col] = np.concatenate([item["preds_test"], item["preds_predict"]])
            else:
                algo_df[run_col] = item["preds_test"]

        if not X_predict_raw.empty and len(best_item["preds_predict"]) > 0:
            algo_df[short_col] = np.concatenate([best_item["preds_test"], best_item["preds_predict"]])
        else:
            algo_df[short_col] = best_item["preds_test"]

        target_path = config.get_preds_path(short_col)
        algo_df.to_parquet(target_path, index=False)
        print(
            f"Preds 26/27 ({short_col}) shape: {algo_df.shape} "
            f"(test={int(algo_df['is_played'].sum())}, next_gw={int((~algo_df['is_played']).sum())}) -> {target_path}"
        )



if __name__ == "__main__":
    all_outputs = []
    all_summaries = []

    for model_name, setup in MODEL_REGISTRY.items():
        p_low, p_high = calculate_scaled_bounds(
            dataloader=setup.train_dl,
            value_idx=setup.value_idx,
            model_type=setup.model_type,
        )

        loss_functions = {
            "FPLLoss": (
                FPLLoss(
                    p_low=p_low,
                    p_high=p_high,
                    value_idx=setup.value_idx,
                    minutes_idx=setup.minutes_idx,
                    model_type=setup.model_type,
                    w_mid=config.W_MID,
                    w_premium=config.W_PREMIUM,
                    eps_min=config.EPS_MIN,
                    minutes_mean=(
                        float(mlp_num_scaler.mean_[mlp_minutes_num_idx]) / 90.0
                        if setup.model_type == "mlp"
                        else 0.0
                    ),
                    minutes_scale=(
                        float(mlp_num_scaler.scale_[mlp_minutes_num_idx]) / 90.0
                        if setup.model_type == "mlp"
                        else 1.0
                    ),
                ),
                {
                    "w_mid": config.W_MID,
                    "w_premium": config.W_PREMIUM,
                    "eps_min": config.EPS_MIN,
                },
            ),
            "MSELoss": (MSELossWrapper(), {}),
        }

        for loss_name, (loss_fn, extra_loss_params) in loss_functions.items():
            output = run_multi_seed_experiment(
                model_name=model_name,
                setup=setup,
                loss_name=loss_name,
                loss_fn=loss_fn,
                p_low=p_low,
                p_high=p_high,
                lr=hipers.lr,
                weight_decay=hipers.weight_decay,
                dropout=hipers.dropout,
                seeds=hipers.seeds,
                num_epochs=hipers.num_epochs,
                tag_stage="baseline",
                extra_params=extra_loss_params,
            )
            all_outputs.append(output)
            summary = {k: v for k, v in output.items() if not k.startswith("preds_")}
            all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries).sort_values("valid_mse_mean", ascending=True).reset_index(drop=True)
    print(f"\n{'=' * 100}")
    print("RANKING MODELI (wybór po Valid MSE, końcowa ewaluacja na Test):")
    print(f"{'=' * 100}")
    for idx, row in summary_df.iterrows():
        print(
            f"{idx + 1:2d}. {row['model']:16s} | {row['loss']:14s} || "
            f"VALID: MSE={row['valid_mse_mean']:.4f}, "
            f"MAE={row['valid_mae_mean']:.4f}, "
            f"AdjR2={row['valid_adj_r2_mean']:.4f} || "
            f"TEST: MSE={row['test_mse_mean']:.4f}, "
            f"MAE={row['test_mae_mean']:.4f}, "
            f"AdjR2={row['test_adj_r2_mean']:.4f}"
        )

    save_nn_predictions(all_outputs)