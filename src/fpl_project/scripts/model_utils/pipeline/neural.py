"""Stage 2b: train neural base models (MLP, Conv1D) in MSE and FPLLoss variants."""

import numpy as np
import torch
import torch.nn as nn

from fpl_project.scripts import config
from fpl_project.scripts.model_utils.losses import FPLLoss, MSELossWrapper
from fpl_project.scripts.model_utils.models import Conv1DRegressor, MLP
from fpl_project.scripts.model_utils.pipeline.containers import (
    FPLWeightConfig,
    NNDataBundle,
    NNTrainingResult,
    PredictionSplit,
    PreparedData,
)
from fpl_project.scripts.model_utils.trainer import Trainer, set_seed


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
    trained_models: dict[str, nn.Module] = {}

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
            trained_models[key] = model_nn
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
        models=trained_models,
    )
