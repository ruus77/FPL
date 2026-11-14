"""Saving and loading of trained base models together with the preprocessing needed to reuse them.

Layout of ``config.TRAINED_MODELS_DIR``:
    glm_mse.joblib, glm_fplloss.joblib, xgboost_mse.joblib, xgboost_fplloss.joblib
        fitted estimators (input: output of the tabular preprocessor)
    mlp_mse.pt, mlp_fplloss.pt, conv1d_mse.pt, conv1d_fplloss.pt
        {"architecture", "init_kwargs", "state_dict"} of the neural networks
    preprocessing.joblib
        fitted tabular preprocessor, target/feature scalers, Conv1D feature lists, price thresholds

Usage in a notebook:
    from fpl_project.scripts.model_utils.pipeline.persistence import load_trained_models
    trained = load_trained_models()
    trained.tabular["xgboost_fplloss"].predict(trained.preprocessing["tabular_preprocessor"].transform(X))
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import torch
from torch import nn

from fpl_project.scripts import config
from fpl_project.scripts.model_utils.models import Conv1DRegressor, MLP
from fpl_project.scripts.model_utils.pipeline.containers import (
    NNDataBundle,
    NNTrainingResult,
    PreparedData,
    TabularTuningResult,
)

PREPROCESSING_FILE = "preprocessing.joblib"
ARCHITECTURES: dict[str, type[nn.Module]] = {"mlp": MLP, "conv1d": Conv1DRegressor}


@dataclass(frozen=True)
class TrainedModels:
    """Base models and preprocessing artifacts restored from disk."""

    tabular: dict[str, object]
    neural: dict[str, nn.Module]
    preprocessing: dict[str, object]


def _architecture_init_kwargs(bundle: NNDataBundle) -> dict:
    """Constructor arguments matching pipeline.neural._instantiate_neural_model."""
    if bundle.name == "mlp":
        return {
            "input_size": bundle.n_features,
            "hidden_sizes": list(config.NN_HIDDEN_SIZES),
            "dropout": config.NN_DROPOUT,
        }
    return {
        "n_features": bundle.n_features,
        "seq_len": config.NN_SEQ_LEN,
        "channels": list(config.NN_CONV_CHANNELS),
        "kernel_size": config.NN_KERNEL_SIZE,
        "dropout": config.NN_DROPOUT,
    }


def _build_preprocessing(data: PreparedData) -> dict[str, object]:
    return {
        "tabular_preprocessor": data.tabular.preprocessor,
        "p_low": data.weights.p_low,
        "p_high": data.weights.p_high,
        "mlp": {"target_scaler": data.mlp_bundle.target_scaler},
        "conv1d": {
            "target_scaler": data.conv_bundle.target_scaler,
            "feature_scaler": data.conv_bundle.feature_scaler,
            "feature_cols": list(data.conv_bundle.feature_cols),
            "context_cols": list(data.conv_bundle.context_cols),
            "seq_len": config.NN_SEQ_LEN,
        },
    }


def _save_neural_model(model: nn.Module, bundle: NNDataBundle, path: Path) -> None:
    init_kwargs = _architecture_init_kwargs(bundle)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # Guard: the stored constructor arguments must rebuild exactly this architecture.
    ARCHITECTURES[bundle.name](**init_kwargs).load_state_dict(state_dict, strict=True)
    torch.save({"architecture": bundle.name, "init_kwargs": init_kwargs, "state_dict": state_dict}, path)


def save_trained_models(
    data: PreparedData,
    tab_result: TabularTuningResult,
    nn_result: NNTrainingResult,
    models_dir: Path = config.TRAINED_MODELS_DIR,
) -> None:
    """Persist all fitted base models and their preprocessing artifacts."""
    models_dir.mkdir(parents=True, exist_ok=True)

    for key, estimator in tab_result.models.items():
        joblib.dump(estimator, models_dir / f"{key}.joblib")

    bundles = {data.mlp_bundle.name: data.mlp_bundle, data.conv_bundle.name: data.conv_bundle}
    for key, model in nn_result.models.items():
        bundle_name = key.rsplit("_", 1)[0]
        _save_neural_model(model, bundles[bundle_name], models_dir / f"{key}.pt")

    joblib.dump(_build_preprocessing(data), models_dir / PREPROCESSING_FILE)
    saved = sorted(tab_result.models) + sorted(nn_result.models)
    print(f"[Modele] Zapisano {len(saved)} modeli bazowych do {models_dir}: {', '.join(saved)}")


def load_trained_models(
    models_dir: Path = config.TRAINED_MODELS_DIR,
    device: str = "cpu",
) -> TrainedModels:
    """Load base models saved by save_trained_models (neural networks are returned in eval mode)."""
    tabular = {path.stem: joblib.load(path) for path in sorted(models_dir.glob("*.joblib"))
               if path.name != PREPROCESSING_FILE}

    neural: dict[str, nn.Module] = {}
    for path in sorted(models_dir.glob("*.pt")):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = ARCHITECTURES[checkpoint["architecture"]](**checkpoint["init_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        neural[path.stem] = model.to(device).eval()

    preprocessing = joblib.load(models_dir / PREPROCESSING_FILE)
    return TrainedModels(tabular=tabular, neural=neural, preprocessing=preprocessing)
