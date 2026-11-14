"""Immutable containers passed between stages of the experiment pipeline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader


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
    preprocessor: ColumnTransformer


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
    feature_scaler: StandardScaler | None = None
    feature_cols: tuple[str, ...] = ()
    context_cols: tuple[str, ...] = ()


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
    models: dict[str, BaseEstimator]


@dataclass(frozen=True)
class NNTrainingResult:
    """Predictions, training histories, and hyperparameters from neural network training."""

    predictions: PredictionSplit
    histories: dict[str, dict[str, list[float]]]
    nn_params: dict[str, dict]
    models: dict[str, nn.Module]


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
