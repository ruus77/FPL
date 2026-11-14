import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fpl_project.scripts.data_utils.features_config import FeaturesConfig


def prepare_sequence_dataframe(
    df: pd.DataFrame,
    cols_map: FeaturesConfig | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add historical points and pre-game match context columns for sequential models."""
    if cols_map is None:
        cols_map = FeaturesConfig()

    out = df.copy()
    out["points_hist"] = out["total_points"].fillna(0.0).astype(np.float32)
    out["was_home_num"] = out["was_home"].astype(np.float32)

    pos_str = out["position"].astype(str)
    for pos_label in ["DEF", "FWD", "MID"]:
        out[f"pos_{pos_label}"] = (pos_str == pos_label).astype(np.float32)

    candidate_hist_cols = (
        ["points_hist"]
        + cols_map.fpl_cols
        + cols_map.match_context_cols
        + cols_map.market_cols
        + cols_map.betting_cols
    )
    seq_feature_cols = list(dict.fromkeys([
        c for c in candidate_hist_cols if c in out.columns and c != "total_points"
    ]))

    betting_present = [c for c in cols_map.betting_cols if c in out.columns]
    context_cols = betting_present + ["was_home_num", "pos_DEF", "pos_FWD", "pos_MID"]

    return out, seq_feature_cols, context_cols


def train_test_split(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    df = df.copy()
    X = df.drop(columns=["total_points"])
    y = df["total_points"]

    played = df["is_played"].astype(bool) if "is_played" in df.columns else y.notna()

    train_mask = (X.season.isin([2223, 2324, 2425])) & played
    valid_mask = (X.season == 2526) & played
    test_mask = (X.season == 2627) & played

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_valid": X[valid_mask],
        "y_valid": y[valid_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "X_predict": X[~played],
    }


def scale_target(
    splits: dict[str, pd.DataFrame | pd.Series],
) -> tuple[dict[str, pd.DataFrame | pd.Series], StandardScaler]:
    splits = splits.copy()
    scaler = StandardScaler()

    scaled_train = scaler.fit_transform(splits["y_train"].values.reshape(-1, 1))
    splits["y_train"] = pd.Series(scaled_train.flatten(), index=splits["y_train"].index)

    for key in ["y_valid", "y_test"]:
        if key in splits:
            y_reshaped = splits[key].values.reshape(-1, 1)
            scaled_val = scaler.transform(y_reshaped)
            splits[key] = pd.Series(scaled_val.flatten(), index=splits[key].index)

    return splits, scaler


class FPLDataPipe:
    """Tabular preprocessing: one-hot encoding of categorical and standardisation of numeric columns."""

    def __init__(
        self,
        num_cols: list[str],
        cat_cols: list[str],
        batch_size: int = 64,
    ):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.batch_size = batch_size
        self.preprocessor = self._build_pipeline()

    def _build_pipeline(self):
        num_transform = Pipeline([
            ("scaler", StandardScaler()),
        ])

        cols_transform = Pipeline([
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
        ])

        preprocessor = ColumnTransformer(
            [
                ("col", cols_transform, self.cat_cols),
                ("num", num_transform, self.num_cols),
            ],
            remainder="drop",
        )

        preprocessor.set_output(transform="pandas")
        return preprocessor
