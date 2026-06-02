import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def train_test_split(df: pd.DataFrame)->dict[str, pd.DataFrame | pd.Series]:
    df = df.copy()
    X = df.drop(columns=["total_points"])
    y = df["total_points"]

    train_mask = (X.season.isin([2223, 2324]))
    valid_mask = (X.season == 2425)
    test_mask = (X.season == 2526)

    return {
        "X_train": X[train_mask], "y_train": y[train_mask],
        "X_valid": X[valid_mask], "y_valid": y[valid_mask],
        "X_test": X[test_mask], "y_test": y[test_mask]
    }


class FPLDataPipe:
    def __init__(self, num_cols: list[str], cat_cols: list[str], batch_size: int = 64):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.batch_size = batch_size
        self.preprocessor = self._build_pipeline()
        self.valid_dataloader, self.train_dataloader, self.test_dataloader = None, None, None

    def _build_pipeline(self):
        num_transform = Pipeline([
            ("scaler", MinMaxScaler()),
        ])

        cols_transform = Pipeline([
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_transform, self.num_cols),
            ("col", cols_transform, self.cat_cols)],
            remainder="drop")

        preprocessor.set_output(transform="pandas")
        return preprocessor

    @staticmethod
    def _to_tensor(X, y):
        return (

            torch.tensor(X.to_numpy(), dtype=torch.float32),
            torch.tensor(y.to_numpy().reshape(-1, 1), dtype=torch.float32)
        )

    def prepare_data(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
        X_train = self.preprocessor.fit_transform(X_train)

        X_valid = self.preprocessor.transform(X_valid)
        X_test = self.preprocessor.transform(X_test)

        X_train_tensor, y_train_tensor = self._to_tensor(X_train, y_train)
        X_valid_tensor, y_valid_tensor = self._to_tensor(X_valid, y_valid)
        X_test_tensor, y_test_tensor = self._to_tensor(X_test, y_test)

        self.train_dataloader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )
        self.valid_dataloader = DataLoader(
            TensorDataset(X_valid_tensor, y_valid_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )
        self.test_dataloader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )

    def get_dataloaders(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader


class SequenceDataset(Dataset):
    """
    Buduje okna sekwencyjne z szeregów czasowych gracza.

    Dla każdego gracza sortuje kolejki i tworzy pary:
        X: cechy z kolejek [t - seq_len, ..., t-1]  →  shape (seq_len, n_features)
        y: total_points z kolejki t

    Okno NIE zawiera kolejki t → zero data leakage.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 feature_cols: list[str],
                 target_col: str = "total_points",
                 seq_len: int = 8,
                 player_col: str = "code",
                 time_col: str = "gw",
                 season_col: str = "season"):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.sequences: list[tuple[np.ndarray, float]] = []

        df = df.copy().sort_values([player_col, season_col, time_col]).reset_index(drop=True)

        for (player, season), grp in df.groupby([player_col, season_col], sort=False):
            feats = grp[feature_cols].values.astype(np.float32)  # (T, F)
            target = grp[target_col].values.astype(np.float32)  # (T,)

            # Normalizacja cech wewnątrz gracza (zapobiega scale shift między sezonami)
            # Używamy min-max na oknie historycznym — fit wyłącznie na przeszłości
            for t in range(seq_len, len(grp)):
                window = feats[t - seq_len: t]  # (seq_len, F) — tylko przeszłość
                label = target[t]  # wartość przyszła
                self.sequences.append((window, label))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        window, label = self.sequences[idx]
        return torch.tensor(window, dtype=torch.float32), torch.tensor([[label]], dtype=torch.float32)


class SequenceDataPipe:

    def __init__(self,
                 feature_cols: list[str],
                 seq_len: int = 8,
                 batch_size: int = 256,
                 model_type: str = "lstm"):
        """
        model_type: 'lstm' → (B, T, F)  |  'conv1d' → (B, F, T)
        """
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.model_type = model_type
        self.scaler = MinMaxScaler()
        self._fitted = False

        self.train_dl: DataLoader | None = None
        self.valid_dl: DataLoader | None = None
        self.test_dl: DataLoader | None = None

    def _scale_df(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        if fit:
            df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
            self._fitted = True
        else:
            assert self._fitted, "Najpierw wywołaj prepare_data z danymi treningowymi."
            df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df

    def _collate(self, batch):
        """Dostosowuje kształt tensora do wymaganego przez model."""
        xs, ys = zip(*batch)
        X = torch.stack(xs)  # (B, T, F)
        y = torch.cat(ys)  # (B, 1)
        if self.model_type == "conv1d":
            X = X.permute(0, 2, 1)  # (B, F, T)
        return X, y

    def prepare_data(self,
                     X_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame,
                     X_valid: pd.DataFrame, y_valid: pd.Series | pd.DataFrame,
                     X_test: pd.DataFrame, y_test: pd.Series | pd.DataFrame):

        def merge(X, y):
            """Łączy X i y — y może być Series lub DataFrame."""
            y_series = y["total_points"] if isinstance(y, pd.DataFrame) else y
            y_series.name = "total_points"
            return X.join(y_series)

        train_df = merge(X_train, y_train)
        valid_df = merge(X_valid, y_valid)
        test_df = merge(X_test, y_test)

        # Skalowanie: fit tylko na treningowych
        train_df = self._scale_df(train_df, fit=True)
        valid_df = self._scale_df(valid_df, fit=False)
        test_df = self._scale_df(test_df, fit=False)

        train_ds = SequenceDataset(train_df, self.feature_cols, seq_len=self.seq_len)
        valid_ds = SequenceDataset(valid_df, self.feature_cols, seq_len=self.seq_len)
        test_ds = SequenceDataset(test_df, self.feature_cols, seq_len=self.seq_len)

        print(f"[SequenceDataPipe/{self.model_type}]  "
              f"train={len(train_ds)}  valid={len(valid_ds)}  test={len(test_ds)}  "
              f"seq_len={self.seq_len}")

        self.train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                                   shuffle=True, collate_fn=self._collate, drop_last=True)
        self.valid_dl = DataLoader(valid_ds, batch_size=self.batch_size,
                                   shuffle=False, collate_fn=self._collate)
        self.test_dl = DataLoader(test_ds, batch_size=self.batch_size,
                                  shuffle=False, collate_fn=self._collate)

    def get_dataloaders(self):
        return self.train_dl, self.valid_dl, self.test_dl



