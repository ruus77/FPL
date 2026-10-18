import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def train_test_split(df: pd.DataFrame)-> dict[str, pd.DataFrame | pd.Series]:
    df = df.copy()
    X = df.drop(columns=["total_points"])
    y = df["total_points"]

    played = df["is_played"].astype(bool) if "is_played" in df.columns else y.notna()

    train_mask = (X.season.isin([2223, 2324, 2425])) & played
    valid_mask = (X.season == 2526) & played
    test_mask = (X.season == 2627) & played

    return {
        "X_train": X[train_mask], "y_train": y[train_mask],
        "X_valid": X[valid_mask], "y_valid": y[valid_mask],
        "X_test": X[test_mask], "y_test": y[test_mask],
        "X_predict": X[~played],
    }


def scale_target(splits: dict[str, pd.DataFrame | pd.Series]) -> tuple[dict[str, pd.DataFrame | pd.Series], MinMaxScaler]:
    splits = splits.copy()
    scaler = MinMaxScaler()
    
    scaled_train = scaler.fit_transform(splits["y_train"].values.reshape(-1, 1))
    
    splits["y_train"] = pd.Series(scaled_train.flatten(), index=splits["y_train"].index)
    
    for key in ["y_valid", "y_test"]:
        if key in splits:
            y_reshaped = splits[key].values.reshape(-1, 1)
            scaled_val = scaler.transform(y_reshaped)
            splits[key] = pd.Series(scaled_val.flatten(), index=splits[key].index)
            
    return splits, scaler


class FPLDataPipe:
    def __init__(self, num_cols: list[str], cat_cols: list[str], batch_size: int = 64, catboost_flag: bool = False):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.batch_size = batch_size
        self.catboost_flag = catboost_flag
        self.preprocessor = self._build_pipeline()
        self.valid_dataloader, self.train_dataloader, self.test_dataloader = None, None, None

    def _build_pipeline(self):
        num_transform = Pipeline([
            ("scaler", MinMaxScaler()),
        ])

        if not self.catboost_flag:
            cols_transform = Pipeline([
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))
            ])
            
            preprocessor = ColumnTransformer([
                ("col", cols_transform, self.cat_cols),
                ("num", num_transform, self.num_cols)
            ], remainder="drop")
        else:
            preprocessor = ColumnTransformer([
                ("col", "passthrough", self.cat_cols),
                ("num", num_transform, self.num_cols)
            ], remainder="drop")
            
        preprocessor.set_output(transform="pandas")
        return preprocessor

    @staticmethod
    def _to_tensor(X, y):
        return (
            torch.tensor(X.to_numpy(), dtype=torch.float32),
            torch.tensor(y.to_numpy().reshape(-1, 1), dtype=torch.float32)
        )

    def prepare_data(self, X_train, y_train, X_valid, y_valid, X_test, y_test, X_predict=None):
        X_train = self.preprocessor.fit_transform(X_train)
        X_valid = self.preprocessor.transform(X_valid)
        X_test = self.preprocessor.transform(X_test)
        X_pred = self.preprocessor.transform(X_predict) if X_predict is not None and not X_predict.empty else None

        if self.catboost_flag:
            X_train[self.cat_cols] = X_train[self.cat_cols].astype(str)
            X_valid[self.cat_cols] = X_valid[self.cat_cols].astype(str)
            X_test[self.cat_cols] = X_test[self.cat_cols].astype(str)
            if X_pred is not None:
                X_pred[self.cat_cols] = X_pred[self.cat_cols].astype(str)

        X_train_tensor, y_train_tensor = self._to_tensor(X_train, y_train)
        X_valid_tensor, y_valid_tensor = self._to_tensor(X_valid, y_valid)
        X_test_tensor, y_test_tensor = self._to_tensor(X_test, y_test)
        self.X_predict_tensor = torch.tensor(X_pred.to_numpy(), dtype=torch.float32) if X_pred is not None else None

        self.train_dataloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=self.batch_size, shuffle=False)
        self.valid_dataloader = DataLoader(TensorDataset(X_valid_tensor, y_valid_tensor), batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=self.batch_size, shuffle=False)
        if self.X_predict_tensor is not None:
            dummy_y = torch.zeros((len(self.X_predict_tensor), 1), dtype=torch.float32)
            self.predict_dataloader = DataLoader(TensorDataset(self.X_predict_tensor, dummy_y), batch_size=self.batch_size, shuffle=False)
        else:
            self.predict_dataloader = None

    def get_dataloaders(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader

class SequenceDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 feature_cols: list[str],
                 target_col: str = "total_points",
                 seq_len: int = 8,
                 player_col: str = "code",
                 time_col: str = "gw",
                 season_col: str = "season",
                 history_df: pd.DataFrame | None = None):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        n_features = len(feature_cols)

        if history_df is not None and not history_df.empty:
            df_indexed = df.copy()
            df_indexed["_orig_pos"] = np.arange(len(df_indexed), dtype=np.int64)
            if target_col not in df_indexed.columns:
                df_indexed[target_col] = 0.0
            else:
                df_indexed[target_col] = df_indexed[target_col].fillna(0.0)

            df_sorted = df_indexed.sort_values([player_col, season_col, time_col]).reset_index(drop=True)

            hist_sorted = history_df.copy().sort_values([player_col, season_col, time_col])
            hist_sorted["_orig_pos"] = -1
            if target_col not in hist_sorted.columns:
                hist_sorted[target_col] = 0.0

            hist_tails = {
                player: grp.tail(seq_len)
                for player, grp in hist_sorted.groupby(player_col, sort=False)
            }

            ordered_sequences: list[tuple[np.ndarray, float] | None] = [None] * len(df_indexed)

            for player, grp in df_sorted.groupby(player_col, sort=False):
                hist_grp = hist_tails.get(player)
                if hist_grp is not None and not hist_grp.empty:
                    combined = pd.concat([hist_grp, grp], ignore_index=True)
                    n_hist = len(hist_grp)
                else:
                    combined = grp.reset_index(drop=True)
                    n_hist = 0

                feats = combined[feature_cols].values.astype(np.float32)
                target = combined[target_col].values.astype(np.float32)
                orig_positions = combined["_orig_pos"].values

                for t in range(n_hist, len(combined)):
                    if t >= seq_len:
                        window = feats[t - seq_len: t]
                    elif t > 0:
                        pad = np.zeros((seq_len - t, n_features), dtype=np.float32)
                        window = np.vstack([pad, feats[:t]])
                    else:
                        window = np.zeros((seq_len, n_features), dtype=np.float32)

                    label = float(target[t])
                    ordered_sequences[int(orig_positions[t])] = (window, label)

            self.sequences = ordered_sequences
        else:
            self.sequences: list[tuple[np.ndarray, float]] = []
            df = df.copy().sort_values([player_col, season_col, time_col]).reset_index(drop=True)
            for (player, season), grp in df.groupby([player_col, season_col], sort=False):
                feats = grp[feature_cols].values.astype(np.float32)
                target = grp[target_col].values.astype(np.float32)

                for t in range(seq_len, len(grp)):
                    window = feats[t - seq_len: t]
                    label = target[t]
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
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.model_type = model_type
        self.scaler = MinMaxScaler()
        self._fitted = False

        self.train_dl: DataLoader | None = None
        self.valid_dl: DataLoader | None = None
        self.test_dl: DataLoader | None = None
        self.predict_dl: DataLoader | None = None

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
        xs, ys = zip(*batch)
        X = torch.stack(xs)
        y = torch.cat(ys)
        if self.model_type == "conv1d":
            X = X.permute(0, 2, 1)
        return X, y

    def prepare_data(self,
                     X_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame,
                     X_valid: pd.DataFrame, y_valid: pd.Series | pd.DataFrame,
                     X_test: pd.DataFrame, y_test: pd.Series | pd.DataFrame,
                     X_predict: pd.DataFrame | None = None):

        def merge(X, y):
            y_series = y["total_points"] if isinstance(y, pd.DataFrame) else y
            y_series = y_series.copy()
            y_series.name = "total_points"
            return X.join(y_series)

        train_df = merge(X_train, y_train)
        valid_df = merge(X_valid, y_valid)
        test_df = merge(X_test, y_test)

        train_df = self._scale_df(train_df, fit=True)
        valid_df = self._scale_df(valid_df, fit=False)
        test_df = self._scale_df(test_df, fit=False)

        train_ds = SequenceDataset(train_df, self.feature_cols, seq_len=self.seq_len)
        valid_ds = SequenceDataset(valid_df, self.feature_cols, seq_len=self.seq_len, history_df=train_df)
        full_hist_for_test = pd.concat([train_df, valid_df], ignore_index=True)
        test_ds = SequenceDataset(
            test_df,
            self.feature_cols,
            seq_len=self.seq_len,
            history_df=full_hist_for_test,
        )

        if X_predict is not None and not X_predict.empty:
            predict_df = X_predict.copy()
            predict_df["total_points"] = 0.0
            predict_df = self._scale_df(predict_df, fit=False)
            full_hist_for_pred = pd.concat([train_df, valid_df, test_df], ignore_index=True)
            predict_ds = SequenceDataset(
                predict_df,
                self.feature_cols,
                seq_len=self.seq_len,
                history_df=full_hist_for_pred,
            )
            self.predict_dl = DataLoader(
                predict_ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self._collate,
            )
        else:
            self.predict_dl = None

        print(f"[SequenceDataPipe/{self.model_type}]  "
              f"train={len(train_ds)}  valid={len(valid_ds)}  test={len(test_ds)}  "
              f"seq_len={self.seq_len}")

        self.train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                                   shuffle=False, collate_fn=self._collate, drop_last=True)
        self.valid_dl = DataLoader(valid_ds, batch_size=self.batch_size,
                                   shuffle=False, collate_fn=self._collate)
        self.test_dl = DataLoader(test_ds, batch_size=self.batch_size,
                                  shuffle=False, collate_fn=self._collate)

    def get_dataloaders(self):
        return self.train_dl, self.valid_dl, self.test_dl



