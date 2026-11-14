"""PyTorch datasets and collate functions for sequential (Conv1D) models."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fpl_project.scripts import config


class AuxSequenceDataset(Dataset):
    """Sequence dataset for Conv1D returning (window, target, [value_ema_3, minutes_ema_3 / 90])."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        context_cols: list[str] | None = None,
        target_col: str = "total_points",
        seq_len: int = config.NN_SEQ_LEN,
        player_col: str = "code",
        time_col: str = "gw",
        season_col: str = "season",
        history_df: pd.DataFrame | None = None,
    ):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.context_cols = context_cols or []
        n_hist = len(feature_cols)

        if history_df is not None and not history_df.empty:
            self.sequences = self._build_with_history(
                df=df,
                history_df=history_df,
                target_col=target_col,
                seq_len=seq_len,
                n_hist=n_hist,
                player_col=player_col,
                season_col=season_col,
                time_col=time_col,
            )
        else:
            self.sequences = self._build_without_history(
                df=df,
                target_col=target_col,
                seq_len=seq_len,
                player_col=player_col,
                season_col=season_col,
                time_col=time_col,
            )

    def _build_with_history(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame,
        target_col: str,
        seq_len: int,
        n_hist: int,
        player_col: str,
        season_col: str,
        time_col: str,
    ) -> list[tuple[np.ndarray, float, np.ndarray]]:
        df_indexed = df.copy()
        df_indexed["_orig_pos"] = np.arange(len(df_indexed), dtype=np.int64)
        df_indexed[target_col] = (
            df_indexed[target_col].fillna(0.0) if target_col in df_indexed.columns else 0.0
        )
        df_sorted = df_indexed.sort_values([player_col, season_col, time_col]).reset_index(drop=True)

        hist_sorted = history_df.copy().sort_values([player_col, season_col, time_col])
        hist_sorted["_orig_pos"] = -1
        if target_col not in hist_sorted.columns:
            hist_sorted[target_col] = 0.0

        hist_tails = {
            player: grp.tail(seq_len)
            for player, grp in hist_sorted.groupby(player_col, sort=False)
        }

        ordered_sequences: list[tuple[np.ndarray, float, np.ndarray] | None] = [None] * len(df_indexed)

        for player, grp in df_sorted.groupby(player_col, sort=False):
            hist_grp = hist_tails.get(player)
            if hist_grp is not None and not hist_grp.empty:
                combined = pd.concat([hist_grp, grp], ignore_index=True)
                n_prefix = len(hist_grp)
            else:
                combined = grp.reset_index(drop=True)
                n_prefix = 0

            feats = combined[self.feature_cols].values.astype(np.float32)
            ctx_feats = (
                combined[self.context_cols].values.astype(np.float32)
                if self.context_cols
                else None
            )
            target = combined[target_col].values.astype(np.float32)
            raw_price = combined["_raw_price"].values.astype(np.float32)
            raw_min = combined["_raw_min"].values.astype(np.float32)
            orig_positions = combined["_orig_pos"].values

            for t in range(n_prefix, len(combined)):
                hist_window = self._slice_or_pad(feats, t, seq_len, n_hist)
                window = self._attach_context(hist_window, ctx_feats, t, seq_len)
                aux = np.array([raw_price[t], raw_min[t]], dtype=np.float32)
                ordered_sequences[int(orig_positions[t])] = (window, float(target[t]), aux)

        return ordered_sequences  # type: ignore[return-value]

    def _build_without_history(
        self,
        df: pd.DataFrame,
        target_col: str,
        seq_len: int,
        player_col: str,
        season_col: str,
        time_col: str,
    ) -> list[tuple[np.ndarray, float, np.ndarray]]:
        sequences: list[tuple[np.ndarray, float, np.ndarray]] = []
        df_sorted = df.copy().sort_values([player_col, season_col, time_col]).reset_index(drop=True)
        for _, grp in df_sorted.groupby([player_col, season_col], sort=False):
            feats = grp[self.feature_cols].values.astype(np.float32)
            ctx_feats = (
                grp[self.context_cols].values.astype(np.float32)
                if self.context_cols
                else None
            )
            target = grp[target_col].values.astype(np.float32)
            raw_price = grp["_raw_price"].values.astype(np.float32)
            raw_min = grp["_raw_min"].values.astype(np.float32)

            for t in range(seq_len, len(grp)):
                hist_window = feats[t - seq_len : t]
                window = self._attach_context(hist_window, ctx_feats, t, seq_len)
                aux = np.array([raw_price[t], raw_min[t]], dtype=np.float32)
                sequences.append((window, float(target[t]), aux))
        return sequences

    @staticmethod
    def _slice_or_pad(feats: np.ndarray, t: int, seq_len: int, n_hist: int) -> np.ndarray:
        if t >= seq_len:
            return feats[t - seq_len : t]
        if t > 0:
            pad = np.zeros((seq_len - t, n_hist), dtype=np.float32)
            return np.vstack([pad, feats[:t]])
        return np.zeros((seq_len, n_hist), dtype=np.float32)

    @staticmethod
    def _attach_context(
        hist_window: np.ndarray,
        ctx_feats: np.ndarray | None,
        t: int,
        seq_len: int,
    ) -> np.ndarray:
        if ctx_feats is None:
            return hist_window
        ctx_step = np.broadcast_to(ctx_feats[t], (seq_len, ctx_feats.shape[1])).astype(np.float32)
        return np.hstack([hist_window, ctx_step])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        window, label, aux = self.sequences[idx]
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor([[label]], dtype=torch.float32),
            torch.tensor(aux, dtype=torch.float32),
        )


def collate_conv1d_with_aux(batch):
    """Collate function stacking sequence windows, targets, and auxiliary vectors."""
    xs, ys, auxs = zip(*batch)
    return torch.stack(xs), torch.cat(ys), torch.stack(auxs)
