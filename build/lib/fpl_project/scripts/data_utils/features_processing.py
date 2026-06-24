import pandas as pd
from fpl_project.scripts.data_utils.features_config import FeaturesConfig


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=["code", "kickoff_time", "season", "gw"]).reset_index(drop=True)


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cols_map: FeaturesConfig = FeaturesConfig()

    def ema(self, window_size: int = 4) -> tuple[pd.DataFrame, list[str]]:
        df = self.df.copy()
        if self.cols_map is None:
            return pd.DataFrame(), []

        ema_features = self.cols_map.fpl_cols + self.cols_map.perf_cols + self.cols_map.target
        ema_features = [col for col in ema_features if col in df.columns]

        df = sort_data(df)

        ema_frame = df.groupby(["code", "season"])[ema_features].transform(
            lambda x: x.ewm(span=window_size, adjust=False).mean().shift(1)
        )

        new_cols = [f"{c}_ema_{window_size}" for c in ema_features]
        df[new_cols] = ema_frame.fillna(0)

        return df, new_cols

    def lag(self, lag_size: int = 1) -> tuple[pd.DataFrame, list[str]]:
        df = self.df.copy()
        if self.cols_map is None:
            return pd.DataFrame(), []

        lag_features = self.cols_map.fpl_cols + self.cols_map.perf_cols + self.cols_map.target
        lag_features = [col for col in lag_features if col in df.columns]

        df = sort_data(df)

        lagged_frame = df.groupby(["code", "season"])[lag_features].transform(
            lambda x: x.shift(lag_size)
        )

        new_cols = [f"{c}_lagged_{lag_size}" for c in lag_features]
        df[new_cols] = lagged_frame.fillna(0)

        return df, new_cols

    def rolling_std(self, window_size: int = 4) -> tuple[pd.DataFrame, list[str]]:

        df = self.df.copy()
        if self.cols_map is None:
            return pd.DataFrame(), []

        std_features = self.cols_map.fpl_cols + self.cols_map.perf_cols + self.cols_map.target

        std_features = [col for col in std_features if col in df.columns]

        df = sort_data(df)

        std_frame = df.groupby(["code", "season"])[std_features].transform(
            lambda x: x.rolling(window_size).std().shift(1)
        )

        new_cols = [f"{c}_std_{window_size}" for c in std_features]
        df[new_cols] = std_frame.fillna(0)

        return df, new_cols

    def features_integration(self, lag_size: int | list[int] = 1, window_size: int | list[int] = 8,
                             std_size: int | list[int] = 4) -> pd.DataFrame:

        df = self.df.copy()
        df = sort_data(df)

        # EMA
        if isinstance(window_size, int):
            df_ema, ema_cols = self.ema(window_size=window_size)
        else:
            ema_frames = []
            ema_cols = []

            for w in window_size:
                tmp_df, tmp_cols = self.ema(window_size=w)
                ema_frames.append(tmp_df[tmp_cols])
                ema_cols.extend(tmp_cols)

            df_ema = pd.concat(ema_frames, axis=1)

        # LAG
        if isinstance(lag_size, int):
            df_lag, lag_cols = self.lag(lag_size=lag_size)
        else:
            lag_frames = []
            lag_cols = []

            for l in lag_size:
                tmp_df, tmp_cols = self.lag(lag_size=l)
                lag_frames.append(tmp_df[tmp_cols])
                lag_cols.extend(tmp_cols)

            df_lag = pd.concat(lag_frames, axis=1)

        # STD
        if isinstance(std_size, int):
            df_std, std_cols = self.rolling_std(window_size=std_size)
        else:
            std_frames = []
            std_cols = []

            for s in std_size:
                tmp_df, tmp_cols = self.rolling_std(window_size=s)
                std_frames.append(tmp_df[tmp_cols])
                std_cols.extend(tmp_cols)

            df_std = pd.concat(std_frames, axis=1)

        # Ensure we have lists
        pre_game_cols = []
        if getattr(self.cols_map, 'pre_game_cols', None) is not None:
            pre_game_cols = list(set(self.cols_map.pre_game_cols))

        target_cols = []
        if getattr(self.cols_map, 'target', None) is not None:
            target_cols = list(set(self.cols_map.target))

        pre_game_cols = pre_game_cols + target_cols
        pre_game_cols = [col for col in pre_game_cols if col in df.columns]

        return pd.concat([df[pre_game_cols], df_ema, df_lag, df_std], axis=1).fillna(0)