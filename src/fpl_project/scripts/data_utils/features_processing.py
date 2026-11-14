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

        ema_features = self.cols_map.LEAKAGE_COLS
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
        
        lag_features = self.cols_map.LEAKAGE_COLS
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


        std_features = self.cols_map.LEAKAGE_COLS

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


        pre_game_cols = self.cols_map.target + self.cols_map.id_cols + self.cols_map.betting_cols + self.cols_map.static_cols

        out = pd.concat([df[pre_game_cols], df_ema, df_lag, df_std], axis=1)
        num_cols = out.select_dtypes(include="number").columns
        played = out["is_played"].astype(bool)
        out[num_cols] = out[num_cols].fillna(out.loc[played, num_cols].median()).fillna(0)
        return out


def build_and_save_features_engineering_data() -> pd.DataFrame:
    import numpy as np
    from fpl_project.scripts import config

    if not config.EDA.exists():
        from fpl_project.data_processing.data_import import fetch_and_save_eda_data
        fetch_and_save_eda_data()

    data = pd.read_parquet(config.EDA).reset_index(drop=True)
    num_cols = data.select_dtypes(include=np.number).columns
    data[num_cols] = data[num_cols].astype(np.float64)
    data.columns = data.columns.str.lower()

    if "transfers_trend" not in data.columns:
        data["transfers_trend"] = (data.transfers_balance / data.selected).replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_eng = FeatureEngineer(df=data)
    data_lag = feature_eng.features_integration(lag_size=[1, 3, 5], window_size=[3, 5], std_size=[3, 5])

    played = data_lag["is_played"].astype(bool)
    corr_matrix = (
        data_lag.loc[played]
        .drop(columns=["total_points"], errors="ignore")
        .select_dtypes(include=np.number)
        .corr()
        .abs()
    )
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop = [col for col in upper_tri.columns if (upper_tri[col] > 0.9).any()]
    if cols_to_drop:
        data_lag.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_lag.to_parquet(config.FEATURES_ENGINEERING, index=False)
    return data_lag
