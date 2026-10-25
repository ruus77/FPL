import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from fpl_project.scripts import config
from fpl_project.scripts.config import SEASONS

@dataclass(frozen=True)
class ContentConfig:
    VAASTAV_GW_URL: str = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{}/gws/gw{}.csv"
    VAASTAV_ID_URL: str = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{}/players_raw.csv"
    BETTING_URL: str = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
    BETTING_FIXTURES_URL: str = "https://www.football-data.co.uk/fixtures.csv"
    FCI_URL: str = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/{}/By%20Gameweek/GW{}/{}.csv"
    FCI_PLAYERSTATS_URL: str = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/{}/playerstats.csv"

    FCI_TARGET_COLUMNS: list[str] = field(default_factory=lambda: [
        'name', 'position', 'xP', 'assists', 'bonus', 'bps', 'clean_sheets',
        'creativity', 'element', 'expected_assists', 'expected_goal_involvements',
        'expected_goals', 'expected_goals_conceded', 'fixture', 'goals_conceded',
        'goals_scored', 'ict_index', 'influence', 'kickoff_time', 'minutes',
        'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 'round',
        'saves', 'selected', 'starts', 'team_a_score', 'team_h_score', 'threat',
        'event_points', 'transfers_balance_event', 'transfers_in_event',
        'transfers_out_event', 'value', 'was_home', 'yellow_cards', 'gw', 'code', 'team',
        'is_played'
    ])

    FCI_CUMULATIVE_COLUMNS: list[str] = field(default_factory=lambda: [
        'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
        'expected_assists', 'expected_goal_involvements', 'expected_goals',
        'expected_goals_conceded', 'goals_conceded', 'goals_scored', 'ict_index',
        'influence', 'minutes', 'own_goals', 'penalties_missed', 'penalties_saved',
        'red_cards', 'saves', 'starts', 'threat', 'yellow_cards'
    ])
    
    SEASONS: list[str] = field(default_factory=lambda: SEASONS)
    BETTING_SEASONS: list[str] = field(
        default_factory=lambda: [s[2:4] + s[5:] for s in SEASONS]
    )

class FetchData(ABC):
    def __init__(self, seasons: list[str]):
        self.seasons = seasons

    @abstractmethod
    def _merge_and_clean(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass
   
class FetchVaastav(FetchData):
    def __init__(self, config_data: ContentConfig):
        super().__init__(seasons=config_data.SEASONS[:-1])
        self.config = config_data
        self.cols_drop = ['modified',
                        'mng_clean_sheets',
                        'mng_draw',
                        'mng_goals_scored',
                        'mng_loss',
                        'mng_underdog_draw',
                        'mng_underdog_win',
                        'mng_win',
                        'clearances_blocks_interceptions',
                        'defensive_contribution',
                        'recoveries',
                        'tackles']
    def _fetch_gameweeks(self) -> pd.DataFrame:
        frames = []
        for season in self.seasons:
            season_frames = []
            for gw in range(1, 39):
                try:
                    url = self.config.VAASTAV_GW_URL.format(season, gw)
                    df = pd.read_csv(url)
                    if not df.empty:
                        season_frames.append(
                            df.assign(season=season, gw=gw).dropna(axis=1, how="all")
                        )
                except Exception as e:
                    raise(e)

            if season_frames:
                frames.extend(season_frames)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_ids(self) -> pd.DataFrame:
        frames = []
        for season in self.seasons:
            try:
                url = self.config.VAASTAV_ID_URL.format(season)
                df = pd.read_csv(
                    url,
                    usecols=["id", "first_name", "second_name", "code"]
                )
                df["season"] = season
                df["name"] = df["first_name"] + " " + df["second_name"]
                frames.append(df.drop(columns=["first_name", "second_name"]))
            except Exception as e:
                raise(e)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _merge_and_clean(self, gw_df: pd.DataFrame, id_df: pd.DataFrame) -> pd.DataFrame:
        if gw_df.empty or id_df.empty:
            return pd.DataFrame()

        df = pd.merge(
            gw_df, id_df,
            how="inner",
            left_on=["element", "season"],
            right_on=["id", "season"]
        ).drop(columns=["id"])

        if "name_y" in df.columns:
            df.drop(columns=["name_y"], inplace=True)
        if "name_x" in df.columns:
            df.rename(columns={"name_x": "name"}, inplace=True)

        if "value" in df.columns:
            df["value"] = df["value"] / 10

        df["selected"] = df.groupby(["season", "gw"])["selected"].transform(
            lambda x: x / x.sum()
        )

        df["kickoff_time"] = (
            df["kickoff_time"]
            .astype(str)
            .str.replace("Z", "", regex=False)
            .str.slice(0, 19)
        )
        kickoff_dt = pd.to_datetime(df['kickoff_time'], errors='coerce')

        df['kickoff_time'] = pd.to_datetime(kickoff_dt.dt.strftime('%d/%m/%Y'), dayfirst=True)

        df["season"] = df["season"].apply(
            lambda x: re.sub(r'^20(\d{2})-(\d{2})$', r'\1\2', str(x))
        )

        df = df.drop_duplicates(subset=["code", "season", "gw"])

        return df
        
    def get_data(self) -> pd.DataFrame:
        gw_df = self._fetch_gameweeks()
        id_df = self._fetch_ids()
        df = self._merge_and_clean(gw_df, id_df)
        df.drop(columns=self.cols_drop, inplace=True, errors='ignore')
        df["is_played"] = True
        return df

class FetchFCI(FetchData):
    def __init__(self, config_data: ContentConfig):
        current_season = config_data.SEASONS[-1]
        
        super().__init__(seasons=[current_season])
        self.config = config_data
        self.current_season = current_season
        yr1, yr2 = current_season.split("-")
        self.fci_season_path = f"{yr1}-20{yr2}"

    def _get_fci_base(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            playerstats_url = self.config.FCI_PLAYERSTATS_URL.format(self.fci_season_path)
            playerstats = pd.read_csv(playerstats_url, low_memory=False)

            teams_url = playerstats_url.replace("playerstats.csv", "teams.csv")
            teams_df = pd.read_csv(teams_url, usecols=["id", "code", "name"])
            teams_df = teams_df.rename(columns={"name": "team", "id": "team_id", "code": "team_code"})
            teams_df["team"] = teams_df["team"].replace({"Ipswich Town": "Ipswich"})

            return playerstats, teams_df
        except Exception as e:
            raise(e)

    def _fetch_gameweeks(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        players_frames, matches_frames = [], []

        for gw in range(1, 39):
            try:
                url_players = self.config.FCI_URL.format(self.fci_season_path, gw, "players")
                df_p = pd.read_csv(
                    url_players, low_memory=False,
                    usecols=["player_id", "first_name", "second_name", "position", "player_code", "team_code"]
                )
                players_frames.append(df_p)

                url_matches = self.config.FCI_URL.format(self.fci_season_path, gw, "matches")
                df_m = pd.read_csv(
                    url_matches, low_memory=False,
                    usecols=["match_id", "gameweek", "kickoff_time", "home_team", "away_team", "home_score", "away_score"]
                )
                matches_frames.append(df_m)
            except Exception as e:
                raise(e)

        if not players_frames or not matches_frames:
            return pd.DataFrame(), pd.DataFrame()

        players = pd.concat(players_frames, ignore_index=True).drop_duplicates(subset=["player_id"])
        matches = pd.concat(matches_frames, ignore_index=True)
        return players, matches

    def _decumulate_playerstats(self, playerstats: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.config.FCI_CUMULATIVE_COLUMNS if c in playerstats.columns]
        ps = playerstats.sort_values(["id", "gw"]).copy()
        ps[cols] = ps[cols].apply(pd.to_numeric, errors="coerce")

        per_gw = ps.groupby("id")[cols].diff()
        ps[cols] = per_gw.fillna(ps[cols]).clip(lower=0).round(2)
        return ps

    def _merge_and_clean(self) -> pd.DataFrame:
        players, matches = self._fetch_gameweeks()
        playerstats, teams_df = self._get_fci_base()

        if players.empty or matches.empty or playerstats.empty or teams_df.empty:
            return pd.DataFrame()

        playerstats = self._decumulate_playerstats(playerstats)

        players["name"] = players["first_name"] + " " + players["second_name"]
        players = players.drop(columns=["first_name", "second_name"])
        players = pd.merge(players, teams_df, on="team_code", how="left")

        df = pd.merge(playerstats, players, how="left", left_on="id", right_on="player_id")
        df = pd.merge(df, matches, how="left", left_on="gw", right_on="gameweek")

        df = df[(df["team_code"] == df["home_team"]) | (df["team_code"] == df["away_team"])].copy()
        df["was_home"] = df["team_code"] == df["home_team"]

        df["opponent_team"] = df["away_team"].where(df["was_home"], df["home_team"])

        for col in ["transfers_in_event", "transfers_out_event"]:
            df[col] = df[col].fillna(0)
        df["transfers_balance_event"] = df["transfers_in_event"] - df["transfers_out_event"]

        df["position"] = df["position"].map({
            "Defender": "DEF", "Forward": "FWD", "Goalkeeper": "GK", "Midfielder": "MID"
        })

        df = df[df["match_id"].astype(str).str.contains("prem", na=False)].copy()

        df["is_played"] = df["home_score"].notna() & df["away_score"].notna()

        df["selected_by_percent"] = df.groupby("gw")["selected_by_percent"].transform(
            lambda x: x / x.sum()
        )

        rename_dict = {
            "id": "element", "ep_this": "xP", "match_id": "fixture", "now_cost": "value",
            "selected_by_percent": "selected", "home_score": "team_h_score",
            "away_score": "team_a_score", "player_code": "code",
        }
        df = df.rename(columns=rename_dict)
        df["round"] = df["gw"]

        try:
            target_cols = list(self.config.FCI_TARGET_COLUMNS)
            if "opponent_team" not in target_cols:
                target_cols.append("opponent_team")
            
            df = df[target_cols].copy()
        except Exception as e:
            raise(e)

        df = df.rename(columns={
            "transfers_out_event": "transfers_out",
            "transfers_balance_event": "transfers_balance",
            "transfers_in_event": "transfers_in",
            "event_points": "total_points",
        })

        return df.assign(season=self.current_season)

    def get_data(self) -> pd.DataFrame:
        return self._merge_and_clean()

class FetchBetting(FetchData):
    def __init__(self, config_data: ContentConfig):
        super().__init__(seasons=config_data.BETTING_SEASONS)
        self.config = config_data
        self.teams_map = {
            'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa', 'Bournemouth': 'Bournemouth', 'Brentford': 'Brentford',
            'Brighton': 'Brighton', 'Burnley': 'Burnley',
            'Coventry': 'Coventry City', 'Hull': 'Hull City',
            'Crystal Palace': 'Crystal Palace', 'Everton': 'Everton', 'Fulham': 'Fulham', 'Ipswich': 'Ipswich',
            'Leeds': 'Leeds', 'Leicester': 'Leicester', 'Liverpool': 'Liverpool',
            'Luton': 'Luton', 'Man City': 'Man City', 'Man United': 'Man Utd', 'Newcastle': 'Newcastle',
            "Nott'm Forest": "Nott'm Forest", 'Sheffield United': 'Sheffield Utd',
            'Southampton': 'Southampton', 'Sunderland': 'Sunderland', 'Tottenham': 'Spurs', 'West Ham': 'West Ham',
            'Wolves': 'Wolves', 'Chelsea': 'Chelsea'
        }

        self.betting_features = [
            'date', 'time', 'hometeam', 'awayteam', 'season',
            'avgh', 'avgd', 'avga',
            'avg>2.5', 'avg<2.5',
            'ahh'
        ]

    def _fetch_fixtures(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.BETTING_FIXTURES_URL)
        if "Div" in df.columns:
            df = df[df["Div"] == "E0"]
        df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
        df["season"] = self.seasons[-1]
        return df

    def _fetch_seasons(self):
        frames = []
        url = self.config.BETTING_URL
        for season in self.seasons:
            try:
                df = pd.read_csv(url.format(season))
                df["season"] = season
                frames.append(df)
            except Exception as e:
                raise(e)

        fixtures = self._fetch_fixtures()
        if not fixtures.empty:
            frames.append(fixtures)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _merge_and_clean(self) -> pd.DataFrame:
        df = self._fetch_seasons()
        if df.empty:
            return df

        df.columns = df.columns.str.lower()
        df = df.drop(columns=["div", "referee"], errors="ignore")

        df.hometeam, df.awayteam = df.hometeam.map(self.teams_map), df.awayteam.map(self.teams_map)

        available_features = [c for c in self.betting_features if c in df.columns]
        df = df[available_features]

        df.columns = df.columns.str.replace(">", "_over_", regex=False).str.replace("<", "_under_", regex=False)
        df.date = pd.to_datetime(df.date, dayfirst=True)

        df = df.drop_duplicates(subset=["date", "hometeam", "awayteam"], keep="first").reset_index(drop=True)
        return df

    def _odds_to_probs(self) -> pd.DataFrame:
        df = self._merge_and_clean()
        if df.empty:
            return df

        match_market_groups = [['avgh', 'avgd', 'avga']]
        goals_market_groups = [['avg_over_2.5', 'avg_under_2.5']]

        for group in match_market_groups:
            if all(col in df.columns for col in group):
                raw_probs = 1 / df[group]
                total_margin = raw_probs.sum(axis=1)
                prob_names = [col.replace('avg', 'prob') for col in group]
                df[prob_names] = raw_probs.div(total_margin, axis=0)

        for group in goals_market_groups:
            if all(col in df.columns for col in group):
                raw_probs = 1 / df[group]
                total_margin = raw_probs.sum(axis=1)
                prob_names = [col.replace('avg', 'prob') for col in group]
                df[prob_names] = raw_probs.div(total_margin, axis=0)

        return df

    def _transform_wide_to_long(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        if df_wide.empty:
            return df_wide

        home_df = pd.DataFrame()
        home_df['date'] = df_wide['date']
        home_df['season'] = df_wide['season']
        home_df['team'] = df_wide['hometeam']
        home_df['opponent_team'] = df_wide['awayteam']
        home_df['was_home'] = True

        home_df['prob_win'] = df_wide['probh']
        home_df['prob_draw'] = df_wide['probd']
        home_df['prob_lose'] = df_wide['proba']

        if 'ahh' in df_wide.columns:
            home_df['ahh'] = df_wide['ahh']

        away_df = pd.DataFrame()
        away_df['date'] = df_wide['date']
        away_df['season'] = df_wide['season']
        away_df['team'] = df_wide['awayteam']
        away_df['opponent_team'] = df_wide['hometeam']
        away_df['was_home'] = False

        away_df['prob_win'] = df_wide['proba']
        away_df['prob_draw'] = df_wide['probd']
        away_df['prob_lose'] = df_wide['probh']

        if 'ahh' in df_wide.columns:
            away_df['ahh'] = -df_wide['ahh']

        for df_part in [home_df, away_df]:
            for col in ['prob_over_2.5', 'prob_under_2.5']:
                if col in df_wide.columns:
                    df_part[col] = df_wide[col]

        df_long = pd.concat([home_df, away_df], ignore_index=True)
        df_long = df_long.sort_values(by=['season', 'date', 'team']).reset_index(drop=True)

        return df_long

    def get_data(self) -> pd.DataFrame:
        df_wide = self._odds_to_probs()
        df_long = self._transform_wide_to_long(df_wide)

        output_cols = ['date', 'season', 'team', 'opponent_team', 'was_home',
                       'prob_win', 'prob_draw', 'prob_lose', 'ahh',
                       'prob_over_2.5', 'prob_under_2.5']

        final_cols = [c for c in output_cols if c in df_long.columns]

        return df_long[final_cols]

class MergeData:
    def __init__(self, fci_df: pd.DataFrame,
                 vaastav_df: pd.DataFrame,
                 betting_df: pd.DataFrame):
        
        self.fci_df = fci_df
        self.vaastav_df = vaastav_df
        self.betting_df = betting_df

    def _vaastav_fci(self) -> pd.DataFrame:
        return pd.concat([self.vaastav_df, self.fci_df], ignore_index=True)

    def _elo_betting_vaastav(self) -> pd.DataFrame:
        fpl_df = self._vaastav_fci()
        betting_df = self.betting_df
        
        fpl_df['kickoff_time'] = pd.to_datetime(fpl_df['kickoff_time']).dt.normalize()
        betting_df['date'] = pd.to_datetime(betting_df['date']).dt.normalize()

        fpl_df['team'] = fpl_df['team'].astype(str)
        betting_df['team'] = betting_df['team'].astype(str)
        
        fpl_df['season'] = fpl_df['season'].astype(str).apply(
            lambda x: re.sub(r'^20(\d{2})-(\d{2})$', r'\1\2', x) if '-' in x else x
        )
        betting_df['season'] = betting_df['season'].astype(str)

        df = pd.merge(
            fpl_df,
            betting_df,
            left_on=['kickoff_time', 'team', 'season'],
            right_on=['date', 'team', 'season'],
            how='inner'
        )
        df['fixture'] = df['fixture'].astype(str)
        df["season"] = df["season"].astype(float)
        df.drop(columns=['was_home_x', 'opponent_team_x'], inplace=True)
        df.rename(columns={'was_home_y': 'was_home',
                           'opponent_team_y': 'opponent_team',
                           'expected_goals_conceded': 'xg_conceded',
                           'expected_goals': 'xg',
                           'expected_goal_involvements': 'xg_involvements',
                           'team_expected_points': 'team_xp',
                           'opp_expected_points': 'opp_xp',
                           'expected_assists': 'xa'}, inplace=True)

        df = df.loc[df["position"] != "AM"].reset_index(drop=True)

        minutes = pd.to_numeric(df["minutes"], errors="coerce").replace(0, np.nan)
        for col in ["xa", "xg_involvements", "xg", "xg_conceded"]:
            if col in df.columns:
                num_col = pd.to_numeric(df[col], errors="coerce")
                res = (num_col / (minutes / 90.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                df[f"{col}_per_90"] = res.astype(float)

        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df = df.sort_values(by=["code", "kickoff_time", "season", "gw"]).reset_index(drop=True)

        return df

    def get_data(self) -> pd.DataFrame:
        df = self._elo_betting_vaastav()
        return df


def fetch_and_save_eda_data() -> pd.DataFrame:
    content_cfg = ContentConfig()

    vaastav_fetcher = FetchVaastav(config_data=content_cfg)
    fci_fetcher = FetchFCI(config_data=content_cfg)
    betting_fetcher = FetchBetting(config_data=content_cfg)

    vaastav_df = vaastav_fetcher.get_data()
    fci_df = fci_fetcher.get_data()
    betting_df = betting_fetcher.get_data()

    merger = MergeData(vaastav_df=vaastav_df, betting_df=betting_df, fci_df=fci_df)
    data = merger.get_data()

    print("--- DIAGNOSTICS ---")
    print(f"Vaastav shape: {vaastav_df.shape}")
    print(f"FCI shape: {fci_df.shape}")
    print(f"Betting shape: {betting_df.shape}")
    print(f"Merged shape: {data.shape}")
    print(f"Seasons in merged data: {data['season'].unique()}")
    latest = data[data["season"] == data["season"].max()]
    print("Rows per gameweek in latest season (is_played):")
    print(latest.groupby(["gw", "is_played"]).size().unstack(fill_value=0))
    print(f"Rows to forecast (is_played=False): {(~data['is_played']).sum()}")
    missing_vals = data.isna().sum()
    print("Missing values:\n", missing_vals[missing_vals > 0])
    print(f"Total Duplicates: {data.duplicated().sum()}")
    print("-------------------")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(config.EDA, index=False)
    return data


if __name__ == "__main__":
    fetch_and_save_eda_data()