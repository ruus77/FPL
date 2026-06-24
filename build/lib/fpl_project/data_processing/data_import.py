import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd
import soccerdata as sd


from fpl_project.scripts import config
from fpl_project.scripts.config import SEASONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.DATA_DIR / "fpl_historical.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContentConfig:
    VAASTAV_GW_URL: str = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{}/gws/gw{}.csv"
    VAASTAV_ID_URL: str = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{}/players_raw.csv"
    BETTING_URL : str = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"

    SEASONS: list[str] = field(default_factory=lambda: SEASONS)
    BETTING_SEASONS: list[str] = field(
        default_factory=lambda: [s.replace("-", "").replace("20", "") for s in SEASONS])

    DROP_COLS: list[str] = field(default_factory=lambda: [
        'modified', 'mng_clean_sheets', 'mng_draw',
        'mng_goals_scored', 'mng_loss', 'mng_underdog_draw',
        'mng_underdog_win', 'mng_win', 'clearances_blocks_interceptions',
        'defensive_contribution',   'recoveries',
        'tackles'])


class FetchData(ABC):
    def __init__(self, seasons: list[str]):
        self.seasons = seasons

    @abstractmethod
    def _merge_and_clean(self)->pd.DataFrame:
        pass

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def run_tests(self, df: pd.DataFrame) -> None:
        pass


class FetchVaastav(FetchData):
    def __init__(self, config_data: ContentConfig):
        super().__init__(seasons=config_data.SEASONS)
        self.config = config_data

    def _fetch_gameweeks(self) -> pd.DataFrame:
        frames = []
        skipped = 0

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
                except Exception:
                    skipped += 1

            if season_frames:
                frames.extend(season_frames)
                logger.info("Pobrano dane GW dla sezonu %s. Liczba kolejek: %d", season, len(season_frames))
            else:
                logger.warning("Brak danych GW dla sezonu %s.", season)

        if skipped > 0:
            logger.warning("Pominięto łącznie %d plików GW (brak pliku na serwerze lub błąd sieciowy).", skipped)

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
                logger.info("Pobrano metadane zawodników dla sezonu %s. Liczba rekordów: %d", season, len(df))
            except Exception as e:
                logger.error("Błąd pobierania metadanych zawodników dla sezonu %s: %s", season, e)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _merge_and_clean(self, gw_df: pd.DataFrame, id_df: pd.DataFrame) -> pd.DataFrame:
        if gw_df.empty or id_df.empty:
            logger.error("Nie można wykonać operacji merge. Jeden ze zbiorów wejściowych jest pusty.")
            return pd.DataFrame()

        logger.info("Rozpoczęto łączenie zbiorów GW oraz IDs.")

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

        df = df.drop(columns=self.config.DROP_COLS, errors="ignore")

        before = len(df)
        df = df.dropna(subset=["team", "season", "was_home", "kickoff_time"])
        dropped = before - len(df)

        if dropped > 0:
            logger.warning(
                "Usunięto %d wierszy z powodu braków danych w kluczowych kolumnach (team, season, was_home, kickoff_time).",
                dropped
            )

        logger.info("Proces czyszczenia zakończony. Wymiary końcowego zbioru: %d wierszy, %d kolumn.", *df.shape)

        return df

    def get_data(self) -> pd.DataFrame:
        logger.info("Rozpoczęto pełny proces pobierania i przetwarzania danych z repozytorium Vaastav.")

        gw_df = self._fetch_gameweeks()
        id_df = self._fetch_ids()

        df = self._merge_and_clean(gw_df, id_df)
        return df

    def run_tests(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.error("[Walidacja] Zbiór danych jest pusty. Przerywam testy.")
            return

        logger.info("Rozpoczęto walidację jakości danych (Data Quality Report).")

        gw_counts = df.groupby("season")["gw"].nunique().sort_index()
        for season, count in gw_counts.items():
            if count < 38:
                logger.warning("[Walidacja] Sezon %s: Zidentyfikowano niepełną liczbę GW (%d). Oczekiwano 38.", season,
                               count)
            else:
                logger.info("[Walidacja] Sezon %s: Kompletność GW (38) potwierdzona.", season)

        for season, count in df.groupby("season").size().sort_index().items():
            logger.info("[Walidacja] Sezon %s: Liczba poprawnych rekordów wynosi %d.", season, count)

        for season, group in df.groupby("season"):
            null_pct = group.isna().mean()
            null_pct = null_pct[null_pct > 0].sort_values(ascending=False)
            if null_pct.empty:
                logger.info("[Walidacja] Sezon %s: Brak wartości pustych w zbiorze.", season)
            else:
                for col, pct in null_pct.items():
                    logger.warning("[Walidacja] Sezon %s: Wykryto braki danych w kolumnie '%s' (%.1f%%).", season, col,
                                   pct * 100)

        cols_per_season = df.groupby("season").apply(lambda g: set(g.columns[g.notna().any()]))
        all_cols = set.union(*cols_per_season.values)
        for season, cols in cols_per_season.items():
            missing = all_cols - cols
            if missing:
                logger.warning("[Walidacja] Sezon %s: Brakujące kolumny względem schematu globalnego: %s", season,
                               sorted(missing))
            else:
                logger.info("[Walidacja] Sezon %s: Schemat kolumn jest spójny z globalnym.", season)

        for season, group in df.groupby("season"):
            if "value" in group.columns:
                v = group["value"]
                logger.info("[Walidacja] Sezon %s: Statystyki wartości 'value' - Min: %.1f, Max: %.1f, Średnia: %.1f",
                            season, v.min(), v.max(), v.mean())
                if v.min() < 3.0 or v.max() > 20.0:
                    logger.warning(
                        "[Walidacja] Sezon %s: Wykryto wartości w kolumnie 'value' poza dopuszczalnym zakresem [3.0, 20.0].",
                        season)

        dupes = df.duplicated(subset=["code", "season", "gw"]).sum()
        if dupes > 0:
            logger.warning(
                "[Walidacja] Wykryto duplikaty na poziomie klucza głównego (code, season, gw). Liczba rekordów: %d.",
                dupes)
        else:
            logger.info("[Walidacja] Test unikalności klucza głównego zakończony pomyślnie. Brak duplikatów.")

        for season, count in df.groupby("season")["code"].nunique().sort_index().items():
            logger.info("[Walidacja] Sezon %s: Liczba unikalnych zawodników wynosi %d.", season, count)

        logger.info("Zakończono walidację jakości danych.")


class FetchBetting(FetchData):
    def __init__(self, config_data: ContentConfig):
        super().__init__(seasons=config_data.BETTING_SEASONS)
        self.config = config_data
        self.teams_map = {
            'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa', 'Bournemouth': 'Bournemouth', 'Brentford': 'Brentford',
            'Brighton': 'Brighton', 'Burnley': 'Burnley',
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

    def _fetch_seasons(self):
        frames = []
        url = self.config.BETTING_URL
        for season in self.seasons:
            try:
                df = pd.read_csv(url.format(season))
                df["season"] = season
                frames.append(df)
            except Exception as e:
                logger.error("Błąd pobierania danych bukmacherskich dla sezonu %s: %s", season, e)
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

        logger.info("Pomyślnie przekonwertowano kursy bukmacherskie na znormalizowane prawdopodobieństwa.")
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

    def run_tests(self, df: pd.DataFrame) -> None:
        pass


class FetchELO(FetchData):
    def __init__(self, config_data: ContentConfig, betting_df: pd.DataFrame):
        super().__init__(seasons=config_data.BETTING_SEASONS)
        self.config = config_data
        self.betting_df = betting_df

        self.betting_to_elo_map = {
            'Spurs': 'Tottenham',
            'Man Utd': 'Man United',
            "Nott'm Forest": 'Forest',
            'Sheffield Utd': 'Sheffield United'
        }
        self.elo_to_betting_map = {v: k for k, v in self.betting_to_elo_map.items()}

    def _fetch_seasons(self) -> pd.DataFrame:
        if self.betting_df.empty:
            return pd.DataFrame()
        self.betting_df.date = pd.to_datetime(self.betting_df.date, dayfirst=True)
        min_date = self.betting_df.date.min() - pd.Timedelta(days=100)

        elo_teams_list = [self.betting_to_elo_map.get(team, team) for team in self.betting_df['team'].unique()]
        elo = sd.ClubElo()
        frames = []
        for team in elo_teams_list:
            try:
                team_history = elo.read_team_history(team)
                team_history = team_history[team_history.index >= min_date]
                team_history = team_history.reset_index()
                frames.append(team_history)
            except Exception as e:
                logger.error(f"Błąd pobierania danych elo dla drużyny {team}: {e}")

        if frames:
            return pd.concat(frames, axis=0, ignore_index=True)
        else:
            return pd.DataFrame()

    def _merge_and_clean(self) -> pd.DataFrame:
        df = self._fetch_seasons()
        if df.empty:
            return pd.DataFrame()

        df.columns = df.columns.str.lower()

        df["from"] = pd.to_datetime(df["from"])
        df["to"] = pd.to_datetime(df["to"])

        df['team'] = df['team'].replace(self.elo_to_betting_map)

        target_cols = ["from", "to", "team", "elo"]
        available_cols = [c for c in target_cols if c in df.columns]
        df = df[available_cols]

        if df.empty:
            return pd.DataFrame()
        df = df.sort_values(by=['team', 'from']).reset_index(drop=True)

        return df

    def get_data(self) -> pd.DataFrame:
        df = self._merge_and_clean()
        return df

    def run_tests(self, df: pd.DataFrame) -> None:
        pass

class MergeData:
    def __init__(self, vaastav_df:pd.DataFrame,
                 betting_df:pd.DataFrame,
                 elo_df:pd.DataFrame):

        self.vaastav_df = vaastav_df
        self.betting_df = betting_df
        self.elo_df = elo_df

    def _elo_betting(self)->pd.DataFrame:
        elo_df = self.elo_df
        betting_df = self.betting_df

        betting_df = betting_df.sort_values("date")
        elo_df = elo_df.sort_values("from")

        if elo_df.empty or betting_df.empty:
            return pd.DataFrame()
        betting_elo_df = pd.merge_asof(betting_df,
                                       elo_df,
                                       left_on="date",
                                       right_on="from",
                                       by="team",
                                       direction="backward")

        if "from" in betting_elo_df.columns:
            betting_elo_df = betting_elo_df.drop(columns=['from'])
        if "to" in betting_elo_df.columns:
            betting_elo_df = betting_elo_df.drop(columns=['to'])

        return betting_elo_df

    def _elo_betting_vaastav(self) -> pd.DataFrame:
        vaastav_df = self.vaastav_df
        elo_betting_df = self._elo_betting()

        if vaastav_df.empty or elo_betting_df.empty:
            return pd.DataFrame()

        vaastav_df = vaastav_df.copy()
        elo_betting_df = elo_betting_df.copy()

        vaastav_df['kickoff_time'] = pd.to_datetime(vaastav_df['kickoff_time'])
        elo_betting_df['date'] = pd.to_datetime(elo_betting_df['date'])

        vaastav_df['team'] = vaastav_df['team'].astype(str)
        elo_betting_df['team'] = elo_betting_df['team'].astype(str)
        vaastav_df['season'] = vaastav_df['season'].astype(str)
        elo_betting_df['season'] = elo_betting_df['season'].astype(str)

        df = pd.merge(
            vaastav_df,
            elo_betting_df,
            left_on=['kickoff_time', 'team', 'season'],
            right_on=['date', 'team', 'season'],
            how='inner'
        )
        df.drop(columns=['was_home_x', 'opponent_team_x'], inplace=True)
        df.rename(columns={'was_home_y': 'was_home',
                           'opponent_team_y': 'opponent_team'}, inplace=True)


        return df

    def get_data(self) -> pd.DataFrame:
        df = self._elo_betting_vaastav()
        return df

if __name__ == "__main__":
    logger.info("Rozpoczęcie potoku przetwarzania danych FPL.")

    content_cfg = ContentConfig()
    vaastav_fetcher = FetchVaastav(config_data=content_cfg)
    betting_fetcher = FetchBetting(config_data=content_cfg)

    vaastav_df = vaastav_fetcher.get_data()
    betting_df = betting_fetcher.get_data()

    elo_fetcher = FetchELO(config_data=content_cfg, betting_df=betting_df)

    elo_df = elo_fetcher.get_data()

    merger = MergeData(vaastav_df=vaastav_df, betting_df=betting_df, elo_df=elo_df)
    data = merger.get_data()

    if not data.empty:

        data.to_parquet(config.FPL_VAASTAV_PATH, index=False)
        logger.info("Pomyślnie wyeksportowano zbiór danych do formatu Parquet: %s", config.FPL_VAASTAV_PATH)
    else:
        logger.error("Potok zatrzymany. Wynikowy zbiór danych (final_data) jest pusty.")

    logger.info("Zakończenie działania procesu.")