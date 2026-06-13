import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd

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
    SEASONS: list[str] = field(default_factory=lambda: SEASONS)
    DROP_COLS: list[str] = field(default_factory=lambda: [
        'modified', 'mng_clean_sheets', 'mng_draw',
        'mng_goals_scored', 'mng_loss', 'mng_underdog_draw',
        'mng_underdog_win', 'mng_win'
    ])


class FetchData(ABC):
    def __init__(self, seasons: list[str]):
        self.seasons = seasons

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

    def merge_and_clean(self, gw_df: pd.DataFrame, id_df: pd.DataFrame) -> pd.DataFrame:
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

        # Łączenie i czyszczenie odbywa się automatycznie wewnątrz get_data
        final_df = self.merge_and_clean(gw_df, id_df)
        return final_df

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


if __name__ == "__main__":
    logger.info("Rozpoczęcie potoku przetwarzania danych FPL.")

    content_cfg = ContentConfig()
    fetcher = FetchVaastav(config_data=content_cfg)

    data = fetcher.get_data()

    if not data.empty:
        fetcher.run_tests(data)

        data.to_parquet(config.FPL_VAASTAV_PATH, index=False)
        logger.info("Pomyślnie wyeksportowano zbiór danych do formatu Parquet: %s", config.FPL_VAASTAV_PATH)
    else:
        logger.error("Potok zatrzymany. Wynikowy zbiór danych (final_data) jest pusty.")

    logger.info("Zakończenie działania procesu.")