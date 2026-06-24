from pathlib import Path
from dotenv import load_dotenv

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

load_dotenv()

# Zmienione: dodano .resolve() oraz czwarty .parent
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT_DIR / "Data"

# Automatyczne tworzenie folderu Data, jeśli nie istnieje (super przydatne przy Gicie!)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Definicje plików
FPL_VAASTAV_PATH = DATA_DIR / "fpl_vaastav.parquet"
FPL_HISTORICAL_PATH = DATA_DIR / "FPL_historical.parquet"
TIDY_DATA_PATH = DATA_DIR / "tidy_data.parquet"

FPL_DATA_PATH = DATA_DIR / "fpl_data.parquet"