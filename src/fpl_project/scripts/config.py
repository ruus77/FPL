from pathlib import Path
from dotenv import load_dotenv

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]

# Stałe funkcji straty FPLLoss
EPS_MIN = 0.05
ETA = EPS_MIN
W_MID = 1.25
W_PREMIUM = 2.0

eps_min = EPS_MIN
eta = EPS_MIN
w_mid = W_MID
w_premium = W_PREMIUM

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT_DIR / "Data"
RESULTS_DIR = ROOT_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EDA = DATA_DIR / "eda.parquet"
FEATURES_ENGINEERING = DATA_DIR / "features_engineering.parquet"

ML_PREDS_PATH = DATA_DIR / "ml_preds.parquet"

PREDS_GLM = DATA_DIR / "preds_glm.parquet"
PREDS_XGBOOST = DATA_DIR / "preds_xgboost.parquet"
PREDS_MLP = DATA_DIR / "preds_mlp.parquet"
PREDS_CONV1D = DATA_DIR / "preds_conv1d.parquet"
PREDS_RF_META = DATA_DIR / "preds_rf_meta.parquet"
PREDS_GLM_META = DATA_DIR / "preds_glm_meta.parquet"


def get_preds_path(algorithm_name: str) -> Path:
    return DATA_DIR / f"preds_{algorithm_name.lower()}.parquet"
