from pathlib import Path
from dotenv import load_dotenv

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]

# Ziarno losowe i parametry walidacji krzyżowej
SEED = 77
CV_SPLITS = 5
N_ITER_SEARCH = 20

# Stałe funkcji straty FPLLoss i progów cenowych
PRICE_QUANTILE_LOW = 0.50
PRICE_QUANTILE_HIGH = 0.955
EPS_MIN = 0.1
W_MID = 1.25
W_PREMIUM = 2.0

eps_min = EPS_MIN
w_mid = W_MID
w_premium = W_PREMIUM

# Zakres obcinania predykcji i regularyzacja meta-modelu
PRED_CLIP_MIN = 0.0
PRED_CLIP_MAX = 15.0
META_RIDGE_ALPHA = 10.0

# Hiperparametry sieci neuronowych (MLP, Conv1D)
NN_BATCH_SIZE = 128
NN_SEQ_LEN = 8
NN_HIDDEN_SIZES = [32, 16]
NN_CONV_CHANNELS = [32, 16]
NN_KERNEL_SIZE = 3
NN_DROPOUT = 0.2
NN_LR = 1e-3
NN_MIN_LR = 1e-5
NN_WEIGHT_DECAY = 0.005
NN_EPOCHS = 150
NN_PATIENCE = 10
NN_SCHEDULER_FACTOR = 0.5
NN_SCHEDULER_PATIENCE = 3
NN_SCHEDULER_COOLDOWN = 2
NN_ES_SEASON = 2425
NN_ES_MIN_GW = 31

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

# Ścieżki plików wynikowych w results/
METRICS_TEST_PATH = RESULTS_DIR / "metrics_test.csv"
METRICS_VALID_PATH = RESULTS_DIR / "metrics_valid.csv"
TABLE_TIERS_PATH = RESULTS_DIR / "table_tiers.csv"
TIER_COUNTS_PATH = RESULTS_DIR / "tier_counts.csv"
META_WEIGHTS_PATH = RESULTS_DIR / "meta_weights.csv"
PREDS_VALID_PATH = RESULTS_DIR / "preds_valid.parquet"
PREDS_TEST_PATH = RESULTS_DIR / "preds_test.parquet"
RESULTS_README_PATH = RESULTS_DIR / "README.md"


def get_preds_path(algorithm_name: str) -> Path:
    return DATA_DIR / f"preds_{algorithm_name.lower()}.parquet"
