from pathlib import Path
from dotenv import load_dotenv
import scipy.stats as stats

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]

# Random seed and cross-validation
SEED = 77
CV_SPLITS = 5
N_ITER_SEARCH = 20

# Hyperparameter search spaces for tabular models (RandomizedSearchCV)
GLM_PARAM_GRID = {
    "alpha": stats.loguniform(1e-5, 0.01),
    "l1_ratio": stats.uniform(loc=0.0, scale=1.0),
    "family": ["poisson", "negative.binomial", "tweedie", "normal"],
    "link": ["auto", "log"],
}

XGB_PARAM_GRID = {
    "n_estimators": stats.randint(150, 500),
    "max_depth": stats.randint(3, 7),
    "learning_rate": stats.uniform(0.01, 0.05),
    "subsample": stats.uniform(0.7, 0.3),
    "colsample_bytree": stats.uniform(0.6, 0.3),
    "min_child_weight": stats.randint(1, 10),
    "gamma": stats.uniform(0.0, 5.0),
    "reg_alpha": stats.uniform(0.0, 2.0),
    "reg_lambda": stats.uniform(0.5, 3.0),
}

# FPLLoss constants and price-tier quantiles
PRICE_QUANTILE_LOW = 0.50
PRICE_QUANTILE_HIGH = 0.955
EPS_MIN = 0.1
W_MID = 1.25
W_PREMIUM = 2.0

# Prediction clipping range and meta-model regularisation
PRED_CLIP_MIN = 0.0
PRED_CLIP_MAX = 15.0
META_RIDGE_ALPHA = 10.0

# Neural network hyperparameters (MLP, Conv1D)
NN_BATCH_SIZE = 128
NN_SEQ_LEN = 8
NN_HIDDEN_SIZES = [128, 64]
NN_CONV_CHANNELS = [128, 64]
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
TRAINED_MODELS_DIR = ROOT_DIR / "trained_models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EDA = DATA_DIR / "eda.parquet"
FEATURES_ENGINEERING = DATA_DIR / "features_engineering.parquet"

PREDS_GLM_META = DATA_DIR / "preds_glm_meta.parquet"

# Output files in results/
METRICS_TEST_PATH = RESULTS_DIR / "metrics_test.csv"
METRICS_VALID_PATH = RESULTS_DIR / "metrics_valid.csv"
TABLE_TIERS_PATH = RESULTS_DIR / "table_tiers.csv"
TIER_COUNTS_PATH = RESULTS_DIR / "tier_counts.csv"
META_WEIGHTS_PATH = RESULTS_DIR / "meta_weights.csv"
PREDS_VALID_PATH = RESULTS_DIR / "preds_valid.parquet"
PREDS_TEST_PATH = RESULTS_DIR / "preds_test.parquet"
PREDS_PREDICT_PATH = RESULTS_DIR / "preds_predict.parquet"
RESULTS_README_PATH = RESULTS_DIR / "README.md"
