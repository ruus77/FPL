# Predicting Fantasy Premier League Player Performance Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.7-yellow.svg)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-red.svg)](https://xgboost.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning research and decision-support project developed as part of an **Engineering Thesis** in **Applied Mathematics (Data Analysis)** at the **Cracow University of Technology (Politechnika Krakowska)**, Faculty of Computer Science and Telecommunications.

* **Author:** Samuel Rus
* **Supervisor:** dr Adam Sebastian Marszałek

---

## 📌 Project Overview

Fantasy Premier League (FPL) is one of the world's most popular fantasy sports platforms, engaging millions of managers globally each season. Predicting player performance (gameweek `total_points`) is a notoriously difficult sports analytics challenge characterized by:
* **High zero-inflation:** non-playing squad members, unused substitutes, and defenders conceding goals frequently score 0 or fewer points.
* **Heavy right-skewed variance:** high-performing "premium" players produce occasional explosive double-digit hauls that dictate overall ranking success.
* **Asymmetric decision risk:** underpredicting an elite captain candidate costs far more transfer and captaincy value than mispredicting a cheap bench player.

This project delivers a robust, reproducible predictive system that combines tabular models, deep sequential architectures, domain-tailored loss functions, stacking meta-models, and team optimization via Integer Linear Programming (ILP).

---

## 🏗️ System Overview

The project pipeline is structured into five core stages:
1. **Multi-Source Data Ingestion:** Merges gameweek records (`vaastav/Fantasy-Premier-League`), expected metrics (`FPL-Core-Insights`), and betting odds (`football-data.co.uk`).
2. **Exploratory Data Analysis:** Evaluates distributions, Spearman rank correlations, PCA dimensionality reduction, and non-parametric Kruskal-Wallis & Dunn post-hoc tests across price tiers.
3. **Leakage-Free Feature Engineering:** Constructs temporal performance indicators (EMA-3, EMA-8, lags, rolling standard deviations) strictly shifted (`shift(1)`) relative to target gameweeks.
4. **Multi-Family Predictive Modeling:** Trains tabular (GLM, XGBoost, CatBoost, MLP) and sequential window architectures (Residual Conv1D, Stacked LSTM) using `StandardScaler` and domain-tailored loss functions (`FPLLoss`).
5. **Stacking Ensemble & Decision Optimization:** Combines out-of-fold predictions via a Ridge/GLM meta-regressor, feeding an Integer Linear Programming (ILP) squad optimization engine presented in an interactive Streamlit dashboard.

---

## 🔬 Key Methodological Principles

### 1. Zero Data Leakage Guarantee
* Match event features (e.g. goals, assists, minutes, expected metrics $xG$, $xA$, bonus points) are strictly shifted by at least one gameweek (`shift(1)`) relative to the target gameweek.
* Pre-match features (bookmaker implied probabilities, home/away status, player position) are used without temporal shifts.
* Feature scaling (`StandardScaler`) is fitted strictly on the training set (`fit_transform`) and subsequently applied (`transform`) to validation and test sets.

### 2. Chronological Train/Val/Test Split
Sports time-series data violates the i.i.d. assumption. To reflect real-world forecasting, random splits are prohibited:
* **Train:** Seasons 2022/23, 2023/24, and 2024/25
* **Validation:** Season 2025/26 (used for early stopping, model checkpointing, hyperparameter tuning)
* **Test:** Season 2026/27 (out-of-sample final evaluation)

### 3. Asymmetric Custom Loss (`FPLLoss`)

To reflect FPL decision dynamics, a custom PyTorch loss function is implemented:

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2 \cdot w^{\text{price}}_i \cdot \big(\ln(1 + \text{minutes}_i) + \varepsilon_{\min}\big)
$$

where:
* **Price weighting ($w^{\text{price}}_i$):** penalizes errors on high-value premium players ($w_{\text{premium}} = 2.0$) compared to mid-priced ($w_{\text{mid}} = 1.25$) and budget players ($w = 1.0$), with empirical thresholds defined by price quantiles ($p_{\text{low}} = 4.5\text{m}$, $p_{\text{high}} = 7.0\text{m}$).
* **Minutes weighting:** suppresses gradient distortion caused by non-playing squad members with 0 minutes.

### 4. Price-Segmented Evaluation Metrics
In addition to global MSE, MAE, and Adjusted $R^2$, metrics are evaluated separately across price classes:
* `MSE@budget`, `MAE@budget` ($\text{price} < 4.5\text{m}$)
* `MSE@mid`, `MAE@mid` ($4.5\text{m} \le \text{price} < 7.0\text{m}$)
* `MSE@premium`, `MAE@premium` ($\text{price} \ge 7.0\text{m}$)

---

## 📂 Repository Structure

```text
FPL-1/
├── src/
│   └── fpl_project/
│       ├── data_processing/
│       │   ├── data_import.py             # Multi-source data scrapers & merge logic
│       │   ├── eda.ipynb                  # Statistical tests (Spearman, PCA, Kruskal-Wallis, Dunn)
│       │   └── features_engineering.ipynb # Leakage-free feature transforms (EMA, lags, rolling std)
│       ├── dashboards/
│       │   └── streamlit_test.py          # Interactive squad selector dashboard
│       ├── models/
│       │   ├── train_nn.py                # Deep learning training script (MLP, Conv1D, LSTM)
│       │   ├── train_ensemble.ipynb       # Tabular models (GLM, XGBoost, CatBoost)
│       │   ├── meta_model.ipynb           # Stacking meta-regressor evaluation
│       │   └── trained_models/            # Serialized model weights (.pth) & predictions (.pt)
│       ├── scripts/
│       │   ├── config.py                  # Project paths, season definitions, and artifacts
│       │   ├── data_utils/
│       │   │   ├── colors_config.py       # Official Premier League team hex palettes
│       │   │   ├── features_config.py     # Feature groupings & leakage column definitions
│       │   │   ├── features_processing.py # Feature engineering classes
│       │   │   └── visuals.py             # Plotting utilities & custom styles
│       │   └── model_utils/
│       │       ├── custom_metrics.py      # TorchMetrics classes (CustomMSE, CustomMAE)
│       │       ├── data.py                # FPLDataPipe, SequenceDataPipe, DataLoader factories
│       │       ├── mlflow_tracker.py      # MLflow experiment logger
│       │       ├── models.py              # PyTorch architectures (MLP, Conv1D, LSTM, FPLLoss)
│       │       ├── selector.py            # RandomizedSearchCV pipeline for tabular models
│       │       └── trainer.py             # Neural network training engine with EarlyStopping
│       └── engineering_thesis.tex         # Complete LaTeX source of the thesis (mwrep class)
├── Data/                                  # Local storage for raw & processed Parquet datasets
├── mlartifacts/                           # MLflow experiment artifacts
├── mlflow.db                              # SQLite database for MLflow experiment logs
├── pyproject.toml                         # Package configuration (fpl_project)
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.11 or higher
* CUDA-capable GPU (recommended for PyTorch training, but CPU execution is fully supported)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ruus77/FPL.git
   cd FPL
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

---

## 📊 Workflow & Usage

### 1. Data Ingestion & Feature Engineering
Download raw gameweek records from Vaastav, FCI, and betting lines from football-data.co.uk:
```python
from fpl_project.data_processing.data_import import fetch_and_save_eda_data
fetch_and_save_eda_data()
```
Run feature engineering to create lagged variables, EMA, and rolling standard deviations:
```python
from fpl_project.data_processing.data_import import build_and_save_features_engineering_data
build_and_save_features_engineering_data()
```

### 2. Deep Learning Training (MLP, Conv1D, LSTM)
Train neural architectures using `train_nn.py`:
```bash
python -m fpl_project.models.train_nn
```
The script automatically:
* Loads chronological splits and standardizes features via `StandardScaler`.
* Calculates dynamic price bounds ($p_{\text{low}}$, $p_{\text{high}}$) from training quantiles.
* Fits models across multiple seeds with `FPLLoss` and standard `MSELoss`.
* Logs metrics (`MSE`, `MAE`, segmented errors) and checkpoints into MLflow.

### 3. Experiment Tracking
Inspect training runs, learning curves, and segmented loss comparisons via MLflow:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open your browser at `http://127.0.0.1:5000`.

### 4. Interactive Dashboard
Launch the Streamlit application for interactive player recommendations and squad optimization:
```bash
streamlit run src/fpl_project/dashboards/streamlit_test.py
```

---

## 📈 Key Research Findings

* **Significance of Price Segmentation:** Kruskal-Wallis testing ($H = 11337.22, p < 0.001$) and post-hoc Dunn tests with Bonferroni correction confirm statistically distinct scoring distributions across quantile-based price tiers (`budget`, `mid`, `premium`).
* **Impact of `FPLLoss`:** Penalizing prediction errors on high-priced players drastically reduces `MSE@premium` without sacrificing overall accuracy on budget assets.
* **Stacking Generalization:** The ensemble meta-regressor out-performs standalone individual models on out-of-sample data by effectively blending tree-based interactions with sequential form capture from recurrent architectures.

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).

If you refer to this research or codebase in academic work, please cite:
```bibtex
@thesis{rus2026fpl,
  author       = {Samuel Rus},
  title        = {Predicting Premier League Player Performance Using Selected Machine Learning Methods},
  type         = {Engineering Thesis},
  institution  = {Cracow University of Technology, Faculty of Computer Science and Telecommunications},
  year         = {2026},
  supervisor   = {dr Adam Sebastian Marszałek}
}
```
