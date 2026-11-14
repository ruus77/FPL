FPL points prediction
=====================

Engineering thesis project: predicting Fantasy Premier League (FPL) player points
with machine learning and combining base models into a meta-model.

Models
------
Base models, each trained with MSE and with FPLLoss (price- and minutes-weighted MSE):
  - GLM (glum), XGBoost, MLP, Conv1D (sequence of the last 8 gameweeks)
Meta-models fitted on validation predictions:
  - NNLS, Ridge with context features, mean ensemble, best single model

Data
----
  - Historical FPL gameweek data (public GitHub dataset), seasons 2022/23-2025/26
  - Current season data (FPL-Core-Insights), season 2026/27
  - Betting odds (football-data.co.uk)
Chronological split: train = 2022/23-2024/25, validation = 2025/26, test = 2026/27.
Match features are shifted by one gameweek (shift(1)) to prevent data leakage.

Structure
---------
  src/fpl_project/data_processing/   data import, EDA and feature engineering notebooks
  src/fpl_project/models/            run_experiments.py (entry point), meta_model.ipynb
  src/fpl_project/scripts/config.py  paths, seeds, hyperparameters
  src/fpl_project/scripts/model_utils/
      data, datasets, models, losses, evaluation, trainer, selector, mlflow_tracker
      pipeline/  data_preparation -> tabular / neural -> meta -> reporting,
                 persistence (saved models), forecast (unplayed fixtures)

Setup
-----
  pip install -r requirements.txt

Usage
-----
  python -m fpl_project.data_processing.data_import      # download data to Data/
  python -m fpl_project.models.run_experiments           # full experiment (~70 min)
(run from the repository root with PYTHONPATH=src)

Outputs
-------
  results/          metrics, price-tier tables, meta-model weights, predictions
                    (preds_valid / preds_test / preds_predict .parquet)
  trained_models/   fitted base models and preprocessing
                    (load with pipeline.persistence.load_trained_models)
  mlflow.db         MLflow experiment tracking
Predictions for the next gameweek (results/preds_predict.parquet) are produced once
betting odds for its fixtures are published.
