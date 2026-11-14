# FPL Points Prediction

Engineering thesis project: predicting Fantasy Premier League (FPL) player points with machine
learning and combining the base models into a meta-model.

## Models

| Level | Models | Training |
| :--- | :--- | :--- |
| Base | GLM (glum), XGBoost, MLP, Conv1D | each with **MSE** and **FPLLoss** (price- and minutes-weighted MSE) |
| Meta | NNLS, Ridge with context features, mean ensemble, best single model | fitted on validation predictions |

Conv1D uses a window of the last 8 gameweeks; the other models use lagged tabular features.

## Data

| Source | Content |
| :--- | :--- |
| Public FPL dataset (GitHub) | gameweek data, seasons 2022/23 – 2025/26 |
| FPL-Core-Insights | current season 2026/27 |
| football-data.co.uk | betting odds |

- **Chronological split:** train = 2022/23 – 2024/25, validation = 2025/26, test = 2026/27.
- **No leakage:** match features are shifted by one gameweek (`shift(1)`).

## Project structure

```
src/fpl_project/
├── data_processing/        data import, EDA and feature engineering notebooks
├── models/
│   ├── run_experiments.py  entry point of the experiment
│   └── meta_model.ipynb    meta-model analysis
└── scripts/
    ├── config.py           paths, seed, hyperparameters
    ├── data_utils/         feature configuration and engineering
    └── model_utils/        data, datasets, models, losses, evaluation, trainer, selector
        └── pipeline/       data_preparation → tabular / neural → meta → reporting,
                            persistence (saved models), forecast (unplayed fixtures)
```

## Getting started

```bash
pip install -r requirements.txt

# run from the repository root
export PYTHONPATH=src
python -m fpl_project.data_processing.data_import   # download data to Data/
python -m fpl_project.models.run_experiments        # full experiment (~70 min)
```

## Outputs

| Path | Content |
| :--- | :--- |
| [`results/`](results/README.md) | metrics, price-tier tables, meta-model weights, predictions |
| `trained_models/` | fitted base models and preprocessing (`pipeline.persistence.load_trained_models`) |
| `mlflow.db` | MLflow experiment tracking |

Predictions for the next gameweek (`results/preds_predict.parquet`) are produced once betting
odds for its fixtures are published.

## Notes

- Code, comments and `results/README.md` are in English; plot titles and labels in the notebooks
  are in Polish (thesis figures).
