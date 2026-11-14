# Experiment results

Test-set performance (season 2026/27, GW 1–10, 6,596 player-matches) of four
base models and four meta-models. Every model is trained twice: with plain **MSE** and with **FPLLoss**,
a weighted MSE that gives more weight to expensive players and players with more expected minutes.

## Key findings

- **Best model:** context Ridge meta-model (MSE training) with test MSE **4.313**, 1.9% lower than the best base model, Conv1D (FPLLoss training), with 4.398.
- **Meta-models help:** the context Ridge beats every base model on MSE (MSE and FPLLoss training).
- **FPLLoss on premium players:** lower premium-tier MSE than plain MSE training for 4 of 4 base models.

## Test results

Lower is better for MSE and MAE, higher for R². **Bold** marks the best value in each column.
Price tiers: **budget** < £4.5m ≤ **mid** < £6.925m ≤ **premium**.

### Trained with MSE

| Model | MSE | MAE | R² | MSE budget | MSE mid | MSE premium |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM | 4.660 | 1.231 | 0.260 | 4.247 | 4.366 | 12.953 |
| XGBoost | 4.452 | **1.154** | 0.293 | 3.828 | 4.256 | 12.458 |
| MLP | 4.460 | 1.191 | 0.291 | 3.725 | 4.269 | 13.131 |
| Conv1D | 4.470 | 1.237 | 0.290 | 3.743 | 4.289 | 12.922 |
| Meta: NNLS | 4.325 | 1.198 | 0.313 | 3.592 | 4.162 | 12.485 |
| Meta: context Ridge | **4.313** | 1.192 | **0.315** | **3.575** | **4.153** | **12.432** |
| Meta: mean ensemble | 4.388 | 1.182 | 0.303 | 3.679 | 4.213 | 12.603 |
| Meta: best single model (XGBoost) | 4.452 | **1.154** | 0.293 | 3.828 | 4.256 | 12.458 |

### Trained with FPLLoss

| Model | MSE | MAE | R² | MSE budget | MSE mid | MSE premium |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM | 4.671 | 1.245 | 0.258 | 4.236 | 4.391 | 12.858 |
| XGBoost | 4.440 | 1.192 | 0.295 | 3.841 | 4.261 | **11.947** |
| MLP | 4.519 | 1.200 | 0.282 | 3.859 | 4.328 | 12.691 |
| Conv1D | 4.398 | 1.195 | 0.301 | 3.657 | 4.224 | 12.814 |
| Meta: NNLS | 4.331 | 1.234 | 0.312 | **3.609** | 4.182 | 12.143 |
| Meta: context Ridge | **4.313** | **1.186** | **0.315** | 3.616 | **4.160** | 12.038 |
| Meta: mean ensemble | 4.401 | 1.190 | 0.301 | 3.723 | 4.231 | 12.291 |
| Meta: best single model (XGBoost) | 4.440 | 1.192 | 0.295 | 3.841 | 4.261 | **11.947** |

MAE per price tier and validation-set metrics are in `metrics_test.csv` and `metrics_valid.csv`.

## Models

- **Base models:** GLM (`glum`), XGBoost and MLP on lagged tabular features; Conv1D on the last
  8 gameweeks.
- **Meta-models**, fitted on validation-set predictions of the four base models:
  - *NNLS* – non-negative linear combination,
  - *context Ridge* – Ridge on predictions, position, price tier, minutes and their interactions,
  - *mean ensemble* – simple average,
  - *best single model* – the base model with the lowest validation error.

## Data

| Split | Seasons | Rows | Budget | Mid | Premium |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `train` | 2022/23 – 2024/25 | 80,618 | 29,611 | 47,376 | 3,631 |
| `valid` | 2025/26 | 29,338 | 10,748 | 17,154 | 1,436 |
| `test` | 2026/27 (GW 1–10) | 6,596 | 1,763 | 4,583 | 250 |

- Price tiers are the 50% and 95.5% quantiles of player price in the training set.
- Neural networks train on 74,533 rows; the last gameweeks of 2024/25 (GW ≥ 31,
  6,085 rows) are held out for early stopping.
- Conv1D skips the first 19,031 training rows, which lack a full 8-gameweek history.
- Rows dropped when joining predictions of all models: 0 (valid), 0 (test).

## FPLLoss

Each observation is weighted by price tier and expected minutes:

```
w = w_tier * (ln(1 + m) + eps_min),   m = clip(minutes_ema_3 / 90, 0, 1)
```

| Constant | Value |
| :--- | ---: |
| `w_tier` budget | 1.0 |
| `w_tier` mid (`W_MID`) | 1.25 |
| `w_tier` premium (`W_PREMIUM`) | 2.0 |
| `eps_min` (`EPS_MIN`) | 0.1 |

Weights are not normalised, so FPLLoss values are on a different scale than MSE.

## Hyperparameters

Tuned with `RandomizedSearchCV` (20 candidates, 5-fold `TimeSeriesSplit` with folds split
between gameweeks). Search spaces: `GLM_PARAM_GRID` and `XGB_PARAM_GRID` in `scripts/config.py`.

<details>
<summary><b>GLM</b></summary>

| Parameter | MSE | FPLLoss |
| :--- | ---: | ---: |
| `alpha` | 0.0032 | 0.0057 |
| `family` | `normal` | `poisson` |
| `l1_ratio` | 0.9235 | 0.3534 |
| `link` | `log` | `log` |

</details>

<details>
<summary><b>XGBoost</b></summary>

| Parameter | MSE | FPLLoss |
| :--- | ---: | ---: |
| `colsample_bytree` | 0.6658 | 0.6071 |
| `gamma` | 0.6578 | 0.0526 |
| `learning_rate` | 0.038 | 0.018 |
| `max_depth` | 3 | 6 |
| `min_child_weight` | 9 | 9 |
| `n_estimators` | 159 | 192 |
| `reg_alpha` | 1.94 | 1.735 |
| `reg_lambda` | 1.915 | 2.416 |
| `subsample` | 0.92 | 0.7996 |

</details>

<details>
<summary><b>Neural networks</b></summary>

| Parameter | MLP | Conv1D |
| :--- | ---: | ---: |
| hidden layers | `[128, 64]` | – |
| conv channels | – | `[128, 64]` |
| kernel size | – | 3 |
| window length (gameweeks) | – | 8 |
| dropout | 0.2 | 0.2 |
| learning rate | 0.001 | 0.001 |
| weight decay | 0.005 | 0.005 |
| batch size | 128 | 128 |
| epochs trained (MSE) | 13 | 15 |
| epochs trained (FPLLoss) | 17 | 15 |

</details>

## Files

| File | Content |
| :--- | :--- |
| `metrics_test.csv`, `metrics_valid.csv` | all metrics on the test and validation sets |
| `table_tiers.csv` | MSE and MAE per price tier (2 decimals) |
| `tier_counts.csv` | number of rows per price tier and split |
| `meta_weights.csv` | coefficients and intercepts of the meta-models |
| `preds_valid.parquet`, `preds_test.parquet` | predictions of all models |
| `preds_predict.parquet` | predictions for unplayed fixtures (once betting odds are available) |
| `../trained_models/` | fitted base models and preprocessing |

Reproduce with `PYTHONPATH=src python -m fpl_project.models.run_experiments` (seed `77`).
This file is generated by `scripts/model_utils/pipeline/results_readme.py`.
