"""Rendering of results/README.md."""

from dataclasses import dataclass
import re

import pandas as pd

from fpl_project.scripts import config
from fpl_project.scripts.model_utils.pipeline.containers import MergedPredictions, PreparedData

BASE_MODELS = {"glm": "GLM", "xgboost": "XGBoost", "mlp": "MLP", "conv1d": "Conv1D"}
META_MODELS = {
    "meta_nnls": "Meta: NNLS",
    "meta_context": "Meta: context Ridge",
    "mean_ensemble": "Meta: mean ensemble",
    "best_single": "Meta: best single model",
}
MODEL_LABELS = {**BASE_MODELS, **META_MODELS}
VARIANT_LABELS = {"mse": "MSE", "fplloss": "FPLLoss"}
NN_PARAM_LABELS = {
    "hidden_sizes": "hidden layers",
    "channels": "conv channels",
    "kernel_size": "kernel size",
    "seq_len": "window length (gameweeks)",
    "dropout": "dropout",
    "lr": "learning rate",
    "weight_decay": "weight decay",
    "batch_size": "batch size",
}


@dataclass(frozen=True)
class ReadmeContext:
    """Dataset facts shown in the README."""

    n_train: int
    n_valid: int
    n_test: int
    test_gw_min: int
    test_gw_max: int
    n_train_sub: int
    n_es_val: int
    n_warmup_dropped: int
    n_dropped_valid: int
    n_dropped_test: int
    p_low: float
    p_high: float


def build_readme_context(data: PreparedData, merged: MergedPredictions) -> ReadmeContext:
    return ReadmeContext(
        n_train=len(data.raw.X_train),
        n_valid=len(merged.valid_df),
        n_test=len(merged.test_df),
        test_gw_min=int(merged.test_df["gw"].min()),
        test_gw_max=int(merged.test_df["gw"].max()),
        n_train_sub=data.mlp_bundle.n_train_sub,
        n_es_val=data.mlp_bundle.n_es_val,
        n_warmup_dropped=data.conv_bundle.n_warmup_dropped,
        n_dropped_valid=len(data.raw.X_valid) - len(merged.valid_df),
        n_dropped_test=len(data.raw.X_test) - len(merged.test_df),
        p_low=data.weights.p_low,
        p_high=data.weights.p_high,
    )


def _md_table(header: list[str], rows: list[list[str]], align: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt_param(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"`{value}`"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.4g}"


def _model_key(model: str) -> str:
    return "best_single" if model.startswith("best_single") else model


def _with_keys(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.assign(key=metrics["model"].map(_model_key))


def _best_single_base(metrics: pd.DataFrame, variant: str) -> str:
    """Base model chosen as 'best single' on the validation set, e.g. 'best_single(xgboost)' -> 'XGBoost'."""
    mask = (metrics["variant"] == variant) & metrics["model"].str.startswith("best_single")
    name = metrics.loc[mask, "model"].iloc[0]
    return BASE_MODELS[re.search(r"\((\w+)\)", name).group(1)]


def _row_label(row: pd.Series) -> str:
    """E.g. 'context Ridge meta-model (MSE training)' or 'Conv1D (FPLLoss training)'."""
    label = MODEL_LABELS[row["key"]]
    if label.startswith("Meta: "):
        label = label.removeprefix("Meta: ") + " meta-model"
    return f"{label} ({VARIANT_LABELS[row['variant']]} training)"


def _price(value: float) -> str:
    return f"£{value:g}m"


def _key_findings(metrics: pd.DataFrame) -> str:
    m = _with_keys(metrics)
    best = m.loc[m["mse"].idxmin()]
    base = m[m["level"] == "base"]
    best_base = base.loc[base["mse"].idxmin()]
    gain = 100.0 * (best_base["mse"] - best["mse"]) / best_base["mse"]

    findings = [
        f"**Best model:** {_row_label(best)} with test MSE **{best['mse']:.3f}**, "
        f"{gain:.1f}% lower than the best base model, {_row_label(best_base)}, with {best_base['mse']:.3f}."
    ]

    variants_won = [
        VARIANT_LABELS[v]
        for v in VARIANT_LABELS
        if m.loc[(m["variant"] == v) & (m["key"] == "meta_context"), "mse"].iloc[0]
        < m.loc[(m["variant"] == v) & (m["level"] == "base"), "mse"].min()
    ]
    if variants_won:
        findings.append(
            "**Meta-models help:** the context Ridge beats every base model on MSE "
            f"({' and '.join(variants_won)} training)."
        )

    premium = base.pivot(index="key", columns="variant", values="mse@premium")
    n_better = int((premium["fplloss"] < premium["mse"]).sum())
    findings.append(
        f"**FPLLoss on premium players:** lower premium-tier MSE than plain MSE training for "
        f"{n_better} of {len(premium)} base models."
    )
    return "\n".join(f"- {f}" for f in findings)


def _results_table(metrics: pd.DataFrame, variant: str) -> str:
    """Test metrics of one training variant; best value per column in bold."""
    m = _with_keys(metrics)
    m = m[m["variant"] == variant].set_index("key")
    columns = [  # (column, header, best value selector)
        ("mse", "MSE", min),
        ("mae", "MAE", min),
        ("r2", "R²", max),
        ("mse@budget", "MSE budget", min),
        ("mse@mid", "MSE mid", min),
        ("mse@premium", "MSE premium", min),
    ]
    rows = []
    for key, label in MODEL_LABELS.items():
        if key == "best_single":
            label = f"{label} ({_best_single_base(metrics, variant)})"
        row = [label]
        for col, _, pick_best in columns:
            value = m.loc[key, col]
            cell = f"{value:.3f}"
            row.append(f"**{cell}**" if value == pick_best(m[col]) else cell)
        rows.append(row)
    header = ["Model"] + [h for _, h, _ in columns]
    return _md_table(header, rows, [":---"] + ["---:"] * len(columns))


def _data_table(ctx: ReadmeContext, tier_counts: pd.DataFrame) -> str:
    seasons = {
        "train": "2022/23 – 2024/25",
        "valid": "2025/26",
        "test": f"2026/27 (GW {ctx.test_gw_min}–{ctx.test_gw_max})",
    }
    rows = [
        [f"`{r['split']}`", seasons[r["split"]]] + [f"{int(r[c]):,}" for c in ("total", "budget", "mid", "premium")]
        for _, r in tier_counts.iterrows()
    ]
    header = ["Split", "Seasons", "Rows", "Budget", "Mid", "Premium"]
    return _md_table(header, rows, [":---", ":---"] + ["---:"] * 4)


def _tabular_params_table(best_params: dict[str, dict], model: str) -> str:
    params = sorted({p for v in VARIANT_LABELS for p in best_params.get(f"{model}_{v}", {})})
    rows = [
        [f"`{p}`"] + [_fmt_param(best_params.get(f"{model}_{v}", {}).get(p, "–")) for v in VARIANT_LABELS]
        for p in params
    ]
    return _md_table(["Parameter"] + list(VARIANT_LABELS.values()), rows, [":---", "---:", "---:"])


def _nn_params_table(nn_params: dict[str, dict]) -> str:
    archs = {"mlp": "MLP", "conv1d": "Conv1D"}
    rows = []
    for key, label in NN_PARAM_LABELS.items():
        values = [nn_params.get(f"{a}_mse", {}).get(key) for a in archs]
        if any(v is not None for v in values):
            rows.append([label] + ["–" if v is None else _fmt_param(v) for v in values])
    for variant, vlabel in VARIANT_LABELS.items():
        rows.append(
            [f"epochs trained ({vlabel})"]
            + [_fmt_param(nn_params.get(f"{a}_{variant}", {}).get("epochs_ran", "–")) for a in archs]
        )
    return _md_table(["Parameter"] + list(archs.values()), rows, [":---", "---:", "---:"])


def render_results_readme(
    ctx: ReadmeContext,
    metrics_test: pd.DataFrame,
    tier_counts: pd.DataFrame,
    best_params: dict[str, dict],
    nn_params: dict[str, dict],
) -> str:
    """Render results/README.md from the result tables."""
    return f"""# Experiment results

Test-set performance (season 2026/27, GW {ctx.test_gw_min}–{ctx.test_gw_max}, {ctx.n_test:,} player-matches) of four
base models and four meta-models. Every model is trained twice: with plain **MSE** and with **FPLLoss**,
a weighted MSE that gives more weight to expensive players and players with more expected minutes.

## Key findings

{_key_findings(metrics_test)}

## Test results

Lower is better for MSE and MAE, higher for R². **Bold** marks the best value in each column.
Price tiers: **budget** < {_price(ctx.p_low)} ≤ **mid** < {_price(ctx.p_high)} ≤ **premium**.

### Trained with MSE

{_results_table(metrics_test, "mse")}

### Trained with FPLLoss

{_results_table(metrics_test, "fplloss")}

MAE per price tier and validation-set metrics are in `metrics_test.csv` and `metrics_valid.csv`.

## Models

- **Base models:** GLM (`glum`), XGBoost and MLP on lagged tabular features; Conv1D on the last
  {config.NN_SEQ_LEN} gameweeks.
- **Meta-models**, fitted on validation-set predictions of the four base models:
  - *NNLS* – non-negative linear combination,
  - *context Ridge* – Ridge on predictions, position, price tier, minutes and their interactions,
  - *mean ensemble* – simple average,
  - *best single model* – the base model with the lowest validation error.

## Data

{_data_table(ctx, tier_counts)}

- Price tiers are the {config.PRICE_QUANTILE_LOW:.0%} and {config.PRICE_QUANTILE_HIGH:.1%} quantiles of player price in the training set.
- Neural networks train on {ctx.n_train_sub:,} rows; the last gameweeks of 2024/25 (GW ≥ {config.NN_ES_MIN_GW},
  {ctx.n_es_val:,} rows) are held out for early stopping.
- Conv1D skips the first {ctx.n_warmup_dropped:,} training rows, which lack a full {config.NN_SEQ_LEN}-gameweek history.
- Rows dropped when joining predictions of all models: {ctx.n_dropped_valid} (valid), {ctx.n_dropped_test} (test).

## FPLLoss

Each observation is weighted by price tier and expected minutes:

```
w = w_tier * (ln(1 + m) + eps_min),   m = clip(minutes_ema_3 / 90, 0, 1)
```

| Constant | Value |
| :--- | ---: |
| `w_tier` budget | 1.0 |
| `w_tier` mid (`W_MID`) | {config.W_MID} |
| `w_tier` premium (`W_PREMIUM`) | {config.W_PREMIUM} |
| `eps_min` (`EPS_MIN`) | {config.EPS_MIN} |

Weights are not normalised, so FPLLoss values are on a different scale than MSE.

## Hyperparameters

Tuned with `RandomizedSearchCV` ({config.N_ITER_SEARCH} candidates, {config.CV_SPLITS}-fold `TimeSeriesSplit` with folds split
between gameweeks). Search spaces: `GLM_PARAM_GRID` and `XGB_PARAM_GRID` in `scripts/config.py`.

<details>
<summary><b>GLM</b></summary>

{_tabular_params_table(best_params, "glm")}

</details>

<details>
<summary><b>XGBoost</b></summary>

{_tabular_params_table(best_params, "xgboost")}

</details>

<details>
<summary><b>Neural networks</b></summary>

{_nn_params_table(nn_params)}

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

Reproduce with `PYTHONPATH=src python -m fpl_project.models.run_experiments` (seed `{config.SEED}`).
This file is generated by `scripts/model_utils/pipeline/results_readme.py`.
"""
