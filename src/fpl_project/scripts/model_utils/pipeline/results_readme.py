"""Rendering of results/README.md (in Polish: it documents the thesis results)."""

from dataclasses import dataclass
import re

import pandas as pd

from fpl_project.scripts import config
from fpl_project.scripts.model_utils.pipeline.containers import MergedPredictions, PreparedData

MODEL_LABELS = {
    "glm": "GLM",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "conv1d": "Conv1D",
    "meta_nnls": "NNLS (meta)",
    "meta_context": "Ridge z kontekstem (meta)",
    "mean_ensemble": "Średnia (meta)",
    "best_single": "Najlepszy pojedynczy",
}
VARIANT_LABELS = {"mse": "MSE", "fplloss": "FPLLoss"}
NN_PARAM_LABELS = {
    "hidden_sizes": "warstwy ukryte",
    "channels": "kanały konwolucji",
    "kernel_size": "rozmiar jądra",
    "seq_len": "długość okna",
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


def _n(value: int) -> str:
    """Integer with a space as thousands separator (Polish convention)."""
    return f"{value:,}".replace(",", " ")


def _fmt_param(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"`{value}`"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.4g}"


def _model_key(model: str) -> str:
    return "best_single" if model.startswith("best_single") else model


def _summary_table(metrics: pd.DataFrame) -> str:
    """Main metrics, MSE and FPLLoss variants side by side; best value per column in bold."""
    columns = [  # (metric, label, best value selector)
        ("mse", "MSE", min),
        ("mae", "MAE", min),
        ("r2", "R²", max),
    ]
    by_model = metrics.assign(key=metrics["model"].map(_model_key)).set_index(["variant", "key"])

    rows = [[label] for label in MODEL_LABELS.values()]
    for metric, _, pick_best in columns:
        for variant in VARIANT_LABELS:
            values = by_model.loc[variant, metric]
            best = pick_best(values)
            for row, key in zip(rows, MODEL_LABELS):
                cell = f"{values[key]:.3f}"
                row.append(f"**{cell}**" if values[key] == best else cell)

    header = ["Model"] + [f"{label} ({VARIANT_LABELS[v]})" for _, label, _ in columns for v in VARIANT_LABELS]
    return _md_table(header, rows, [":---"] + ["---:"] * (len(header) - 1))


def _best_single_note(metrics: pd.DataFrame) -> str:
    picks = []
    for variant, label in VARIANT_LABELS.items():
        model = metrics.loc[(metrics["variant"] == variant) & metrics["model"].str.startswith("best_single"), "model"].iloc[0]
        base_model = re.search(r"\((\w+)\)", model).group(1)
        picks.append(f"{label}: {MODEL_LABELS[base_model]}")
    return "„Najlepszy pojedynczy” to model bazowy o najniższym błędzie na zbiorze walidacyjnym (" + ", ".join(picks) + ")."


def _tier_table(metrics: pd.DataFrame) -> str:
    rows = []
    for _, r in metrics.iterrows():
        rows.append(
            [MODEL_LABELS[_model_key(r["model"])], VARIANT_LABELS[r["variant"]]]
            + [f"{r[f'{m}@{t}']:.2f}" for m in ("mse", "mae") for t in ("budget", "mid", "premium")]
        )
    header = ["Model", "Wariant", "MSE budget", "MSE mid", "MSE premium", "MAE budget", "MAE mid", "MAE premium"]
    return _md_table(header, rows, [":---", ":---"] + ["---:"] * 6)


def _data_table(ctx: ReadmeContext, tier_counts: pd.DataFrame) -> str:
    seasons = {
        "train": "2022/23 – 2024/25",
        "valid": "2025/26",
        "test": f"2026/27 (kolejki {ctx.test_gw_min}–{ctx.test_gw_max})",
    }
    rows = []
    for _, r in tier_counts.iterrows():
        rows.append([f"`{r['split']}`", seasons[r["split"]]] + [_n(int(r[c])) for c in ("total", "budget", "mid", "premium")])
    header = ["Zbiór", "Sezony", "Obserwacje", "budget", "mid", "premium"]
    return _md_table(header, rows, [":---", ":---"] + ["---:"] * 4)


def _tabular_params_table(best_params: dict[str, dict], model: str) -> str:
    params = sorted({p for v in VARIANT_LABELS for p in best_params.get(f"{model}_{v}", {})})
    rows = [[f"`{p}`"] + [_fmt_param(best_params.get(f"{model}_{v}", {}).get(p, "–")) for v in VARIANT_LABELS] for p in params]
    return _md_table(["Parametr"] + list(VARIANT_LABELS.values()), rows, [":---", ":---", ":---"])


def _nn_params_table(nn_params: dict[str, dict]) -> str:
    archs = {"mlp": "MLP", "conv1d": "Conv1D"}
    rows = []
    for key, label in NN_PARAM_LABELS.items():
        values = [nn_params.get(f"{a}_mse", {}).get(key) for a in archs]
        if any(v is not None for v in values):
            rows.append([label] + ["–" if v is None else _fmt_param(v) for v in values])
    for variant, vlabel in VARIANT_LABELS.items():
        rows.append([f"liczba epok treningu ({vlabel})"] + [_fmt_param(nn_params.get(f"{a}_{variant}", {}).get("epochs_ran", "–")) for a in archs])
    return _md_table(["Parametr"] + list(archs.values()), rows, [":---", "---:", "---:"])


def render_results_readme(
    ctx: ReadmeContext,
    metrics_test: pd.DataFrame,
    tier_counts: pd.DataFrame,
    best_params: dict[str, dict],
    nn_params: dict[str, dict],
) -> str:
    """Render results/README.md from the result tables."""
    return f"""# Wyniki eksperymentu

Predykcja punktów FPL: cztery modele bazowe (GLM, XGBoost, MLP, Conv1D), każdy uczony z funkcją straty
**MSE** oraz **FPLLoss**, i meta-modele dopasowane na predykcjach ze zbioru walidacyjnego.
Wszystkie metryki dotyczą zbioru testowego (sezon 2026/27).

> Plik generowany automatycznie przez `run_experiments.py`. Zmiany wprowadzaj w
> `src/fpl_project/scripts/model_utils/pipeline/results_readme.py`.

```bash
PYTHONPATH=src python -m fpl_project.models.run_experiments
```

## Wyniki na zbiorze testowym

{_summary_table(metrics_test)}

Pogrubienie oznacza najlepszy wynik w kolumnie. {_best_single_note(metrics_test)}

## Błąd w grupach cenowych

Grupy według ceny zawodnika (`value_ema_3`): **budget** < {ctx.p_low:.2f} mln £ ≤ **mid** < {ctx.p_high:.3f} mln £ ≤ **premium**.

{_tier_table(metrics_test)}

## Dane

{_data_table(ctx, tier_counts)}

- Progi cenowe to kwantyle {config.PRICE_QUANTILE_LOW:.2f} i {config.PRICE_QUANTILE_HIGH:.3f} ceny na zbiorze treningowym.
- Sieci (MLP, Conv1D) uczą się na {_n(ctx.n_train_sub)} obserwacjach; ostatnie kolejki sezonu 2024/25
  (`gw ≥ {config.NN_ES_MIN_GW}`, {_n(ctx.n_es_val)} obserwacji) służą do wczesnego zatrzymania.
- Conv1D pomija {_n(ctx.n_warmup_dropped)} początkowych obserwacji, dla których nie ma pełnego okna {config.NN_SEQ_LEN} kolejek.
- Po złączeniu predykcji wszystkich modeli odrzucono {ctx.n_dropped_valid} wierszy na `valid` i {ctx.n_dropped_test} na `test`.

## Funkcja straty FPLLoss

$$w_i = w_{{\\text{{cena}},i}} \\cdot \\left(\\ln(1 + m_i) + \\varepsilon_{{\\min}}\\right), \\qquad m_i = \\text{{clip}}\\left(\\tfrac{{\\text{{minutes\\_ema\\_3}}}}{{90}}, 0, 1\\right)$$

| Stała | Wartość |
| :--- | ---: |
| waga `budget` | 1.0 |
| waga `mid` (`W_MID`) | {config.W_MID} |
| waga `premium` (`W_PREMIUM`) | {config.W_PREMIUM} |
| $\\varepsilon_{{\\min}}$ (`EPS_MIN`) | {config.EPS_MIN} |

Wagi nie są normalizowane, dlatego wartości FPLLoss mają inną skalę niż MSE. Modele tabelaryczne
otrzymują wagi jako `sample_weight` w treningu i w ocenie foldów walidacji krzyżowej.

## Hiperparametry

Strojenie: `RandomizedSearchCV` ({config.N_ITER_SEARCH} losowań, {config.CV_SPLITS} foldów `TimeSeriesSplit` z granicami
między kolejkami). Przestrzenie przeszukiwania: `GLM_PARAM_GRID` i `XGB_PARAM_GRID` w `scripts/config.py`.

**GLM**

{_tabular_params_table(best_params, "glm")}

**XGBoost**

{_tabular_params_table(best_params, "xgboost")}

**Sieci neuronowe**

{_nn_params_table(nn_params)}

## Pliki

| Plik | Zawartość |
| :--- | :--- |
| `metrics_test.csv`, `metrics_valid.csv` | wszystkie metryki na zbiorze testowym i walidacyjnym |
| `table_tiers.csv` | MSE i MAE w grupach cenowych (2 miejsca po przecinku) |
| `tier_counts.csv` | liczebności grup cenowych w zbiorach |
| `meta_weights.csv` | wagi i wyrazy wolne meta-modeli |
| `preds_valid.parquet`, `preds_test.parquet` | predykcje wszystkich modeli |
| `preds_predict.parquet` | predykcje na nierozegrane mecze (gdy dostępne są kursy) |
| `../trained_models/` | zapisane modele bazowe i preprocessing |

Ziarno losowe: `SEED = {config.SEED}`.
"""
