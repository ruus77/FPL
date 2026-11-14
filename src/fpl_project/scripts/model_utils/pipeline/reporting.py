"""Stage 4: save result tables, render results/README.md, and log runs to MLflow."""

from pathlib import Path

from glum._glm import get_family, get_link
import pandas as pd

from fpl_project.scripts import config
from fpl_project.scripts.model_utils.evaluation import compute_tier_counts_table
from fpl_project.scripts.model_utils.mlflow_tracker import MlflowTracker
from fpl_project.scripts.model_utils.pipeline.containers import (
    MergedPredictions,
    MetaEvaluationResult,
    NNTrainingResult,
    PreparedData,
)


def build_table_tiers(df_metrics_test: pd.DataFrame) -> pd.DataFrame:
    """Build publication-ready table_tiers.csv rounded to 2 decimal places."""
    models_order = ["glm", "xgboost", "mlp", "conv1d", "meta_nnls", "meta_context", "mean_ensemble"]
    rows = []
    for variant, variant_label in [("mse", "MSE"), ("fplloss", "FPLLoss")]:
        for m_name in models_order:
            r = df_metrics_test[
                (df_metrics_test["model"] == m_name) & (df_metrics_test["variant"] == variant)
            ].iloc[0]
            rows.append({
                "Model": m_name,
                "Wariant": variant_label,
                "MSE": round(float(r["mse"]), 2),
                "MSE budget": round(float(r["mse@budget"]), 2),
                "MSE mid": round(float(r["mse@mid"]), 2),
                "MSE premium": round(float(r["mse@premium"]), 2),
                "MAE": round(float(r["mae"]), 2),
                "MAE budget": round(float(r["mae@budget"]), 2),
                "MAE mid": round(float(r["mae@mid"]), 2),
                "MAE premium": round(float(r["mae@premium"]), 2),
            })
    return pd.DataFrame(rows)


def log_results_to_mlflow(
    eval_result: MetaEvaluationResult,
    best_params_log: dict[str, dict],
    nn_result: NNTrainingResult,
    data: PreparedData,
    artifact_paths: list[Path],
) -> None:
    """Log experiment runs, parameters, valid/test metrics, and result artifacts to MLflow."""
    import mlflow

    tracker = MlflowTracker()
    tracker.set_tracking()

    metric_cols = [
        "mse", "mae", "r2", "adj_r2", "fplloss",
        "mse@budget", "mse@mid", "mse@premium",
        "mae@budget", "mae@mid", "mae@premium",
    ]

    df_metrics_test = eval_result.metrics_test
    df_metrics_valid = eval_result.metrics_valid

    for idx in range(len(df_metrics_test)):
        row_t = df_metrics_test.iloc[idx]
        row_v = df_metrics_valid.iloc[idx]
        m_name = str(row_t["model"])
        variant = str(row_t["variant"])
        level = str(row_t["level"])
        run_name = f"{m_name}_{variant}_{level}"
        model_key = f"{m_name}_{variant}"

        with tracker.start_run(run_name=run_name):
            mlflow.set_tag("model", m_name)
            mlflow.set_tag("variant", variant)
            mlflow.set_tag("level", level)

            params = {
                "model": m_name,
                "variant": variant,
                "level": level,
                "p_features": int(row_t["p"]),
                "seed": config.SEED,
                "p_low": data.weights.p_low,
                "p_high": data.weights.p_high,
                "eps_min": config.EPS_MIN,
                "w_mid": config.W_MID,
                "w_premium": config.W_PREMIUM,
            }
            if model_key in best_params_log:
                for k, v in best_params_log[model_key].items():
                    params[f"hp_{k}"] = v
            if model_key in nn_result.nn_params:
                for k, v in nn_result.nn_params[model_key].items():
                    params[f"nn_{k}"] = v
            tracker.log_params(params)

            if model_key in nn_result.histories:
                hist = nn_result.histories[model_key]
                for ep in range(len(hist["train_loss"])):
                    tracker.log_metrics(
                        {
                            "train_loss": float(hist["train_loss"][ep]),
                            "es_val_loss": float(hist["test_loss"][ep]),
                            "train_mse": float(hist["train_mse"][ep]),
                            "es_val_mse": float(hist["test_mse"][ep]),
                            "train_mae": float(hist["train_mae"][ep]),
                            "es_val_mae": float(hist["test_mae"][ep]),
                        },
                        step=ep,
                    )

            metrics_to_log = {}
            for mc in metric_cols:
                clean_mc = mc.replace("@", "_")
                metrics_to_log[f"valid_{clean_mc}"] = float(row_v[mc])
                metrics_to_log[f"test_{clean_mc}"] = float(row_t[mc])
            if model_key in nn_result.nn_params:
                metrics_to_log["n_epochs_trained"] = float(nn_result.nn_params[model_key]["epochs_ran"])
            tracker.log_metrics(metrics_to_log)

            if idx == len(df_metrics_test) - 1:
                for ap in artifact_paths:
                    if ap.exists():
                        mlflow.log_artifact(str(ap))

    print(f"[MLflow] Zalogowano {len(df_metrics_test)} przebiegów eksperymentalnych do MLflow.")


def _render_results_readme(
    data: PreparedData,
    merged: MergedPredictions,
    eval_result: MetaEvaluationResult,
    df_tier_counts: pd.DataFrame,
    best_params_log: dict[str, dict],
) -> str:
    tc_train = df_tier_counts[df_tier_counts["split"] == "train"].iloc[0]
    tc_valid = df_tier_counts[df_tier_counts["split"] == "valid"].iloc[0]
    tc_test = df_tier_counts[df_tier_counts["split"] == "test"].iloc[0]
    n_test_gws = int(merged.test_df["gw"].nunique())

    nb_default = get_family("negative.binomial")
    tw_default = get_family("tweedie")
    auto_links = {
        fam_name: type(get_link("auto", get_family(fam_name))).__name__
        for fam_name in ["poisson", "negative.binomial", "tweedie", "normal"]
    }
    glm_params_mse = best_params_log.get("glm_mse", {})
    glm_params_fpl = best_params_log.get("glm_fplloss", {})

    metrics_md_lines = [
        "| Model | Wariant | Poziom | $p$ | MSE | MAE | $R^2$ | Adj $R^2$ | FPLLoss | MSE budget | MSE mid | MSE premium | MAE budget | MAE mid | MAE premium |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in eval_result.metrics_test.iterrows():
        metrics_md_lines.append(
            f"| `{r['model']}` | `{r['variant']}` | `{r['level']}` | {int(r['p'])} | "
            f"{r['mse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} | {r['adj_r2']:.4f} | {r['fplloss']:.4f} | "
            f"{r['mse@budget']:.4f} | {r['mse@mid']:.4f} | {r['mse@premium']:.4f} | "
            f"{r['mae@budget']:.4f} | {r['mae@mid']:.4f} | {r['mae@premium']:.4f} |"
        )
    metrics_md_table = "\n".join(metrics_md_lines)
    p_low, p_high = data.weights.p_low, data.weights.p_high

    return f"""# Eksperymenty Predykcji Punktów FPL — Modele Bazowe, FPLLoss i Meta-Modele

## 1. Uruchomienie eksperymentu
Cały potok eksperymentalny uruchamiany jest jednym poleceniem z katalogu głównego repozytorium:

```bash
PYTHONPATH=src python -m fpl_project.models.run_experiments
```

Ziarno losowe zostało ustalone na `SEED = {config.SEED}` (`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`).

## 2. Podział danych i liczebności grup cenowych (`tier_counts.csv`)
- **Train (`2022/23`, `2023/24`, `2024/25`)**: `{len(data.raw.X_train)}` obserwacji (posortowanych globalnie po `(season, gw, code)`).
  - Wewnętrzny chronologiczny zbiór wczesnego zatrzymania dla sieci (`MLP`, `Conv1D`): ostatnie 8 kolejek sezonu `2024/25` (`gw >= {config.NN_ES_MIN_GW}`, `{data.mlp_bundle.n_es_val}` obserwacji); trening właściwy sieci na wcześniejszych kolejkach (`{data.mlp_bundle.n_train_sub}` obserwacji).
- **Valid (`2025/26`)**: `{len(merged.valid_df)}` obserwacji.
- **Test (`2026/27`, `is_played == True`)**: `{len(merged.test_df)}` obserwacji (`{n_test_gws}` kolejek).
- **Złączenie wewnętrzne (`inner join` po `(code, season, gw)`)**: `0` odrzuconych wierszy na `valid` oraz `0` na `test` (w `train_sub` dla `Conv1D` pominięto `{data.conv_bundle.n_warmup_dropped}` początkowych wierszy rozgrzewki okna `seq_len = {config.NN_SEQ_LEN}`).

Progi cenowe na zbiorze `train`: $p_{{\\text{{low}}}} = {p_low:.4f}$ mln £ (kwantyl `{config.PRICE_QUANTILE_LOW:.2f}`), $p_{{\\text{{high}}}} = {p_high:.4f}$ mln £ (kwantyl `{config.PRICE_QUANTILE_HIGH:.3f}`).

| Zbiór (`split`) | `budget` ($< {p_low:.2f}$) | `mid` ($[{p_low:.2f}, {p_high:.3f})$) | `premium` ($\\ge {p_high:.3f}$) | Razem (`total`) |
| :--- | ---: | ---: | ---: | ---: |
| `train` | {int(tc_train['budget'])} | {int(tc_train['mid'])} | {int(tc_train['premium'])} | {int(tc_train['total'])} |
| `valid` | {int(tc_valid['budget'])} | {int(tc_valid['mid'])} | {int(tc_valid['premium'])} | {int(tc_valid['total'])} |
| `test` | {int(tc_test['budget'])} | {int(tc_test['mid'])} | {int(tc_test['premium'])} | {int(tc_test['total'])} |

## 3. Funkcja straty `FPLLoss` i stałe konfiguracyjne
Stałe funkcji straty zdefiniowane w `src/fpl_project/scripts/config.py`:
- `EPS_MIN` ($\\varepsilon_{{\\min}}$) = `{config.EPS_MIN}`
- `W_MID` ($w_{{\\text{{mid}}}}$) = `{config.W_MID}`
- `W_PREMIUM` ($w_{{\\text{{premium}}}}$) = `{config.W_PREMIUM}`

Wzór wagi obserwacji (gdzie $m_i = \\text{{clip}}(\\text{{minutes\\_ema\\_3}} / 90, 0, 1)$):
$$w_i = w_{{\\text{{cena}}, i}} \\cdot (\\ln(1 + m_i) + \\varepsilon_{{\\min}}), \\quad w_{{\\text{{cena}}, i}} \\in \\{{1.0, {config.W_MID}, {config.W_PREMIUM}\\}}$$

Ponieważ wagi $w_i$ nie są dzielone przez stałą normalizującą, wartości `FPLLoss` w tabelach są wyrażone w innej skali niż `MSE`.

## 4. Konfiguracja GLM (`glum`) i realizacja wariantu (B)
- **Przestrzeń przeszukiwania GLM (`GLM_PARAM_GRID`)**:
  - `alpha`: `stats.loguniform(1e-5, 0.01)`
  - `l1_ratio`: `stats.uniform(loc=0.0, scale=1.0)`
  - `family`: `["poisson", "negative.binomial", "tweedie", "normal"]`
  - `link`: `["auto", "log"]`
- **Domyślne parametry w bibliotece `glum` (`glum._glm.get_family` oraz `glum._glm.get_link`)**:
  - `family="negative.binomial"` (bez podanego parametru) przyjmuje $\\theta = {float(nb_default.theta):.1f}$ (`NegativeBinomialDistribution(1.0)`),
  - `family="tweedie"` (bez podanego parametru) przyjmuje moc $p = {float(tw_default.power):.1f}$ (`TweedieDistribution(1.5)`),
  - `link="auto"` zwraca dla poszczególnych rodzin:
    - `"poisson"` $\\rightarrow$ `{auto_links['poisson']}` (`log`),
    - `"negative.binomial"` $\\rightarrow$ `{auto_links['negative.binomial']}` (`log`),
    - `"tweedie"` $\\rightarrow$ `{auto_links['tweedie']}` (`log`),
    - `"normal"` $\\rightarrow$ `{auto_links['normal']}` (`identity`).
- **Wybrane hiperparametry GLM w `RandomizedSearchCV`**:
  - Wariant `mse`: `{glm_params_mse}`
  - Wariant `fplloss`: `{glm_params_fpl}`
- **Wariant (B) dla GLM i XGBoost**:
  - **XGBoost (`objective="reg:squarederror"`)**: przekazanie `sample_weight = w_i` do `fit()` minimalizuje ważony błąd średniokwadratowy $\\sum_i w_i (y_i - \\hat{{y}}_i)^2$.
  - **GLM (`glum.GeneralizedLinearRegressor`)**: przekazanie `sample_weight = w_i` maksymalizuje ważoną log-wiarygodność wybranej rodziny rozkładu.
  - W `TimeSeriesSplit` CV (granice foldów pomiędzy kolejkami `(season, gw)`) wagi `w_i` przekazywane są do `estimator.fit()` oraz do ważonego kryterium oceny foldu (`neg_weighted_mse`).

## 5. Wyniki na zbiorze testowym (`metrics_test.csv`)
{metrics_md_table}

## 6. Pliki wynikowe w katalogu `results/`
- `metrics_test.csv`, `metrics_valid.csv` — metryki na zbiorze testowym i walidacyjnym,
- `tier_counts.csv` — liczebności grup `budget`, `mid`, `premium` dla `train`, `valid`, `test`,
- `table_tiers.csv` — tabela MSE i MAE w podziale na grupy cenowe (zaokrąglona do 2 miejsc po przecinku),
- `meta_weights.csv` — wagi i wyrazy wolne wyuczonych meta-modeli,
- `preds_valid.parquet`, `preds_test.parquet` — predykcje wszystkich modeli na zbiorze walidacyjnym i testowym,
- `experiment_run.log` — pełny zapis przebiegu eksperymentu.
"""


def save_results(
    data: PreparedData,
    merged: MergedPredictions,
    eval_result: MetaEvaluationResult,
    best_params_log: dict[str, dict],
    nn_result: NNTrainingResult,
) -> None:
    """Save all result CSV/Parquet files, update results/README.md, and log to MLflow."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_tier_counts = compute_tier_counts_table(
        splits_dict={"train": data.raw.X_train, "valid": merged.valid_df, "test": merged.test_df},
        p_low=data.weights.p_low,
        p_high=data.weights.p_high,
    )
    df_table_tiers = build_table_tiers(eval_result.metrics_test)

    eval_result.metrics_test.to_csv(config.METRICS_TEST_PATH, index=False)
    eval_result.metrics_valid.to_csv(config.METRICS_VALID_PATH, index=False)
    merged.valid_df.to_parquet(config.PREDS_VALID_PATH, index=False)
    merged.test_df.to_parquet(config.PREDS_TEST_PATH, index=False)
    eval_result.meta_weights.to_csv(config.META_WEIGHTS_PATH, index=False)
    df_tier_counts.to_csv(config.TIER_COUNTS_PATH, index=False)
    df_table_tiers.to_csv(config.TABLE_TIERS_PATH, index=False, float_format="%.2f")

    readme_content = _render_results_readme(
        data=data,
        merged=merged,
        eval_result=eval_result,
        df_tier_counts=df_tier_counts,
        best_params_log=best_params_log,
    )
    config.RESULTS_README_PATH.write_text(readme_content, encoding="utf-8")

    log_results_to_mlflow(
        eval_result=eval_result,
        best_params_log=best_params_log,
        nn_result=nn_result,
        data=data,
        artifact_paths=[
            config.METRICS_TEST_PATH,
            config.METRICS_VALID_PATH,
            config.META_WEIGHTS_PATH,
            config.TIER_COUNTS_PATH,
            config.TABLE_TIERS_PATH,
            config.RESULTS_README_PATH,
        ],
    )

    print("\n" + "=" * 115)
    print(f"LICZEBNOŚCI GRUP CENOWYCH ({config.TIER_COUNTS_PATH.name}):")
    print(df_tier_counts.to_string(index=False))
    print("\n" + "=" * 115)
    print(f"TABELA GOTOWA DO PRACY ({config.TABLE_TIERS_PATH.name}):")
    print(df_table_tiers.to_string(index=False))
    print("\n" + "=" * 115)
    print(f"TABELA WYNIKÓW NA ZBIORZE TESTOWYM ({config.METRICS_TEST_PATH.name}):")
    print("=" * 115)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    print(eval_result.metrics_test.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 115)
