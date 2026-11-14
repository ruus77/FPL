"""Stage 4: save result tables, render results/README.md, and log runs to MLflow."""

from pathlib import Path

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
from fpl_project.scripts.model_utils.pipeline.results_readme import build_readme_context, render_results_readme


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

    readme_content = render_results_readme(
        ctx=build_readme_context(data, merged),
        metrics_test=eval_result.metrics_test,
        tier_counts=df_tier_counts,
        best_params=best_params_log,
        nn_params=nn_result.nn_params,
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
