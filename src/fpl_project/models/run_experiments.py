"""Entry point of the FPL experiment pipeline.

Usage (from the repository root):
    PYTHONPATH=src python -m fpl_project.models.run_experiments

Stages:
    1. data_preparation.load_data        -> PreparedData
    2. tabular.tune_tabular_models       -> GLM, XGBoost (MSE / FPLLoss)
       neural.train_neural_networks      -> MLP, Conv1D (MSE / FPLLoss)
    3. meta.merge_base_predictions, meta.fit_meta_models
    4. reporting.save_results, persistence.save_trained_models
    5. forecast.predict_unplayed_fixtures -> results/preds_predict.parquet (is_played == False)
"""

import sys
import time
import warnings

from fpl_project.scripts import config
from fpl_project.scripts.model_utils.pipeline.containers import PredictionSplit
from fpl_project.scripts.model_utils.pipeline.data_preparation import load_data
from fpl_project.scripts.model_utils.pipeline.forecast import (
    predict_unplayed_fixtures,
    save_unplayed_predictions,
)
from fpl_project.scripts.model_utils.pipeline.meta import fit_meta_models, merge_base_predictions
from fpl_project.scripts.model_utils.pipeline.neural import train_neural_networks
from fpl_project.scripts.model_utils.pipeline.persistence import save_trained_models
from fpl_project.scripts.model_utils.pipeline.reporting import save_results
from fpl_project.scripts.model_utils.pipeline.sanity_checks import run_verification_tests
from fpl_project.scripts.model_utils.pipeline.tabular import tune_tabular_models
from fpl_project.scripts.model_utils.selector import initialize_device
from fpl_project.scripts.model_utils.trainer import set_seed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    """Run the complete FPL experiment pipeline."""
    total_start = time.time()
    set_seed(config.SEED)
    run_verification_tests()

    device = initialize_device()
    print(f"Urządzenie obliczeniowe: {device} | Ziarno losowe (SEED): {config.SEED}")

    data = load_data()
    tab_result = tune_tabular_models(data, device)
    nn_result = train_neural_networks(data, device)

    combined_preds = PredictionSplit(
        valid={**tab_result.predictions.valid, **nn_result.predictions.valid},
        test={**tab_result.predictions.test, **nn_result.predictions.test},
    )

    merged = merge_base_predictions(data, combined_preds)
    eval_result = fit_meta_models(data, merged)
    save_results(
        data=data,
        merged=merged,
        eval_result=eval_result,
        best_params_log=tab_result.best_params,
        nn_result=nn_result,
    )
    save_trained_models(data=data, tab_result=tab_result, nn_result=nn_result)

    preds_predict = predict_unplayed_fixtures(data, tab_result, nn_result, device)
    save_unplayed_predictions(preds_predict)
    print(f"Czas całkowity wykonania eksperymentu: {time.time() - total_start:.2f} s")


if __name__ == "__main__":
    main()
