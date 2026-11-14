from datetime import datetime
import sys
import urllib.request
import mlflow
import mlflow.pytorch

from fpl_project.scripts import config


class MlflowTracker:
    @staticmethod
    def set_tracking(experiment_name: str | None = None):
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass
        uri = "http://localhost:5000"
        try:
            urllib.request.urlopen(uri, timeout=1.5)
        except Exception:
            db_path = (config.ROOT_DIR / "mlflow.db").as_posix()
            uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(uri)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = experiment_name or f"FPL-Points-Predictions_{current_time}"
        mlflow.set_experiment(exp_name)

    @staticmethod
    def start_run(run_name: str):
        return mlflow.start_run(run_name=run_name)

    @staticmethod
    def log_params(params: dict):
        clean_params = {
            k: (round(float(v), 4) if isinstance(v, (float, int)) and not isinstance(v, bool) else str(v))
            for k, v in params.items()
        }
        mlflow.log_params(clean_params)

    @staticmethod
    def log_metrics(metrics: dict, step: int = None):
        clean_metrics = {
            k: round(float(v), 6)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(clean_metrics, step=step)

    @staticmethod
    def log_model(model, artifact_path: str = None, name: str = None, input_example=None):
        model_name = name or artifact_path or "model"
        try:
            if input_example is not None:
                mlflow.pytorch.log_model(model, name=model_name, input_example=input_example)
            else:
                mlflow.pytorch.log_model(model, name=model_name, serialization_format="pickle")
        except Exception:
            try:
                mlflow.pytorch.log_model(model, artifact_path=model_name, serialization_format="pickle")
            except Exception as e:
                print(f"Warning: Failed to log model to MLflow: {e}")
