from datetime import datetime
import mlflow
import mlflow.pytorch

class MlflowTracker: 
    @staticmethod
    def set_tracking():
        mlflow.set_tracking_uri("http://localhost:5000")   
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mlflow.set_experiment(f"FPL-Points-Predictions_{current_time}")

    @staticmethod
    def start_run(run_name: str):
        return mlflow.start_run(run_name=run_name)

    @staticmethod
    def log_params(params: dict):
        clean_params = {
            k: (round(v, 4) if isinstance(v, float) else v) 
            for k, v in params.items()
        }
        mlflow.log_params(clean_params)
    
    @staticmethod
    def log_metrics(metrics: dict, step: int = None):
        clean_metrics = {
            k: (round(v, 4) if isinstance(v, float) else v) 
            for k, v in metrics.items()
        }
        mlflow.log_metrics(clean_metrics, step=step)

    @staticmethod
    def log_model(model, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path=artifact_path)
