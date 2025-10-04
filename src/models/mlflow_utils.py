"""
MLflow Integration Module

This module provides utilities for experiment tracking, model logging,
and artifact management using MLflow. All experiments and models are tracked
for reproducibility and comparison.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import logging
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Wrapper class for MLflow experiment tracking and model management.
    """

    def __init__(self, experiment_name: str = "Telco_Churn_Prediction",
                 tracking_uri: str = "file:./mlruns"):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name (str): Name of the MLflow experiment
            tracking_uri (str): URI for MLflow tracking server
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        self.experiment = mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow experiment: {experiment_name}")
        logger.info(f"Tracking URI: {tracking_uri}")

    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """
        Start a new MLflow run.

        Args:
            run_name (str): Name for this run
            tags (Dict[str, str]): Tags to associate with the run
        """
        mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run_name}")

    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.

        Args:
            params (Dict[str, Any]): Dictionary of parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

        logger.info(f"Logged {len(params)} parameters to MLflow")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to MLflow.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics to log
            step (int): Optional step number for metric tracking
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

        logger.info(f"Logged {len(metrics)} metrics to MLflow")

    def log_model(self, model: Any, model_name: str, artifact_path: str = "model"):
        """
        Log a trained model to MLflow.

        Args:
            model: Trained model object
            model_name (str): Name/type of the model
            artifact_path (str): Path within MLflow to store model
        """
        # Determine model flavor based on model type
        model_type = type(model).__name__

        logger.info(f"Logging {model_type} model as {model_name}")

        if 'XGB' in model_type:
            mlflow.xgboost.log_model(model, artifact_path)
        elif 'LGBM' in model_type or 'LightGBM' in model_type:
            mlflow.lightgbm.log_model(model, artifact_path)
        else:
            # Sklearn models (LogisticRegression, RandomForest, etc.)
            mlflow.sklearn.log_model(model, artifact_path)

        logger.info(f"Model logged to {artifact_path}")

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Log an artifact (file) to MLflow.

        Args:
            local_path (str): Path to local file
            artifact_path (str): Optional subdirectory in artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")

    def log_dict_as_artifact(self, data: Dict, filename: str):
        """
        Log a dictionary as JSON artifact.

        Args:
            data (Dict): Dictionary to log
            filename (str): Name for the JSON file
        """
        # Create temporary file
        temp_path = Path(f"temp_{filename}")
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Log to MLflow
        mlflow.log_artifact(str(temp_path))

        # Clean up
        temp_path.unlink()

        logger.info(f"Logged dictionary as {filename}")

    def log_dataframe_as_artifact(self, df: pd.DataFrame, filename: str):
        """
        Log a pandas DataFrame as CSV artifact.

        Args:
            df (pd.DataFrame): DataFrame to log
            filename (str): Name for the CSV file
        """
        # Create temporary file
        temp_path = Path(f"temp_{filename}")
        df.to_csv(temp_path, index=False)

        # Log to MLflow
        mlflow.log_artifact(str(temp_path))

        # Clean up
        temp_path.unlink()

        logger.info(f"Logged DataFrame as {filename}")

    def save_feature_columns(self, feature_cols: List[str]):
        """
        Save feature column names as artifact for serving consistency.

        Args:
            feature_cols (List[str]): List of feature column names
        """
        # Save to temp file
        temp_path = Path("temp_feature_columns.txt")
        with open(temp_path, 'w') as f:
            f.write('\n'.join(feature_cols))

        # Log to MLflow
        mlflow.log_artifact(str(temp_path), "feature_columns.txt")

        # Clean up
        temp_path.unlink()

        logger.info(f"Saved {len(feature_cols)} feature columns to MLflow")

    def save_preprocessing_pipeline(self, preprocessing_obj: Any):
        """
        Save preprocessing pipeline as artifact.

        Args:
            preprocessing_obj: Preprocessing object/pipeline to save
        """
        # Save to temp file
        temp_path = Path("temp_preprocessing.pkl")
        joblib.dump(preprocessing_obj, temp_path)

        # Log to MLflow
        mlflow.log_artifact(str(temp_path), "preprocessing.pkl")

        # Clean up
        temp_path.unlink()

        logger.info("Saved preprocessing pipeline to MLflow")

    def load_model(self, run_id: str, model_path: str = "model") -> Any:
        """
        Load a model from MLflow.

        Args:
            run_id (str): MLflow run ID
            model_path (str): Path to model within run artifacts

        Returns:
            Any: Loaded model object
        """
        model_uri = f"runs:/{run_id}/{model_path}"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info(f"Loaded model from run {run_id}")

        return model

    def get_best_run(self, metric: str = "f1", ascending: bool = False) -> Dict[str, Any]:
        """
        Get the best run from the experiment based on a metric.

        Args:
            metric (str): Metric to optimize
            ascending (bool): Whether lower is better

        Returns:
            Dict[str, Any]: Best run information
        """
        # Search for runs in the experiment
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )

        if len(runs) == 0:
            raise ValueError("No runs found in experiment")

        best_run = runs.iloc[0].to_dict()

        logger.info(f"Best run by {metric}: {best_run['run_id']}")
        logger.info(f"  {metric}: {best_run.get(f'metrics.{metric}', 'N/A')}")

        return best_run

    def log_complete_experiment(self, model_name: str, model: Any,
                                 params: Dict[str, Any], metrics: Dict[str, float],
                                 feature_cols: List[str], run_name: str = None):
        """
        Log a complete experiment with model, params, metrics, and artifacts.

        Args:
            model_name (str): Name of the model
            model: Trained model object
            params (Dict[str, Any]): Model parameters
            metrics (Dict[str, float]): Evaluation metrics
            feature_cols (List[str]): List of feature column names
            run_name (str): Optional name for the run
        """
        # Start run
        self.start_run(run_name=run_name or f"{model_name}_run")

        try:
            # Log model type as tag
            mlflow.set_tag("model_type", model_name)

            # Log parameters
            self.log_params(params)

            # Log metrics
            self.log_metrics(metrics)

            # Log model
            self.log_model(model, model_name)

            # Log feature columns
            self.save_feature_columns(feature_cols)

            logger.info(f"Complete experiment logged for {model_name}")

        finally:
            # Always end run, even if there's an error
            self.end_run()


if __name__ == "__main__":
    # Example usage
    tracker = MLflowTracker(experiment_name="Test_Experiment")

    # Log a test run
    tracker.start_run(run_name="test_run")
    tracker.log_params({"param1": "value1", "param2": 42})
    tracker.log_metrics({"accuracy": 0.85, "f1": 0.80})
    tracker.end_run()

    print("MLflow test completed successfully")
