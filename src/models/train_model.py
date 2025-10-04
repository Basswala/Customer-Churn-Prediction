"""
Model Training Module

This module trains and compares multiple machine learning models for churn prediction.
It supports Logistic Regression, Random Forest, XGBoost, and LightGBM.
All experiments are tracked using MLflow for reproducibility.
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import xgboost as xgb
import lightgbm as lgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """
    Unified class for training and evaluating multiple churn prediction models.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def prepare_train_test_split(self, X: pd.DataFrame, y: pd.Series,
                                   test_size: float = 0.2) -> Tuple:
        """
        Split data into training and testing sets.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing

        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y  # Maintain class distribution in splits
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Train churn rate: {y_train.mean():.2%}")
        logger.info(f"Test churn rate: {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test

    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """
        Train Logistic Regression model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target

        Returns:
            LogisticRegression: Trained model
        """
        logger.info("Training Logistic Regression model")

        # Initialize model with balanced class weights for imbalanced data
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced',  # Handle class imbalance
            solver='lbfgs'
        )

        # Train model and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        logger.info(f"Logistic Regression trained in {train_time:.2f} seconds")

        self.models['logistic_regression'] = model
        return model

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Train Random Forest model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target

        Returns:
            RandomForestClassifier: Trained model
        """
        logger.info("Training Random Forest model")

        # Initialize model with optimized hyperparameters
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=self.random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1  # Use all available cores
        )

        # Train model and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        logger.info(f"Random Forest trained in {train_time:.2f} seconds")

        self.models['random_forest'] = model
        return model

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """
        Train XGBoost model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target

        Returns:
            xgb.XGBClassifier: Trained model
        """
        logger.info("Training XGBoost model")

        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Initialize model with optimized hyperparameters
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=3,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # Handle class imbalance
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )

        # Train model and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        logger.info(f"XGBoost trained in {train_time:.2f} seconds")

        self.models['xgboost'] = model
        return model

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
        """
        Train LightGBM model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target

        Returns:
            lgb.LGBMClassifier: Trained model
        """
        logger.info("Training LightGBM model")

        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Initialize model with optimized hyperparameters
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # Handle class imbalance
            random_state=self.random_state,
            verbose=-1  # Suppress training logs
        )

        # Train model and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        logger.info(f"LightGBM trained in {train_time:.2f} seconds")

        self.models['lightgbm'] = model
        return model

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                       model_name: str) -> Dict[str, float]:
        """
        Evaluate model performance on test set.

        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model for logging

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")

        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        pred_time = time.time() - start_time

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'prediction_time': pred_time
        }

        # Log results
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        # Store results
        self.results[model_name] = metrics

        return metrics

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all available models.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target

        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        logger.info("=" * 50)
        logger.info("Training all models")
        logger.info("=" * 50)

        # Train each model
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)

        logger.info(f"Completed training {len(self.models)} models")

        return self.models

    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate and compare all trained models.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target

        Returns:
            pd.DataFrame: Comparison table of model performances
        """
        logger.info("=" * 50)
        logger.info("Comparing all models")
        logger.info("=" * 50)

        # Evaluate each model
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)

        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.sort_values('f1', ascending=False)

        logger.info("\n" + "=" * 50)
        logger.info("Model Comparison (sorted by F1 score)")
        logger.info("=" * 50)
        print(comparison_df.to_string())

        return comparison_df

    def get_best_model(self, metric: str = 'f1') -> Tuple[str, Any]:
        """
        Get the best performing model based on specified metric.

        Args:
            metric (str): Metric to use for selection (default: 'f1')

        Returns:
            Tuple[str, Any]: Model name and model object
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")

        # Find best model
        best_model_name = max(self.results, key=lambda x: self.results[x][metric])
        best_model = self.models[best_model_name]
        best_score = self.results[best_model_name][metric]

        logger.info(f"Best model: {best_model_name} ({metric}={best_score:.4f})")

        return best_model_name, best_model


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_raw_data, split_features_target
    from src.data.preprocess import clean_data
    from src.features.build_features import transform_features

    # Load and prepare data
    df = load_raw_data("data/raw/Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
    df_clean = df_clean.drop(columns=['customerID'])
    df_transformed = transform_features(df_clean, is_training=True)

    # Split features and target
    X = df_transformed.drop(columns=['Churn'])
    y = df_transformed['Churn']

    # Train and compare models
    trainer = ChurnModelTrainer(random_state=42)
    X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
    trainer.train_all_models(X_train, y_train)
    comparison = trainer.compare_models(X_test, y_test)
    best_name, best_model = trainer.get_best_model(metric='f1')
