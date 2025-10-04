"""
Model Inference Module

This module handles loading the trained model and making predictions on new data.
It ensures feature transformations are consistent with training time.
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Union
import mlflow.pyfunc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Binary mapping - MUST match training exactly
BINARY_MAP = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'PaperlessBilling': {'Yes': 1, 'No': 0}
}


class ChurnPredictor:
    """
    Production-ready predictor for customer churn.
    """

    def __init__(self, model_path: str = "artifacts/model",
                 feature_cols_path: str = "artifacts/feature_columns.json"):
        """
        Initialize predictor with trained model and feature definitions.

        Args:
            model_path (str): Path to saved model directory
            feature_cols_path (str): Path to feature columns JSON file
        """
        self.model = None
        self.feature_cols = None

        # Load model
        if Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning(f"Model not found at {model_path}")

        # Load feature columns
        if Path(feature_cols_path).exists():
            self.load_feature_columns(feature_cols_path)
        else:
            logger.warning(f"Feature columns not found at {feature_cols_path}")

    def load_model(self, model_path: str):
        """
        Load trained model from disk.

        Args:
            model_path (str): Path to model directory
        """
        try:
            # Try loading as MLflow model first
            self.model = mlflow.pyfunc.load_model(model_path)
            logger.info(f"Loaded MLflow model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def load_feature_columns(self, feature_cols_path: str):
        """
        Load feature column names from JSON file.

        Args:
            feature_cols_path (str): Path to feature columns JSON
        """
        with open(feature_cols_path, 'r') as f:
            self.feature_cols = json.load(f)

        logger.info(f"Loaded {len(self.feature_cols)} feature columns")

    def preprocess_input(self, customer_data: Dict) -> pd.DataFrame:
        """
        Preprocess raw customer data into model-ready features.
        This MUST match the training-time transformations exactly.

        Args:
            customer_data (Dict): Raw customer attributes

        Returns:
            pd.DataFrame: Preprocessed features ready for prediction
        """
        # Create DataFrame from input dict
        df = pd.DataFrame([customer_data])

        # Step 1: Encode binary features using fixed mapping
        for col, mapping in BINARY_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # Step 2: Create engineered features (same as training)
        # Average monthly charge per tenure
        df['AvgMonthlyChargePerTenure'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']
        )

        # Tenure bins
        df['TenureBin'] = pd.cut(
            df['tenure'],
            bins=[-1, 12, 24, 48, 72],
            labels=['0-1yr', '1-2yr', '2-4yr', '4yr+']
        )
        df = pd.get_dummies(df, columns=['TenureBin'], drop_first=True, dtype=int)

        # Charges category
        df['ChargesCategory'] = pd.cut(
            df['MonthlyCharges'],
            bins=[0, 35, 70, np.inf],
            labels=['Low', 'Medium', 'High']
        )
        df = pd.get_dummies(df, columns=['ChargesCategory'], drop_first=True, dtype=int)

        # Step 3: One-hot encode categorical features
        categorical_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
        ]

        existing_cats = [col for col in categorical_cols if col in df.columns]
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True, dtype=int)

        # Step 4: Align with training feature columns
        # Add missing columns with value 0
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Remove extra columns and reorder to match training
        df = df[self.feature_cols]

        return df

    def predict(self, customer_data: Union[Dict, List[Dict]],
                return_proba: bool = False) -> Union[str, List[str], float, List[float]]:
        """
        Predict churn for customer(s).

        Args:
            customer_data: Single customer dict or list of customer dicts
            return_proba: If True, return probability instead of binary prediction

        Returns:
            Prediction(s): "Likely to churn"/"Not likely to churn" or probability
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot make predictions.")

        # Handle single customer vs batch
        is_single = isinstance(customer_data, dict)
        if is_single:
            customer_data = [customer_data]

        # Preprocess each customer
        processed_inputs = []
        for customer in customer_data:
            processed = self.preprocess_input(customer)
            processed_inputs.append(processed)

        # Combine into single DataFrame
        X = pd.concat(processed_inputs, ignore_index=True)

        # Make prediction
        if return_proba:
            # Return probability of churning
            predictions = self.model.predict(X)
            # If model returns probabilities, use them; otherwise use binary predictions
            if len(predictions.shape) > 1 and predictions.shape[1] == 2:
                predictions = predictions[:, 1]
        else:
            # Return binary prediction
            predictions = self.model.predict(X)

            # Convert to human-readable labels
            if predictions.dtype in [np.int32, np.int64]:
                predictions = np.where(predictions == 1, "Likely to churn", "Not likely to churn")
            elif predictions.dtype == object:
                # Already in string format
                pass

        # Return single value or list
        if is_single:
            return predictions[0] if isinstance(predictions, (list, np.ndarray)) else predictions
        else:
            return predictions.tolist() if isinstance(predictions, np.ndarray) else predictions


def create_sample_customer() -> Dict:
    """
    Create a sample customer for testing.

    Returns:
        Dict: Sample customer data
    """
    return {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.0,
        'TotalCharges': 1020.0
    }


if __name__ == "__main__":
    # Test predictor with sample data
    predictor = ChurnPredictor()

    sample_customer = create_sample_customer()
    prediction = predictor.predict(sample_customer)

    print(f"Sample customer prediction: {prediction}")
