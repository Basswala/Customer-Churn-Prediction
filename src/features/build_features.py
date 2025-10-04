"""
Feature Engineering Module

This module transforms raw customer data into features ready for machine learning.
Critical: Training and serving MUST use identical transformations for consistency.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from typing import Tuple, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Binary mapping dictionary for consistent encoding across training and serving
BINARY_MAP = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'Churn': {'Yes': 1, 'No': 0}
}


def encode_binary_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Encode binary categorical features (Yes/No, Male/Female) as 0/1.

    Args:
        df (pd.DataFrame): Input dataframe
        is_training (bool): Whether this is training data (for logging purposes)

    Returns:
        pd.DataFrame: Dataframe with binary features encoded
    """
    df_encoded = df.copy()

    logger.info(f"Encoding binary features ({'training' if is_training else 'serving'})")

    # Apply binary mappings
    for column, mapping in BINARY_MAP.items():
        if column in df_encoded.columns:
            # Store original values for validation
            original_values = set(df_encoded[column].unique())

            # Apply mapping
            df_encoded[column] = df_encoded[column].map(mapping)

            # Validate that all values were mapped
            if df_encoded[column].isna().any():
                unmapped = original_values - set(mapping.keys())
                logger.warning(f"Column {column} has unmapped values: {unmapped}")

            logger.info(f"  {column}: {mapping}")

    return df_encoded


def encode_multiclass_features(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    """
    One-hot encode multi-class categorical features.

    Args:
        df (pd.DataFrame): Input dataframe
        categorical_cols (List[str]): List of categorical columns to encode.
                                       If None, will auto-detect.

    Returns:
        pd.DataFrame: Dataframe with one-hot encoded features
    """
    df_encoded = df.copy()

    # Define multi-class categorical columns
    if categorical_cols is None:
        categorical_cols = [
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'Contract',
            'PaymentMethod'
        ]

    logger.info(f"One-hot encoding {len(categorical_cols)} categorical features")

    # Filter to only existing columns
    existing_cats = [col for col in categorical_cols if col in df_encoded.columns]

    # One-hot encode with drop_first=True to avoid multicollinearity
    df_encoded = pd.get_dummies(
        df_encoded,
        columns=existing_cats,
        drop_first=True,  # Critical: prevents dummy variable trap
        dtype=int
    )

    logger.info(f"After encoding: {df_encoded.shape[1]} total features")

    return df_encoded


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional engineered features from existing columns.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with additional engineered features
    """
    df_eng = df.copy()

    logger.info("Creating engineered features")

    # 1. Average monthly charge per tenure month
    # Avoid division by zero for new customers
    df_eng['AvgMonthlyChargePerTenure'] = np.where(
        df_eng['tenure'] > 0,
        df_eng['TotalCharges'] / df_eng['tenure'],
        df_eng['MonthlyCharges']
    )

    # 2. Tenure bins (customer lifecycle stages)
    df_eng['TenureBin'] = pd.cut(
        df_eng['tenure'],
        bins=[-1, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4yr+']
    )

    # One-hot encode tenure bins
    df_eng = pd.get_dummies(df_eng, columns=['TenureBin'], drop_first=True, dtype=int)

    # 3. Charges category (low, medium, high spender)
    df_eng['ChargesCategory'] = pd.cut(
        df_eng['MonthlyCharges'],
        bins=[0, 35, 70, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    # One-hot encode charges category
    df_eng = pd.get_dummies(df_eng, columns=['ChargesCategory'], drop_first=True, dtype=int)

    # 4. Service count (how many services the customer has)
    # Count non-zero/non-"No" service columns
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']

    # This requires services to already be encoded as binary
    # For simplicity, we'll count after one-hot encoding in the main pipeline

    logger.info(f"Engineered features created. New shape: {df_eng.shape}")

    return df_eng


def transform_features(df: pd.DataFrame, target_col: str = 'Churn',
                        is_training: bool = True) -> pd.DataFrame:
    """
    Complete feature transformation pipeline.

    This function applies all transformations in the correct order:
    1. Binary encoding
    2. Multi-class encoding
    3. Feature engineering

    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column (default: 'Churn')
        is_training (bool): Whether this is training data

    Returns:
        pd.DataFrame: Fully transformed feature matrix
    """
    logger.info(f"Starting feature transformation pipeline ({'training' if is_training else 'serving'})")

    # Make a copy to avoid modifying original
    df_transformed = df.copy()

    # Step 1: Encode binary features
    df_transformed = encode_binary_features(df_transformed, is_training)

    # Step 2: Create engineered features (before one-hot encoding)
    df_transformed = create_engineered_features(df_transformed)

    # Step 3: One-hot encode multi-class categorical features
    df_transformed = encode_multiclass_features(df_transformed)

    logger.info(f"Feature transformation complete. Final shape: {df_transformed.shape}")

    return df_transformed


def save_feature_columns(feature_cols: List[str], output_path: str = "artifacts/feature_columns.json"):
    """
    Save feature column names for serving consistency.

    Args:
        feature_cols (List[str]): List of feature column names
        output_path (str): Path to save feature columns
    """
    import json

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)

    logger.info(f"Saved {len(feature_cols)} feature columns to {output_path}")


def load_feature_columns(input_path: str = "artifacts/feature_columns.json") -> List[str]:
    """
    Load feature column names for serving.

    Args:
        input_path (str): Path to feature columns file

    Returns:
        List[str]: List of feature column names
    """
    import json

    with open(input_path, 'r') as f:
        feature_cols = json.load(f)

    logger.info(f"Loaded {len(feature_cols)} feature columns from {input_path}")

    return feature_cols


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_raw_data
    from src.data.preprocess import clean_data

    df = load_raw_data("data/raw/Telco-Customer-Churn.csv")
    df_clean = clean_data(df)

    # Drop customerID
    df_clean = df_clean.drop(columns=['customerID'])

    # Transform features
    df_transformed = transform_features(df_clean, is_training=True)

    print(f"Original shape: {df.shape}")
    print(f"Transformed shape: {df_transformed.shape}")
    print(f"Feature columns: {df_transformed.columns.tolist()}")
