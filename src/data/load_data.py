"""
Data Loading Module

This module handles loading and initial validation of the Telco Customer Churn dataset.
It provides functions to read CSV data and perform basic sanity checks.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw customer churn data from CSV file.

    Args:
        file_path (str): Path to the CSV file containing customer data

    Returns:
        pd.DataFrame: Loaded dataframe with customer information

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        # Verify file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load CSV data
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        # Log basic information
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {', '.join(df.columns.tolist())}")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def validate_data_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the dataframe contains expected columns for churn prediction.

    Args:
        df (pd.DataFrame): Input dataframe to validate

    Returns:
        bool: True if validation passes, raises ValueError otherwise

    Raises:
        ValueError: If required columns are missing
    """
    # Required columns for churn prediction
    required_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]

    # Check for missing columns
    missing_cols = set(required_columns) - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    logger.info("Data schema validation passed")
    return True


def split_features_target(df: pd.DataFrame, target_col: str = 'Churn') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features (X) and target variable (y).

    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column (default: 'Churn')

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target variable
    """
    # Drop customerID as it's not a feature for prediction
    X = df.drop(columns=[target_col, 'customerID'])
    y = df[target_col]

    logger.info(f"Split data into {X.shape[1]} features and target variable")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    return X, y


if __name__ == "__main__":
    # Example usage
    data_path = "data/raw/Telco-Customer-Churn.csv"
    df = load_raw_data(data_path)
    validate_data_schema(df)
    X, y = split_features_target(df)
