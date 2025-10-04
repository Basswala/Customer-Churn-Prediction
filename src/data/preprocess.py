"""
Data Preprocessing Module

This module handles data cleaning, type conversion, and preparation for feature engineering.
It ensures data quality and consistency before model training.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data by handling missing values and type conversions.

    Args:
        df (pd.DataFrame): Raw input dataframe

    Returns:
        pd.DataFrame: Cleaned dataframe ready for feature engineering
    """
    df_clean = df.copy()

    logger.info("Starting data cleaning process")

    # 1. Handle TotalCharges column (sometimes stored as object instead of float)
    # Convert empty strings to NaN, then to numeric
    if df_clean['TotalCharges'].dtype == 'object':
        logger.info("Converting TotalCharges from object to numeric")
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

    # 2. Handle missing values in TotalCharges
    # For customers with 0 tenure, set TotalCharges to 0
    missing_count = df_clean['TotalCharges'].isna().sum()
    if missing_count > 0:
        logger.info(f"Found {missing_count} missing values in TotalCharges")
        df_clean.loc[df_clean['tenure'] == 0, 'TotalCharges'] = 0

        # For any remaining NaNs, fill with median
        remaining_nans = df_clean['TotalCharges'].isna().sum()
        if remaining_nans > 0:
            median_val = df_clean['TotalCharges'].median()
            df_clean['TotalCharges'].fillna(median_val, inplace=True)
            logger.info(f"Filled {remaining_nans} remaining NaNs with median: {median_val}")

    # 3. Ensure SeniorCitizen is integer (already 0/1 in dataset)
    df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(int)

    # 4. Strip whitespace from string columns
    string_columns = df_clean.select_dtypes(include=['object']).columns
    for col in string_columns:
        df_clean[col] = df_clean[col].str.strip()

    # 5. Check for duplicates
    duplicate_count = df_clean.duplicated(subset='customerID').sum()
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate customer IDs - removing duplicates")
        df_clean = df_clean.drop_duplicates(subset='customerID', keep='first')

    logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")

    return df_clean


def validate_numeric_ranges(df: pd.DataFrame) -> bool:
    """
    Validate that numeric columns are within expected ranges.

    Args:
        df (pd.DataFrame): Dataframe to validate

    Returns:
        bool: True if all validations pass

    Raises:
        ValueError: If data contains invalid values
    """
    logger.info("Validating numeric ranges")

    # Check tenure (should be >= 0)
    if (df['tenure'] < 0).any():
        raise ValueError("Found negative values in tenure column")

    # Check MonthlyCharges (should be positive)
    if (df['MonthlyCharges'] <= 0).any():
        raise ValueError("Found non-positive values in MonthlyCharges")

    # Check TotalCharges (should be >= 0)
    if (df['TotalCharges'] < 0).any():
        raise ValueError("Found negative values in TotalCharges")

    # Check SeniorCitizen (should be 0 or 1)
    if not df['SeniorCitizen'].isin([0, 1]).all():
        raise ValueError("SeniorCitizen contains values other than 0 or 1")

    logger.info("All numeric range validations passed")
    return True


def get_data_statistics(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the dataset.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        dict: Dictionary containing various statistics
    """
    stats = {
        'total_customers': len(df),
        'churn_count': (df['Churn'] == 'Yes').sum() if 'Churn' in df.columns else 0,
        'churn_rate': (df['Churn'] == 'Yes').mean() if 'Churn' in df.columns else 0,
        'avg_tenure': df['tenure'].mean(),
        'avg_monthly_charges': df['MonthlyCharges'].mean(),
        'avg_total_charges': df['TotalCharges'].mean(),
        'senior_citizen_pct': df['SeniorCitizen'].mean(),
        'missing_values': df.isnull().sum().to_dict()
    }

    logger.info(f"Dataset statistics: {stats['total_customers']} customers, "
                f"{stats['churn_rate']:.2%} churn rate")

    return stats


if __name__ == "__main__":
    # Example usage
    from load_data import load_raw_data

    df = load_raw_data("data/raw/Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
    validate_numeric_ranges(df_clean)
    stats = get_data_statistics(df_clean)
    print(stats)
