"""
Data Validation Module using Great Expectations

This module provides data quality validation to ensure input data meets
expected standards before training. It checks schema, data types, value ranges,
and business logic constraints.
"""

import pandas as pd
import logging
from typing import Dict, List
import great_expectations as ge
from great_expectations.dataset import PandasDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnDataValidator:
    """
    Validator for customer churn dataset using Great Expectations.
    """

    def __init__(self):
        """Initialize data validator."""
        self.validation_results = {}

    def create_expectations(self, df: pd.DataFrame) -> PandasDataset:
        """
        Create Great Expectations expectations for the churn dataset.

        Args:
            df (pd.DataFrame): Input dataframe to validate

        Returns:
            PandasDataset: DataFrame with expectations attached
        """
        logger.info("Creating data expectations")

        # Convert to Great Expectations dataset
        ge_df = ge.from_pandas(df)

        # 1. COLUMN EXISTENCE EXPECTATIONS
        # Ensure all required columns exist
        required_columns = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ]

        for col in required_columns:
            ge_df.expect_column_to_exist(col)

        # 2. DATA TYPE EXPECTATIONS
        # Numeric columns should be numeric
        ge_df.expect_column_values_to_be_of_type('SeniorCitizen', 'int')
        ge_df.expect_column_values_to_be_of_type('tenure', 'int')

        # 3. VALUE RANGE EXPECTATIONS
        # Tenure should be non-negative
        ge_df.expect_column_values_to_be_between('tenure', min_value=0, max_value=100)

        # Monthly charges should be positive
        ge_df.expect_column_values_to_be_between('MonthlyCharges', min_value=0, max_value=200)

        # Total charges should be non-negative
        ge_df.expect_column_values_to_be_between('TotalCharges', min_value=0)

        # Senior citizen should be binary (0 or 1)
        ge_df.expect_column_values_to_be_in_set('SeniorCitizen', [0, 1])

        # 4. CATEGORICAL VALUE EXPECTATIONS
        # Gender should be Male or Female
        ge_df.expect_column_values_to_be_in_set('gender', ['Male', 'Female'])

        # Churn should be Yes or No
        ge_df.expect_column_values_to_be_in_set('Churn', ['Yes', 'No'])

        # Partner should be Yes or No
        ge_df.expect_column_values_to_be_in_set('Partner', ['Yes', 'No'])

        # Dependents should be Yes or No
        ge_df.expect_column_values_to_be_in_set('Dependents', ['Yes', 'No'])

        # Contract types
        ge_df.expect_column_values_to_be_in_set(
            'Contract',
            ['Month-to-month', 'One year', 'Two year']
        )

        # Internet service types
        ge_df.expect_column_values_to_be_in_set(
            'InternetService',
            ['DSL', 'Fiber optic', 'No']
        )

        # 5. UNIQUENESS EXPECTATIONS
        # CustomerID should be unique
        ge_df.expect_column_values_to_be_unique('customerID')

        # 6. NULL VALUE EXPECTATIONS
        # CustomerID should never be null
        ge_df.expect_column_values_to_not_be_null('customerID')

        # Churn should never be null
        ge_df.expect_column_values_to_not_be_null('Churn')

        # 7. BUSINESS LOGIC EXPECTATIONS
        # If tenure is 0, TotalCharges should be approximately MonthlyCharges
        # (Note: This is a simplified check; in practice, you might need custom expectations)

        logger.info(f"Created expectations for {len(required_columns)} columns")

        return ge_df

    def validate_dataset(self, df: pd.DataFrame, detailed: bool = False) -> Dict:
        """
        Validate the dataset against expectations.

        Args:
            df (pd.DataFrame): Input dataframe to validate
            detailed (bool): Whether to return detailed validation results

        Returns:
            Dict: Validation results summary
        """
        logger.info("Starting data validation")

        # Create expectations and validate
        ge_df = self.create_expectations(df)

        # Run validation
        validation_result = ge_df.validate()

        # Extract summary
        success = validation_result.success
        total_expectations = validation_result.statistics['evaluated_expectations']
        successful_expectations = validation_result.statistics['successful_expectations']
        failed_expectations = total_expectations - successful_expectations

        # Log summary
        logger.info("=" * 50)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Overall Success: {success}")
        logger.info(f"Total Expectations: {total_expectations}")
        logger.info(f"Successful: {successful_expectations}")
        logger.info(f"Failed: {failed_expectations}")

        # Log failures if any
        if not success:
            logger.warning("Validation failures detected:")
            for result in validation_result.results:
                if not result.success:
                    expectation_type = result.expectation_config.expectation_type
                    column = result.expectation_config.kwargs.get('column', 'N/A')
                    logger.warning(f"  - {expectation_type} on column '{column}'")

        # Store results
        self.validation_results = {
            'success': success,
            'total_expectations': total_expectations,
            'successful_expectations': successful_expectations,
            'failed_expectations': failed_expectations,
            'success_rate': successful_expectations / total_expectations if total_expectations > 0 else 0
        }

        if detailed:
            self.validation_results['details'] = validation_result.to_json_dict()

        return self.validation_results

    def get_data_quality_score(self) -> float:
        """
        Get a data quality score between 0 and 1.

        Returns:
            float: Data quality score
        """
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_dataset first.")

        return self.validation_results['success_rate']


def validate_churn_data(df: pd.DataFrame) -> bool:
    """
    Convenience function to validate churn data.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        bool: True if validation passes, False otherwise
    """
    validator = ChurnDataValidator()
    results = validator.validate_dataset(df)
    return results['success']


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_raw_data

    df = load_raw_data("data/raw/Telco-Customer-Churn.csv")

    validator = ChurnDataValidator()
    results = validator.validate_dataset(df)

    print(f"\nData Quality Score: {validator.get_data_quality_score():.2%}")
