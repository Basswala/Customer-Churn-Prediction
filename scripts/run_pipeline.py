"""
Complete ML Training Pipeline

This script runs the end-to-end machine learning pipeline:
1. Load and validate data
2. Preprocess and engineer features
3. Train multiple models (LogReg, RF, XGBoost, LightGBM)
4. Compare models and select the best
5. Log everything to MLflow for tracking

Usage:
    python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.load_data import load_raw_data, validate_data_schema
from src.data.preprocess import clean_data, validate_numeric_ranges, get_data_statistics
from src.features.build_features import transform_features, save_feature_columns
from src.models.train_model import ChurnModelTrainer
from src.models.mlflow_utils import MLflowTracker
from src.utils.validate_data import ChurnDataValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_pipeline(input_path: str, target_col: str = 'Churn',
                          test_size: float = 0.2, experiment_name: str = "Telco_Churn_Prediction"):
    """
    Run the complete ML pipeline from data loading to model training.

    Args:
        input_path (str): Path to raw data CSV file
        target_col (str): Name of target column
        test_size (float): Proportion of data for testing
        experiment_name (str): Name for MLflow experiment
    """
    logger.info("=" * 70)
    logger.info("STARTING COMPLETE ML PIPELINE")
    logger.info("=" * 70)

    # =========================================================================
    # STEP 1: DATA LOADING AND VALIDATION
    # =========================================================================
    logger.info("\n[STEP 1/6] Loading and validating raw data...")

    # Load raw data
    df_raw = load_raw_data(input_path)

    # Validate schema
    validate_data_schema(df_raw)

    # Validate data quality with Great Expectations
    validator = ChurnDataValidator()
    validation_results = validator.validate_dataset(df_raw)
    data_quality_pass = validation_results['success']
    data_quality_score = validator.get_data_quality_score()

    logger.info(f"Data quality validation: {'PASSED' if data_quality_pass else 'FAILED'}")
    logger.info(f"Data quality score: {data_quality_score:.2%}")

    # =========================================================================
    # STEP 2: DATA PREPROCESSING
    # =========================================================================
    logger.info("\n[STEP 2/6] Cleaning and preprocessing data...")

    # Clean data
    df_clean = clean_data(df_raw)

    # Validate numeric ranges
    validate_numeric_ranges(df_clean)

    # Get statistics
    stats = get_data_statistics(df_clean)
    logger.info(f"Dataset statistics: {stats['total_customers']} customers, "
                f"{stats['churn_rate']:.2%} churn rate")

    # Save processed data
    processed_path = Path("data/processed/Telco-Customer-Churn-Processed.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    logger.info(f"Saved processed data to {processed_path}")

    # =========================================================================
    # STEP 3: FEATURE ENGINEERING
    # =========================================================================
    logger.info("\n[STEP 3/6] Engineering features...")

    # Drop customerID (not a feature for prediction)
    df_features = df_clean.drop(columns=['customerID'])

    # Transform features
    df_transformed = transform_features(df_features, target_col=target_col, is_training=True)

    # Split features and target
    X = df_transformed.drop(columns=[target_col])
    y = df_transformed[target_col]

    # Save feature column names for serving consistency
    feature_cols = X.columns.tolist()
    save_feature_columns(feature_cols, "artifacts/feature_columns.json")

    logger.info(f"Feature engineering complete: {len(feature_cols)} features")

    # =========================================================================
    # STEP 4: MODEL TRAINING
    # =========================================================================
    logger.info("\n[STEP 4/6] Training multiple models...")

    # Initialize trainer
    trainer = ChurnModelTrainer(random_state=42)

    # Split data
    X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(
        X, y, test_size=test_size
    )

    # Train all models
    trainer.train_all_models(X_train, y_train)

    # =========================================================================
    # STEP 5: MODEL EVALUATION AND COMPARISON
    # =========================================================================
    logger.info("\n[STEP 5/6] Evaluating and comparing models...")

    # Compare models
    comparison_df = trainer.compare_models(X_test, y_test)

    # Get best model
    best_model_name, best_model = trainer.get_best_model(metric='f1')

    # =========================================================================
    # STEP 6: MLFLOW LOGGING
    # =========================================================================
    logger.info("\n[STEP 6/6] Logging experiments to MLflow...")

    # Initialize MLflow tracker
    mlflow_tracker = MLflowTracker(experiment_name=experiment_name)

    # Log each model as a separate run
    for model_name in trainer.models.keys():
        model = trainer.models[model_name]
        metrics = trainer.results[model_name]

        # Prepare parameters
        params = {
            'model_type': model_name,
            'test_size': test_size,
            'random_state': 42,
            'n_features': len(feature_cols),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

        # Add model-specific hyperparameters
        if hasattr(model, 'get_params'):
            model_params = model.get_params()
            # Add only important hyperparameters (avoid logging all defaults)
            important_params = ['n_estimators', 'learning_rate', 'max_depth',
                                'min_samples_split', 'min_samples_leaf']
            for key in important_params:
                if key in model_params:
                    params[f'model_{key}'] = model_params[key]

        # Add data quality metrics
        metrics['data_quality_score'] = data_quality_score
        metrics['data_quality_pass'] = 1.0 if data_quality_pass else 0.0

        # Log complete experiment
        mlflow_tracker.log_complete_experiment(
            model_name=model_name,
            model=model,
            params=params,
            metrics=metrics,
            feature_cols=feature_cols,
            run_name=f"{model_name}_{test_size}"
        )

    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"F1 Score: {trainer.results[best_model_name]['f1']:.4f}")
    logger.info(f"ROC AUC: {trainer.results[best_model_name]['roc_auc']:.4f}")
    logger.info(f"\nView results: mlflow ui --backend-store-uri file:./mlruns")
    logger.info("=" * 70)

    return {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'comparison': comparison_df,
        'feature_columns': feature_cols,
        'data_quality_score': data_quality_score
    }


def main():
    """Main entry point for the pipeline script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run complete ML training pipeline')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to raw data CSV file')
    parser.add_argument('--target', type=str, default='Churn',
                        help='Name of target column (default: Churn)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--experiment', type=str, default='Telco_Churn_Prediction',
                        help='MLflow experiment name')

    args = parser.parse_args()

    # Run pipeline
    results = run_complete_pipeline(
        input_path=args.input,
        target_col=args.target,
        test_size=args.test_size,
        experiment_name=args.experiment
    )

    return results


if __name__ == "__main__":
    main()
