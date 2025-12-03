"""
Main Pipeline Script for Rossmann Sales Forecasting.

This script orchestrates the complete machine learning pipeline:
1. Load and merge data
2. Preprocess features
3. Train multiple models
4. Evaluate and compare results
5. Generate reports

Usage:
    python main.py [--data-dir DATA_DIR] [--validation-weeks N] [--generate-submission]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader, get_data_summary
from preprocessing import (
    DataPreprocessor,
    create_time_based_split,
    prepare_features_and_target
)
from models import ModelFactory
from evaluation import ModelEvaluator, generate_submission_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Rossmann Sales Forecasting Pipeline'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing train.csv, test.csv, and store.csv'
    )
    
    parser.add_argument(
        '--validation-weeks',
        type=int,
        default=6,
        help='Number of weeks to use for validation (default: 6)'
    )
    
    parser.add_argument(
        '--generate-submission',
        action='store_true',
        help='Generate a submission file for the test set'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save outputs (models, reports, submissions)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to train (default: all available)'
    )
    
    return parser.parse_args()


def run_pipeline(
    data_dir: str = 'data',
    validation_weeks: int = 6,
    generate_submission: bool = False,
    output_dir: str = 'output',
    model_names: list = None
):
    """
    Run the complete sales forecasting pipeline.
    
    Args:
        data_dir: Path to data directory.
        validation_weeks: Number of weeks for validation.
        generate_submission: Whether to generate a Kaggle submission.
        output_dir: Directory for output files.
        model_names: List of specific models to train.
        
    Returns:
        Dictionary with pipeline results.
    """
    logger.info("=" * 60)
    logger.info("ROSSMANN SALES FORECASTING PIPELINE")
    logger.info("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load and Merge Data
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 40)
    
    loader = DataLoader(data_dir=data_dir)
    
    try:
        merged_train = loader.load_and_merge_train()
        get_data_summary(merged_train, "Merged Training Data")
    except FileNotFoundError as e:
        logger.error(f"Could not load training data: {e}")
        raise
    
    # =========================================================================
    # STEP 2: Preprocess Data
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 2: Preprocessing Data")
    logger.info("=" * 40)
    
    preprocessor = DataPreprocessor()
    train_processed = preprocessor.preprocess(merged_train, is_train=True)
    
    # =========================================================================
    # STEP 3: Create Time-Based Split
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 3: Creating Time-Based Split")
    logger.info("=" * 40)
    
    train_df, val_df = create_time_based_split(
        train_processed,
        validation_weeks=validation_weeks
    )
    
    # Prepare features and target
    feature_columns = preprocessor.get_feature_columns()
    X_train, y_train = prepare_features_and_target(train_df, feature_columns)
    X_val, y_val = prepare_features_and_target(val_df, feature_columns)
    
    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    
    # Get actual feature names (those that exist in the data)
    feature_names = X_train.columns.tolist()
    
    # =========================================================================
    # STEP 4: Train Models
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 4: Training Models")
    logger.info("=" * 40)
    
    # Determine which models to train
    if model_names is None:
        model_names = ModelFactory.get_available_models()
    
    logger.info(f"Models to train: {model_names}")
    
    evaluator = ModelEvaluator()
    trained_models = {}
    
    for model_name in model_names:
        logger.info(f"\n--- Training {model_name} ---")
        
        try:
            model = ModelFactory.create_model(model_name)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            # Evaluate the model
            evaluator.evaluate_model(
                model.name,
                y_train.values, y_pred_train,
                y_val.values, y_pred_val
            )
            
            # Store feature importance
            importance_df = model.get_feature_importance(feature_names)
            if not importance_df.empty:
                evaluator.add_feature_importance(model.name, importance_df)
            
            # Save model
            model_path = output_path / f"{model_name}_model.joblib"
            model.save_model(str(model_path))
            
            # Store model by its display name (used by evaluator.get_best_model())
            trained_models[model.name] = model
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    # =========================================================================
    # STEP 5: Compare Models and Generate Reports
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 5: Model Comparison and Reports")
    logger.info("=" * 40)
    
    # Print comparison table
    evaluator.print_comparison_table()
    
    # Print feature importance for best model
    evaluator.print_feature_importance()
    
    # Generate full report
    report_path = output_path / "evaluation_report.txt"
    report = evaluator.generate_full_report(str(report_path))
    print(report)
    
    # =========================================================================
    # STEP 6: Generate Submission (Optional)
    # =========================================================================
    if generate_submission:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 6: Generating Submission")
        logger.info("=" * 40)
        
        try:
            # Load and preprocess test data
            merged_test = loader.load_and_merge_test()
            test_processed = preprocessor.preprocess(merged_test, is_train=False)
            
            X_test, _ = prepare_features_and_target(test_processed, feature_columns)
            
            # Use best model for prediction
            best_model_name = evaluator.get_best_model()
            if best_model_name and best_model_name in trained_models:
                best_model = trained_models[best_model_name]
                
                # Make predictions
                test_predictions = best_model.predict(X_test)
                
                # Handle closed stores (Open=0) - predict 0 sales
                if 'Open' in test_processed.columns:
                    test_predictions[test_processed['Open'] == 0] = 0
                
                # Generate submission file
                test_ids = test_processed['Id'].values
                submission_path = output_path / "submission.csv"
                generate_submission_file(
                    test_predictions,
                    test_ids,
                    str(submission_path)
                )
                
                logger.info(f"Submission generated using {best_model_name}")
            else:
                logger.warning("No trained model available for submission")
                
        except FileNotFoundError as e:
            logger.warning(f"Could not generate submission: {e}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Models trained: {list(trained_models.keys())}")
    logger.info(f"Best model: {evaluator.get_best_model()}")
    logger.info(f"Output directory: {output_path.absolute()}")
    
    return {
        'evaluator': evaluator,
        'trained_models': trained_models,
        'preprocessor': preprocessor,
        'feature_columns': feature_columns
    }


def main():
    """Main entry point for the pipeline."""
    args = parse_arguments()
    
    try:
        results = run_pipeline(
            data_dir=args.data_dir,
            validation_weeks=args.validation_weeks,
            generate_submission=args.generate_submission,
            output_dir=args.output_dir,
            model_names=args.models
        )
        
        logger.info("Pipeline executed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
