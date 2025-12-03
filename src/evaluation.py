"""
Evaluation Module for Rossmann Sales Forecasting.

This module provides functions to calculate RMSPE (Root Mean Square Percentage Error)
and generate comprehensive performance reports for model comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Percentage Error (RMSPE).
    
    RMSPE = sqrt(1/n * sum((y_true - y_pred) / y_true)^2)
    
    This metric penalizes percentage errors, giving equal weight to
    errors regardless of the actual sales value.
    
    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.
        
    Returns:
        RMSPE score (lower is better).
        
    Raises:
        ValueError: If arrays have different shapes or contain invalid values.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )
    
    # Filter out zero actual values to avoid division by zero
    mask = y_true > 0
    if not mask.any():
        logger.warning("All actual values are zero, returning 0")
        return 0.0
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Calculate RMSPE
    percentage_errors = ((y_true_filtered - y_pred_filtered) / y_true_filtered) ** 2
    rmspe_value = np.sqrt(np.mean(percentage_errors))
    
    return rmspe_value


def rmspe_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate RMSPE score suitable for sklearn scoring.
    
    Returns negative RMSPE so it can be used with sklearn's scoring
    (where higher is better).
    
    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.
        
    Returns:
        Negative RMSPE score.
    """
    return -rmspe(y_true, y_pred)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics.
    
    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.
        
    Returns:
        Dictionary of metric names to values.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Filter for valid values
    mask = y_true > 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    metrics = {}
    
    # RMSPE (primary metric)
    metrics['rmspe'] = rmspe(y_true, y_pred)
    
    # RMSE (Root Mean Square Error)
    metrics['rmse'] = np.sqrt(np.mean((y_true_filtered - y_pred_filtered) ** 2))
    
    # MAE (Mean Absolute Error)
    metrics['mae'] = np.mean(np.abs(y_true_filtered - y_pred_filtered))
    
    # MAPE (Mean Absolute Percentage Error)
    metrics['mape'] = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    
    # R-squared
    ss_res = np.sum((y_true_filtered - y_pred_filtered) ** 2)
    ss_tot = np.sum((y_true_filtered - np.mean(y_true_filtered)) ** 2)
    metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Mean Bias Error
    metrics['bias'] = np.mean(y_pred_filtered - y_true_filtered)
    
    return metrics


class ModelEvaluator:
    """
    A class to evaluate and compare multiple models.
    
    Provides methods for comprehensive model evaluation and comparison.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.results = {}
        self.feature_importances = {}
        logger.info("ModelEvaluator initialized")
    
    def evaluate_model(
        self,
        model_name: str,
        y_true_train: np.ndarray,
        y_pred_train: np.ndarray,
        y_true_val: np.ndarray,
        y_pred_val: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a model on training and validation data.
        
        Args:
            model_name: Name of the model.
            y_true_train: Actual training values.
            y_pred_train: Predicted training values.
            y_true_val: Actual validation values.
            y_pred_val: Predicted validation values.
            
        Returns:
            Dictionary with train and validation metrics.
        """
        logger.info(f"Evaluating {model_name}...")
        
        train_metrics = calculate_metrics(y_true_train, y_pred_train)
        val_metrics = calculate_metrics(y_true_val, y_pred_val)
        
        result = {
            'train': train_metrics,
            'validation': val_metrics
        }
        
        self.results[model_name] = result
        
        logger.info(f"{model_name} - Train RMSPE: {train_metrics['rmspe']:.4f}, "
                   f"Val RMSPE: {val_metrics['rmspe']:.4f}")
        
        return result
    
    def add_feature_importance(
        self,
        model_name: str,
        importance_df: pd.DataFrame
    ):
        """
        Store feature importance for a model.
        
        Args:
            model_name: Name of the model.
            importance_df: DataFrame with feature importance scores.
        """
        self.feature_importances[model_name] = importance_df
    
    def get_comparison_table(
        self,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate a comparison table for all evaluated models.
        
        Args:
            metrics: List of metrics to include. Defaults to RMSPE only.
            
        Returns:
            DataFrame with model comparison.
        """
        if not self.results:
            logger.warning("No models have been evaluated yet")
            return pd.DataFrame()
        
        metrics = metrics or ['rmspe']
        
        comparison_data = []
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            
            for metric in metrics:
                train_val = result['train'].get(metric, np.nan)
                val_val = result['validation'].get(metric, np.nan)
                
                row[f'Train_{metric.upper()}'] = train_val
                row[f'Val_{metric.upper()}'] = val_val
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by validation RMSPE (ascending - lower is better)
        if 'Val_RMSPE' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('Val_RMSPE')
        
        return comparison_df
    
    def print_comparison_table(
        self,
        metrics: Optional[List[str]] = None,
        tablefmt: str = 'pretty'
    ):
        """
        Print a formatted comparison table.
        
        Args:
            metrics: List of metrics to include.
            tablefmt: Table format for tabulate.
        """
        comparison_df = self.get_comparison_table(metrics)
        
        if comparison_df.empty:
            print("No models to compare.")
            return
        
        print("\n" + "=" * 60)
        print("MODEL COMPARISON RESULTS")
        print("=" * 60)
        
        # Format numeric columns
        for col in comparison_df.columns:
            if col != 'Model':
                comparison_df[col] = comparison_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
                )
        
        print(tabulate(comparison_df, headers='keys', tablefmt=tablefmt, showindex=False))
        print("=" * 60)
        
        # Identify best model
        best_model = comparison_df.iloc[0]['Model']
        print(f"\n🏆 Best Model (by Validation RMSPE): {best_model}")
    
    def get_best_model(self) -> str:
        """
        Get the name of the best performing model.
        
        Returns:
            Name of the best model based on validation RMSPE.
        """
        comparison_df = self.get_comparison_table(['rmspe'])
        if comparison_df.empty:
            return None
        return comparison_df.iloc[0]['Model']
    
    def get_feature_importance_report(
        self,
        model_name: Optional[str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance report for a model.
        
        Args:
            model_name: Name of the model. If None, uses best model.
            top_n: Number of top features to include.
            
        Returns:
            DataFrame with feature importance.
        """
        if model_name is None:
            model_name = self.get_best_model()
        
        if model_name not in self.feature_importances:
            logger.warning(f"No feature importance available for {model_name}")
            return pd.DataFrame()
        
        importance_df = self.feature_importances[model_name].head(top_n)
        return importance_df
    
    def print_feature_importance(
        self,
        model_name: Optional[str] = None,
        top_n: int = 15
    ):
        """
        Print feature importance for a model.
        
        Args:
            model_name: Name of the model. If None, uses best model.
            top_n: Number of top features to display.
        """
        if model_name is None:
            model_name = self.get_best_model()
        
        importance_df = self.get_feature_importance_report(model_name, top_n)
        
        if importance_df.empty:
            print(f"No feature importance available for {model_name}")
            return
        
        print(f"\n{'=' * 60}")
        print(f"FEATURE IMPORTANCE - {model_name}")
        print("=" * 60)
        
        # Normalize importance for display
        total_importance = importance_df['importance'].sum()
        importance_df = importance_df.copy()
        importance_df['importance_pct'] = (
            importance_df['importance'] / total_importance * 100
        )
        
        for idx, row in importance_df.iterrows():
            bar_length = int(row['importance_pct'] / 2)
            bar = "█" * bar_length
            print(f"{row['feature']:25s} {row['importance_pct']:6.2f}% {bar}")
        
        print("=" * 60)
    
    def generate_full_report(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Optional file path to save the report.
            
        Returns:
            Report as a string.
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("ROSSMANN SALES FORECASTING - MODEL EVALUATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Model comparison
        report_lines.append("1. MODEL COMPARISON")
        report_lines.append("-" * 40)
        
        comparison_df = self.get_comparison_table(['rmspe', 'rmse', 'mape', 'r2'])
        if not comparison_df.empty:
            for col in comparison_df.columns:
                if col != 'Model':
                    comparison_df[col] = comparison_df[col].apply(
                        lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
                    )
            report_lines.append(tabulate(
                comparison_df, 
                headers='keys', 
                tablefmt='grid',
                showindex=False
            ))
        
        report_lines.append("")
        
        # Best model
        best_model = self.get_best_model()
        if best_model:
            report_lines.append(f"2. BEST MODEL: {best_model}")
            report_lines.append("-" * 40)
            
            if best_model in self.results:
                val_metrics = self.results[best_model]['validation']
                for metric, value in val_metrics.items():
                    report_lines.append(f"   {metric.upper():10s}: {value:.4f}")
        
        report_lines.append("")
        
        # Feature importance
        if best_model and best_model in self.feature_importances:
            report_lines.append("3. TOP FEATURE IMPORTANCE")
            report_lines.append("-" * 40)
            
            importance_df = self.feature_importances[best_model].head(10)
            total = importance_df['importance'].sum()
            
            for _, row in importance_df.iterrows():
                pct = row['importance'] / total * 100
                report_lines.append(f"   {row['feature']:25s}: {pct:6.2f}%")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


def generate_submission_file(
    predictions: np.ndarray,
    test_ids: np.ndarray,
    output_path: str = "submission.csv"
) -> pd.DataFrame:
    """
    Generate a submission file for Kaggle.
    
    Args:
        predictions: Array of predicted sales values.
        test_ids: Array of test IDs.
        output_path: Path to save the submission file.
        
    Returns:
        Submission DataFrame.
    """
    submission = pd.DataFrame({
        'Id': test_ids,
        'Sales': predictions
    })
    
    # Ensure non-negative predictions
    submission['Sales'] = submission['Sales'].clip(lower=0)
    
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission file saved to {output_path}")
    
    return submission


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    y_true = np.random.randint(1000, 10000, 100)
    y_pred = y_true + np.random.randn(100) * 500
    
    # Calculate RMSPE
    rmspe_value = rmspe(y_true, y_pred)
    print(f"RMSPE: {rmspe_value:.4f}")
    
    # Calculate all metrics
    metrics = calculate_metrics(y_true, y_pred)
    print("\nAll Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test evaluator
    evaluator = ModelEvaluator()
    
    # Simulate multiple models
    for model_name in ['LinearRegression', 'RandomForest', 'XGBoost']:
        noise_scale = {'LinearRegression': 800, 'RandomForest': 400, 'XGBoost': 200}
        y_pred_train = y_true + np.random.randn(100) * noise_scale[model_name]
        y_pred_val = y_true + np.random.randn(100) * noise_scale[model_name] * 1.2
        
        evaluator.evaluate_model(
            model_name,
            y_true, y_pred_train,
            y_true, y_pred_val
        )
        
        # Add fake feature importance
        importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(10)],
            'importance': np.random.rand(10)
        }).sort_values('importance', ascending=False)
        evaluator.add_feature_importance(model_name, importance_df)
    
    # Print comparison
    evaluator.print_comparison_table()
    evaluator.print_feature_importance()
    
    # Generate report
    print(evaluator.generate_full_report())
