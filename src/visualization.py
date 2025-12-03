"""
Visualization Module for Rossmann Sales Forecasting.

This module provides functions to create feature importance plots
and other visualizations for model analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a horizontal bar plot of feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns.
        title: Plot title.
        top_n: Number of top features to display.
        figsize: Figure size as (width, height).
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib Figure object.
    """
    # Get top N features
    plot_df = importance_df.head(top_n).copy()
    
    # Normalize importance to percentage
    total = plot_df['importance'].sum()
    plot_df['importance_pct'] = plot_df['importance'] / total * 100
    
    # Reverse order for horizontal bar plot (highest at top)
    plot_df = plot_df.iloc[::-1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    colors = sns.color_palette("viridis", n_colors=len(plot_df))
    bars = ax.barh(
        plot_df['feature'],
        plot_df['importance_pct'],
        color=colors[::-1]
    )
    
    # Add value labels
    for bar, val in zip(bars, plot_df['importance_pct']):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%',
            va='center',
            fontsize=9
        )
    
    # Formatting
    ax.set_xlabel('Importance (%)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, plot_df['importance_pct'].max() * 1.15)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'RMSPE',
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a grouped bar plot comparing model performance.
    
    Args:
        comparison_df: DataFrame with model comparison data.
        metric: Metric to plot (e.g., 'RMSPE').
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib Figure object.
    """
    train_col = f'Train_{metric}'
    val_col = f'Val_{metric}'
    
    if train_col not in comparison_df.columns:
        logger.warning(f"Column {train_col} not found in comparison data")
        return None
    
    # Prepare data
    models = comparison_df['Model'].tolist()
    
    # Convert to numeric if needed
    train_scores = pd.to_numeric(comparison_df[train_col], errors='coerce')
    val_scores = pd.to_numeric(comparison_df[val_col], errors='coerce')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(models))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, train_scores, width, label='Train', color='steelblue')
    bars2 = ax.bar(x + width/2, val_scores, width, label='Validation', color='coral')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9
            )
    
    # Formatting
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.legend()
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    return fig


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    sample_size: int = 5000,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a scatter plot of predictions vs actual values.
    
    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        model_name: Name of the model for the title.
        sample_size: Number of points to sample for visualization.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib Figure object.
    """
    # Sample data if too large
    if len(y_true) > sample_size:
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true_sample, y_pred_sample, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(y_true_sample.min(), y_pred_sample.min())
    max_val = max(y_true_sample.max(), y_pred_sample.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Formatting
    ax.set_xlabel('Actual Sales', fontsize=12)
    ax.set_ylabel('Predicted Sales', fontsize=12)
    ax.set_title(f'{model_name}: Predictions vs Actual', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction plot saved to {save_path}")
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create residual plots for model diagnostics.
    
    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        model_name: Name of the model.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib Figure object.
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.3, s=5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Sales')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    
    # Residuals distribution
    ax2 = axes[1]
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution')
    
    fig.suptitle(f'{model_name} - Residual Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Residuals plot saved to {save_path}")
    
    return fig


def plot_sales_distribution(
    train_sales: pd.Series,
    val_sales: Optional[pd.Series] = None,
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of sales in training and validation sets.
    
    Args:
        train_sales: Training sales values.
        val_sales: Optional validation sales values.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        
    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training distribution
    ax.hist(train_sales, bins=50, alpha=0.7, label='Training', edgecolor='black')
    
    # Plot validation distribution if provided
    if val_sales is not None:
        ax.hist(val_sales, bins=50, alpha=0.7, label='Validation', edgecolor='black')
        ax.legend()
    
    ax.set_xlabel('Sales')
    ax.set_ylabel('Frequency')
    ax.set_title('Sales Distribution')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sales distribution plot saved to {save_path}")
    
    return fig


def generate_all_plots(
    evaluator,
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_val: np.ndarray,
    y_pred_val: np.ndarray,
    output_dir: str = "output/plots"
):
    """
    Generate all visualization plots and save them.
    
    Args:
        evaluator: ModelEvaluator instance with results.
        y_train: Actual training values.
        y_pred_train: Predicted training values.
        y_val: Actual validation values.
        y_pred_val: Predicted validation values.
        output_dir: Directory to save plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Best model name
    best_model = evaluator.get_best_model()
    
    # Feature importance
    if best_model in evaluator.feature_importances:
        plot_feature_importance(
            evaluator.feature_importances[best_model],
            title=f"{best_model} - Feature Importance",
            save_path=str(output_path / "feature_importance.png")
        )
    
    # Model comparison
    comparison_df = evaluator.get_comparison_table()
    if not comparison_df.empty:
        plot_model_comparison(
            comparison_df,
            save_path=str(output_path / "model_comparison.png")
        )
    
    # Predictions vs Actual (for best model)
    plot_prediction_vs_actual(
        y_val,
        y_pred_val,
        model_name=best_model,
        save_path=str(output_path / "predictions_vs_actual.png")
    )
    
    # Residuals
    plot_residuals(
        y_val,
        y_pred_val,
        model_name=best_model,
        save_path=str(output_path / "residuals.png")
    )
    
    logger.info(f"All plots saved to {output_path}")
    
    plt.close('all')


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate sample feature importance
    importance_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(20)],
        'importance': np.random.rand(20)
    }).sort_values('importance', ascending=False)
    
    # Create sample plot
    fig = plot_feature_importance(
        importance_df,
        title="Sample Feature Importance",
        top_n=15
    )
    plt.show()
