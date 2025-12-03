"""
Script to generate comparison plots for the README.
Run this after training models to create visualization images.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = Path("output/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model results data
results_data = {
    'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'Linear Regression'],
    'Train_RMSPE': [0.1801, 0.2900, 0.3136, 0.5356],
    'Val_RMSPE': [0.1489, 0.2625, 0.2690, 0.5099],
    'Train_R2': [0.9522, 0.8047, 0.7424, 0.1932],
    'Val_R2': [0.8995, 0.7725, 0.6943, 0.2174],
    'Training_Time': [4, 4.5, 17, 0.1]
}

results_df = pd.DataFrame(results_data)

# Feature importance data (from XGBoost)
feature_importance = {
    'Feature': ['Promo', 'CompetitionDistance', 'Promo2', 'StoreType', 'Store', 
                'Assortment', 'CompetitionMonthsOpen', 'DayOfWeek', 'StateHoliday', 
                'IsMonthEnd', 'IsWeekend', 'Promo2WeeksActive', 'DayOfYear', 
                'IsMonthStart', 'Day'],
    'Importance': [25.30, 14.11, 13.96, 10.97, 10.50, 9.79, 4.08, 3.99, 3.65, 
                   3.64, 2.78, 2.19, 2.16, 2.05, 1.95]
}
feature_df = pd.DataFrame(feature_importance)


def plot_model_comparison_rmspe():
    """Create a grouped bar chart comparing RMSPE across models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results_df['Model']))
    width = 0.35
    
    colors = ['#2ecc71', '#e74c3c']  # Green for train, red for validation
    
    bars1 = ax.bar(x - width/2, results_df['Train_RMSPE'] * 100, width, 
                   label='Training RMSPE', color=colors[0], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, results_df['Val_RMSPE'] * 100, width, 
                   label='Validation RMSPE', color=colors[1], edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSPE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Training vs Validation RMSPE', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 60)
    
    # Add a horizontal line at the best validation RMSPE
    ax.axhline(y=14.89, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7, label='Best: 14.89%')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison_rmspe.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'model_comparison_rmspe.png'}")


def plot_validation_rmspe_horizontal():
    """Create a horizontal bar chart of validation RMSPE."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Sort by validation RMSPE
    sorted_df = results_df.sort_values('Val_RMSPE', ascending=True)
    
    # Color gradient based on performance
    colors = ['#27ae60', '#f39c12', '#e67e22', '#c0392b']
    
    bars = ax.barh(sorted_df['Model'], sorted_df['Val_RMSPE'] * 100, 
                   color=colors, edgecolor='black', linewidth=0.5, height=0.6)
    
    # Add value labels
    for bar, val in zip(bars, sorted_df['Val_RMSPE'] * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', 
                va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Validation RMSPE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Ranking (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 60)
    
    # Add winner badge
    ax.annotate('🏆 Best', xy=(14.89, 0), xytext=(25, 0),
                fontsize=12, fontweight='bold', color='#27ae60',
                va='center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'validation_rmspe_ranking.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'validation_rmspe_ranking.png'}")


def plot_feature_importance():
    """Create a horizontal bar chart of feature importance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Take top 15 features
    top_features = feature_df.head(15).iloc[::-1]  # Reverse for horizontal bar
    
    # Color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))[::-1]
    
    bars = ax.barh(top_features['Feature'], top_features['Importance'], 
                   color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, top_features['Importance']):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('XGBoost Feature Importance: Top 15 Features', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 30)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'feature_importance.png'}")


def plot_r2_comparison():
    """Create a comparison chart of R² scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results_df['Model']))
    width = 0.35
    
    colors = ['#3498db', '#9b59b6']
    
    bars1 = ax.bar(x - width/2, results_df['Train_R2'] * 100, width, 
                   label='Training R²', color=colors[0], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, results_df['Val_R2'] * 100, width, 
                   label='Validation R²', color=colors[1], edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: R² Score (Variance Explained)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'r2_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'r2_comparison.png'}")


def plot_training_time():
    """Create a bar chart of training times."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c']
    
    bars = ax.bar(results_df['Model'], results_df['Training_Time'], 
                  color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, results_df['Training_Time']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val}s', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_time.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'training_time.png'}")


def plot_metrics_radar():
    """Create a radar/spider chart comparing models across metrics."""
    # Normalize metrics to 0-1 scale (inverted for RMSPE so higher is better)
    metrics = ['RMSPE\n(inverted)', 'R² Score', 'Speed\n(inverted)']
    
    # Normalize: lower RMSPE is better, so invert
    rmspe_norm = 1 - (results_df['Val_RMSPE'] / results_df['Val_RMSPE'].max())
    r2_norm = results_df['Val_R2']
    speed_norm = 1 - (results_df['Training_Time'] / results_df['Training_Time'].max())
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#27ae60', '#f39c12', '#3498db', '#e74c3c']
    
    for i, (model, color) in enumerate(zip(results_df['Model'], colors)):
        values = [rmspe_norm.iloc[i], r2_norm.iloc[i], speed_norm.iloc[i]]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison Radar Chart\n(Higher is Better)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_radar.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'model_radar.png'}")


def plot_feature_category_pie():
    """Create a pie chart showing feature importance by category."""
    categories = {
        'Promotions': 25.30 + 13.96 + 2.19,  # Promo + Promo2 + Promo2WeeksActive
        'Store Characteristics': 10.97 + 10.50 + 9.79,  # StoreType + Store + Assortment
        'Competition': 14.11 + 4.08,  # CompetitionDistance + CompetitionMonthsOpen
        'Time/Date': 3.99 + 3.65 + 3.64 + 2.78 + 2.16 + 2.05 + 1.95  # DayOfWeek + StateHoliday + IsMonthEnd + IsWeekend + DayOfYear + IsMonthStart + Day
    }
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
    explode = (0.05, 0, 0, 0)  # Explode the largest slice
    
    wedges, texts, autotexts = ax.pie(
        categories.values(), 
        labels=categories.keys(),
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1}
    )
    
    # Style the text
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax.set_title('Feature Importance by Category', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_categories.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'feature_categories.png'}")


if __name__ == "__main__":
    print("Generating comparison plots...\n")
    
    plot_model_comparison_rmspe()
    plot_validation_rmspe_horizontal()
    plot_feature_importance()
    plot_r2_comparison()
    plot_training_time()
    plot_metrics_radar()
    plot_feature_category_pie()
    
    print(f"\n✅ All plots saved to: {OUTPUT_DIR.absolute()}")
