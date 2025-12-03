"""
Preprocessing Module for Rossmann Sales Forecasting.

This module provides functionality for data cleaning, handling missing values,
feature engineering, and preparing data for model training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class to preprocess Rossmann sales data.
    
    Handles data cleaning, missing value imputation, and feature engineering.
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor with label encoders."""
        self.label_encoders = {}
        self.feature_columns = []
        self.categorical_columns = ['StateHoliday', 'StoreType', 'Assortment']
        logger.info("DataPreprocessor initialized")
    
    def clean_data(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Clean the dataset by handling special cases and filtering.
        
        Args:
            df: DataFrame to clean.
            is_train: Whether this is training data (affects Sales handling).
            
        Returns:
            Cleaned DataFrame.
        """
        logger.info("Starting data cleaning...")
        df = df.copy()
        
        # Handle Open column: When store is closed (Open=0), Sales should be 0
        # For training, we can filter out closed stores or keep them
        if is_train:
            # Filter out rows where store is closed (Sales would be 0)
            # This helps the model focus on predicting actual sales
            initial_rows = len(df)
            df = df[df['Open'] == 1]
            logger.info(f"Filtered closed store days: {initial_rows - len(df)} rows removed")
            
            # Also filter out rows with 0 sales (anomalies when store was open)
            df = df[df['Sales'] > 0]
            logger.info(f"Final training rows after cleaning: {len(df)}")
        else:
            # For test data, fill NaN in Open column with 1 (assume open)
            if 'Open' in df.columns and df['Open'].isnull().any():
                df['Open'] = df['Open'].fillna(1)
                logger.info("Filled missing 'Open' values with 1 for test data")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values.
            
        Returns:
            DataFrame with imputed missing values.
        """
        logger.info("Handling missing values...")
        df = df.copy()
        
        # Competition Distance: Fill with a large number (no nearby competition)
        if 'CompetitionDistance' in df.columns:
            median_dist = df['CompetitionDistance'].median()
            df['CompetitionDistance'] = df['CompetitionDistance'].fillna(median_dist)
            logger.info(f"Filled CompetitionDistance NaN with median: {median_dist:.2f}")
        
        # Competition Open Since Month/Year: Fill with 0 (no competition info)
        if 'CompetitionOpenSinceMonth' in df.columns:
            df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
        
        if 'CompetitionOpenSinceYear' in df.columns:
            df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
        
        # Promo2 Since Week/Year: Fill with 0 (no promo2 info)
        if 'Promo2SinceWeek' in df.columns:
            df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
        
        if 'Promo2SinceYear' in df.columns:
            df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
        
        # PromoInterval: Fill with empty string
        if 'PromoInterval' in df.columns:
            df['PromoInterval'] = df['PromoInterval'].fillna('')
        
        # Log remaining missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            logger.warning(f"Remaining missing values:\n{missing}")
        else:
            logger.info("All missing values handled successfully")
        
        return df
    
    def extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract date-based features from the Date column.
        
        Args:
            df: DataFrame with Date column.
            
        Returns:
            DataFrame with additional date features.
        """
        logger.info("Extracting date features...")
        df = df.copy()
        
        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract date components
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        # Note: Don't overwrite DayOfWeek - the original data has it as 1-7 (Mon-Sun)
        # which is more meaningful than pandas' 0-6
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
        
        # Additional useful date features
        df['DayOfYear'] = df['Date'].dt.dayofyear
        # Use original DayOfWeek (1-7) for IsWeekend calculation
        df['IsWeekend'] = (df['DayOfWeek'] >= 6).astype(int)  # 6=Sat, 7=Sun in original
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        
        logger.info("Date features extracted: Year, Month, Day, WeekOfYear, "
                   "DayOfYear, IsWeekend, IsMonthStart, IsMonthEnd")
        
        return df
    
    def calculate_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate competition-related features.
        
        Args:
            df: DataFrame with competition columns.
            
        Returns:
            DataFrame with competition features.
        """
        logger.info("Calculating competition features...")
        df = df.copy()
        
        # Calculate months since competition opened
        # Create a competition open date from year and month
        df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
        df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
        
        # Calculate time since competition started (in months)
        # Only calculate where we have valid competition data
        df['CompetitionMonthsOpen'] = 0
        
        mask = (df['CompetitionOpenSinceYear'] > 0) & (df['CompetitionOpenSinceMonth'] > 0)
        
        df.loc[mask, 'CompetitionMonthsOpen'] = (
            (df.loc[mask, 'Year'] - df.loc[mask, 'CompetitionOpenSinceYear']) * 12 +
            (df.loc[mask, 'Month'] - df.loc[mask, 'CompetitionOpenSinceMonth'])
        )
        
        # Ensure non-negative values
        df['CompetitionMonthsOpen'] = df['CompetitionMonthsOpen'].clip(lower=0)
        
        # Create binary feature: Has competition
        df['HasCompetition'] = (df['CompetitionDistance'] > 0).astype(int)
        
        logger.info("Competition features calculated: CompetitionMonthsOpen, HasCompetition")
        
        return df
    
    def calculate_promo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate promotion-related features.
        
        Args:
            df: DataFrame with promo columns.
            
        Returns:
            DataFrame with promo features.
        """
        logger.info("Calculating promo features...")
        df = df.copy()
        
        # Calculate weeks since Promo2 started
        df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
        df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
        
        df['Promo2WeeksActive'] = 0
        mask = (df['Promo2SinceYear'] > 0) & (df['Promo2SinceWeek'] > 0)
        
        df.loc[mask, 'Promo2WeeksActive'] = (
            (df.loc[mask, 'Year'] - df.loc[mask, 'Promo2SinceYear']) * 52 +
            (df.loc[mask, 'WeekOfYear'] - df.loc[mask, 'Promo2SinceWeek'])
        )
        
        # Ensure non-negative values
        df['Promo2WeeksActive'] = df['Promo2WeeksActive'].clip(lower=0)
        
        # Check if current month is in PromoInterval
        month_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        
        df['PromoInterval'] = df['PromoInterval'].fillna('')
        df['IsPromo2Month'] = 0
        
        for month_num, month_name in month_map.items():
            mask = (df['Month'] == month_num) & (df['PromoInterval'].str.contains(month_name, na=False))
            df.loc[mask, 'IsPromo2Month'] = 1
        
        logger.info("Promo features calculated: Promo2WeeksActive, IsPromo2Month")
        
        return df
    
    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df: DataFrame with categorical features.
            fit: Whether to fit the encoders (True for training data).
            
        Returns:
            DataFrame with encoded categorical features.
        """
        logger.info("Encoding categorical features...")
        df = df.copy()
        
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
                
            # Convert to string to handle mixed types
            df[col] = df[col].astype(str)
            
            if fit:
                # Fit and transform
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                logger.info(f"Fitted and transformed '{col}': {len(le.classes_)} classes")
            else:
                # Transform only
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen labels
                    df[col] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                    logger.info(f"Transformed '{col}' using existing encoder")
                else:
                    logger.warning(f"No encoder found for '{col}', skipping")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get the list of feature columns for model training.
        
        Returns:
            List of feature column names.
        """
        feature_cols = [
            # Store info
            'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday',
            
            # Encoded categoricals
            'StateHoliday', 'StoreType', 'Assortment',
            
            # Date features
            'Year', 'Month', 'Day', 'WeekOfYear', 'DayOfYear',
            'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
            
            # Store metadata
            'CompetitionDistance', 'Promo2',
            
            # Calculated features
            'CompetitionMonthsOpen', 'HasCompetition',
            'Promo2WeeksActive', 'IsPromo2Month'
        ]
        
        self.feature_columns = feature_cols
        return feature_cols
    
    def preprocess(
        self,
        df: pd.DataFrame,
        is_train: bool = True
    ) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to the data.
        
        Args:
            df: Raw DataFrame to preprocess.
            is_train: Whether this is training data.
            
        Returns:
            Preprocessed DataFrame ready for modeling.
        """
        logger.info(f"Starting preprocessing pipeline (is_train={is_train})...")
        
        # Step 1: Clean data
        df = self.clean_data(df, is_train=is_train)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Extract date features
        df = self.extract_date_features(df)
        
        # Step 4: Calculate competition features
        df = self.calculate_competition_features(df)
        
        # Step 5: Calculate promo features
        df = self.calculate_promo_features(df)
        
        # Step 6: Encode categorical features
        df = self.encode_categorical_features(df, fit=is_train)
        
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        
        return df


def create_time_based_split(
    df: pd.DataFrame,
    date_column: str = 'Date',
    train_ratio: float = 0.8,
    validation_weeks: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a time-based train/validation split.
    
    For time-series data, we should not use random splitting.
    Instead, we train on earlier dates and validate on later dates.
    
    Args:
        df: DataFrame with date column.
        date_column: Name of the date column.
        train_ratio: Ratio of data for training (alternative to validation_weeks).
        validation_weeks: Number of weeks to use for validation (takes precedence).
        
    Returns:
        Tuple of (train_df, validation_df).
    """
    logger.info("Creating time-based train/validation split...")
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df = df.sort_values(date_column)
    
    # Get date range
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    logger.info(f"Date range: {min_date.date()} to {max_date.date()}")
    
    # Calculate split date
    if validation_weeks is not None:
        # Use last N weeks for validation
        split_date = max_date - pd.Timedelta(weeks=validation_weeks)
        logger.info(f"Using last {validation_weeks} weeks for validation")
    else:
        # Use ratio-based split
        date_range = (max_date - min_date).days
        split_days = int(date_range * train_ratio)
        split_date = min_date + pd.Timedelta(days=split_days)
    
    logger.info(f"Split date: {split_date.date()}")
    
    # Split the data
    train_df = df[df[date_column] < split_date].copy()
    val_df = df[df[date_column] >= split_date].copy()
    
    logger.info(f"Training set: {len(train_df)} rows "
               f"({train_df[date_column].min().date()} to {train_df[date_column].max().date()})")
    logger.info(f"Validation set: {len(val_df)} rows "
               f"({val_df[date_column].min().date()} to {val_df[date_column].max().date()})")
    
    return train_df, val_df


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'Sales'
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Prepare feature matrix and target vector.
    
    Args:
        df: Preprocessed DataFrame.
        feature_columns: List of feature column names.
        target_column: Name of the target column.
        
    Returns:
        Tuple of (X features DataFrame, y target Series or None).
    """
    # Check which features are available
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    X = df[available_features].copy()
    
    # Get target if available
    y = None
    if target_column in df.columns:
        y = df[target_column].copy()
    
    logger.info(f"Features shape: {X.shape}, Target: {'available' if y is not None else 'not available'}")
    
    return X, y


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader(data_dir="data")
    
    try:
        merged_train, _ = loader.load_and_merge_all()
        
        # Preprocess
        preprocessor = DataPreprocessor()
        train_processed = preprocessor.preprocess(merged_train, is_train=True)
        
        # Create time-based split
        train_df, val_df = create_time_based_split(
            train_processed,
            validation_weeks=6
        )
        
        # Prepare features
        feature_cols = preprocessor.get_feature_columns()
        X_train, y_train = prepare_features_and_target(train_df, feature_cols)
        X_val, y_val = prepare_features_and_target(val_df, feature_cols)
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Validation features shape: {X_val.shape}")
        
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
