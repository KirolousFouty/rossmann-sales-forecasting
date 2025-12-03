"""
Data Loader Module for Rossmann Sales Forecasting.

This module provides functionality to load raw CSV files and merge
store data with training and test datasets.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class to load and merge Rossmann store sales data.
    
    Attributes:
        data_dir (Path): Directory containing the data files.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataLoader with the data directory.
        
        Args:
            data_dir: Path to the directory containing CSV files.
        """
        self.data_dir = Path(data_dir)
        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")
    
    def load_train_data(self) -> pd.DataFrame:
        """
        Load the training data from train.csv.
        
        Returns:
            DataFrame containing training data.
            
        Raises:
            FileNotFoundError: If train.csv is not found.
        """
        train_path = self.data_dir / "train.csv"
        logger.info(f"Loading training data from {train_path}")
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        
        # Specify dtypes to avoid mixed type warnings (StateHoliday has 0 and 'a','b','c')
        dtype_spec = {
            'Store': 'int32',
            'DayOfWeek': 'int8',
            'Sales': 'int32',
            'Customers': 'int32',
            'Open': 'int8',
            'Promo': 'int8',
            'StateHoliday': 'str',  # Mixed: 0, 'a', 'b', 'c'
            'SchoolHoliday': 'int8'
        }
        
        train_df = pd.read_csv(
            train_path,
            parse_dates=['Date'],
            dtype=dtype_spec,
            low_memory=False
        )
        logger.info(f"Loaded training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
        return train_df
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load the test data from test.csv.
        
        Returns:
            DataFrame containing test data.
            
        Raises:
            FileNotFoundError: If test.csv is not found.
        """
        test_path = self.data_dir / "test.csv"
        logger.info(f"Loading test data from {test_path}")
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        # Specify dtypes to avoid mixed type warnings
        dtype_spec = {
            'Id': 'int32',
            'Store': 'int32',
            'DayOfWeek': 'int8',
            'Open': 'float32',  # Has NaN values
            'Promo': 'int8',
            'StateHoliday': 'str',
            'SchoolHoliday': 'int8'
        }
        
        test_df = pd.read_csv(
            test_path,
            parse_dates=['Date'],
            dtype=dtype_spec,
            low_memory=False
        )
        logger.info(f"Loaded test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
        return test_df
    
    def load_store_data(self) -> pd.DataFrame:
        """
        Load the store metadata from store.csv.
        
        Returns:
            DataFrame containing store metadata.
            
        Raises:
            FileNotFoundError: If store.csv is not found.
        """
        store_path = self.data_dir / "store.csv"
        logger.info(f"Loading store data from {store_path}")
        
        if not store_path.exists():
            raise FileNotFoundError(f"Store file not found: {store_path}")
        
        store_df = pd.read_csv(store_path, low_memory=False)
        logger.info(f"Loaded store data: {store_df.shape[0]} rows, {store_df.shape[1]} columns")
        return store_df
    
    def merge_with_store(
        self,
        df: pd.DataFrame,
        store_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge a dataset with store metadata.
        
        Args:
            df: DataFrame (train or test) to merge.
            store_df: Store metadata DataFrame.
            
        Returns:
            Merged DataFrame.
        """
        logger.info("Merging dataset with store data...")
        merged_df = pd.merge(df, store_df, on='Store', how='left')
        logger.info(f"Merged dataset shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        return merged_df
    
    def load_and_merge_train(self) -> pd.DataFrame:
        """
        Load training data and merge with store metadata.
        
        Returns:
            Merged training DataFrame.
        """
        train_df = self.load_train_data()
        store_df = self.load_store_data()
        return self.merge_with_store(train_df, store_df)
    
    def load_and_merge_test(self) -> pd.DataFrame:
        """
        Load test data and merge with store metadata.
        
        Returns:
            Merged test DataFrame.
        """
        test_df = self.load_test_data()
        store_df = self.load_store_data()
        return self.merge_with_store(test_df, store_df)
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets: train, test, and store.
        
        Returns:
            Tuple of (train_df, test_df, store_df).
        """
        train_df = self.load_train_data()
        test_df = self.load_test_data()
        store_df = self.load_store_data()
        return train_df, test_df, store_df
    
    def load_and_merge_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and merge all data.
        
        Returns:
            Tuple of (merged_train_df, merged_test_df).
        """
        train_df, test_df, store_df = self.load_all_data()
        merged_train = self.merge_with_store(train_df, store_df)
        merged_test = self.merge_with_store(test_df, store_df)
        return merged_train, merged_test


def get_data_summary(df: pd.DataFrame, name: str = "Dataset") -> dict:
    """
    Generate a summary of the dataset.
    
    Args:
        df: DataFrame to summarize.
        name: Name of the dataset for display.
        
    Returns:
        Dictionary containing dataset summary.
    """
    summary = {
        'name': name,
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'column_types': df.dtypes.value_counts().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    logger.info(f"\n{name} Summary:")
    logger.info(f"  - Rows: {summary['rows']:,}")
    logger.info(f"  - Columns: {summary['columns']}")
    logger.info(f"  - Missing Values: {summary['missing_values']:,} ({summary['missing_percentage']:.2f}%)")
    logger.info(f"  - Memory Usage: {summary['memory_usage_mb']:.2f} MB")
    
    return summary


if __name__ == "__main__":
    # Example usage
    loader = DataLoader(data_dir="data")
    
    try:
        merged_train, merged_test = loader.load_and_merge_all()
        get_data_summary(merged_train, "Merged Training Data")
        get_data_summary(merged_test, "Merged Test Data")
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
