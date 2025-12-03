"""
Models Module for Rossmann Sales Forecasting.

This module provides a class-based structure for defining and training
different regression models for sales prediction.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import joblib
from pathlib import Path

# Scikit-learn models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# XGBoost / LightGBM - defer imports to avoid load-time errors
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
xgb = None
lgb = None

def _check_xgboost():
    """Check if XGBoost is available and can be loaded."""
    global XGBOOST_AVAILABLE, xgb
    if xgb is not None:
        return XGBOOST_AVAILABLE
    try:
        import xgboost as _xgb
        xgb = _xgb
        XGBOOST_AVAILABLE = True
    except (ImportError, Exception) as e:
        XGBOOST_AVAILABLE = False
        logging.getLogger(__name__).warning(f"XGBoost not available: {e}")
    return XGBOOST_AVAILABLE

def _check_lightgbm():
    """Check if LightGBM is available and can be loaded."""
    global LIGHTGBM_AVAILABLE, lgb
    if lgb is not None:
        return LIGHTGBM_AVAILABLE
    try:
        import lightgbm as _lgb
        lgb = _lgb
        LIGHTGBM_AVAILABLE = True
    except (ImportError, Exception) as e:
        LIGHTGBM_AVAILABLE = False
        logging.getLogger(__name__).warning(f"LightGBM not available: {e}")
    return LIGHTGBM_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.
    
    Provides a common interface for model training, prediction, and evaluation.
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.
        
        Args:
            name: Name of the model for identification.
            params: Hyperparameters for the model.
        """
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_importances_ = None
        logger.info(f"Initialized {self.name} model")
    
    @abstractmethod
    def _create_model(self):
        """Create the underlying model instance. Must be implemented by subclasses."""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            
        Returns:
            Self for method chaining.
        """
        logger.info(f"Training {self.name} on {X.shape[0]} samples with {X.shape[1]} features...")
        
        # Create model if not exists
        if self.model is None:
            self.model = self._create_model()
        
        # Keep DataFrame to preserve feature names for tree-based models
        # This avoids warnings from LightGBM and XGBoost
        if isinstance(X, pd.DataFrame):
            X_train = X
            self._feature_names = X.columns.tolist()
        else:
            X_train = X
            self._feature_names = None
        
        y_train = y.values if isinstance(y, pd.Series) else y
        
        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_)
        
        logger.info(f"{self.name} training complete")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix (DataFrame or array).
            
        Returns:
            Array of predictions.
            
        Raises:
            ValueError: If model is not fitted.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model is not fitted. Call fit() first.")
        
        # Keep as DataFrame to preserve feature names (avoids LightGBM warning)
        # Only convert to values for models that don't support DataFrames
        if isinstance(X, pd.DataFrame):
            X_pred = X
        else:
            X_pred = X
        
        predictions = self.model.predict(X_pred)
        
        # Ensure non-negative predictions for sales
        predictions = np.clip(predictions, 0, None)
        
        return predictions
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names for labeling.
            
        Returns:
            DataFrame with feature importance scores.
        """
        if self.feature_importances_ is None:
            logger.warning(f"No feature importances available for {self.name}")
            return pd.DataFrame()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: File path to save the model.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str):
        """
        Load the model from disk.
        
        Args:
            path: File path to load the model from.
        """
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
    
    def __repr__(self) -> str:
        return f"{self.name}(params={self.params}, is_fitted={self.is_fitted})"


class LinearRegressionModel(BaseModel):
    """
    Linear Regression model for sales forecasting.
    
    Serves as a simple baseline model.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Linear Regression model.
        
        Args:
            params: Hyperparameters for LinearRegression.
        """
        default_params = {
            'fit_intercept': True,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        super().__init__(name="LinearRegression", params=default_params)
    
    def _create_model(self):
        """Create the LinearRegression model instance."""
        return LinearRegression(**self.params)


class RandomForestModel(BaseModel):
    """
    Random Forest Regressor for sales forecasting.
    
    Good for capturing non-linear relationships.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Random Forest model.
        
        Args:
            params: Hyperparameters for RandomForestRegressor.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'n_jobs': -1,
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__(name="RandomForest", params=default_params)
    
    def _create_model(self):
        """Create the RandomForestRegressor model instance."""
        return RandomForestRegressor(**self.params)


class XGBoostModel(BaseModel):
    """
    XGBoost Regressor for sales forecasting.
    
    State-of-the-art gradient boosting model.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.
        
        Args:
            params: Hyperparameters for XGBRegressor.
            
        Raises:
            ImportError: If XGBoost is not installed or can't be loaded.
        """
        if not _check_xgboost():
            raise ImportError(
                "XGBoost is not available. On macOS, run: brew install libomp && pip install xgboost"
            )
        
        default_params = {
            'n_estimators': 500,
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 1
        }
        if params:
            default_params.update(params)
        super().__init__(name="XGBoost", params=default_params)
    
    def _create_model(self):
        """Create the XGBRegressor model instance."""
        return xgb.XGBRegressor(**self.params)


class LightGBMModel(BaseModel):
    """
    LightGBM Regressor for sales forecasting.
    
    Efficient gradient boosting framework.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM model.
        
        Args:
            params: Hyperparameters for LGBMRegressor.
            
        Raises:
            ImportError: If LightGBM is not installed or can't be loaded.
        """
        if not _check_lightgbm():
            raise ImportError(
                "LightGBM is not available. Install with: pip install lightgbm"
            )
        
        default_params = {
            'n_estimators': 500,
            'max_depth': 10,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': -1
        }
        if params:
            default_params.update(params)
        super().__init__(name="LightGBM", params=default_params)
    
    def _create_model(self):
        """Create the LGBMRegressor model instance."""
        return lgb.LGBMRegressor(**self.params)


class ModelFactory:
    """
    Factory class to create model instances.
    
    Provides a unified interface for model creation.
    """
    
    _model_registry = {
        'linear_regression': LinearRegressionModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        params: Optional[Dict[str, Any]] = None
    ) -> BaseModel:
        """
        Create a model instance by type.
        
        Args:
            model_type: Type of model ('linear_regression', 'random_forest', 'xgboost', 'lightgbm').
            params: Hyperparameters for the model.
            
        Returns:
            Model instance.
            
        Raises:
            ValueError: If model type is not recognized.
        """
        model_type = model_type.lower()
        
        if model_type not in cls._model_registry:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {list(cls._model_registry.keys())}"
            )
        
        return cls._model_registry[model_type](params)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List of available model type names.
        """
        available = []
        for name, model_class in cls._model_registry.items():
            try:
                # Check if dependencies are available using deferred import checks
                if name == 'xgboost' and not _check_xgboost():
                    continue
                if name == 'lightgbm' and not _check_lightgbm():
                    continue
                available.append(name)
            except ImportError:
                pass
        return available
    
    @classmethod
    def create_all_models(
        cls,
        custom_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[BaseModel]:
        """
        Create instances of all available models.
        
        Args:
            custom_params: Dictionary mapping model type to custom parameters.
            
        Returns:
            List of model instances.
        """
        custom_params = custom_params or {}
        models = []
        
        for model_type in cls.get_available_models():
            params = custom_params.get(model_type, None)
            try:
                model = cls.create_model(model_type, params)
                models.append(model)
            except ImportError as e:
                logger.warning(f"Could not create {model_type}: {e}")
        
        return models


if __name__ == "__main__":
    # Example usage
    print("Available models:", ModelFactory.get_available_models())
    
    # Create sample data
    np.random.seed(42)
    X_sample = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    y_sample = pd.Series(np.random.randint(1000, 10000, 100))
    
    # Test each model
    for model_type in ModelFactory.get_available_models():
        print(f"\nTesting {model_type}...")
        model = ModelFactory.create_model(model_type)
        model.fit(X_sample, y_sample)
        predictions = model.predict(X_sample)
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
