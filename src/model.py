import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import joblib
import os
import logging
from typing import Dict, List, Any, Optional, Tuple

try:
    from .exceptions import ModelError, ModelNotTrainedError, ModelLoadError, DataValidationError
    from .data_validators import validate_file_path
except ImportError:
    from exceptions import ModelError, ModelNotTrainedError, ModelLoadError, DataValidationError
    from data_validators import validate_file_path

from config.config import config

logger = logging.getLogger(__name__)

class HousingPriceModel:
    
    SUPPORTED_MODELS = {
        'xgboost': xgb.XGBRegressor,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'linear': LinearRegression
    }
    
    def __init__(self, model_type: str = None):
        self.model_type = model_type or config.model.model_type
        self.model = None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.training_metrics: Dict[str, float] = {}
        
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ModelError(f"Unsupported model type: {self.model_type}. "
                           f"Supported types: {list(self.SUPPORTED_MODELS.keys())}")
        
        logger.info(f"HousingPriceModel initialized with type: {self.model_type}")
        
    def create_model(self) -> None:
        try:
            if self.model_type == 'xgboost':
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=config.model.random_state,
                    n_jobs=-1,
                    verbosity=0  
                )
            elif self.model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=config.model.random_state,
                    n_jobs=-1
                )
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=config.model.random_state
                )
            elif self.model_type == 'linear':
                self.model = LinearRegression()
            
            logger.info(f"Model {self.model_type} created successfully")
            
        except Exception as e:
            raise ModelError(f"Failed to create model: {e}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: Optional[List[str]] = None) -> None:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Target variable
            feature_names: Feature names
            
        Raises:
            ModelError: On training error
            DataValidationError: On incorrect data
        """
        if X_train is None or y_train is None:
            raise DataValidationError("Training data cannot be None")
        
        if len(X_train) == 0 or len(y_train) == 0:
            raise DataValidationError("Training data cannot be empty")
        
        if len(X_train) != len(y_train):
            raise DataValidationError("X_train and y_train must have the same length")
        
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            raise DataValidationError("Training data contains NaN values")
        
        try:
            if self.model is None:
                self.create_model()
            
            if feature_names is not None:
                self.feature_names = feature_names.copy()
            
            logger.info(f"Starting training with {len(X_train)} samples, {X_train.shape[1]} features")
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            y_pred_train = self.model.predict(X_train)
            self.training_metrics = {
                'train_r2': r2_score(y_train, y_pred_train),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'train_mae': mean_absolute_error(y_train, y_pred_train)
            }
            
            logger.info(f"Model {self.model_type} trained successfully. "
                       f"Train R²: {self.training_metrics['train_r2']:.4f}")
            
        except Exception as e:
            self.is_trained = False
            raise ModelError(f"Training failed: {e}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict prices
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted prices
            
        Raises:
            ModelNotTrainedError: If model is not trained
            ModelError: On prediction error
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Model must be trained before prediction")
        
        if X is None:
            raise DataValidationError("Input data cannot be None")
        
        if len(X) == 0:
            raise DataValidationError("Input data cannot be empty")
        
        try:
            predictions = self.model.predict(X)
            
            if np.any(np.isnan(predictions)):
                raise ModelError("Model produced NaN predictions")
            
            if np.any(predictions < 0):
                logger.warning("Model produced negative predictions, clipping to 0")
                predictions = np.clip(predictions, 0, None)
            
            return predictions
            
        except Exception as e:
            if isinstance(e, (ModelNotTrainedError, DataValidationError)):
                raise
            raise ModelError(f"Prediction failed: {e}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Evaluate model quality
        
        Args:
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Tuple of (metrics, predictions)
            
        Raises:
            ModelNotTrainedError: If model is not trained
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Model must be trained before evaluation")
        
        try:
            y_pred = self.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            logger.info(f"Model evaluation completed. R²: {metrics['r2']:.4f}, "
                       f"RMSE: {metrics['rmse']:.2f}")
            
            return metrics, y_pred
            
        except Exception as e:
            if isinstance(e, ModelNotTrainedError):
                raise
            raise ModelError(f"Evaluation failed: {e}")
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = None) -> Dict[str, Any]:
        """
        Cross-validate the model
        
        Args:
            X: Features
            y: Target variable
            cv: Number of folds
            
        Returns:
            Cross-validation results
        """
        if cv is None:
            cv = config.model.cv_folds
        
        try:
            if self.model is None:
                self.create_model()
            
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
            
            results = {
                'mean_r2': scores.mean(),
                'std_r2': scores.std(),
                'scores': scores.tolist(),
                'cv_folds': cv
            }
            
            logger.info(f"Cross-validation completed. Mean R²: {results['mean_r2']:.4f} "
                       f"(±{results['std_r2']:.4f})")
            
            return results
            
        except Exception as e:
            raise ModelError(f"Cross-validation failed: {e}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance
        
        Returns:
            DataFrame with feature importance
            
        Raises:
            ModelNotTrainedError: If model is not trained
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Model must be trained to get feature importance")
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_)
            else:
                raise ModelError("Model does not support feature importance")
            
            if len(self.feature_names) == len(importance):
                feature_names = self.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
                logger.warning("Feature names length mismatch, using generic names")
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
            
        except Exception as e:
            if isinstance(e, ModelNotTrainedError):
                raise
            raise ModelError(f"Failed to get feature importance: {e}")
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                            param_grid: Optional[Dict] = None) -> None:
        """
        Hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Target variable
            param_grid: Parameter grid for search
        """
        try:
            if self.model is None:
                self.create_model()
            
            if param_grid is None:
                param_grid = self._get_default_param_grid()
            
            if not param_grid:
                logger.info("No hyperparameter tuning available for this model type")
                return
            
            logger.info(f"Starting hyperparameter tuning with {len(param_grid)} parameter combinations")
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=config.model.cv_folds, 
                scoring='r2', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best R²: {grid_search.best_score_:.4f}")
            
            self.model = grid_search.best_estimator_
            self.is_trained = True
            
        except Exception as e:
            raise ModelError(f"Hyperparameter tuning failed: {e}")
    
    def _get_default_param_grid(self) -> Dict:
        """Get default parameter grid"""
        if self.model_type == 'xgboost':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        else:
            return {}
    
    def save_model(self, filepath: str = None) -> None:
        """
        Save the model
        
        Args:
            filepath: Path to save
            
        Raises:
            ModelNotTrainedError: If model is not trained
            ModelError: On save error
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Cannot save untrained model")
        
        if filepath is None:
            filepath = config.get_model_path()
        
        try:
            validate_file_path(filepath)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            raise ModelError(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str = None) -> None:
        """
        Load the model
        
        Args:
            filepath: Path to model file
            
        Raises:
            ModelLoadError: On load error
        """
        if filepath is None:
            filepath = config.get_model_path()
        
        try:
            validate_file_path(filepath, must_exist=True)
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            self.training_metrics = model_data.get('training_metrics', {})
            
            logger.info(f"Model loaded from {filepath}")
            
        except FileNotFoundError:
            raise ModelLoadError(f"Model file not found: {filepath}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def predict_single(self, features_dict: Dict[str, Any]) -> float:
        """
        Predict for a single object
        
        Args:
            features_dict: Dictionary with features
            
        Returns:
            Predicted price
        """
        features_df = pd.DataFrame([features_dict])
        prediction = self.predict(features_df.values)[0]
        return float(prediction)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names.copy(),
            'training_metrics': self.training_metrics.copy()
        } 