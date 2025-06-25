import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import joblib
import os

class HousingPriceModel:
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.model_path = 'models/housing_price_model.pkl'
        
    def create_model(self):
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, feature_names=None):
        if self.model is None:
            self.create_model()
        
        if feature_names is not None:
            self.feature_names = feature_names
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print(f"Model {self.model_type} trained successfully!")
        
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model is not trained. Run train() first")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        if not self.is_trained:
            raise ValueError("Model is not trained. Run train() first")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics, y_pred
    
    def cross_validate(self, X, y, cv=5):
        if self.model is None:
            self.create_model()
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        return {
            'mean_r2': scores.mean(),
            'std_r2': scores.std(),
            'scores': scores
        }
    
    def get_feature_importance(self):
        if not self.is_trained:
            raise ValueError("Model is not trained. Run train() first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            return None
        
        if len(self.feature_names) == len(importance):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None):
        if self.model is None:
            self.create_model()
        
        if param_grid is None:
            if self.model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2]
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                print("Hyperparameter tuning available only for XGBoost and Random Forest")
                return
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='r2', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best R²: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
    def save_model(self, filepath=None):
        if not self.is_trained:
            raise ValueError("Model is not trained. Nothing to save")
        
        if filepath is None:
            filepath = self.model_path
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        if filepath is None:
            filepath = self.model_path
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def predict_single(self, features_dict):
        if not self.is_trained:
            raise ValueError("Model is not trained. Run train() first")
        
        features_df = pd.DataFrame([features_dict])
        return self.predict(features_df)[0] 