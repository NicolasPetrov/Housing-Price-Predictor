"""
Tests for the model
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import HousingPriceModel
from exceptions import ModelError, ModelNotTrainedError, DataValidationError

class TestHousingPriceModel:
    """Tests for HousingPriceModel"""
    
    def test_model_initialization(self):
        """Model initialization test"""
        model = HousingPriceModel(model_type='xgboost')
        
        assert model.model_type == 'xgboost'
        assert model.model is None
        assert not model.is_trained
        assert model.feature_names == []
    
    def test_unsupported_model_type(self):
        """Initialization test with unsupported model type"""
        with pytest.raises(ModelError, match="Unsupported model type"):
            HousingPriceModel(model_type='unsupported_model')
    
    def test_create_model(self):
        """Model creation test"""
        model = HousingPriceModel(model_type='xgboost')
        model.create_model()
        
        assert model.model is not None
        assert hasattr(model.model, 'fit')
        assert hasattr(model.model, 'predict')
    
    def test_train_with_valid_data(self):
        """Test learning with correct data"""
        model = HousingPriceModel(model_type='linear')
        
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100) * 100000
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        model.train(X_train, y_train, feature_names)
        
        assert model.is_trained
        assert model.feature_names == feature_names
        assert len(model.training_metrics) > 0
    
    def test_train_with_invalid_data(self):
        """Test learning with incorrect data"""
        model = HousingPriceModel(model_type='linear')
        
        with pytest.raises(DataValidationError, match="Training data cannot be None"):
            model.train(None, None)
        
        with pytest.raises(DataValidationError, match="Training data cannot be empty"):
            model.train(np.array([]), np.array([]))
        
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(50)
        
        with pytest.raises(DataValidationError, match="must have the same length"):
            model.train(X_train, y_train)
    
    def test_predict_without_training(self):
        """Untrained prediction test"""
        model = HousingPriceModel(model_type='linear')
        X_test = np.random.rand(10, 5)
        
        with pytest.raises(ModelNotTrainedError, match="Model must be trained"):
            model.predict(X_test)
    
    def test_predict_with_trained_model(self):
        """Prediction test with trained model"""
        model = HousingPriceModel(model_type='linear')
        
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100) * 100000
        model.train(X_train, y_train)
        
        X_test = np.random.rand(10, 5)
        predictions = model.predict(X_test)
        
        assert len(predictions) == 10
        assert all(pred >= 0 for pred in predictions)
    
    def test_evaluate_model(self):
        """Model evaluation test"""
        model = HousingPriceModel(model_type='linear')
        
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100) * 100000
        model.train(X_train, y_train)
        
        X_test = np.random.rand(20, 5)
        y_test = np.random.rand(20) * 100000
        metrics, predictions = model.evaluate(X_test, y_test)
        
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert len(predictions) == 20
    
    def test_get_feature_importance(self):
        """Feature Importance Extraction Test"""
        model = HousingPriceModel(model_type='linear')
        
        with pytest.raises(ModelNotTrainedError):
            model.get_feature_importance()
        
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100) * 100000
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        model.train(X_train, y_train, feature_names)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == 5
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
    
    @patch('joblib.dump')
    @patch('os.makedirs')
    def test_save_model(self, mock_makedirs, mock_dump):
        """Model Preservation Test"""
        model = HousingPriceModel(model_type='linear')
        
        with pytest.raises(ModelNotTrainedError):
            model.save_model()
        
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100) * 1000000
        model.train(X_train, y_train)
        
        model.save_model('test_model.pkl')
        
        mock_makedirs.assert_called_once()
        mock_dump.assert_called_once()
    
    @patch('joblib.load')
    @patch('os.path.exists')
    def test_load_model(self, mock_exists, mock_load):
        """Model loading test"""
        model = HousingPriceModel(model_type='linear')
        
        mock_exists.return_value = True
        mock_model_data = {
            'model': MagicMock(),
            'model_type': 'linear',
            'feature_names': ['feature_1', 'feature_2'],
            'is_trained': True,
            'training_metrics': {'r2': 0.8}
        }
        mock_load.return_value = mock_model_data
        
        model.load_model('test_model.pkl')
        
        assert model.is_trained
        assert model.model_type == 'linear'
        assert model.feature_names == ['feature_1', 'feature_2']
        assert model.training_metrics == {'r2': 0.8}
    
    def test_get_model_info(self):
        """Model Information Retrieval Test"""
        model = HousingPriceModel(model_type='xgboost')
        
        info = model.get_model_info()
        
        assert 'model_type' in info
        assert 'is_trained' in info
        assert 'feature_count' in info
        assert 'feature_names' in info
        assert 'training_metrics' in info
        
        assert info['model_type'] == 'xgboost'
        assert not info['is_trained']
        assert info['feature_count'] == 0

if __name__ == "__main__":
    pytest.main([__file__])