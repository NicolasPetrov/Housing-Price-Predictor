"""
Data processing tests
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataProcessor, SafeLabelEncoder
from exceptions import DataProcessingError, DataValidationError

class TestSafeLabelEncoder:
    """Tests for SafeLabelEncoder"""
    
    def test_fit_transform(self):
        """Learning and Conversion Test"""
        encoder = SafeLabelEncoder()
        data = ['A', 'B', 'C', 'A', 'B']
        
        encoded = encoder.fit_transform(data)
        
        assert encoder.is_fitted
        assert len(encoded) == 5
        assert all(isinstance(x, (int, np.integer)) for x in encoded)
    
    def test_transform_unknown_values(self):
        """Transformation test with unknown values"""
        encoder = SafeLabelEncoder()
        
        encoder.fit(['A', 'B', 'C'])
        
        result = encoder.transform(['A', 'D', 'B', 'E'])
        
        assert result[1] == encoder.unknown_value
        assert result[3] == encoder.unknown_value
    
    def test_inverse_transform(self):
        """Inverse transform test"""
        encoder = SafeLabelEncoder()
        data = ['A', 'B', 'C']
        
        encoded = encoder.fit_transform(data)
        decoded = encoder.inverse_transform(encoded)
        
        assert list(decoded) == data

class TestDataProcessor:
    """Tests for DataProcessor"""
    
    def test_initialization(self):
        """Initialization test"""
        processor = DataProcessor()
        
        assert not processor.is_fitted
        assert len(processor.feature_names) == 0
        assert len(processor.label_encoders) == 0
        assert len(processor.imputers) == 0
    
    def test_generate_sample_data_valid(self):
        """Test of generating correct data"""
        processor = DataProcessor()
        
        df = processor.generate_sample_data(n_samples=100)
        
        assert len(df) == 100
        assert 'price' in df.columns
        assert 'total_area' in df.columns
        assert 'district' in df.columns
        
        assert df['total_area'].min() >= 20
        assert df['total_area'].max() <= 200
        assert df['rooms'].min() >= 1
        assert df['rooms'].max() <= 5
    
    def test_generate_sample_data_invalid_size(self):
        """Test generation with incorrect size"""
        processor = DataProcessor()
        
        with pytest.raises(DataValidationError, match="n_samples must be positive"):
            processor.generate_sample_data(n_samples=0)
        
        with pytest.raises(DataValidationError, match="n_samples must be positive"):
            processor.generate_sample_data(n_samples=-10)
    
    def test_preprocess_data_fit_mode(self):
        """Preprocessing test in training mode"""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            'total_area': [75.0, 80.0, 90.0],
            'living_area': [52.5, 56.0, 63.0],
            'kitchen_area': [12.0, 10.0, 15.0],
            'rooms': [2, 3, 2],
            'floor': [5, 7, 3],
            'total_floors': [12, 15, 9],
            'year_built': [2010, 2015, 2005],
            'ceiling_height': [2.7, 2.8, 2.6],
            'building_prestige': ['standard', 'elite', 'economy'],
            'district': ['Central', 'Northern', 'Southern'],
            'metro_distance': [1.5, 2.0, 3.0],
            'parking': [1, 0, 1],
            'elevator': [1, 1, 0],
            'renovation': ['European renovation', 'Cosmetic', 'No renovation'],
            'house_type': ['Monolithic', 'Brick', 'Panel'],
            'bathrooms_count': [1, 2, 1],
            'bathroom_type': ['separate', 'combined', 'separate'],
            'balcony_type': ['loggia', 'balcony', 'none'],
            'balcony_count': [1, 1, 0],
            'balcony_glazed': ['yes', 'no', 'no'],
            'balcony_area': [6.0, 4.0, 0.0],
            'air_quality_index': [70, 65, 80],
            'noise_level': [40, 50, 35],
            'green_zone_distance': [1.0, 2.0, 0.5],
            'water_body_distance': [3.0, 5.0, 1.0],
            'industrial_distance': [8.0, 6.0, 10.0],
            'crime_rate': [20, 30, 15],
            'avg_income_index': [120, 100, 90],
            'population_density': [4000, 6000, 3000],
            'elderly_ratio': [0.2, 0.3, 0.25],
            'hospital_distance': [1.5, 2.5, 1.0],
            'clinics_count_3km': [3, 2, 4],
            'pharmacy_distance': [0.3, 0.8, 0.5],
            'emergency_time': [10, 15, 8],
            'supermarket_distance': [0.5, 1.2, 0.3],
            'shops_count_1km': [20, 10, 25],
            'mall_distance': [2.0, 4.0, 1.5],
            'market_distance': [1.0, 2.0, 0.8],
            'services_count_1km': [8, 5, 12],
            'useful_area_ratio': [0.7, 0.7, 0.7],
            'price': [50000, 60000, 40000]
        })
        
        df_processed = processor.preprocess_data(df, fit=True)
        
        assert len(processor.label_encoders) > 0
        assert len(processor.imputers) > 0
        assert len(df_processed) == 3
        
        X, y, feature_names = processor.prepare_features(df_processed)
        assert processor.is_fitted
    
    def test_preprocess_data_transform_mode(self):
        """Preprocessing test in transform mode"""
        processor = DataProcessor()
        
        train_df = pd.DataFrame({
            'total_area': [75.0, 80.0],
            'living_area': [52.5, 56.0],
            'kitchen_area': [12.0, 10.0],
            'rooms': [2, 3],
            'floor': [5, 7],
            'total_floors': [12, 15],
            'year_built': [2010, 2015],
            'ceiling_height': [2.7, 2.8],
            'building_prestige': ['standard', 'elite'],
            'district': ['Central', 'Northern'],
            'metro_distance': [1.5, 2.0],
            'parking': [1, 0],
            'elevator': [1, 1],
            'renovation': ['European renovation', 'Cosmetic'],
            'house_type': ['Monolithic', 'Brick'],
            'bathrooms_count': [1, 2],
            'bathroom_type': ['separate', 'combined'],
            'balcony_type': ['loggia', 'balcony'],
            'balcony_count': [1, 1],
            'balcony_glazed': ['yes', 'no'],
            'balcony_area': [6.0, 4.0],
            'air_quality_index': [70, 65],
            'noise_level': [40, 50],
            'green_zone_distance': [1.0, 2.0],
            'water_body_distance': [3.0, 5.0],
            'industrial_distance': [8.0, 6.0],
            'crime_rate': [20, 30],
            'avg_income_index': [120, 100],
            'population_density': [4000, 6000],
            'elderly_ratio': [0.2, 0.3],
            'hospital_distance': [1.5, 2.5],
            'clinics_count_3km': [3, 2],
            'pharmacy_distance': [0.3, 0.8],
            'emergency_time': [10, 15],
            'supermarket_distance': [0.5, 1.2],
            'shops_count_1km': [20, 10],
            'mall_distance': [2.0, 4.0],
            'market_distance': [1.0, 2.0],
            'services_count_1km': [8, 5],
            'useful_area_ratio': [0.7, 0.7],
            'price': [50000, 60000]
        })
        
        processor.preprocess_data(train_df, fit=True)
        
        test_df = pd.DataFrame({
            'total_area': [85.0],
            'living_area': [59.5],
            'kitchen_area': [11.0],
            'rooms': [2],
            'floor': [6],
            'total_floors': [14],
            'year_built': [2012],
            'ceiling_height': [2.7],
            'building_prestige': ['standard'],
            'district': ['Central'],
            'metro_distance': [1.8],
            'parking': [1],
            'elevator': [1],
            'renovation': ['European renovation'],
            'house_type': ['Monolithic'],
            'bathrooms_count': [1],
            'bathroom_type': ['separate'],
            'balcony_type': ['none'],
            'balcony_count': [0],
            'balcony_glazed': ['no'],
            'balcony_area': [0.0],
            'air_quality_index': [75],
            'noise_level': [35],
            'green_zone_distance': [0.8],
            'water_body_distance': [2.5],
            'industrial_distance': [9.0],
            'crime_rate': [18],
            'avg_income_index': [110],
            'population_density': [4500],
            'elderly_ratio': [0.22],
            'hospital_distance': [1.2],
            'clinics_count_3km': [4],
            'pharmacy_distance': [0.4],
            'emergency_time': [9],
            'supermarket_distance': [0.6],
            'shops_count_1km': [18],
            'mall_distance': [2.5],
            'market_distance': [1.2],
            'services_count_1km': [10],
            'useful_area_ratio': [0.7],
            'house_type': ['Monolithic']
        })
        
        df_processed = processor.preprocess_data(test_df, fit=False)
        
        assert len(df_processed) == 1
    
    def test_prepare_features(self):
        """Feature Preparation Test"""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'target': [10.0, 20.0, 30.0]
        })
        
        X, y, feature_names = processor.prepare_features(df, target_column='target')
        
        assert X.shape == (3, 2)
        assert len(y) == 3
        assert feature_names == ['feature1', 'feature2']
        assert processor.is_fitted
    
    def test_prepare_features_without_target(self):
        """Feature preparation test without target variable"""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
        
        X, y, feature_names = processor.prepare_features(df, target_column='target')
        
        assert X.shape == (3, 2)
        assert y is None
        assert feature_names == ['feature1', 'feature2']
    
    def test_split_data(self):
        """Data partitioning test"""
        processor = DataProcessor()
        
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2, random_state=42)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    @patch('joblib.dump')
    @patch('os.makedirs')
    def test_save_preprocessor(self, mock_makedirs, mock_dump):
        """Preprocessor Saving Test"""
        processor = DataProcessor()
        processor.is_fitted = True
        
        processor.save_preprocessor('test_preprocessor.pkl')
        
        mock_makedirs.assert_called_once()
        mock_dump.assert_called_once()
    
    @patch('joblib.load')
    @patch('os.path.exists')
    def test_load_preprocessor(self, mock_exists, mock_load):
        """Preprocessor loading test"""
        processor = DataProcessor()
        
        mock_exists.return_value = True
        mock_data = {
            'scaler': MagicMock(),
            'label_encoders': {'district': MagicMock()},
            'imputers': {'area': MagicMock()},
            'feature_names': ['area', 'rooms'],
            'is_fitted': True,
            'data_stats': {'n_samples': 100}
        }
        mock_load.return_value = mock_data
        
        processor.load_preprocessor('test_preprocessor.pkl')
        
        assert processor.is_fitted
        assert processor.feature_names == ['area', 'rooms']
        assert 'district' in processor.label_encoders
    
    def test_get_data_statistics(self):
        """Data Statistics Obtaining Test"""
        processor = DataProcessor()
        
        stats = processor.get_data_statistics()
        assert isinstance(stats, dict)
        
        df = pd.DataFrame({
            'total_area': [75.0, 80.0, 90.0],
            'rooms': [2, 3, 2],
            'district': ['Central', 'Northern', 'Southern'],
            'price': [50000, 60000, 40000]
        })
        
        df_minimal = df[['total_area', 'rooms']].copy()
        df_minimal['living_area'] = [52.5, 56.0, 63.0]
        df_minimal['kitchen_area'] = [12.0, 10.0, 15.0]
        df_minimal['floor'] = [5, 7, 3]
        df_minimal['total_floors'] = [12, 15, 9]
        df_minimal['year_built'] = [2010, 2015, 2005]
        df_minimal['ceiling_height'] = [2.7, 2.8, 2.6]
        df_minimal['building_prestige'] = ['standard', 'elite', 'economy']
        df_minimal['district'] = ['Central', 'Northern', 'Southern']
        df_minimal['metro_distance'] = [1.5, 2.0, 3.0]
        df_minimal['parking'] = [1, 0, 1]
        df_minimal['elevator'] = [1, 1, 0]
        df_minimal['renovation'] = ['European renovation', 'Cosmetic', 'No renovation']
        df_minimal['house_type'] = ['Monolithic', 'Brick', 'Panel']
        df_minimal['bathrooms_count'] = [1, 2, 1]
        df_minimal['bathroom_type'] = ['separate', 'combined', 'separate']
        df_minimal['balcony_type'] = ['loggia', 'balcony', 'none']
        df_minimal['balcony_count'] = [1, 1, 0]
        df_minimal['balcony_glazed'] = ['yes', 'no', 'no']
        df_minimal['balcony_area'] = [6.0, 4.0, 0.0]
        df_minimal['air_quality_index'] = [70, 65, 80]
        df_minimal['noise_level'] = [40, 50, 35]
        df_minimal['green_zone_distance'] = [1.0, 2.0, 0.5]
        df_minimal['water_body_distance'] = [3.0, 5.0, 1.0]
        df_minimal['industrial_distance'] = [8.0, 6.0, 10.0]
        df_minimal['crime_rate'] = [20, 30, 15]
        df_minimal['avg_income_index'] = [120, 100, 90]
        df_minimal['population_density'] = [4000, 6000, 3000]
        df_minimal['elderly_ratio'] = [0.2, 0.3, 0.25]
        df_minimal['hospital_distance'] = [1.5, 2.5, 1.0]
        df_minimal['clinics_count_3km'] = [3, 2, 4]
        df_minimal['pharmacy_distance'] = [0.3, 0.8, 0.5]
        df_minimal['emergency_time'] = [10, 15, 8]
        df_minimal['supermarket_distance'] = [0.5, 1.2, 0.3]
        df_minimal['shops_count_1km'] = [20, 10, 25]
        df_minimal['mall_distance'] = [2.0, 4.0, 1.5]
        df_minimal['market_distance'] = [1.0, 2.0, 0.8]
        df_minimal['services_count_1km'] = [8, 5, 12]
        df_minimal['useful_area_ratio'] = [0.7, 0.7, 0.7]
        df_minimal['price'] = [50000, 60000, 40000]
        
        processor.preprocess_data(df_minimal, fit=True)
        stats = processor.get_data_statistics()
        
        assert 'n_samples' in stats
        assert stats['n_samples'] == 3

if __name__ == "__main__":
    pytest.main([__file__])