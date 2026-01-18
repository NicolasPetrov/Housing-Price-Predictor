"""
Tests for validators
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_validators import DataValidator, InputValidator
from exceptions import DataValidationError

class TestDataValidator:
    """Tests for DataValidator"""
    
    def test_valid_dataframe(self):
        """Validation test of a correct DataFrame"""
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
        
        DataValidator.validate_dataframe(df, check_target=True)
    
    def test_empty_dataframe(self):
        """Empty DataFrame Validation Test"""
        df = pd.DataFrame()
        
        with pytest.raises(DataValidationError, match="DataFrame is empty"):
            DataValidator.validate_dataframe(df)
    
    def test_missing_columns(self):
        """DataFrame validation test with missing columns"""
        df = pd.DataFrame({
            'total_area': [75.0],
            'rooms': [2]
        })
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            DataValidator.validate_dataframe(df)
    
    def test_invalid_numeric_ranges(self):
        """Test for validation of invalid numerical ranges"""
        df = pd.DataFrame({
            'total_area': [-10.0],
            'living_area': [52.5],
            'kitchen_area': [12.0],
            'rooms': [2],
            'floor': [5],
            'total_floors': [12],
            'year_built': [2010],
            'ceiling_height': [2.7],
            'building_prestige': ['standard'],
            'district': ['Central'],
            'metro_distance': [1.5],
            'parking': [1],
            'elevator': [1],
            'renovation': ['European renovation'],
            'house_type': ['Monolithic'],
            'bathrooms_count': [1],
            'bathroom_type': ['separate'],
            'balcony_type': ['loggia'],
            'balcony_count': [1],
            'balcony_glazed': ['yes'],
            'balcony_area': [6.0],
            'air_quality_index': [70],
            'noise_level': [40],
            'green_zone_distance': [1.0],
            'water_body_distance': [3.0],
            'industrial_distance': [8.0],
            'crime_rate': [20],
            'avg_income_index': [120],
            'population_density': [4000],
            'elderly_ratio': [0.2],
            'hospital_distance': [1.5],
            'clinics_count_3km': [3],
            'pharmacy_distance': [0.3],
            'emergency_time': [10],
            'supermarket_distance': [0.5],
            'shops_count_1km': [20],
            'mall_distance': [2.0],
            'market_distance': [1.0],
            'services_count_1km': [8],
            'useful_area_ratio': [0.7],
            'price': [50000]
        })
        
        with pytest.raises(DataValidationError, match="outside valid range"):
            DataValidator.validate_dataframe(df)
    
    def test_invalid_categorical_values(self):
        """Validation test for incorrect categorical values"""
        df = pd.DataFrame({
            'total_area': [75.0],
            'living_area': [52.5],
            'kitchen_area': [12.0],
            'rooms': [2],
            'floor': [5],
            'total_floors': [12],
            'year_built': [2010],
            'ceiling_height': [2.7],
            'building_prestige': ['standard'],
            'district': ['InvalidDistrict'],
            'metro_distance': [1.5],
            'parking': [1],
            'elevator': [1],
            'renovation': ['European renovation'],
            'house_type': ['Monolithic'],
            'bathrooms_count': [1],
            'bathroom_type': ['separate'],
            'balcony_type': ['loggia'],
            'balcony_count': [1],
            'balcony_glazed': ['yes'],
            'balcony_area': [6.0],
            'air_quality_index': [70],
            'noise_level': [40],
            'green_zone_distance': [1.0],
            'water_body_distance': [3.0],
            'industrial_distance': [8.0],
            'crime_rate': [20],
            'avg_income_index': [120],
            'population_density': [4000],
            'elderly_ratio': [0.2],
            'hospital_distance': [1.5],
            'clinics_count_3km': [3],
            'pharmacy_distance': [0.3],
            'emergency_time': [10],
            'supermarket_distance': [0.5],
            'shops_count_1km': [20],
            'mall_distance': [2.0],
            'market_distance': [1.0],
            'services_count_1km': [8],
            'useful_area_ratio': [0.7],
            'price': [50000]
        })
        
        with pytest.raises(DataValidationError, match="contains invalid values"):
            DataValidator.validate_dataframe(df)
    
    def test_logical_constraints(self):
        """Validation test of logical constraints"""
        df = pd.DataFrame({
            'total_area': [75.0],
            'living_area': [52.5],
            'kitchen_area': [12.0],
            'rooms': [2],
            'floor': [15],
            'total_floors': [10],
            'year_built': [2010],
            'ceiling_height': [2.7],
            'building_prestige': ['standard'],
            'district': ['Central'],
            'metro_distance': [1.5],
            'parking': [1],
            'elevator': [1],
            'renovation': ['European renovation'],
            'house_type': ['Monolithic'],
            'bathrooms_count': [1],
            'bathroom_type': ['separate'],
            'balcony_type': ['loggia'],
            'balcony_count': [1],
            'balcony_glazed': ['yes'],
            'balcony_area': [6.0],
            'air_quality_index': [70],
            'noise_level': [40],
            'green_zone_distance': [1.0],
            'water_body_distance': [3.0],
            'industrial_distance': [8.0],
            'crime_rate': [20],
            'avg_income_index': [120],
            'population_density': [4000],
            'elderly_ratio': [0.2],
            'hospital_distance': [1.5],
            'clinics_count_3km': [3],
            'pharmacy_distance': [0.3],
            'emergency_time': [10],
            'supermarket_distance': [0.5],
            'shops_count_1km': [20],
            'mall_distance': [2.0],
            'market_distance': [1.0],
            'services_count_1km': [8],
            'useful_area_ratio': [0.7],
            'price': [50000]
        })
        
        with pytest.raises(DataValidationError, match="floor > total_floors"):
            DataValidator.validate_dataframe(df)

class TestInputValidator:
    """Tests for InputValidator"""
    
    def test_valid_input(self):
        """Input validation test"""
        features = {
            'total_area': 75.0,
            'living_area': 52.5,
            'kitchen_area': 12.0,
            'rooms': 2,
            'floor': 5,
            'total_floors': 12,
            'year_built': 2010,
            'ceiling_height': 2.7,
            'building_prestige': 'standard',
            'district': 'Central',
            'metro_distance': 1.5,
            'parking': 1,
            'elevator': 1,
            'renovation': 'European renovation',
            'house_type': 'Monolithic',
            'bathrooms_count': 1,
            'bathroom_type': 'separate',
            'balcony_type': 'loggia',
            'balcony_count': 1,
            'balcony_glazed': 'yes',
            'balcony_area': 6.0,
            'air_quality_index': 70,
            'noise_level': 40,
            'green_zone_distance': 1.0,
            'water_body_distance': 3.0,
            'industrial_distance': 8.0,
            'crime_rate': 20,
            'avg_income_index': 120,
            'population_density': 4000,
            'elderly_ratio': 0.2,
            'hospital_distance': 1.5,
            'clinics_count_3km': 3,
            'pharmacy_distance': 0.3,
            'emergency_time': 10,
            'supermarket_distance': 0.5,
            'shops_count_1km': 20,
            'mall_distance': 2.0,
            'market_distance': 1.0,
            'services_count_1km': 8,
            'useful_area_ratio': 0.7
        }
        
        validated = InputValidator.validate_prediction_input(features)
        
        assert validated is not None
        assert validated['total_area'] == 75.0
        assert validated['rooms'] == 2
    
    def test_empty_input(self):
        """Empty input validation test"""
        features = {}
        
        with pytest.raises(DataValidationError, match="Features dictionary is empty"):
            InputValidator.validate_prediction_input(features)
    
    def test_input_type_conversion(self):
        """Input data type conversion test"""
        features = {
            'total_area': 75.5,
            'living_area': 52.5,
            'kitchen_area': 12.0,
            'rooms': 2,
            'floor': 5,
            'total_floors': 12,
            'year_built': 2010,
            'ceiling_height': 2.7,
            'building_prestige': 'standard',
            'district': 'Central',
            'metro_distance': 1.5,
            'parking': 1,
            'elevator': 1,
            'renovation': 'European renovation',
            'house_type': 'Monolithic',
            'bathrooms_count': 1,
            'bathroom_type': 'separate',
            'balcony_type': 'loggia',
            'balcony_count': 1,
            'balcony_glazed': 'yes',
            'balcony_area': 6.0,
            'air_quality_index': 70,
            'noise_level': 40,
            'green_zone_distance': 1.0,
            'water_body_distance': 3.0,
            'industrial_distance': 8.0,
            'crime_rate': 20,
            'avg_income_index': 120,
            'population_density': 4000,
            'elderly_ratio': 0.2,
            'hospital_distance': 1.5,
            'clinics_count_3km': 3,
            'pharmacy_distance': 0.3,
            'emergency_time': 10,
            'supermarket_distance': 0.5,
            'shops_count_1km': 20,
            'mall_distance': 2.0,
            'market_distance': 1.0,
            'services_count_1km': 8,
            'useful_area_ratio': 0.7
        }
        
        validated = InputValidator.validate_prediction_input(features)
        
        assert isinstance(validated['total_area'], float)
        assert isinstance(validated['rooms'], int)
        assert isinstance(validated['parking'], int)
        assert validated['parking'] == 1

if __name__ == "__main__":
    pytest.main([__file__])