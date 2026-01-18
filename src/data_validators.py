"""
Data validators for validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

try:
    from .exceptions import DataValidationError
except ImportError:
    from exceptions import DataValidationError

logger = logging.getLogger(__name__)

class DataValidator:
    """Class for real estate data validation"""
    
    VALID_DISTRICTS = {'Central', 'Northern', 'Southern', 'Western', 'Eastern'}
    VALID_RENOVATIONS = {'No renovation', 'Cosmetic', 'European renovation'}
    VALID_HOUSE_TYPES = {'Panel', 'Brick', 'Monolithic', 'Block', 'Wooden'}
    VALID_BUILDING_PRESTIGE = {'economy', 'standard', 'elite', 'premium'}
    VALID_BATHROOM_TYPES = {'combined', 'separate'}
    VALID_BALCONY_TYPES = {'none', 'balcony', 'loggia', 'terrace'}
    VALID_BALCONY_GLAZED = {'no', 'yes', 'panoramic'}
    
    RANGES = {
        'total_area': (10, 500),
        'living_area': (8, 400),
        'kitchen_area': (4, 50),
        'rooms': (1, 10),
        'floor': (1, 50),
        'total_floors': (1, 50),
        'year_built': (1900, 2030),
        'ceiling_height': (2.0, 5.0),
        'useful_area_ratio': (0.4, 0.9),
        
        'bathrooms_count': (1, 5),
        'balcony_count': (0, 3),
        'balcony_area': (0, 30),
        
        'metro_distance': (0.0, 20.0),
        
        'air_quality_index': (0, 100),
        'noise_level': (15, 90),
        'green_zone_distance': (0.0, 10.0),
        'water_body_distance': (0.0, 20.0),
        'industrial_distance': (0.0, 30.0),
        
        'crime_rate': (0, 100),
        'avg_income_index': (20, 300),
        'population_density': (500, 20000),
        'elderly_ratio': (0.05, 0.6),
        
        'hospital_distance': (0.1, 15.0),
        'clinics_count_3km': (0, 20),
        'pharmacy_distance': (0.05, 5.0),
        'emergency_time': (3, 40),
        
        'supermarket_distance': (0.05, 8.0),
        'shops_count_1km': (1, 100),
        'mall_distance': (0.2, 20.0),
        'market_distance': (0.1, 15.0),
        'services_count_1km': (0, 50),
        
        'price': (1000, 1000000)
    }
    
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, check_target: bool = True) -> None:
        """
        Validate DataFrame with real estate data
        
        Args:
            df: DataFrame to validate
            check_target: Whether to check target variable presence
            
        Raises:
            DataValidationError: On validation errors
        """
        if df is None or df.empty:
            raise DataValidationError("DataFrame is empty or None")
        
        required_columns = [
            'total_area', 'rooms', 'floor', 'total_floors', 'year_built',
            'district', 'metro_distance', 'parking', 'elevator', 
            'renovation', 'house_type', 'building_prestige'
        ]
        
        if check_target:
            required_columns.append('price')
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        cls._validate_data_types(df)
        
        cls._validate_numeric_ranges(df, check_target)
        
        cls._validate_categorical_values(df)
        
        cls._validate_logical_constraints(df)
        
        logger.info(f"DataFrame validation passed for {len(df)} records")
    
    @classmethod
    def _validate_data_types(cls, df: pd.DataFrame) -> None:
        """Check data types"""
        numeric_columns = [
            'total_area', 'living_area', 'kitchen_area', 'rooms', 'floor', 'total_floors', 
            'year_built', 'ceiling_height', 'useful_area_ratio', 'bathrooms_count', 
            'balcony_count', 'balcony_area', 'metro_distance', 'air_quality_index', 
            'noise_level', 'green_zone_distance', 'water_body_distance', 'industrial_distance',
            'crime_rate', 'avg_income_index', 'population_density', 'elderly_ratio',
            'hospital_distance', 'clinics_count_3km', 'pharmacy_distance', 'emergency_time',
            'supermarket_distance', 'shops_count_1km', 'mall_distance', 'market_distance', 
            'services_count_1km'
        ]
        if 'price' in df.columns:
            numeric_columns.append('price')
        
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise DataValidationError(f"Column '{col}' must be numeric")
                
                if df[col].isna().any():
                    logger.warning(f"Column '{col}' contains NaN values")
    
    @classmethod
    def _validate_numeric_ranges(cls, df: pd.DataFrame, check_target: bool) -> None:
        """Check numeric value ranges"""
        for col, (min_val, max_val) in cls.RANGES.items():
            if col in df.columns:
                if col == 'price' and not check_target:
                    continue
                
                invalid_mask = (df[col] < min_val) | (df[col] > max_val)
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum()
                    raise DataValidationError(
                        f"Column '{col}' has {invalid_count} values outside valid range "
                        f"[{min_val}, {max_val}]"
                    )
    
    @classmethod
    def _validate_categorical_values(cls, df: pd.DataFrame) -> None:
        """Check categorical values"""
        categorical_validations = {
            'district': cls.VALID_DISTRICTS,
            'renovation': cls.VALID_RENOVATIONS,
            'house_type': cls.VALID_HOUSE_TYPES,
            'building_prestige': cls.VALID_BUILDING_PRESTIGE,
            'bathroom_type': cls.VALID_BATHROOM_TYPES,
            'balcony_type': cls.VALID_BALCONY_TYPES,
            'balcony_glazed': cls.VALID_BALCONY_GLAZED
        }
        
        for col, valid_values in categorical_validations.items():
            if col in df.columns:
                invalid_values = set(df[col].unique()) - valid_values
                if invalid_values:
                    raise DataValidationError(
                        f"Column '{col}' contains invalid values: {invalid_values}. "
                        f"Valid values: {valid_values}"
                    )
    
    @classmethod
    def _validate_logical_constraints(cls, df: pd.DataFrame) -> None:
        """Check logical constraints"""
        if 'floor' in df.columns and 'total_floors' in df.columns:
            invalid_floors = df['floor'] > df['total_floors']
            if invalid_floors.any():
                raise DataValidationError(
                    f"Found {invalid_floors.sum()} records where floor > total_floors"
                )
        
        if 'living_area' in df.columns and 'total_area' in df.columns:
            invalid_living = df['living_area'] > df['total_area']
            if invalid_living.any():
                raise DataValidationError(
                    f"Found {invalid_living.sum()} records where living_area > total_area"
                )
        
        if 'kitchen_area' in df.columns and 'total_area' in df.columns:
            invalid_kitchen = df['kitchen_area'] > df['total_area']
            if invalid_kitchen.any():
                raise DataValidationError(
                    f"Found {invalid_kitchen.sum()} records where kitchen_area > total_area"
                )
        
        if 'rooms' in df.columns and 'total_area' in df.columns:
            min_area_per_room = df['total_area'] / df['rooms']
            invalid_area = (min_area_per_room < 8) | (min_area_per_room > 200)
            if invalid_area.any():
                logger.warning(
                    f"Found {invalid_area.sum()} records with unusual area/rooms ratio"
                )
        
        if 'balcony_count' in df.columns and 'rooms' in df.columns:
            invalid_balcony = df['balcony_count'] > (df['rooms'] + 1)
            if invalid_balcony.any():
                logger.warning(
                    f"Found {invalid_balcony.sum()} records with too many balconies"
                )

class InputValidator:
    """Validator for user input"""
    
    @staticmethod
    def validate_prediction_input(features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data for prediction
        
        Args:
            features: Dictionary with real estate characteristics
            
        Returns:
            Validated and cleaned dictionary
            
        Raises:
            DataValidationError: On incorrect data
        """
        if not features:
            raise DataValidationError("Features dictionary is empty")
        
        validated_features = features.copy()
        
        for key in ['area', 'metro_distance']:
            if key in validated_features:
                try:
                    validated_features[key] = round(float(validated_features[key]), 2)
                except (ValueError, TypeError):
                    raise DataValidationError(f"Invalid value for {key}: {validated_features[key]}")
        
        for key in ['rooms', 'floor', 'total_floors', 'year_built']:
            if key in validated_features:
                try:
                    validated_features[key] = int(float(validated_features[key]))
                except (ValueError, TypeError):
                    raise DataValidationError(f"Invalid value for {key}: {validated_features[key]}")
        
        for key in ['parking', 'elevator', 'balcony']:
            if key in validated_features:
                validated_features[key] = int(bool(validated_features[key]))
        
        df = pd.DataFrame([validated_features])
        
        try:
            DataValidator.validate_dataframe(df, check_target=False)
        except DataValidationError as e:
            raise DataValidationError(f"Input validation failed: {e}")
        
        logger.info("Input validation passed")
        return validated_features

def validate_file_path(filepath: str, must_exist: bool = False) -> str:
    """
    Validate file path
    
    Args:
        filepath: Path to file
        must_exist: Whether file must exist
        
    Returns:
        Validated path
        
    Raises:
        DataValidationError: On incorrect path
    """
    import os
    
    if not filepath or not isinstance(filepath, str):
        raise DataValidationError("File path must be a non-empty string")
    
    if must_exist and not os.path.exists(filepath):
        raise DataValidationError(f"File does not exist: {filepath}")
    
    if '..' in filepath or filepath.startswith('/'):
        raise DataValidationError(f"Potentially unsafe file path: {filepath}")
    
    return filepath