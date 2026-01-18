import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings
import logging
from typing import Tuple, Optional, Dict, Any, List

try:
    from .exceptions import DataProcessingError, DataValidationError, ModelLoadError
    from .data_validators import DataValidator, validate_file_path
except ImportError:
    from exceptions import DataProcessingError, DataValidationError, ModelLoadError
    from data_validators import DataValidator, validate_file_path

from config.config import config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class SafeLabelEncoder:
    
    def __init__(self, handle_unknown: str = 'use_encoded_value', unknown_value: int = -1):
        self.encoder = LabelEncoder()
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.is_fitted = False
        self._classes = None
    
    def fit(self, y):
        self.encoder.fit(y)
        self._classes = set(self.encoder.classes_)
        self.is_fitted = True
        return self
    
    def transform(self, y):
        if not self.is_fitted:
            raise ValueError("Encoder is not fitted")
        
        y_array = np.array(y)
        result = np.full(len(y_array), self.unknown_value, dtype=int)
        
        mask = np.isin(y_array, list(self._classes))
        if mask.any():
            result[mask] = self.encoder.transform(y_array[mask])
        
        unknown_count = (~mask).sum()
        if unknown_count > 0:
            unknown_values = set(y_array[~mask])
            logger.warning(f"Found {unknown_count} unknown values: {unknown_values}")
        
        return result
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        if not self.is_fitted:
            raise ValueError("Encoder is not fitted")
        
        y_array = np.array(y)
        result = np.full(len(y_array), 'unknown', dtype=object)
        
        valid_mask = (y_array >= 0) & (y_array < len(self.encoder.classes_))
        if valid_mask.any():
            result[valid_mask] = self.encoder.inverse_transform(y_array[valid_mask])
        
        return result

class DataProcessor:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, SafeLabelEncoder] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.feature_names: List[str] = []
        self.is_fitted = False
        self._data_stats: Dict[str, Any] = {}
        
        logger.info("DataProcessor initialized")
        
    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        if n_samples <= 0:
            raise DataValidationError("n_samples must be positive")
        
        if n_samples > 100000:
            logger.warning(f"Large sample size requested: {n_samples}")
        
        logger.info(f"Generating {n_samples} synthetic samples")
        
        try:
            np.random.seed(config.data.random_state)
            
            year_built = np.random.randint(1930, 2024, n_samples)
            
            data = {
                'total_area': np.random.normal(80, 30, n_samples).clip(20, 200),
                'living_area': 0,
                'kitchen_area': 0,
                'rooms': np.random.randint(1, 6, n_samples),
                'floor': np.random.randint(1, 21, n_samples),
                'total_floors': np.random.randint(5, 25, n_samples),
                'year_built': year_built,
                'ceiling_height': np.random.normal(2.7, 0.3, n_samples).clip(2.2, 4.0),
                
                'building_prestige': self._generate_building_prestige(year_built),
                'house_type': np.random.choice(['Panel', 'Brick', 'Monolithic', 'Block', 'Wooden'], n_samples),
                
                'district': np.random.choice(['Central', 'Northern', 'Southern', 'Western', 'Eastern'], n_samples),
                'metro_distance': np.random.exponential(2, n_samples).clip(0.1, 10),
                
                'bathrooms_count': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1]),
                'bathroom_type': np.random.choice(['combined', 'separate'], n_samples, p=[0.4, 0.6]),
                
                'balcony_type': np.random.choice(['none', 'balcony', 'loggia', 'terrace'], n_samples, p=[0.2, 0.4, 0.35, 0.05]),
                'balcony_count': 0,
                'balcony_glazed': np.random.choice(['no', 'yes', 'panoramic'], n_samples, p=[0.3, 0.6, 0.1]),
                'balcony_area': 0,
                
                'parking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                'elevator': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
                'renovation': np.random.choice(['No renovation', 'Cosmetic', 'European renovation'], n_samples),
                
                'air_quality_index': np.random.normal(50, 20, n_samples).clip(10, 100),
                'noise_level': np.random.normal(45, 15, n_samples).clip(20, 80),
                'green_zone_distance': np.random.exponential(1, n_samples).clip(0.1, 5),
                'water_body_distance': np.random.exponential(2, n_samples).clip(0.2, 10),
                'industrial_distance': np.random.exponential(3, n_samples).clip(0.5, 15),
                
                'crime_rate': np.random.normal(30, 15, n_samples).clip(5, 80),
                'avg_income_index': np.random.normal(100, 30, n_samples).clip(40, 200),
                'population_density': np.random.normal(5000, 2000, n_samples).clip(1000, 15000),
                'elderly_ratio': np.random.normal(0.25, 0.1, n_samples).clip(0.1, 0.5),
                
                'hospital_distance': np.random.exponential(2, n_samples).clip(0.3, 8),
                'clinics_count_3km': np.random.poisson(3, n_samples).clip(0, 10),
                'pharmacy_distance': np.random.exponential(0.5, n_samples).clip(0.1, 2),
                'emergency_time': np.random.normal(12, 5, n_samples).clip(5, 25),
                
                'supermarket_distance': np.random.exponential(0.8, n_samples).clip(0.1, 3),
                'shops_count_1km': np.random.poisson(15, n_samples).clip(2, 50),
                'mall_distance': np.random.exponential(3, n_samples).clip(0.5, 12),
                'market_distance': np.random.exponential(2, n_samples).clip(0.3, 8),
                'services_count_1km': np.random.poisson(8, n_samples).clip(1, 25)
            }
            
            df = pd.DataFrame(data)
            
            df = self._calculate_dependent_fields(df)
            
            df.loc[df['floor'] > df['total_floors'], 'floor'] = df['total_floors']
            
            df['price'] = self._calculate_enhanced_synthetic_price(df)
            
            DataValidator.validate_dataframe(df, check_target=True)
            
            logger.info(f"Successfully generated {len(df)} samples with enhanced features")
            return df
            
        except Exception as e:
            raise DataProcessingError(f"Failed to generate sample data: {e}")
    
    def _generate_building_prestige(self, year_built: np.ndarray) -> np.ndarray:
        prestige = np.full(len(year_built), 'standard', dtype=object)
        
        stalin_mask = (year_built >= 1930) & (year_built <= 1960)
        prestige[stalin_mask] = np.random.choice(['elite', 'standard'], 
                                               np.sum(stalin_mask), p=[0.7, 0.3])
        
        khrush_mask = (year_built > 1960) & (year_built <= 1990)
        prestige[khrush_mask] = np.random.choice(['standard', 'economy'], 
                                               np.sum(khrush_mask), p=[0.8, 0.2])
        
        nineties_mask = (year_built > 1990) & (year_built <= 2010)
        prestige[nineties_mask] = np.random.choice(['standard', 'economy', 'elite'], 
                                                 np.sum(nineties_mask), p=[0.6, 0.3, 0.1])
        
        modern_mask = year_built > 2010
        prestige[modern_mask] = np.random.choice(['elite', 'standard', 'premium'], 
                                               np.sum(modern_mask), p=[0.4, 0.4, 0.2])
        
        return prestige
    
    def _calculate_dependent_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        living_ratio = np.random.uniform(0.6, 0.8, len(df))
        df['living_area'] = (df['total_area'] * living_ratio).round(1)
        
        kitchen_base = np.where(df['total_area'] < 50, 
                               np.random.uniform(6, 9, len(df)),
                               np.where(df['total_area'] < 80,
                                       np.random.uniform(8, 12, len(df)),
                                       np.random.uniform(10, 18, len(df))))
        df['kitchen_area'] = kitchen_base.round(1)
        
        df['balcony_count'] = np.where(df['balcony_type'] == 'none', 0,
                                     np.where(df['rooms'] <= 2, 1,
                                            np.random.choice([1, 2], len(df), p=[0.7, 0.3])))
        
        df['balcony_area'] = np.where(df['balcony_count'] == 0, 0,
                                    np.where(df['balcony_type'] == 'balcony',
                                           np.random.uniform(3, 6, len(df)),
                                           np.where(df['balcony_type'] == 'loggia',
                                                   np.random.uniform(4, 8, len(df)),
                                                   np.random.uniform(8, 15, len(df)))))
        df['balcony_area'] = df['balcony_area'].round(1)
        
        df['useful_area_ratio'] = (df['living_area'] / df['total_area']).round(3)
        
        return df

    def _calculate_enhanced_synthetic_price(self, df: pd.DataFrame) -> pd.Series:
        """Calculate synthetic price based on all features with non-linear dependencies"""
        
        price_base = 600
        
        price_area = df['total_area'] * price_base
        
        prestige_multipliers = {
            'economy': 0.8, 'standard': 1.0, 'elite': 1.6, 'premium': 2.2
        }
        price_prestige = price_area * df['building_prestige'].map(prestige_multipliers)
        
        year_factor = self._calculate_year_prestige_factor(df['year_built'])
        price_year = price_prestige * year_factor
        
        district_multipliers = {
            'Central': 1.8, 'Northern': 1.3, 'Southern': 1.1,
            'Western': 1.4, 'Eastern': 1.0
        }
        price_district = price_year * df['district'].map(district_multipliers)
        
        house_type_multipliers = {
            'Monolithic': 1.4, 'Brick': 1.3, 'Panel': 1.0,
            'Block': 1.1, 'Wooden': 0.8
        }
        price_house_type = price_district * df['house_type'].map(house_type_multipliers)
        
        price_rooms = price_house_type * (1 + df['rooms'] * 0.08)
        price_floor = price_rooms * (1 + (df['floor'] / df['total_floors']) * 0.15)
        price_ceiling = price_floor * (1 + (df['ceiling_height'] - 2.5) * 0.1)
        price_useful_area = price_ceiling * (1 + (df['useful_area_ratio'] - 0.7) * 0.3)
        
        bathroom_factor = 1 + (df['bathrooms_count'] - 1) * 0.12
        bathroom_factor *= np.where(df['bathroom_type'] == 'separate', 1.05, 1.0)
        price_bathrooms = price_useful_area * bathroom_factor
        
        balcony_multipliers = {'none': 1.0, 'balcony': 1.08, 'loggia': 1.12, 'terrace': 1.25}
        balcony_factor = df['balcony_type'].map(balcony_multipliers)
        balcony_factor *= (1 + df['balcony_count'] * 0.03)
        glazed_multipliers = {'no': 1.0, 'yes': 1.03, 'panoramic': 1.08}
        balcony_factor *= df['balcony_glazed'].map(glazed_multipliers)
        price_balcony = price_bathrooms * balcony_factor
        
        eco_factor = (
            (1 + (100 - df['air_quality_index']) / 100 * 0.2) * 
            (1 - (df['noise_level'] - 30) / 50 * 0.15) *    
            (1 - df['green_zone_distance'] * 0.05) *        
            (1 - df['water_body_distance'] * 0.02) *        
            (1 + df['industrial_distance'] * 0.03)           
        )
        price_eco = price_balcony * eco_factor.clip(0.7, 1.4)
        
        social_factor = (
            (1 - df['crime_rate'] / 100 * 0.25) *           
            (1 + (df['avg_income_index'] - 100) / 100 * 0.15) * 
            (1 - (df['population_density'] - 5000) / 10000 * 0.1) 
        )
        price_social = price_eco * social_factor.clip(0.8, 1.3)
        
        medical_factor = (
            (1 - df['hospital_distance'] * 0.02) *          
            (1 + df['clinics_count_3km'] * 0.01) *          
            (1 - df['pharmacy_distance'] * 0.03) *           
            (1 - (df['emergency_time'] - 10) / 20 * 0.08)    
        )
        price_medical = price_social * medical_factor.clip(0.9, 1.15)
        
        shopping_factor = (
            (1 - df['supermarket_distance'] * 0.05) *        
            (1 + df['shops_count_1km'] / 50 * 0.1) *        
            (1 - df['mall_distance'] * 0.01) *               
            (1 + df['services_count_1km'] / 25 * 0.08)      
        )
        price_shopping = price_medical * shopping_factor.clip(0.85, 1.2)
        
        price_metro = price_shopping * (1 - df['metro_distance'] * 0.05)
        price_parking = price_metro * (1 + df['parking'] * 0.12)
        price_elevator = price_parking * (1 + df['elevator'] * 0.06)
        
        renovation_multipliers = {
            'No renovation': 1.0, 'Cosmetic': 1.15, 'European renovation': 1.35
        }
        price_renovation = price_elevator * df['renovation'].map(renovation_multipliers)
        
        noise = np.random.normal(0, 0.08, len(df))
        final_price = price_renovation * (1 + noise)
        
        return final_price.clip(20000, 500000).round(0)  
    
    def _calculate_year_prestige_factor(self, year_built: pd.Series) -> pd.Series:
        """Non-linear function for year built prestige factor"""
        factor = np.ones(len(year_built))
        
        stalin_mask = (year_built >= 1930) & (year_built <= 1960)
        stalin_factor = 1.2 + 0.3 * np.exp(-((year_built[stalin_mask] - 1950) / 10) ** 2)
        factor[stalin_mask] = stalin_factor
        
        khrush_mask = (year_built > 1960) & (year_built <= 1980)
        factor[khrush_mask] = 0.9 - (year_built[khrush_mask] - 1960) / 100 * 0.1
        
        brezh_mask = (year_built > 1980) & (year_built <= 1990)
        factor[brezh_mask] = 0.95
        
        nineties_mask = (year_built > 1990) & (year_built <= 2010)
        factor[nineties_mask] = 0.9 + (year_built[nineties_mask] - 1990) / 20 * 0.3
        
        modern_mask = year_built > 2010
        modern_factor = 1.2 + (year_built[modern_mask] - 2010) / 20 * 0.4
        factor[modern_mask] = modern_factor.clip(1.2, 1.8)
        
        return pd.Series(factor, index=year_built.index)
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = None) -> pd.DataFrame:
        """
        Data preprocessing with validation
        
        Args:
            df: DataFrame to process
            fit: Whether to fit preprocessors (None = auto-detect)
            
        Returns:
            Processed DataFrame
            
        Raises:
            DataProcessingError: On processing error
        """
        if df is None or df.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        if fit is None:
            fit = not self.is_fitted
        
        logger.info(f"Preprocessing data: {len(df)} records, fit={fit}")
        
        try:
            df_processed = df.copy()
            
            has_target = 'price' in df_processed.columns
            DataValidator.validate_dataframe(df_processed, check_target=has_target)
            
            categorical_columns = ['district', 'renovation', 'house_type', 'building_prestige', 
                                 'bathroom_type', 'balcony_type', 'balcony_glazed']
            for col in categorical_columns:
                if col in df_processed.columns:
                    df_processed[col] = self._process_categorical_column(
                        df_processed[col], col, fit
                    )
            
            numerical_columns = [
                'total_area', 'living_area', 'kitchen_area', 'rooms', 'floor', 'total_floors', 
                'year_built', 'ceiling_height', 'useful_area_ratio',
                
                'bathrooms_count', 'balcony_count', 'balcony_area',
                
                'metro_distance',
                
                'parking', 'elevator',
                
                'air_quality_index', 'noise_level', 'green_zone_distance', 
                'water_body_distance', 'industrial_distance',
                
                'crime_rate', 'avg_income_index', 'population_density', 'elderly_ratio',
                
                'hospital_distance', 'clinics_count_3km', 'pharmacy_distance', 'emergency_time',
                
                'supermarket_distance', 'shops_count_1km', 'mall_distance', 
                'market_distance', 'services_count_1km'
            ]
            
            if 'price' in df_processed.columns:
                initial_len = len(df_processed)
                df_processed = df_processed.dropna(subset=['price'])
                dropped = initial_len - len(df_processed)
                if dropped > 0:
                    logger.warning(f"Dropped {dropped} rows with missing price")
            
            for col in numerical_columns:
                if col in df_processed.columns:
                    df_processed[col] = self._process_numerical_column(
                        df_processed[col], col, fit
                    )
            
            if fit:
                self._save_data_statistics(df_processed)
            
            logger.info(f"Preprocessing completed: {len(df_processed)} records")
            return df_processed
            
        except Exception as e:
            raise DataProcessingError(f"Data preprocessing failed: {e}")
    
    def _process_categorical_column(self, series: pd.Series, col_name: str, fit: bool) -> pd.Series:
        """Process categorical feature"""
        if fit:
            encoder = SafeLabelEncoder()
            encoded = encoder.fit_transform(series.astype(str))
            self.label_encoders[col_name] = encoder
        else:
            if col_name not in self.label_encoders:
                logger.warning(f"No encoder found for column {col_name}, creating new one")
                encoder = SafeLabelEncoder()
                encoded = encoder.fit_transform(series.astype(str))
                self.label_encoders[col_name] = encoder
            else:
                encoded = self.label_encoders[col_name].transform(series.astype(str))
        
        return pd.Series(encoded, index=series.index)
    
    def _process_numerical_column(self, series: pd.Series, col_name: str, fit: bool) -> pd.Series:
        """Process numerical feature"""
        if fit:
            imputer = SimpleImputer(strategy='median')
            filled = imputer.fit_transform(series.values.reshape(-1, 1)).flatten()
            self.imputers[col_name] = imputer
        else:
            if col_name in self.imputers:
                filled = self.imputers[col_name].transform(series.values.reshape(-1, 1)).flatten()
            else:
                median_val = series.median()
                filled = series.fillna(median_val).values
                logger.warning(f"No imputer found for {col_name}, using median: {median_val}")
        
        return pd.Series(filled, index=series.index)
    
    def _save_data_statistics(self, df: pd.DataFrame) -> None:
        """Save data statistics"""
        self._data_stats = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'numeric_stats': df.select_dtypes(include=[np.number]).describe().to_dict(),
            'categorical_stats': {
                col: df[col].value_counts().to_dict()
                for col in df.select_dtypes(include=['object']).columns
            }
        }
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'price') -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Prepare features for model
        
        Args:
            df: DataFrame with data
            target_column: Target variable name
            
        Returns:
            Tuple of (X, y, feature_names)
            
        Raises:
            DataProcessingError: On feature preparation error
        """
        try:
            if target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column].values
            else:
                X = df
                y = None
            
            if not self.is_fitted:
                self.feature_names = X.columns.tolist()
                X_scaled = self.scaler.fit_transform(X)
                self.is_fitted = True
                logger.info("Scaler fitted on training data")
            else:
                if self.feature_names:
                    missing_features = set(self.feature_names) - set(X.columns)
                    if missing_features:
                        logger.warning(f"Missing features: {missing_features}, filling with zeros")
                        for feature in missing_features:
                            X[feature] = 0
                    
                    X = X[self.feature_names]
                else:
                    self.feature_names = X.columns.tolist()
                
                X_scaled = self.scaler.transform(X)
                logger.info("Applied fitted scaler to data")
            
            return X_scaled, y, self.feature_names
            
        except Exception as e:
            raise DataProcessingError(f"Feature preparation failed: {e}")
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = None, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and test sets"""
        if test_size is None:
            test_size = config.data.test_size
        if random_state is None:
            random_state = config.data.random_state
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def inverse_transform_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Inverse transform categorical features"""
        df_transformed = df.copy()
        for col in columns:
            if col in self.label_encoders:
                df_transformed[col] = self.label_encoders[col].inverse_transform(df_transformed[col])
        return df_transformed
    
    def save_preprocessor(self, filepath: str = None) -> None:
        """
        Save data preprocessor
        
        Args:
            filepath: Path to save
            
        Raises:
            DataProcessingError: On save error
        """
        if filepath is None:
            filepath = config.get_preprocessor_path()
        
        try:
            validate_file_path(filepath)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            preprocessor_data = {
                'scaler': self.scaler,
                'label_encoders': dict(self.label_encoders),  
                'imputers': dict(self.imputers),  
                'feature_names': list(self.feature_names),  
                'is_fitted': self.is_fitted,
                'data_stats': dict(self._data_stats)  
            }
            
            joblib.dump(preprocessor_data, filepath)
            logger.info(f"Data preprocessor saved to {filepath}")
            
        except Exception as e:
            raise DataProcessingError(f"Failed to save preprocessor: {e}")
    
    def load_preprocessor(self, filepath: str = None) -> None:
        """
        Load data preprocessor
        
        Args:
            filepath: Path to file
            
        Raises:
            ModelLoadError: On load error
        """
        if filepath is None:
            filepath = config.get_preprocessor_path()
        
        try:
            validate_file_path(filepath, must_exist=True)
            
            preprocessor_data = joblib.load(filepath)
            
            self.scaler = preprocessor_data['scaler']
            self.label_encoders = preprocessor_data['label_encoders']
            self.imputers = preprocessor_data.get('imputers', {})
            self.feature_names = preprocessor_data['feature_names']
            self.is_fitted = preprocessor_data['is_fitted']
            self._data_stats = preprocessor_data.get('data_stats', {})
            
            logger.info(f"Data preprocessor loaded from {filepath}")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load preprocessor: {e}")
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get data statistics"""
        return self._data_stats.copy() 