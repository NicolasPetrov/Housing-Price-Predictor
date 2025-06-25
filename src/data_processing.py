import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}  # Отдельный imputer для каждого признака
        self.feature_names = []
        self.is_fitted = False
        
    def generate_sample_data(self, n_samples=1000):
        np.random.seed(42)
        
        data = {
            'area': np.random.normal(80, 30, n_samples).clip(20, 200),
            'rooms': np.random.randint(1, 6, n_samples),
            'floor': np.random.randint(1, 21, n_samples),
            'total_floors': np.random.randint(5, 25, n_samples),
            'year_built': np.random.randint(1960, 2024, n_samples),
            'district': np.random.choice(['Central', 'Northern', 'Southern', 'Western', 'Eastern'], n_samples),
            'metro_distance': np.random.exponential(2, n_samples).clip(0.1, 10),
            'parking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'elevator': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'balcony': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'renovation': np.random.choice(['No renovation', 'Cosmetic', 'European renovation'], n_samples),
            'house_type': np.random.choice(['Panel', 'Brick', 'Monolithic', 'Block', 'Wooden'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        price_base = 50000
        
        price_area = df['area'] * price_base
        
        district_multipliers = {
            'Central': 1.8,
            'Northern': 1.3,
            'Southern': 1.1,
            'Western': 1.4,
            'Eastern': 1.0
        }
        price_district = price_area * df['district'].map(district_multipliers)
        
        house_type_multipliers = {
            'Monolithic': 1.4,
            'Brick': 1.3,
            'Panel': 1.0,
            'Block': 1.1,
            'Wooden': 0.8
        }
        price_house_type = price_district * df['house_type'].map(house_type_multipliers)
        
        price_rooms = price_house_type * (1 + df['rooms'] * 0.1)
        price_floor = price_rooms * (1 + (df['floor'] / df['total_floors']) * 0.2)
        price_year = price_floor * (1 + (df['year_built'] - 1960) / 100 * 0.5)
        price_metro = price_year * (1 - df['metro_distance'] * 0.05)
        price_parking = price_metro * (1 + df['parking'] * 0.1)
        price_elevator = price_parking * (1 + df['elevator'] * 0.05)
        price_balcony = price_elevator * (1 + df['balcony'] * 0.08)
        
        renovation_multipliers = {
            'No renovation': 1.0,
            'Cosmetic': 1.15,
            'European renovation': 1.3
        }
        price_renovation = price_balcony * df['renovation'].map(renovation_multipliers)
        
        noise = np.random.normal(0, 0.1, n_samples)
        df['price'] = price_renovation * (1 + noise)
        
        df['price'] = df['price'].round(-3)
        
        return df
    
    def preprocess_data(self, df):
        df_processed = df.copy()
        
        categorical_columns = ['district', 'renovation', 'house_type']
        for col in categorical_columns:
            if col in df_processed.columns:
                if not self.is_fitted:
                    # При обучении - создаем новый encoder
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # При предсказании - используем сохраненный encoder
                    if col in self.label_encoders:
                        # Обрабатываем новые значения
                        unique_values = df_processed[col].unique()
                        for value in unique_values:
                            if value not in self.label_encoders[col].classes_:
                                # Добавляем новое значение в encoder
                                self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, value)
                        
                        df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
                    else:
                        # Если encoder не найден, создаем новый
                        le = LabelEncoder()
                        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        self.label_encoders[col] = le
        
        numerical_columns = ['area', 'rooms', 'floor', 'total_floors', 'year_built', 
                           'metro_distance', 'parking', 'elevator', 'balcony']
        
        if 'price' in df_processed.columns:
            df_processed = df_processed.dropna(subset=['price'])
        
        for col in numerical_columns:
            if col in df_processed.columns:
                if not self.is_fitted:
                    # При обучении - создаем и обучаем imputer для каждого признака
                    imputer = SimpleImputer(strategy='median')
                    df_processed[col] = imputer.fit_transform(df_processed[[col]]).flatten()
                    self.imputers[col] = imputer
                else:
                    # При предсказании - используем обученный imputer
                    if col in self.imputers:
                        df_processed[col] = self.imputers[col].transform(df_processed[[col]]).flatten()
                    else:
                        # Если imputer не найден, используем медиану
                        median_val = df_processed[col].median()
                        df_processed[col] = df_processed[col].fillna(median_val)
        
        return df_processed
    
    def prepare_features(self, df, target_column='price'):
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df
            y = None
        
        self.feature_names = X.columns.tolist()
        
        if not self.is_fitted:
            # При обучении - обучаем scaler
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            # При предсказании - используем обученный scaler
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y, self.feature_names
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def inverse_transform_categorical(self, df, columns):
        df_transformed = df.copy()
        for col in columns:
            if col in self.label_encoders:
                df_transformed[col] = self.label_encoders[col].inverse_transform(df_transformed[col])
        return df_transformed
    
    def save_preprocessor(self, filepath='models/data_preprocessor.pkl'):
        """Сохраняет предобработчик данных"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputers': self.imputers,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Data preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/data_preprocessor.pkl'):
        """Загружает предобработчик данных"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        preprocessor_data = joblib.load(filepath)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.imputers = preprocessor_data.get('imputers', {})  # Обратная совместимость
        self.feature_names = preprocessor_data['feature_names']
        self.is_fitted = preprocessor_data['is_fitted']
        
        print(f"Data preprocessor loaded from {filepath}") 