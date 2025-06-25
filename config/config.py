import os
from typing import Dict, List, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    """Конфигурация моделей"""
    model_type: str = "xgboost"
    optimization_trials: int = 100
    cv_folds: int = 5

@dataclass
class DataConfig:
    """Конфигурация данных"""
    sample_size: int = 10000  # Увеличенный размер выборки
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class ValidationConfig:
    """Конфигурация валидации"""
    enable_outlier_detection: bool = True
    outlier_threshold: float = 3.0
    confidence_level: float = 0.95
    min_confidence: float = 0.7

class Config:
    """Основная конфигурация приложения"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.validation = ValidationConfig()
        
        # Пути к файлам
        self.data_path = "data"
        self.models_path = "models"
        self.reports_path = "reports"
        
        # Создаем директории если их нет
        for path in [self.data_path, self.models_path, self.reports_path]:
            os.makedirs(path, exist_ok=True)

# Глобальный экземпляр конфигурации
config = Config() 