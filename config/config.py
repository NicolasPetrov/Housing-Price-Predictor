import os
from typing import Dict, List, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import logging

load_dotenv()

@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = os.getenv("MODEL_TYPE", "xgboost")
    optimization_trials: int = int(os.getenv("OPTIMIZATION_TRIALS", "100"))
    cv_folds: int = int(os.getenv("CV_FOLDS", "5"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))

@dataclass
class DataConfig:
    """Data configuration"""
    sample_size: int = int(os.getenv("SAMPLE_SIZE", "10000"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))

@dataclass
class ValidationConfig:
    """Validation configuration"""
    enable_outlier_detection: bool = os.getenv("ENABLE_OUTLIER_DETECTION", "true").lower() == "true"
    outlier_threshold: float = float(os.getenv("OUTLIER_THRESHOLD", "3.0"))
    confidence_level: float = float(os.getenv("CONFIDENCE_LEVEL", "0.95"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.7"))

@dataclass
class PathConfig:
    """File paths configuration"""
    data_path: str = os.getenv("DATA_PATH", "data")
    models_path: str = os.getenv("MODELS_PATH", "models")
    reports_path: str = os.getenv("REPORTS_PATH", "reports")
    logs_path: str = os.getenv("LOGS_PATH", "logs")
    
    housing_data_file: str = os.getenv("HOUSING_DATA_FILE", "housing_data.csv")
    model_file: str = os.getenv("MODEL_FILE", "housing_price_model.pkl")
    preprocessor_file: str = os.getenv("PREPROCESSOR_FILE", "data_preprocessor.pkl")

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_enabled: bool = os.getenv("LOG_FILE_ENABLED", "true").lower() == "true"

class Config:
    """Main application configuration"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.validation = ValidationConfig()
        self.paths = PathConfig()
        self.logging = LoggingConfig()
        
        self._create_directories()
        
        self._setup_logging()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.paths.data_path,
            self.paths.models_path,
            self.paths.reports_path,
            self.paths.logs_path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format=self.logging.format,
            handlers=[]
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(self.logging.format)
        console_handler.setFormatter(console_formatter)
        
        handlers = [console_handler]
        if self.logging.file_enabled:
            log_file = os.path.join(self.paths.logs_path, "housing_predictor.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(console_formatter)
            handlers.append(file_handler)
        
        root_logger = logging.getLogger()
        root_logger.handlers = handlers
    
    def get_model_path(self) -> str:
        """Get full path to model file"""
        return os.path.join(self.paths.models_path, self.paths.model_file)
    
    def get_preprocessor_path(self) -> str:
        """Get full path to preprocessor file"""
        return os.path.join(self.paths.models_path, self.paths.preprocessor_file)
    
    def get_data_path(self) -> str:
        """Get full path to data file"""
        return os.path.join(self.paths.data_path, self.paths.housing_data_file)

config = Config() 