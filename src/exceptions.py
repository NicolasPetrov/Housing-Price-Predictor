"""
Custom exclusions for the Housing Price Predictor project
"""

class HousingPredictorError(Exception):
    """Base class for all project errors"""
    pass

class ModelError(HousingPredictorError):
    """Model related errors"""
    pass

class ModelNotTrainedError(ModelError):
    """The model is not trained"""
    pass

class ModelLoadError(ModelError):
    """Error loading model"""
    pass

class DataError(HousingPredictorError):
    """Data related errors"""
    pass

class DataValidationError(DataError):
    """Data validation error"""
    pass

class DataProcessingError(DataError):
    """Data processing error"""
    pass

class ConfigurationError(HousingPredictorError):
    """Configuration errors"""
    pass