# Housing Price Predictor
**Housing Price Predictor** is an advanced machine learning system for predicting real estate prices using XGBoost and ensemble methods. Built with a comprehensive web interface powered by Streamlit, this application provides accurate price predictions with detailed model explanations and interactive visualizations. 

This project serves as an example of building a production-ready machine learning application with web interface, model interpretability, and comprehensive data processing. It includes robust error handling, automated model training, and modular code structure, making it an excellent starting point for learning ML deployment or extending with new features.

#### Features
- **XGBoost Model**: Optimized gradient boosting with 10,000+ training samples and hyperparameter tuning via Optuna.
- **Interactive Web Interface**: Streamlit application with real-time predictions, data analysis dashboard, and model explanations.
- **Model Interpretability**: SHAP and LIME explanations for feature importance and local model behavior.
- **Multi-language Support**: Complete interface translation between English and Russian with dynamic language switching.
- **Advanced Data Processing**: Extended features including house type, infrastructure, and environmental factors with outlier handling.
- **Comprehensive Evaluation**: Multiple performance metrics (R², RMSE, MAE, MAPE) with detailed visualizations.

#### Technical Details
- **Language**: Python 3.8+
- **Main Libraries**:
  - `xgboost`: Gradient boosting algorithm for price prediction.
  - `streamlit`: Web application framework with interactive components.
  - `shap`: Model interpretability and feature importance analysis.
  - `optuna`: Hyperparameter optimization and automated tuning.
  - `plotly`: Interactive visualizations and data exploration.
- **Structure**:
  - `app.py`: Main Streamlit web application with multi-language support.
  - `train_model.py`: Automated model training with data generation and optimization.
  - `config/config.py`: Configuration settings for model and data parameters.
  - `src/model.py`: XGBoost model implementation with prediction and evaluation methods.
  - `src/data_processing.py`: Data preprocessing, validation, and feature engineering.
  - `src/visualization.py`: Plotting utilities for model performance and data analysis.
  - `src/explainer.py`: SHAP and LIME model explanation implementations.
- **Key Components**:
  - **Automated Training**: Generates 10,000+ synthetic housing records and trains optimized models.
  - **Feature Engineering**: 12+ carefully engineered features including location, infrastructure, and property characteristics.
  - `src/data_processing.py`: Data preprocessing, validation, and feature engineering.
  - **Model Persistence**: Saves trained models, explainers, and preprocessing components.
  - **Web Interface**: Real-time predictions with instant explanations and interactive data exploration.
  - **Performance Monitoring**: Continuous model evaluation with comprehensive metrics and visualizations.
- **Error Handling**: 
  - Robust data validation at all processing stages.
  - Graceful handling of missing data and outliers using IQR method.
  - Comprehensive logging for debugging and monitoring.
- **Persistence**: Saves trained models, explainers, and configuration in organized directory structure.

#### Installation & Quick Start
```bash
# Clone and setup
git clone <repository-url>
cd Housing-Price-Predictor
pip install -r requirements.txt

# Train model (generates data, optimizes parameters, saves components)
python train_model.py

# Launch web application
streamlit run app.py
```

#### Usage Examples
```python
# Basic prediction
from src.model import HousingPriceModel
model = HousingPriceModel()
model.load_model()
prediction = model.predict(X_new)

# Feature importance
feature_importance = model.get_feature_importance()

# Model explanation
from src.explainer import ModelExplainer
explainer = ModelExplainer(model.model, feature_names)
explanation = explainer.explain_price_factors(X_sample, features)
```

#### Configuration
Edit `config/config.py` to customize model and data parameters:
```python
@dataclass
class ModelConfig:
    model_type: str = "xgboost"
    optimization_trials: int = 100
    cv_folds: int = 5

@dataclass
class DataConfig:
    sample_size: int = 10000  # Large dataset
    test_size: float = 0.2
    random_state: int = 42
```

#### Results and Reports
After training, the system generates:
- Model performance plots and feature importance visualizations
- SHAP summary plots for model interpretability
- Data distribution analysis and trend reports
- Trained model components saved for production use