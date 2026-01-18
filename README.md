# Housing Price Predictor
**Housing Price Predictor** is an advanced machine learning system for predicting real estate prices using XGBoost with comprehensive factor analysis. Built with a web interface powered by Streamlit, this application provides highly accurate price predictions with detailed model explanations and interactive visualizations using **41 comprehensive factors** instead of basic property characteristics.

<img width="1432" height="672" alt="housingpricepredictor" src="https://github.com/user-attachments/assets/44dc03b1-5e23-48fa-9164-d8e620464052" />

This project serves as an example of building a production-ready machine learning application with advanced feature engineering, comprehensive environmental and social factor analysis, and extensive real estate market modeling.

## 🏠 Revolutionary Real Estate Analysis

### 🏢 Advanced Building Assessment
- **Building Prestige Classification** with 4-tier system (economy/standard/elite/premium)
- **Non-linear Year Analysis** with historical value peaks for Stalin-era buildings (1930-1960) and modern constructions (2010+)
- **Architectural Heritage Value** integrated into pricing algorithms
- **Detailed Property Characteristics** including ceiling height, room layouts, and space efficiency ratios

### 🌍 Environmental Impact Analysis
- **Air Quality Index** (0-100 scale) with pollution impact modeling
- **Noise Level Assessment** (20-80 dB) from traffic, airports, and industrial sources
- **Green Zone Proximity** to parks, forests, and recreational areas
- **Water Body Access** to rivers, lakes, and waterfront properties
- **Industrial Distance** from factories, waste facilities, and heavy industry

### 👥 Social & Economic Factors
- **Crime Rate Statistics** by district with safety impact analysis
- **Average Income Demographics** of neighborhood residents
- **Population Density** optimization for urban planning
- **Age Demographics** including elderly ratio and family composition

### 🏥 Infrastructure Accessibility
- **Healthcare Proximity** to hospitals, clinics, and emergency services
- **Medical Facility Density** within 3km radius
- **Pharmacy Accessibility** and healthcare convenience
- **Emergency Response Time** for critical services

### 🛒 Commercial & Service Infrastructure
- **Retail Accessibility** to supermarkets, shopping centers, and local stores
- **Service Availability** including banks, post offices, and professional services
- **Market Proximity** to traditional and modern shopping facilities
- **Transportation Connectivity** beyond just metro access

### 🏠 Detailed Property Analysis
- **Multi-Area Breakdown** (total/living/kitchen areas with efficiency ratios)
- **Bathroom Configuration** (count, type: separate/combined, facilities)
- **Balcony & Outdoor Spaces** (type: balcony/loggia/terrace, glazing, area)
- **Ceiling Height Impact** on property value and living comfort

## ✨ System Features

### 🔒 Security & Validation
- **Comprehensive data validation** with 41-factor validation system
- **Safe categorical encoding** with unknown value handling for all property types
- **Input sanitization** and type checking across all environmental factors
- **Robust error handling** with custom exception hierarchy for real estate data

### 🚀 Performance Optimizations
- **Efficient Streamlit caching** for complex multi-factor models
- **Optimized data loading** with smart fallbacks and automatic feature completion
- **Memory-efficient processing** for 41-factor datasets and large property databases

### 🧪 Testing & Quality Assurance
- **Comprehensive test suite** with pytest covering all 41 factors
- **Unit tests** for environmental, social, and infrastructure components
- **Validation tests** for real estate data integrity and market logic
- **Mock testing** for external data sources and API dependencies

### ⚙️ Configuration Management
- **Environment-based configuration** with .env support for all factor weights
- **Flexible file paths** and settings for different market regions
- **Centralized logging** with configurable levels for production monitoring

#### Advanced Features
- **XGBoost Model**: Optimized gradient boosting with 10,000+ training samples using 41 comprehensive factors achieving 90.19% accuracy (R²).
- **Advanced Web Interface**: Full control over all 41 factors with tabbed organization for comprehensive real estate analysis.
- **Model Interpretability**: SHAP explanations showing impact of environmental, social, and infrastructure factors.
- **Multi-language Support**: Complete interface translation between English, Russian, and French with dynamic language switching.
- **Comprehensive Factor Analysis**: 41 factors including building prestige, environmental quality, social demographics, and infrastructure accessibility.
- **Quality Scoring System**: Automatic calculation of Environmental Score, Infrastructure Score, and Building Quality Score (0-100 scales).
- **Production-Ready**: Robust error handling, comprehensive logging, and extensive testing with automatic feature completion.

#### Technical Details
- **Language**: Python 3.8+
- **Main Libraries**:
  - `xgboost`: Advanced gradient boosting algorithm for 41-factor price prediction.
  - `streamlit`: Web application framework with tabbed interface for complex factor input.
  - `shap`: Model interpretability for environmental, social, and infrastructure factor analysis.
  - `optuna`: Hyperparameter optimization for multi-factor model tuning.
  - `plotly`: Interactive visualizations for comprehensive factor analysis and quality scoring.
  - `pytest`: Comprehensive testing framework covering all 41 factors.
- **Architecture**:
  - **Advanced Interface Design**: Comprehensive 41-factor analysis system
  - **Quality Scoring System**: Environmental, Infrastructure, and Building quality indices
- **Structure**:
  - `app.py`: Advanced Streamlit application with full 41-factor control and tabbed organization.
  - `train_model.py`: Automated model training with 41-factor data generation and comprehensive validation.
  - `config/config.py`: Environment-based configuration with factor weight management.
  - `src/model.py`: XGBoost model implementation with 41-factor support and comprehensive error handling.
  - `src/data_processing.py`: Advanced data preprocessing with environmental, social, and infrastructure factor engineering.
  - `src/visualization.py`: Enhanced plotting utilities for multi-factor analysis and quality scoring visualization.
  - `src/explainer.py`: SHAP model explanations for complex factor interactions and importance analysis.
  - `src/data_validators.py`: Comprehensive validation for all 41 factors with real estate market logic.
  - `src/exceptions.py`: Custom exception hierarchy for advanced real estate data handling.
  - `tests/`: Complete test suite with 41-factor validation and market logic testing.

#### Installation & Quick Start
```bash
# Clone and setup
git clone <repository-url>
cd Housing-Price-Predictor
pip install -r requirements.txt

# Copy environment configuration (optional)
cp .env.example .env

# Run tests to verify installation
python run_tests.py

# Train model with 41-factor system (generates comprehensive data, optimizes parameters)
python train_model.py

# Launch web application
streamlit run app.py
```

#### Interface Features
The application provides a comprehensive interface for real estate analysis:

**Advanced Interface** (`app.py`):
- Full control over all 41 factors
- Tabbed organization: Basic Properties, Building Details, Environment, Infrastructure
- Real-time quality scoring (Environmental, Infrastructure, Building Quality)
- Comprehensive factor visualization and analysis
- Professional real estate evaluation tool

#### Testing
The project includes a comprehensive test suite:

```bash
# Run all tests
python run_tests.py

# Run specific test files
pytest tests/test_model.py -v
pytest tests/test_validators.py -v
pytest tests/test_data_processing.py -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

#### Usage Examples
```python
# Advanced prediction with 41-factor validation
from src.model import HousingPriceModel
from src.data_validators import InputValidator

model = HousingPriceModel()
model.load_model()

# Comprehensive feature set with environmental and social factors
features = {
    # Basic property characteristics
    'total_area': 75.0,
    'living_area': 52.5,
    'kitchen_area': 12.0,
    'rooms': 2,
    'ceiling_height': 2.7,
    
    # Building and location
    'building_prestige': 'elite',
    'district': 'Central',
    'year_built': 1950,  # Stalin-era building with historical value
    
    # Environmental factors
    'air_quality_index': 85,
    'noise_level': 35,
    'green_zone_distance': 0.3,
    
    # Social and infrastructure
    'crime_rate': 15,
    'hospital_distance': 1.2,
    'supermarket_distance': 0.4,
    
    # Detailed property features
    'bathrooms_count': 2,
    'bathroom_type': 'separate',
    'balcony_type': 'loggia',
    'balcony_glazed': 'panoramic',
    # ... up to 41 total factors
}

# Automatic validation and prediction
validated_features = InputValidator.validate_prediction_input(features)
prediction = model.predict_single(validated_features)

# Quality scoring analysis
from app import calculate_scores
quality_scores = calculate_scores(features)
print(f"Environmental Score: {quality_scores['environmental']:.1f}/100")
print(f"Infrastructure Score: {quality_scores['infrastructure']:.1f}/100")
print(f"Building Quality Score: {quality_scores['building_quality']:.1f}/100")

# Advanced feature importance for 41-factor analysis
try:
    feature_importance = model.get_feature_importance()
    print("Top 10 factors affecting price:")
    print(feature_importance.head(10))
except ModelNotTrainedError:
    print("Model needs to be trained first")

# Comprehensive model explanation with environmental factors
from src.explainer import ModelExplainer
explainer = ModelExplainer(model.model, model.feature_names)
explanation = explainer.explain_price_factors(X_sample, features)
```

#### Advanced Factor Analysis
```python
# Environmental impact analysis
environmental_factors = {
    'air_quality_index': 90,      # Excellent air quality
    'noise_level': 30,            # Quiet area
    'green_zone_distance': 0.2,   # Close to park
    'industrial_distance': 8.0    # Far from industry
}

# Social demographics analysis
social_factors = {
    'crime_rate': 10,             # Very safe area
    'avg_income_index': 150,      # High-income neighborhood
    'population_density': 3000    # Optimal density
}

# Infrastructure accessibility analysis
infrastructure_factors = {
    'hospital_distance': 0.8,     # Close to medical care
    'metro_distance': 0.5,        # Excellent transport
    'supermarket_distance': 0.3,  # Convenient shopping
    'clinics_count_3km': 5        # Good medical coverage
}
```

#### Configuration
Edit `.env` file or set environment variables to customize 41-factor analysis:
```bash
# Model configuration for comprehensive analysis
MODEL_TYPE=xgboost
OPTIMIZATION_TRIALS=100
SAMPLE_SIZE=10000

# Factor weights and importance
ENVIRONMENTAL_WEIGHT=0.25
SOCIAL_WEIGHT=0.20
INFRASTRUCTURE_WEIGHT=0.15
BUILDING_WEIGHT=0.40

# Paths configuration
DATA_PATH=data
MODELS_PATH=models
LOGS_PATH=logs

# Logging configuration for production monitoring
LOG_LEVEL=INFO
LOG_FILE_ENABLED=true

# Interface configuration
ENABLE_ADVANCED_INTERFACE=true
DEFAULT_LANGUAGE=ru
QUALITY_SCORING_ENABLED=true
```

#### Model Performance & Accuracy
The 41-factor system achieves superior accuracy compared to traditional models:

- **R² Score**: 0.9019 (90.19% variance explained)
- **RMSE**: $38,032 (Root Mean Square Error)
- **MAE**: $26,228 (Mean Absolute Error)
- **MAPE**: 15.99% (Mean Absolute Percentage Error)

**Top 10 Most Important Factors**:
1. **Year Built** (16.54%) - with non-linear historical value modeling
2. **Building Prestige** (15.39%) - new comprehensive classification
3. **Total Area** (14.42%) - enhanced with living/kitchen area ratios
4. **District** (7.22%) - expanded location analysis
5. **House Type** (6.87%) - detailed construction material impact
6. **Renovation** (5.42%) - comprehensive condition assessment
7. **Living Area** (4.43%) - new detailed space analysis
8. **Rooms** (3.03%) - enhanced with layout efficiency
9. **Metro Distance** (2.87%) - part of comprehensive transport analysis
10. **Bathrooms Count** (2.45%) - new detailed facility analysis

#### Error Handling & Reliability
The application includes comprehensive error handling for complex real estate data:

- **Custom Exception Hierarchy**: Specific exceptions for environmental, social, and infrastructure data errors
- **41-Factor Input Validation**: Comprehensive validation of all property, environmental, and social inputs
- **Graceful Degradation**: Application continues with intelligent defaults when some factors are unavailable
- **Detailed Logging**: Comprehensive logging for debugging complex factor interactions and monitoring
- **Automatic Feature Completion**: Smart defaults for missing factors based on available data
- **Market Logic Validation**: Ensures realistic relationships between factors (e.g., living area < total area)

#### Security Features
- **Input Sanitization**: All 41 factors are validated and sanitized with real estate market constraints
- **Safe File Paths**: Protection against path traversal attacks in data and model storage
- **Comprehensive Data Validation**: Market logic validation for all environmental and social factors
- **Error Information Filtering**: Sensitive market data is not exposed in error messages
- **Factor Range Validation**: All factors validated against realistic market ranges and relationships

#### Results and Reports
After training with the 41-factor system, the system generates:
- **Comprehensive Model Performance**: Plots showing impact of environmental, social, and infrastructure factors
- **Advanced Feature Importance**: Visualizations for all 41 factors with category groupings
- **SHAP Analysis**: Detailed explanations of factor interactions and non-linear relationships
- **Quality Score Analysis**: Environmental, Infrastructure, and Building Quality score distributions
- **Market Trend Reports**: Analysis of factor correlations and market patterns
- **Factor Validation Reports**: Comprehensive validation results for all 41 factors
- **Production-Ready Components**: Trained models with comprehensive factor analysis capabilities

## 🔧 Development

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Project Structure
```
Housing-Price-Predictor/
├── src/                           # Source code
│   ├── exceptions.py              # Custom exceptions for real estate data
│   ├── data_validators.py         # 41-factor validation system
│   ├── model.py                   # XGBoost implementation with 41-factor support
│   ├── data_processing.py         # Advanced preprocessing with environmental/social factors
│   ├── visualization.py           # Multi-factor plotting and quality score visualization
│   └── explainer.py               # SHAP explanations for complex factor interactions
├── tests/                         # Comprehensive test suite
│   ├── test_model.py              # Model tests with 41-factor validation
│   ├── test_validators.py         # Validation tests for all factor categories
│   └── test_data_processing.py    # Data processing tests with market logic
├── config/                        # Configuration management
├── data/                          # Enhanced datasets with 41 factors
├── models/                        # Trained models with factor completion
├── logs/                          # Comprehensive logging
├── reports/                       # Generated analysis reports
├── app.py                         # Advanced interface with full 41-factor control
└── train_model.py                 # 41-factor model training pipeline
```

## 🏆 Real Estate Market Examples

### Elite Building in Central District
```python
elite_property = {
    'total_area': 120, 'building_prestige': 'elite', 'year_built': 1950,
    'district': 'Central', 'ceiling_height': 3.2, 'air_quality_index': 85,
    'crime_rate': 10, 'hospital_distance': 0.8, 'bathrooms_count': 2
}
# Expected: ~$400,000+ (high historical value + excellent location)
```

### Modern Apartment in Developing Area
```python
modern_property = {
    'total_area': 75, 'building_prestige': 'standard', 'year_built': 2020,
    'district': 'Eastern', 'air_quality_index': 70, 'crime_rate': 25,
    'metro_distance': 2.0, 'supermarket_distance': 1.2
}
# Expected: ~$150,000 (modern construction, developing infrastructure)
```

### Economy Housing in Industrial Zone
```python
economy_property = {
    'total_area': 50, 'building_prestige': 'economy', 'year_built': 1970,
    'district': 'Eastern', 'air_quality_index': 40, 'noise_level': 65,
    'industrial_distance': 0.8, 'crime_rate': 45
}
# Expected: ~$80,000 (environmental challenges, older construction)
```
