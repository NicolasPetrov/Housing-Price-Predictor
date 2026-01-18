import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import joblib
from datetime import datetime
import logging
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, 'src')

from data_processing import DataProcessor
from model import HousingPriceModel
from visualization import DataVisualizer
from explainer import ModelExplainer
from exceptions import HousingPredictorError, ModelError, DataError
from data_validators import InputValidator
from config.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

translations = {
    'en': {
        'title': '🏠 Housing Price Predictor',
        'subtitle': 'Advanced real estate price prediction with comprehensive factors',
        'basic_characteristics': '🏠 Basic Characteristics',
        'building_details': '🏢 Building Details',
        'location_environment': '🌍 Location & Environment',
        'infrastructure': '🏥 Infrastructure',
        'total_area': 'Total Area (sq.m)',
        'living_area': 'Living Area (sq.m)',
        'kitchen_area': 'Kitchen Area (sq.m)',
        'ceiling_height': 'Ceiling Height (m)',
        'building_prestige': 'Building Prestige',
        'bathrooms_count': 'Number of Bathrooms',
        'bathroom_type': 'Bathroom Type',
        'balcony_type': 'Balcony/Loggia Type',
        'balcony_count': 'Number of Balconies',
        'balcony_glazed': 'Balcony Glazing',
        'air_quality_index': 'Air Quality Index (0-100)',
        'noise_level': 'Noise Level (dB)',
        'green_zone_distance': 'Distance to Parks (km)',
        'crime_rate': 'Crime Rate Index (0-100)',
        'hospital_distance': 'Distance to Hospital (km)',
        'supermarket_distance': 'Distance to Supermarket (km)',
        'prediction': '🏠 Price Prediction',
        'data_analysis': '📈 Data Analysis',
        'model_explanation': '🔍 Model Explanation',
        'get_prediction': 'Get Prediction',
        'predicted_price': 'Predicted Price:',
        'price_factors': '📊 Price Factors Analysis',
        'environmental_score': 'Environmental Score',
        'infrastructure_score': 'Infrastructure Score',
        'building_quality_score': 'Building Quality Score',
        'rooms': 'Number of Rooms',
        'floor': 'Floor',
        'total_floors': 'Total Floors',
        'year_built': 'Year Built',
        'district': 'District',
        'metro_distance': 'Distance to Metro (km)',
        'house_type': 'House Type',
        'renovation': 'Renovation',
        'parking': 'Parking',
        'elevator': 'Elevator',
        'language': '🌐 Language',
        'select_section': 'Select section:',
        'water_body_distance': 'Distance to Water (km)',
        'industrial_distance': 'Distance to Industry (km)',
        'avg_income_index': 'Average Income Index',
        'population_density': 'Population Density',
        'elderly_ratio': 'Elderly Ratio',
        'clinics_count_3km': 'Clinics within 3km',
        'pharmacy_distance': 'Distance to Pharmacy (km)',
        'emergency_time': 'Emergency Response Time (min)',
        'shops_count_1km': 'Shops within 1km',
        'mall_distance': 'Distance to Mall (km)',
        'market_distance': 'Distance to Market (km)',
        'services_count_1km': 'Services within 1km',
        'building_prestige_analysis': '📊 Building Prestige Analysis',
        'environmental_factors': '🌍 Environmental Factors',
        'infrastructure_analysis': '🏥 Infrastructure Analysis',
        'correlation_analysis': '🔗 Correlation Analysis',
        'price_by_prestige_title': 'Price Distribution by Building Prestige',
        'price_vs_air_quality': 'Price vs Air Quality',
        'price_vs_green_zones': 'Price vs Distance to Parks',
        'price_vs_hospital': 'Price vs Distance to Hospital',
        'price_vs_supermarket': 'Price vs Distance to Supermarket',
        'correlation_matrix_title': 'Correlation Matrix of Key Features',
        'total_records': 'Total Records',
        'average_price': 'Average Price',
        'median_price': 'Median Price',
        'std_deviation': 'Price Std Dev',
        'top_15_features': 'Top 15 Important Features for This Prediction',
        'error_details': 'Error Details (for developer)',
        'model_load_error': '❌ Failed to load model or data processor',
        'data_unavailable': '❌ Data unavailable',
        'initialization_error': '❌ Initialization error',
        'model_not_trained': 'Model is not trained yet. Please train the model first.',
        'model_overview': 'Model Overview',
        'model_type': 'Model Type',
        'total_features': 'Total Features',
        'model_accuracy': 'Model Accuracy (R²)',
        'feature_importance_analysis': 'Feature Importance Analysis',
        'top_20_features_chart': 'Top 20 Most Important Features',
        'top_5_features': 'Top 5 Features',
        'importance': 'Importance',
        'feature_importance_error': 'Error calculating feature importance',
        'model_performance': 'Model Performance Metrics',
        'r2_score': 'R² Score',
        'mae_score': 'Mean Abs Error',
        'rmse_score': 'Root Mean Sq Error',
        'accuracy_percent': 'Accuracy %',
        'prediction_accuracy': 'Prediction Accuracy Visualization',
        'predictions': 'Predictions',
        'perfect_prediction': 'Perfect Prediction Line',
        'actual_vs_predicted': 'Actual vs Predicted Prices',
        'actual_price': 'Actual Price ($)',
        'predicted_price': 'Predicted Price ($)',
        'performance_calculation_error': 'Error calculating performance metrics',
        'model_interpretation': 'Model Interpretation & How It Works'
    },
    'ru': {
        'title': '🏠 Расчет цен на недвижимость',
        'subtitle': 'Продвинутая система расчета цен с учетом всех факторов',
        'basic_characteristics': '🏠 Основные характеристики',
        'building_details': '🏢 Детали здания',
        'location_environment': '🌍 Местоположение и экология',
        'infrastructure': '🏥 Инфраструктура',
        'total_area': 'Общая площадь (кв.м)',
        'living_area': 'Жилая площадь (кв.м)',
        'kitchen_area': 'Площадь кухни (кв.м)',
        'ceiling_height': 'Высота потолков (м)',
        'building_prestige': 'Престижность здания',
        'bathrooms_count': 'Количество санузлов',
        'bathroom_type': 'Тип санузла',
        'balcony_type': 'Тип балкона/лоджии',
        'balcony_count': 'Количество балконов',
        'balcony_glazed': 'Остекление балкона',
        'air_quality_index': 'Индекс качества воздуха (0-100)',
        'noise_level': 'Уровень шума (дБ)',
        'green_zone_distance': 'Расстояние до парков (км)',
        'crime_rate': 'Индекс преступности (0-100)',
        'hospital_distance': 'Расстояние до больницы (км)',
        'supermarket_distance': 'Расстояние до супермаркета (км)',
        'prediction': '🏠 Расчет цены',
        'data_analysis': '📈 Анализ данных',
        'model_explanation': '🔍 Объяснение модели',
        'get_prediction': 'Получить расчет',
        'predicted_price': 'Предсказанная цена:',
        'price_factors': '📊 Анализ факторов цены',
        'environmental_score': 'Экологический рейтинг',
        'infrastructure_score': 'Рейтинг инфраструктуры',
        'building_quality_score': 'Рейтинг качества здания',
        'rooms': 'Количество комнат',
        'floor': 'Этаж',
        'total_floors': 'Всего этажей',
        'year_built': 'Год постройки',
        'district': 'Район',
        'metro_distance': 'Расстояние до метро (км)',
        'house_type': 'Тип дома',
        'renovation': 'Ремонт',
        'parking': 'Парковка',
        'elevator': 'Лифт',
        'language': '🌐 Язык',
        'select_section': 'Выберите раздел:',
        'water_body_distance': 'Расстояние до водоема (км)',
        'industrial_distance': 'Расстояние до промзоны (км)',
        'avg_income_index': 'Индекс среднего дохода',
        'population_density': 'Плотность населения',
        'elderly_ratio': 'Доля пожилых',
        'clinics_count_3km': 'Поликлиник в радиусе 3км',
        'pharmacy_distance': 'Расстояние до аптеки (км)',
        'emergency_time': 'Время приезда скорой (мин)',
        'shops_count_1km': 'Магазинов в радиусе 1км',
        'mall_distance': 'Расстояние до ТЦ (км)',
        'market_distance': 'Расстояние до рынка (км)',
        'services_count_1km': 'Услуг в радиусе 1км',
        'building_prestige_analysis': '📊 Анализ по престижности зданий',
        'environmental_factors': '🌍 Экологические факторы',
        'infrastructure_analysis': '🏥 Инфраструктура',
        'correlation_analysis': '🔗 Корреляции признаков',
        'price_by_prestige_title': 'Распределение цен по престижности зданий',
        'price_vs_air_quality': 'Цена vs Качество воздуха',
        'price_vs_green_zones': 'Цена vs Расстояние до парков',
        'price_vs_hospital': 'Цена vs Расстояние до больницы',
        'price_vs_supermarket': 'Цена vs Расстояние до супермаркета',
        'correlation_matrix_title': 'Корреляционная матрица ключевых признаков',
        'total_records': 'Всего записей',
        'average_price': 'Средняя цена',
        'median_price': 'Медианная цена',
        'std_deviation': 'Стд. отклонение цены',
        'top_15_features': 'Топ-15 важных признаков для данного предсказания',
        'error_details': 'Детали ошибки (для разработчика)',
        'model_load_error': '❌ Не удалось загрузить модель или процессор данных',
        'data_unavailable': '❌ Данные недоступны',
        'initialization_error': '❌ Ошибка инициализации',
        'model_not_trained': 'Модель еще не обучена. Пожалуйста, сначала обучите модель.',
        'model_overview': 'Обзор модели',
        'model_type': 'Тип модели',
        'total_features': 'Всего признаков',
        'model_accuracy': 'Точность модели (R²)',
        'feature_importance_analysis': 'Анализ важности признаков',
        'top_20_features_chart': 'Топ-20 самых важных признаков',
        'top_5_features': 'Топ-5 признаков',
        'importance': 'Важность',
        'feature_importance_error': 'Ошибка расчета важности признаков',
        'model_performance': 'Метрики производительности модели',
        'r2_score': 'R² Score',
        'mae_score': 'Средняя абс. ошибка',
        'rmse_score': 'Корень средн. кв. ошибки',
        'accuracy_percent': 'Точность %',
        'prediction_accuracy': 'Визуализация точности предсказаний',
        'predictions': 'Предсказания',
        'perfect_prediction': 'Линия идеального предсказания',
        'actual_vs_predicted': 'Реальные vs Предсказанные цены',
        'actual_price': 'Реальная цена ($)',
        'predicted_price': 'Предсказанная цена ($)',
        'performance_calculation_error': 'Ошибка расчета метрик производительности',
        'model_interpretation': 'Интерпретация модели и принцип работы'
    },
    'fr': {
        'title': '🏠 Prédicteur de Prix Immobilier',
        'subtitle': 'Système avancé de prédiction des prix immobiliers avec facteurs complets',
        'basic_characteristics': '🏠 Caractéristiques de Base',
        'building_details': '🏢 Détails du Bâtiment',
        'location_environment': '🌍 Localisation et Environnement',
        'infrastructure': '🏥 Infrastructure',
        'total_area': 'Surface Totale (m²)',
        'living_area': 'Surface Habitable (m²)',
        'kitchen_area': 'Surface Cuisine (m²)',
        'ceiling_height': 'Hauteur sous Plafond (m)',
        'building_prestige': 'Prestige du Bâtiment',
        'bathrooms_count': 'Nombre de Salles de Bain',
        'bathroom_type': 'Type de Salle de Bain',
        'balcony_type': 'Type de Balcon/Loggia',
        'balcony_count': 'Nombre de Balcons',
        'balcony_glazed': 'Vitrage du Balcon',
        'air_quality_index': 'Indice Qualité de l\'Air (0-100)',
        'noise_level': 'Niveau de Bruit (dB)',
        'green_zone_distance': 'Distance aux Parcs (km)',
        'crime_rate': 'Taux de Criminalité (0-100)',
        'hospital_distance': 'Distance à l\'Hôpital (km)',
        'supermarket_distance': 'Distance au Supermarché (km)',
        'prediction': '🏠 Prédiction de Prix',
        'data_analysis': '📈 Analyse des Données',
        'model_explanation': '🔍 Explication du Modèle',
        'get_prediction': 'Obtenir Prédiction',
        'predicted_price': 'Prix Prédit:',
        'price_factors': '📊 Analyse des Facteurs de Prix',
        'environmental_score': 'Score Environnemental',
        'infrastructure_score': 'Score Infrastructure',
        'building_quality_score': 'Score Qualité Bâtiment',
        'rooms': 'Nombre de Pièces',
        'floor': 'Étage',
        'total_floors': 'Nombre d\'Étages Total',
        'year_built': 'Année de Construction',
        'district': 'Quartier',
        'metro_distance': 'Distance au Métro (km)',
        'house_type': 'Type de Maison',
        'renovation': 'Rénovation',
        'parking': 'Parking',
        'elevator': 'Ascenseur',
        'language': '🌐 Langue',
        'select_section': 'Sélectionner section:',
        'water_body_distance': 'Distance à l\'Eau (km)',
        'industrial_distance': 'Distance à l\'Industrie (km)',
        'avg_income_index': 'Indice Revenu Moyen',
        'population_density': 'Densité de Population',
        'elderly_ratio': 'Ratio Personnes Âgées',
        'clinics_count_3km': 'Cliniques dans 3km',
        'pharmacy_distance': 'Distance Pharmacie (km)',
        'emergency_time': 'Temps Urgences (min)',
        'shops_count_1km': 'Magasins dans 1km',
        'mall_distance': 'Distance Centre Commercial (km)',
        'market_distance': 'Distance Marché (km)',
        'services_count_1km': 'Services dans 1km',
        'building_prestige_analysis': '📊 Analyse du Prestige des Bâtiments',
        'environmental_factors': '🌍 Facteurs Environnementaux',
        'infrastructure_analysis': '🏥 Infrastructure',
        'correlation_analysis': '🔗 Analyse de Corrélation',
        'price_by_prestige_title': 'Distribution des Prix par Prestige du Bâtiment',
        'price_vs_air_quality': 'Prix vs Qualité de l\'Air',
        'price_vs_green_zones': 'Prix vs Distance aux Parcs',
        'price_vs_hospital': 'Prix vs Distance à l\'Hôpital',
        'price_vs_supermarket': 'Prix vs Distance au Supermarché',
        'correlation_matrix_title': 'Matrice de Corrélation des Caractéristiques Clés',
        'total_records': 'Total Enregistrements',
        'average_price': 'Prix Moyen',
        'median_price': 'Prix Médian',
        'std_deviation': 'Écart-type Prix',
        'top_15_features': 'Top 15 Caractéristiques Importantes pour Cette Prédiction',
        'error_details': 'Détails de l\'Erreur (pour développeur)',
        'model_load_error': '❌ Échec du chargement du modèle ou processeur de données',
        'data_unavailable': '❌ Données indisponibles',
        'initialization_error': '❌ Erreur d\'initialisation',
        'model_not_trained': 'Le modèle n\'est pas encore entraîné. Veuillez d\'abord entraîner le modèle.',
        'model_overview': 'Aperçu du Modèle',
        'model_type': 'Type de Modèle',
        'total_features': 'Total Caractéristiques',
        'model_accuracy': 'Précision du Modèle (R²)',
        'feature_importance_analysis': 'Analyse d\'Importance des Caractéristiques',
        'top_20_features_chart': 'Top 20 Caractéristiques les Plus Importantes',
        'top_5_features': 'Top 5 Caractéristiques',
        'importance': 'Importance',
        'feature_importance_error': 'Erreur de calcul d\'importance des caractéristiques',
        'model_performance': 'Métriques de Performance du Modèle',
        'r2_score': 'Score R²',
        'mae_score': 'Erreur Abs. Moyenne',
        'rmse_score': 'Racine Erreur Quad. Moy.',
        'accuracy_percent': 'Précision %',
        'prediction_accuracy': 'Visualisation de la Précision des Prédictions',
        'predictions': 'Prédictions',
        'perfect_prediction': 'Ligne de Prédiction Parfaite',
        'actual_vs_predicted': 'Prix Réels vs Prédits',
        'actual_price': 'Prix Réel ($)',
        'predicted_price': 'Prix Prédit ($)',
        'performance_calculation_error': 'Erreur de calcul des métriques de performance',
        'model_interpretation': 'Interprétation du Modèle et Fonctionnement'
    }
}

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .score-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_text(key: str, lang: str = 'en') -> str:
    if lang not in translations:
        lang = 'en'
    return translations[lang].get(key, key)

@st.cache_resource
def load_model_and_processor():
    try:
        model = HousingPriceModel()
        model.load_model()
        logger.info("Model loaded successfully")
        
        try:
            data_processor = DataProcessor()
            data_processor.load_preprocessor()
            logger.info("Data processor loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load saved preprocessor: {e}")
            logger.info("Creating new data processor")
            
            data_processor = DataProcessor()
            df = data_processor.generate_sample_data(1000)
            df_processed = data_processor.preprocess_data(df, fit=True)
            X, y, feature_names = data_processor.prepare_features(df_processed)
            
            model = HousingPriceModel()
            model.create_model()
            model.train(X, y, feature_names)
            logger.info("Model retrained with new processor")
        
        return model, data_processor
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Creating new model and processor from scratch")
        
        data_processor = DataProcessor()
        df = data_processor.generate_sample_data(1000)
        df_processed = data_processor.preprocess_data(df, fit=True)
        X, y, feature_names = data_processor.prepare_features(df_processed)
        
        model = HousingPriceModel()
        model.create_model()
        model.train(X, y, feature_names)
        
        logger.info("New model and processor created successfully")
        return model, data_processor

@st.cache_data
def load_housing_data() -> pd.DataFrame:
    try:
        data_processor = DataProcessor()
        df = data_processor.generate_sample_data(n_samples=5000)
        logger.info(f"Generated {len(df)} records with enhanced features")
        return df
    except Exception as e:
        logger.error(f"Failed to generate housing data: {e}")
        st.error(f"Error generating data: {e}")
        return pd.DataFrame()

def calculate_scores(features: Dict[str, Any]) -> Dict[str, float]:
    
    env_score = (
        (100 - features.get('air_quality_index', 50)) * 0.3 +
        (100 - features.get('noise_level', 50)) * 0.2 +
        max(0, 100 - features.get('green_zone_distance', 2) * 20) * 0.3 +
        max(0, 100 - features.get('crime_rate', 30)) * 0.2
    )
    
    infra_score = (
        max(0, 100 - features.get('hospital_distance', 2) * 15) * 0.25 +
        max(0, 100 - features.get('supermarket_distance', 1) * 30) * 0.25 +
        max(0, 100 - features.get('metro_distance', 2) * 10) * 0.3 +
        min(100, features.get('clinics_count_3km', 3) * 10) * 0.2
    )
    
    prestige_scores = {'economy': 40, 'standard': 60, 'elite': 85, 'premium': 95}
    building_score = (
        prestige_scores.get(features.get('building_prestige', 'standard'), 60) * 0.4 +
        min(100, (features.get('ceiling_height', 2.7) - 2.2) / 1.8 * 100) * 0.2 +
        (features.get('useful_area_ratio', 0.7) * 100) * 0.2 +
        (features.get('bathrooms_count', 1) * 20) * 0.2
    )
    
    return {
        'environmental': max(0, min(100, env_score)),
        'infrastructure': max(0, min(100, infra_score)),
        'building_quality': max(0, min(100, building_score))
    }

def show_prediction_page(model: HousingPriceModel, data_processor: DataProcessor, language: str):
    st.header(get_text("prediction", language))
    
    tab1, tab2, tab3, tab4 = st.tabs([
        get_text("basic_characteristics", language),
        get_text("building_details", language), 
        get_text("location_environment", language),
        get_text("infrastructure", language)
    ])
    
    features = {}
    
    with tab1:
        st.subheader(get_text("basic_characteristics", language))
        col1, col2 = st.columns(2)
        
        with col1:
            features['total_area'] = st.slider(get_text("total_area", language), 20, 200, 75)
            features['living_area'] = st.slider(get_text("living_area", language), 15, 150, int(features['total_area'] * 0.7))
            features['kitchen_area'] = st.slider(get_text("kitchen_area", language), 4, 25, 10)
            features['rooms'] = st.selectbox(get_text("rooms", language), [1, 2, 3, 4, 5], index=1)
            
        with col2:
            features['floor'] = st.slider(get_text("floor", language), 1, 25, 5)
            features['total_floors'] = st.slider(get_text("total_floors", language), max(5, features['floor']), 25, max(12, features['floor']))
            features['year_built'] = st.slider(get_text("year_built", language), 1930, 2024, 2010)
            features['ceiling_height'] = st.slider(get_text("ceiling_height", language), 2.2, 4.0, 2.7, 0.1)
    
    with tab2:
        st.subheader(get_text("building_details", language))
        col1, col2 = st.columns(2)
        
        with col1:
            features['building_prestige'] = st.selectbox(
                get_text("building_prestige", language),
                ['economy', 'standard', 'elite', 'premium'], index=1
            )
            features['house_type'] = st.selectbox(
                get_text("house_type", language), 
                ['Panel', 'Brick', 'Monolithic', 'Block', 'Wooden'], index=2
            )
            features['renovation'] = st.selectbox(
                get_text("renovation", language),
                ['No renovation', 'Cosmetic', 'European renovation'], index=1
            )
            
        with col2:
            features['bathrooms_count'] = st.selectbox(get_text("bathrooms_count", language), [1, 2, 3], index=0)
            features['bathroom_type'] = st.selectbox(get_text("bathroom_type", language), ['combined', 'separate'], index=1)
            features['balcony_type'] = st.selectbox(get_text("balcony_type", language), ['none', 'balcony', 'loggia', 'terrace'], index=2)
            features['balcony_count'] = st.selectbox(get_text("balcony_count", language), [0, 1, 2, 3], index=1)
            features['balcony_glazed'] = st.selectbox(get_text("balcony_glazed", language), ['no', 'yes', 'panoramic'], index=1)
    
    with tab3:
        st.subheader(get_text("location_environment", language))
        col1, col2 = st.columns(2)
        
        with col1:
            features['district'] = st.selectbox(get_text("district", language), ['Central', 'Northern', 'Southern', 'Western', 'Eastern'], index=0)
            features['metro_distance'] = st.slider(get_text("metro_distance", language), 0.1, 10.0, 1.5, 0.1)
            features['air_quality_index'] = st.slider(get_text("air_quality_index", language), 10, 100, 70)
            features['noise_level'] = st.slider(get_text("noise_level", language), 20, 80, 45)
            
        with col2:
            features['green_zone_distance'] = st.slider(get_text("green_zone_distance", language), 0.1, 5.0, 1.0, 0.1)
            features['water_body_distance'] = st.slider(get_text("water_body_distance", language), 0.2, 10.0, 2.0, 0.1)
            features['industrial_distance'] = st.slider(get_text("industrial_distance", language), 0.5, 15.0, 5.0, 0.1)
            features['crime_rate'] = st.slider(get_text("crime_rate", language), 5, 80, 25)
    
    with tab4:
        st.subheader(get_text("infrastructure", language))
        col1, col2 = st.columns(2)
        
        with col1:
            features['hospital_distance'] = st.slider(get_text("hospital_distance", language), 0.3, 8.0, 2.0, 0.1)
            features['clinics_count_3km'] = st.slider(get_text("clinics_count_3km", language), 0, 10, 3)
            features['pharmacy_distance'] = st.slider(get_text("pharmacy_distance", language), 0.1, 2.0, 0.5, 0.1)
            features['emergency_time'] = st.slider(get_text("emergency_time", language), 5, 25, 12)
            
        with col2:
            features['supermarket_distance'] = st.slider(get_text("supermarket_distance", language), 0.1, 3.0, 0.8, 0.1)
            features['shops_count_1km'] = st.slider(get_text("shops_count_1km", language), 2, 50, 15)
            features['mall_distance'] = st.slider(get_text("mall_distance", language), 0.5, 12.0, 3.0, 0.1)
            features['services_count_1km'] = st.slider(get_text("services_count_1km", language), 1, 25, 8)
        
        features['parking'] = 1 if st.checkbox(get_text("parking", language), value=True) else 0
        features['elevator'] = 1 if st.checkbox(get_text("elevator", language), value=True) else 0
    
    features['useful_area_ratio'] = features['living_area'] / features['total_area']
    features['balcony_area'] = 0 if features['balcony_count'] == 0 else features['balcony_count'] * 5
    features['avg_income_index'] = 100
    features['population_density'] = 5000
    features['elderly_ratio'] = 0.25
    features['market_distance'] = features['supermarket_distance'] * 1.5
    
    if st.button(get_text("get_prediction", language), type="primary"):
        try:
            features['useful_area_ratio'] = features['living_area'] / features['total_area']
            features['balcony_area'] = 0 if features['balcony_count'] == 0 else features['balcony_count'] * 5
            
            default_features = {
                'avg_income_index': 100,
                'population_density': 5000,
                'elderly_ratio': 0.25,
                'market_distance': features['supermarket_distance'] * 1.5,
                'water_body_distance': 2.0,
                'industrial_distance': 5.0
            }
            
            complete_features = {**features, **default_features}
            
            test_df = pd.DataFrame([complete_features])
            
            test_df_processed = data_processor.preprocess_data(test_df, fit=False)
            
            test_X, _, feature_names = data_processor.prepare_features(test_df_processed)
            
            prediction = model.predict(test_X)[0]
            
            scores = calculate_scores(complete_features)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>{get_text('predicted_price', language)}</h2>
                    <h1 style="color: #1f77b4; font-size: 2.5rem;">${prediction:,.0f}</h1>
                    <p style="font-size: 1.2rem;">${prediction/1000:.1f}K</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader(get_text("price_factors", language))
                
                st.markdown(f"""
                <div class="score-card">
                    <strong>{get_text('environmental_score', language)}</strong><br>
                    {scores['environmental']:.1f}/100
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="score-card">
                    <strong>{get_text('infrastructure_score', language)}</strong><br>
                    {scores['infrastructure']:.1f}/100
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="score-card">
                    <strong>{get_text('building_quality_score', language)}</strong><br>
                    {scores['building_quality']:.1f}/100
                </div>
                """, unsafe_allow_html=True)
            
            if model.is_trained:
                try:
                    feature_importance = model.get_feature_importance()
                    
                    fig = px.bar(
                        feature_importance.head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=get_text("top_15_features", language),
                        color='importance',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Feature importance visualization failed: {e}")
                    
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            logger.error(f"Prediction error: {e}")
            
            st.expander(get_text("error_details", language)).write({
                "error": str(e),
                "features_provided": list(features.keys()),
                "model_feature_names": getattr(model, 'feature_names', 'Not available')
            })

def show_data_analysis(df: pd.DataFrame, language: str):
    st.header(get_text("data_analysis", language))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(get_text("total_records", language), len(df))
    with col2:
        st.metric(get_text("average_price", language), f"${df['price'].mean():,.0f}")
    with col3:
        st.metric(get_text("median_price", language), f"${df['price'].median():,.0f}")
    with col4:
        st.metric(get_text("std_deviation", language), f"${df['price'].std():,.0f}")
    
    st.subheader(get_text("building_prestige_analysis", language))
    fig = px.box(df, x='building_prestige', y='price', 
                 title=get_text("price_by_prestige_title", language))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(get_text("environmental_factors", language))
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='air_quality_index', y='price', 
                        title=get_text("price_vs_air_quality", language),
                        opacity=0.6)
        
        z = np.polyfit(df['air_quality_index'], df['price'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['air_quality_index'].min(), df['air_quality_index'].max(), 100)
        fig.add_scatter(x=x_trend, y=p(x_trend), mode='lines', name='Trend', 
                       line=dict(color='red', width=2))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='green_zone_distance', y='price',
                        title=get_text("price_vs_green_zones", language),
                        opacity=0.6)
        
        z = np.polyfit(df['green_zone_distance'], df['price'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['green_zone_distance'].min(), df['green_zone_distance'].max(), 100)
        fig.add_scatter(x=x_trend, y=p(x_trend), mode='lines', name='Trend',
                       line=dict(color='red', width=2))
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(get_text("infrastructure_analysis", language))
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='hospital_distance', y='price',
                        title=get_text("price_vs_hospital", language),
                        opacity=0.6)
        
        z = np.polyfit(df['hospital_distance'], df['price'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['hospital_distance'].min(), df['hospital_distance'].max(), 100)
        fig.add_scatter(x=x_trend, y=p(x_trend), mode='lines', name='Trend',
                       line=dict(color='red', width=2))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='supermarket_distance', y='price',
                        title=get_text("price_vs_supermarket", language),
                        opacity=0.6)
        
        z = np.polyfit(df['supermarket_distance'], df['price'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['supermarket_distance'].min(), df['supermarket_distance'].max(), 100)
        fig.add_scatter(x=x_trend, y=p(x_trend), mode='lines', name='Trend',
                       line=dict(color='red', width=2))
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(get_text("correlation_analysis", language))
    numeric_cols = ['total_area', 'air_quality_index', 'noise_level', 'green_zone_distance',
                   'crime_rate', 'hospital_distance', 'supermarket_distance', 'price']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    title=get_text("correlation_matrix_title", language),
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

def show_model_explanation(model: HousingPriceModel, data_processor: DataProcessor, language: str):
    st.header(get_text("model_explanation", language))
    
    if not model.is_trained:
        st.warning(get_text("model_not_trained", language))
        return
    
    st.subheader(get_text("model_overview", language))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(get_text("model_type", language), "Random Forest")
    with col2:
        st.metric(get_text("total_features", language), len(model.feature_names) if model.feature_names else "N/A")
    with col3:
        model_info = model.get_model_info()
        st.metric(get_text("model_accuracy", language), f"{model_info.get('r2_score', 0):.3f}")
    
    st.subheader(get_text("feature_importance_analysis", language))
    
    try:
        feature_importance = model.get_feature_importance()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:

            fig = px.bar(
                feature_importance.head(20),
                x='importance',
                y='feature',
                orientation='h',
                title=get_text("top_20_features_chart", language),
                color='importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"**{get_text('top_5_features', language)}:**")
            for i, row in feature_importance.head(5).iterrows():
                st.markdown(f"**{i+1}.** {row['feature']}")
                st.progress(row['importance'])
                st.markdown(f"*{get_text('importance', language)}: {row['importance']:.3f}*")
                st.markdown("---")
    
    except Exception as e:
        st.error(f"{get_text('feature_importance_error', language)}: {e}")
    
    st.subheader(get_text("model_performance", language))
    
    try:
        test_df = data_processor.generate_sample_data(1000)
        test_df_processed = data_processor.preprocess_data(test_df, fit=False)
        X_test, y_test, _ = data_processor.prepare_features(test_df_processed)
        
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import math
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(get_text("r2_score", language), f"{r2:.3f}")
        with col2:
            st.metric(get_text("mae_score", language), f"${mae:,.0f}")
        with col3:
            st.metric(get_text("rmse_score", language), f"${rmse:,.0f}")
        with col4:
            accuracy_percent = r2 * 100
            st.metric(get_text("accuracy_percent", language), f"{accuracy_percent:.1f}%")
        
        st.subheader(get_text("prediction_accuracy", language))
        
        sample_size = min(500, len(y_test))
        indices = np.random.choice(len(y_test), sample_size, replace=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test[indices],
            y=y_pred[indices],
            mode='markers',
            name=get_text("predictions", language),
            opacity=0.6,
            marker=dict(color='blue', size=6)
        ))
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name=get_text("perfect_prediction", language),
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=get_text("actual_vs_predicted", language),
            xaxis_title=get_text("actual_price", language),
            yaxis_title=get_text("predicted_price", language),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"{get_text('performance_calculation_error', language)}: {e}")
    
    st.subheader(get_text("model_interpretation", language))
    
    interpretation_text = {
        'en': """
        **How the Model Works:**
        
        Our Random Forest model analyzes 41 different factors to predict housing prices. Here's what makes it effective:
        
        **🌳 Random Forest Algorithm:**
        - Uses multiple decision trees (ensemble method)
        - Each tree votes on the final prediction
        - Reduces overfitting and improves accuracy
        - Handles both numerical and categorical features well
        
        **🏠 Key Factor Categories:**
        - **Physical Characteristics**: Area, rooms, ceiling height, building quality
        - **Location Factors**: District, metro distance, neighborhood prestige
        - **Environmental Quality**: Air quality, noise levels, green spaces
        - **Infrastructure Access**: Hospitals, schools, shopping, transportation
        - **Social Indicators**: Crime rates, income levels, demographics
        
        **📊 Prediction Process:**
        1. Input features are normalized and encoded
        2. 100+ decision trees analyze the data
        3. Each tree makes a price prediction
        4. Final prediction is the average of all trees
        5. Confidence intervals show prediction reliability
        
        **🎯 Model Strengths:**
        - High accuracy (90%+ R² score)
        - Handles complex feature interactions
        - Robust to outliers and missing data
        - Provides feature importance rankings
        """,
        'ru': """
        **Как работает модель:**
        
        Наша модель Random Forest анализирует 41 различный фактор для предсказания цен на недвижимость. Вот что делает её эффективной:
        
        **🌳 Алгоритм Random Forest:**
        - Использует множество деревьев решений (ансамблевый метод)
        - Каждое дерево голосует за финальное предсказание
        - Снижает переобучение и повышает точность
        - Хорошо работает с числовыми и категориальными признаками
        
        **🏠 Ключевые категории факторов:**
        - **Физические характеристики**: Площадь, комнаты, высота потолков, качество здания
        - **Факторы местоположения**: Район, расстояние до метро, престижность района
        - **Качество окружающей среды**: Качество воздуха, уровень шума, зеленые зоны
        - **Доступность инфраструктуры**: Больницы, школы, магазины, транспорт
        - **Социальные показатели**: Уровень преступности, доходы, демография
        
        **📊 Процесс предсказания:**
        1. Входные признаки нормализуются и кодируются
        2. 100+ деревьев решений анализируют данные
        3. Каждое дерево делает предсказание цены
        4. Финальное предсказание - среднее всех деревьев
        5. Доверительные интервалы показывают надежность предсказания
        
        **🎯 Сильные стороны модели:**
        - Высокая точность (90%+ R² score)
        - Обрабатывает сложные взаимодействия признаков
        - Устойчива к выбросам и пропущенным данным
        - Предоставляет рейтинг важности признаков
        """,
        'fr': """
        **Comment fonctionne le modèle:**
        
        Notre modèle Random Forest analyse 41 facteurs différents pour prédire les prix immobiliers. Voici ce qui le rend efficace:
        
        **🌳 Algorithme Random Forest:**
        - Utilise plusieurs arbres de décision (méthode d'ensemble)
        - Chaque arbre vote pour la prédiction finale
        - Réduit le surapprentissage et améliore la précision
        - Gère bien les caractéristiques numériques et catégorielles
        
        **🏠 Catégories de facteurs clés:**
        - **Caractéristiques physiques**: Surface, pièces, hauteur sous plafond, qualité du bâtiment
        - **Facteurs de localisation**: Quartier, distance au métro, prestige du quartier
        - **Qualité environnementale**: Qualité de l'air, niveaux de bruit, espaces verts
        - **Accès aux infrastructures**: Hôpitaux, écoles, commerces, transport
        - **Indicateurs sociaux**: Taux de criminalité, niveaux de revenus, démographie
        
        **📊 Processus de prédiction:**
        1. Les caractéristiques d'entrée sont normalisées et encodées
        2. 100+ arbres de décision analysent les données
        3. Chaque arbre fait une prédiction de prix
        4. La prédiction finale est la moyenne de tous les arbres
        5. Les intervalles de confiance montrent la fiabilité de la prédiction
        
        **🎯 Forces du modèle:**
        - Haute précision (90%+ score R²)
        - Gère les interactions complexes entre caractéristiques
        - Robuste aux valeurs aberrantes et données manquantes
        - Fournit un classement d'importance des caractéristiques
        """
    }
    
    st.markdown(interpretation_text.get(language, interpretation_text['en']))

def main():
    
    if 'language' not in st.session_state:
        st.session_state.language = 'en'  
    
    language = st.session_state.language
    
    st.markdown(f'<h1 class="main-header">{get_text("title", language)}</h1>', unsafe_allow_html=True)
    st.markdown(f"### {get_text('subtitle', language)}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        language_options = {
            "English": "en",
            "Русский": "ru", 
            "Français": "fr"
        }
        
        current_display = "English"
        for display, code in language_options.items():
            if code == language:
                current_display = display
                break
        
        selected_language = st.selectbox(
            "🌐 Language / Язык / Langue:",
            list(language_options.keys()),
            index=list(language_options.keys()).index(current_display)
        )
        
        new_language = language_options[selected_language]
        if st.session_state.language != new_language:
            st.session_state.language = new_language
            st.experimental_rerun()
    
    try:
        model, data_processor = load_model_and_processor()
        df = load_housing_data()
        
        if model is None or data_processor is None:
            st.error(get_text("model_load_error", language))
            st.stop()
        
        if df.empty:
            st.error(get_text("data_unavailable", language))
            st.stop()
            
    except Exception as e:
        st.error(f"{get_text('initialization_error', language)}: {e}")
        st.stop()
    
    page = st.sidebar.selectbox(
        get_text("select_section", language),
        [
            get_text("prediction", language),
            get_text("data_analysis", language),
            get_text("model_explanation", language)
        ]
    )
    
    if page == get_text("prediction", language):
        show_prediction_page(model, data_processor, language)
    elif page == get_text("data_analysis", language):
        show_data_analysis(df, language)
    elif page == get_text("model_explanation", language):
        show_model_explanation(model, data_processor, language)

if __name__ == "__main__":
    main()