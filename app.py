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

sys.path.append('src')

from data_processing import DataProcessor
from model import HousingPriceModel
from visualization import DataVisualizer
from explainer import ModelExplainer

st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

translations = {
    'en': {
        'title': '🏠 Housing Price Predictor',
        'subtitle': 'Intelligent real estate price prediction system',
        'navigation': '📊 Navigation',
        'language_selector': '🌐 Language',
        'language_selector_label': 'Choose language:',
        'select_section': 'Select section:',
        'prediction': '🏠 Price Prediction',
        'data_analysis': '📈 Data Analysis',
        'model_explanation': '🔍 Model Explanation',
        'dashboard': '📊 Dashboard',
        'property_characteristics': '📋 Property Characteristics',
        'area': 'Area (sq.m)',
        'rooms': 'Number of rooms',
        'floor': 'Floor',
        'total_floors': 'Total floors in building',
        'year_built': 'Year built',
        'district': 'District',
        'metro_distance': 'Distance to metro (km)',
        'parking': 'Parking',
        'elevator': 'Elevator',
        'balcony': 'Balcony',
        'renovation': 'Renovation',
        'house_type': 'House type',
        'get_prediction': 'Get Prediction',
        'prediction_result': '💰 Prediction Result',
        'predicted_price': 'Predicted price:',
        'explanation': '🔍 Prediction Explanation',
        'enter_characteristics': '👆 Enter property characteristics and click the button to get prediction',
        'data_analysis_title': '📈 Real Estate Data Analysis',
        'total_records': 'Total records',
        'average_price': 'Average price',
        'median_price': 'Median price',
        'std_deviation': 'Standard deviation',
        'price_distribution': '📊 Price Distribution',
        'price_vs_area': '🏠 Price vs Area',
        'prices_by_district': '🗺️ Prices by District',
        'correlation_matrix': '🔗 Correlation Matrix',
        'interactive_dashboard': '📊 Interactive Dashboard',
        'model_explanation_title': '🔍 Model Explanation',
        'model_info': '📋 Model Information',
        'model_type': 'Model type:',
        'status': 'Status:',
        'feature_count': 'Number of features:',
        'quality_metrics': '📊 Quality Metrics',
        'get_quality_metrics': 'Run model training to get quality metrics',
        'feature_importance': '🎯 Feature Importance',
        'top_important_features': 'Top-10 Important Features',
        'model_not_trained': 'Model is not trained. Please train the model first.',
        'shap_explanations': '🔬 SHAP Explanations',
        'shap_unavailable': 'SHAP explanations are not available for untrained model.',
        'dashboard_title': '📊 Real Estate Analysis Dashboard',
        'filters': '🔍 Filters',
        'districts': 'Districts',
        'price_range': 'Price range (thousand ₽)',
        'area_range': 'Area range (sq.m)',
        'filtered_statistics': '📈 Statistics for selected filters',
        'records': 'Records',
        'average_area': 'Average area',
        'interactive_charts': '📊 Interactive Charts',
        'price_vs_area_by_district': 'Price vs Area by Districts',
        'price_distribution_by_district': 'Price Distribution by Districts',
        'average_prices_by_rooms': 'Average Prices by Number of Rooms',
        'data': '📋 Data',
        'help_area': 'Total apartment area',
        'help_metro': 'Distance to nearest metro station',
        'help_renovation': 'Type of renovation in the apartment',
        'help_house_type': 'Type of building construction',
        'house_types': {
            'Panel': 'Panel',
            'Brick': 'Brick',
            'Monolithic': 'Monolithic',
            'Block': 'Block',
            'Wooden': 'Wooden'
        },
        'renovation_types': {
            'No renovation': 'No renovation',
            'Cosmetic': 'Cosmetic',
            'European renovation': 'European renovation'
        },
        'districts': {
            'Central': 'Central',
            'Northern': 'Northern',
            'Southern': 'Southern',
            'Western': 'Western',
            'Eastern': 'Eastern'
        }
    },
    'ru': {
        'title': '🏠 Расчет цен на недвижимость',
        'subtitle': 'Интеллектуальная система расчета цен на недвижимость',
        'navigation': '📊 Навигация',
        'language_selector': '🌐 Язык',
        'language_selector_label': 'Выберите язык:',
        'select_section': 'Выберите раздел:',
        'prediction': '🏠 Расчет цен',
        'data_analysis': '📈 Анализ данных',
        'model_explanation': '🔍 Объяснение модели',
        'dashboard': '📊 Дашборд',
        'property_characteristics': '📋 Характеристики недвижимости',
        'area': 'Площадь (кв.м)',
        'rooms': 'Количество комнат',
        'floor': 'Этаж',
        'total_floors': 'Всего этажей в доме',
        'year_built': 'Год постройки',
        'district': 'Район',
        'metro_distance': 'Расстояние до метро (км)',
        'parking': 'Парковка',
        'elevator': 'Лифт',
        'balcony': 'Балкон',
        'renovation': 'Ремонт',
        'house_type': 'Тип дома',
        'get_prediction': 'Получить расчет',
        'prediction_result': '💰 Результат расчета',
        'predicted_price': 'Предсказанная цена:',
        'explanation': '🔍 Объяснение расчета',
        'enter_characteristics': '👆 Введите характеристики недвижимости и нажмите кнопку для получения расчета',
        'data_analysis_title': '📈 Анализ данных о недвижимости',
        'total_records': 'Всего записей',
        'average_price': 'Средняя цена',
        'median_price': 'Медианная цена',
        'std_deviation': 'Стандартное отклонение',
        'price_distribution': '📊 Распределение цен',
        'price_vs_area': '🏠 Зависимость цены от площади',
        'prices_by_district': '🗺️ Цены по районам',
        'correlation_matrix': '🔗 Корреляционная матрица',
        'interactive_dashboard': '📊 Интерактивный дашборд',
        'model_explanation_title': '🔍 Объяснение модели',
        'model_info': '📋 Информация о модели',
        'model_type': 'Тип модели:',
        'status': 'Статус:',
        'feature_count': 'Количество признаков:',
        'quality_metrics': '📊 Метрики качества',
        'get_quality_metrics': 'Для получения метрик качества запустите обучение модели',
        'feature_importance': '🎯 Важность признаков',
        'top_important_features': 'Топ-10 важных признаков',
        'model_not_trained': 'Модель не обучена. Сначала обучите модель.',
        'shap_explanations': '🔬 SHAP объяснения',
        'shap_unavailable': 'SHAP объяснения недоступны для необученной модели.',
        'dashboard_title': '📊 Дашборд анализа недвижимости',
        'filters': '🔍 Фильтры',
        'districts': 'Районы',
        'price_range': 'Диапазон цен (тыс. ₽)',
        'area_range': 'Диапазон площади (кв.м)',
        'filtered_statistics': '📈 Статистика по выбранным фильтрам',
        'records': 'Записей',
        'average_area': 'Средняя площадь',
        'interactive_charts': '📊 Интерактивные графики',
        'price_vs_area_by_district': 'Цена vs Площадь по районам',
        'price_distribution_by_district': 'Распределение цен по районам',
        'average_prices_by_rooms': 'Средние цены по количеству комнат',
        'data': '📋 Данные',
        'help_area': 'Общая площадь квартиры',
        'help_metro': 'Расстояние до ближайшей станции метро',
        'help_renovation': 'Тип ремонта в квартире',
        'help_house_type': 'Тип конструкции здания',
        'house_types': {
            'Панельный': 'Панельный',
            'Кирпичный': 'Кирпичный',
            'Монолитный': 'Монолитный',
            'Блочный': 'Блочный',
            'Деревянный': 'Деревянный'
        },
        'renovation_types': {
            'Без ремонта': 'No renovation',
            'Косметический': 'Cosmetic',
            'Евроремонт': 'European renovation'
        },
        'districts': {
            'Центральный': 'Central',
            'Северный': 'Northern',
            'Южный': 'Southern',
            'Западный': 'Western',
            'Восточный': 'Eastern'
        }
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .feature-input {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .language-selector {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_text(key, lang='en'):
    """Get translated text"""
    if lang is None:
        lang = 'en'
    if lang not in translations:
        lang = 'en'
    result = translations[lang].get(key, key)
    if result is None:
        return key
    return str(result)

@st.cache_resource
def load_model_and_data():
    """Load model and data"""
    try:

        model = HousingPriceModel()
        model.load_model()
        
        data_processor = DataProcessor()
        try:
            data_processor.load_preprocessor()
            print("Data preprocessor loaded successfully")
        except FileNotFoundError:
            print("Data preprocessor not found, will use default settings")
        
        if os.path.exists('data/housing_data.csv'):
            df = pd.read_csv('data/housing_data.csv')
        else:
            data_processor = DataProcessor()
            df = data_processor.generate_sample_data(n_samples=1000)
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/housing_data.csv', index=False)
        
        df_processed = data_processor.preprocess_data(df)
        X, _, feature_names = data_processor.prepare_features(df_processed)
        explainer = ModelExplainer(model.model, feature_names)
        explainer.create_explainer(X[:100], model_type='tree')
        
        return model, df, data_processor, explainer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first by running: python train_model.py")
        return None, None, None, None

def main():
    """Main application function"""

    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    
    model, df, data_processor, explainer = load_model_and_data()
    if model is None:
        st.stop()
    
    language = st.session_state.get('language', 'en')
    
    st.sidebar.markdown("---")
    st.sidebar.subheader(get_text("language_selector", language))
    selected_language = st.sidebar.selectbox(
        get_text("language_selector_label", language),
        ["English", "Русский"],
        index=0 if language == 'en' else 1,
        key="language_selector"
    )
    
    new_language = 'en' if selected_language == "English" else 'ru'
    
    if st.session_state.language != new_language:
        st.session_state.language = new_language

        try:
            st.experimental_rerun()
        except AttributeError:

            pass
    
    language = st.session_state.language
    
    st.sidebar.title(get_text("navigation", language))
    
    page_title = get_text("title", language)
    st.markdown(f"""
    <script>
        document.title = "{page_title}";
    </script>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<h1 class="main-header">{get_text("title", language)}</h1>', unsafe_allow_html=True)
    st.markdown(f"### {get_text('subtitle', language)}")
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        get_text("select_section", language),
        [
            get_text("prediction", language),
            get_text("data_analysis", language),
            get_text("model_explanation", language),
            get_text("dashboard", language)
        ]
    )
    
    if page == get_text("prediction", language):
        show_prediction_page(model, data_processor, explainer, language)
    elif page == get_text("data_analysis", language):
        show_data_analysis_page(df, language)
    elif page == get_text("model_explanation", language):
        show_model_explanation_page(model, df, data_processor, explainer, language)
    elif page == get_text("dashboard", language):
        show_dashboard_page(df, model, data_processor, explainer, language)

def show_prediction_page(model, data_processor, explainer, language):
    st.header(get_text("prediction", language))
    st.markdown(get_text("enter_characteristics", language))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(get_text("property_characteristics", language))
        area = st.slider(get_text("area", language), 20, 200, 75, help=get_text("help_area", language))
        rooms = st.selectbox(get_text("rooms", language), [1, 2, 3, 4, 5])
        floor = st.slider(get_text("floor", language), 1, 25, 5)
        total_floors = st.slider(get_text("total_floors", language), 5, 25, 12)
        year_built = st.slider(get_text("year_built", language), 1960, 2024, 2010)
        
        if language == 'ru':
            district_options = ['Центральный', 'Северный', 'Южный', 'Западный', 'Восточный']
            district_mapping = {
                'Центральный': 'Central', 'Северный': 'Northern', 'Южный': 'Southern',
                'Западный': 'Western', 'Восточный': 'Eastern'
            }
        else:
            district_options = ['Central', 'Northern', 'Southern', 'Western', 'Eastern']
            district_mapping = {d: d for d in district_options}
        
        district_display = st.selectbox(get_text("district", language), district_options)
        district = district_mapping[district_display]
        
        metro_distance = st.slider(get_text("metro_distance", language), 0.1, 10.0, 1.5, 0.1, help=get_text("help_metro", language))
        parking = st.checkbox(get_text("parking", language), value=True)
        elevator = st.checkbox(get_text("elevator", language), value=True)
        balcony = st.checkbox(get_text("balcony", language), value=True)
        
        if language == 'ru':
            renovation_options = ['Без ремонта', 'Косметический', 'Евроремонт']
            renovation_mapping = {
                'Без ремонта': 'No renovation', 'Косметический': 'Cosmetic', 'Евроремонт': 'European renovation'
            }
        else:
            renovation_options = ['No renovation', 'Cosmetic', 'European renovation']
            renovation_mapping = {r: r for r in renovation_options}
        
        renovation_display = st.selectbox(get_text("renovation", language), renovation_options, help=get_text("help_renovation", language))
        renovation = renovation_mapping[renovation_display]
        
        if language == 'ru':
            house_type_options = ['Панельный', 'Кирпичный', 'Монолитный', 'Блочный', 'Деревянный']
            house_type_mapping = {
                'Панельный': 'Panel', 'Кирпичный': 'Brick', 'Монолитный': 'Monolithic',
                'Блочный': 'Block', 'Деревянный': 'Wooden'
            }
        else:
            house_type_options = ['Panel', 'Brick', 'Monolithic', 'Block', 'Wooden']
            house_type_mapping = {h: h for h in house_type_options}
        
        house_type_display = st.selectbox(get_text("house_type", language), house_type_options, help=get_text("help_house_type", language))
        house_type = house_type_mapping[house_type_display]
        
        if st.button(get_text("get_prediction", language), type="primary"):
            features = {
                'area': area,
                'rooms': rooms,
                'floor': floor,
                'total_floors': total_floors,
                'year_built': year_built,
                'district': district,
                'metro_distance': metro_distance,
                'parking': 1 if parking else 0,
                'elevator': 1 if elevator else 0,
                'balcony': 1 if balcony else 0,
                'renovation': renovation,
                'house_type': house_type
            }
            
            test_df = pd.DataFrame([features])
            test_df_processed = data_processor.preprocess_data(test_df)
            expected_features = model.feature_names
            for col in expected_features:
                if col not in test_df_processed.columns:
                    test_df_processed[col] = 0
            test_df_processed = test_df_processed[expected_features]
            test_X, _, _ = data_processor.prepare_features(test_df_processed)
            prediction = model.predict(test_X)[0]
            
            if explainer is not None:
                try:
                    explanation = explainer.explain_price_factors(test_X[0], features)
                except Exception as e:
                    explanation = f"Model explanation not available: {e}"
            else:
                explanation = 'Model explanation is not available.'
            
            st.session_state.prediction = prediction
            st.session_state.features = features
            st.session_state.explanation = explanation
            st.session_state.test_X = test_X
    
    with col2:
        st.subheader(get_text("prediction_result", language))
        if 'prediction' in st.session_state:
            st.markdown(f"""
            <div class="prediction-card">
                <h2>{get_text('predicted_price', language)}</h2>
                <h1 style="color: #1f77b4; font-size: 2.5rem;">{st.session_state.prediction:,.0f} ₽</h1>
                <p style="font-size: 1.2rem;">{st.session_state.prediction/1000:.1f} thousand ₽</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader(get_text("explanation", language))
            st.markdown(st.session_state.explanation)
            
            if 'test_X' in st.session_state and explainer is not None:
                try:
                    test_data = st.session_state.test_X[0].reshape(1, -1)
                    if test_data.shape[1] == len(model.feature_names):
                        fig = explainer.create_interactive_explanation(test_data[0])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Data shape mismatch for SHAP explanation")
                except Exception as e:
                    st.warning(f"SHAP explanation not available: {e}")
        else:
            st.info(get_text("enter_characteristics", language))

def show_data_analysis_page(df, language):
    """Data analysis page"""
    st.header(get_text("data_analysis_title", language))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(get_text("total_records", language), len(df))
    with col2:
        st.metric(get_text("average_price", language), f"{df['price'].mean():,.0f} ₽")
    with col3:
        st.metric(get_text("median_price", language), f"{df['price'].median():,.0f} ₽")
    with col4:
        st.metric(get_text("std_deviation", language), f"{df['price'].std():,.0f} ₽")
    
    visualizer = DataVisualizer()
    
    st.subheader(get_text("price_distribution", language))
    fig = visualizer.plot_price_distribution(df)
    st.pyplot(fig)
    
    st.subheader(get_text("price_vs_area", language))
    fig = visualizer.plot_price_vs_area(df)
    st.pyplot(fig)
    
    st.subheader(get_text("prices_by_district", language))
    fig = visualizer.plot_price_by_district(df)
    st.pyplot(fig)
    
    st.subheader(get_text("correlation_matrix", language))
    fig = visualizer.plot_correlation_matrix(df)
    st.pyplot(fig)
    
    st.subheader(get_text("interactive_dashboard", language))
    fig = visualizer.create_price_analysis_dashboard(df)
    st.plotly_chart(fig, use_container_width=True)

def show_model_explanation_page(model, df, data_processor, explainer, language):
    """Model explanation page"""
    st.header(get_text("model_explanation_title", language))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(get_text("model_info", language))
        st.info(f"**{get_text('model_type', language)}** {model.model_type}")
        st.info(f"**{get_text('status', language)}** {'Trained' if model.is_trained else 'Not trained'}")
        st.info(f"**{get_text('feature_count', language)}** {len(model.feature_names) if model.feature_names else 12}")
    
    with col2:
        st.subheader(get_text("quality_metrics", language))
        st.info(get_text("get_quality_metrics", language))
    
    st.subheader(get_text("feature_importance", language))
    if model.is_trained:
        try:
            feature_importance = model.get_feature_importance()
            
            fig = px.bar(
                feature_importance.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title=get_text("top_important_features", language),
                color='importance',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(feature_importance, use_container_width=True)
        except Exception as e:
            st.warning(f"Feature importance not available: {e}")
    else:
        st.warning(get_text("model_not_trained", language))
    
    st.subheader(get_text("shap_explanations", language))
    if model.is_trained and explainer is not None:
        try:
            df_processed = data_processor.preprocess_data(df)
            X, _, _ = data_processor.prepare_features(df_processed)
            fig = explainer.plot_summary(X[:100])
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP explanations not available: {e}")
    else:
        st.warning(get_text("shap_unavailable", language))

def show_dashboard_page(df, model, data_processor, explainer, language):
    """Dashboard page"""
    if language is None:
        language = 'en'
    
    st.header(get_text("dashboard_title", language) or "Dashboard")
    
    if language == 'ru':
        district_display_mapping = {
            'Central': 'Центральный',
            'Northern': 'Северный', 
            'Southern': 'Южный',
            'Western': 'Западный',
            'Eastern': 'Восточный'
        }
        district_value_mapping = {v: k for k, v in district_display_mapping.items()}
    else:
        district_display_mapping = {d: d for d in df['district'].unique()}
        district_value_mapping = {d: d for d in df['district'].unique()}
    
    st.subheader(get_text("filters", language) or "Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:

        district_options = [district_display_mapping.get(d, d) for d in df['district'].unique()]
        selected_districts_display = st.multiselect(
            get_text("districts", language) or "Districts",
            options=district_options,
            default=district_options
        )

        selected_districts = [district_value_mapping[d] for d in selected_districts_display]
    
    with col2:
        min_price, max_price = st.slider(
            get_text("price_range", language) or "Price range",
            min_value=int(df['price'].min()/1000),
            max_value=int(df['price'].max()/1000),
            value=(int(df['price'].min()/1000), int(df['price'].max()/1000))
        )
    
    with col3:
        min_area, max_area = st.slider(
            get_text("area_range", language) or "Area range",
            min_value=int(df['area'].min()),
            max_value=int(df['area'].max()),
            value=(int(df['area'].min()), int(df['area'].max()))
        )
    
    filtered_df = df[
        (df['district'].isin(selected_districts)) &
        (df['price'] >= min_price * 1000) &
        (df['price'] <= max_price * 1000) &
        (df['area'] >= min_area) &
        (df['area'] <= max_area)
    ]
    
    st.subheader(get_text("filtered_statistics", language) or "Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(get_text("records", language) or "Records", len(filtered_df))
    with col2:
        st.metric(get_text("average_price", language) or "Average price", f"{filtered_df['price'].mean():,.0f} ₽")
    with col3:
        st.metric(get_text("median_price", language) or "Median price", f"{filtered_df['price'].median():,.0f} ₽")
    with col4:
        st.metric(get_text("average_area", language) or "Average area", f"{filtered_df['area'].mean():.1f} sq.m")
    
    st.subheader(get_text("interactive_charts", language) or "Interactive Charts")
    
    display_df = filtered_df.copy()
    display_df['district_display'] = display_df['district'].map(district_display_mapping)
    
    fig = px.scatter(
        display_df,
        x='area',
        y='price',
        color='district_display',
        size='rooms',
        hover_data=['floor', 'year_built', 'metro_distance'],
        title=get_text("price_vs_area_by_district", language) or "Price vs Area by Districts"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.box(
        display_df,
        x='district_display',
        y='price',
        title=get_text("price_distribution_by_district", language) or "Price Distribution by Districts"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.bar(
        display_df.groupby('rooms')['price'].mean().reset_index(),
        x='rooms',
        y='price',
        title=get_text("average_prices_by_rooms", language) or "Average Prices by Number of Rooms"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(get_text("data", language) or "Data")
    st.dataframe(display_df.drop('district_display', axis=1), use_container_width=True)

if __name__ == "__main__":
    main() 