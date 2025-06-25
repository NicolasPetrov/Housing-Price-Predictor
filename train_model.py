#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

from data_processing import DataProcessor
from model import HousingPriceModel
from visualization import DataVisualizer
from explainer import ModelExplainer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("🏠 Housing Price Predictor - Enhanced XGBoost Training")
    print("=" * 60)
    
    print("1. Initializing components...")
    data_processor = DataProcessor()
    model = HousingPriceModel(model_type='xgboost')
    visualizer = DataVisualizer()
    
    print("2. Generating large synthetic dataset...")

    df = data_processor.generate_sample_data(n_samples=10000)
    print(f"   Created {len(df)} records")
    print(f"   Features: {list(df.columns)}")
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/housing_data.csv', index=False)
    print("   Data saved to data/housing_data.csv")
    
    print("3. Preprocessing data...")
    df_processed = data_processor.preprocess_data(df)
    X, y, feature_names = data_processor.prepare_features(df_processed)
    
    print(f"   Feature size: {X.shape}")
    print(f"   Feature names: {feature_names}")
    
    print("4. Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
    print(f"   Training set: {X_train.shape[0]} records")
    print(f"   Test set: {X_test.shape[0]} records")
    
    print("5. Training enhanced XGBoost model...")

    model.train(X_train, y_train, feature_names)
    
    print("6. Evaluating model quality...")
    metrics, y_pred = model.evaluate(X_test, y_test)
    
    print("   Quality metrics:")
    print(f"   - R² Score: {metrics['r2']:.4f}")
    print(f"   - RMSE: {metrics['rmse']:.2f}")
    print(f"   - MAE: {metrics['mae']:.2f}")
    print(f"   - MAPE: {metrics['mape']:.2f}%")
    
    print("7. Analyzing feature importance...")
    feature_importance = model.get_feature_importance()
    print("   Top-10 important features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    print("8. Creating model explainer...")
    explainer = ModelExplainer(model.model, feature_names)
    explainer.create_explainer(X_train[:100], model_type='tree')
    
    print("9. Creating enhanced visualizations...")
    

    os.makedirs('reports', exist_ok=True)
    
    fig1 = visualizer.plot_price_distribution(df)
    fig1.savefig('models/price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    fig2 = visualizer.plot_price_vs_area(df)
    fig2.savefig('models/price_vs_area.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    fig3 = visualizer.plot_price_by_district(df)
    fig3.savefig('models/price_by_district.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    fig4 = visualizer.plot_feature_importance(feature_importance)
    fig4.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    fig5 = visualizer.plot_model_performance(y_test, y_pred)
    fig5.savefig('models/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    fig6 = explainer.plot_summary(X_test[:100])
    fig6.savefig('models/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig6)
    
    print("   Charts saved to models/ folder")
    
    print("10. Saving enhanced model and preprocessor...")
    model.save_model()
    data_processor.save_preprocessor()  
    
    print("11. Test prediction...")
    
    test_cases = [
        {
            'name': 'Средняя квартира',
            'features': {
                'area': 75.0, 'rooms': 2, 'floor': 5, 'total_floors': 12,
                'year_built': 2010, 'district': 'Central', 'metro_distance': 1.5,
                'parking': 1, 'elevator': 1, 'balcony': 1,
                'renovation': 'European renovation', 'house_type': 'Monolithic'
            }
        },
        {
            'name': 'Дорогая квартира',
            'features': {
                'area': 120.0, 'rooms': 3, 'floor': 15, 'total_floors': 25,
                'year_built': 2020, 'district': 'Central', 'metro_distance': 0.5,
                'parking': 1, 'elevator': 1, 'balcony': 1,
                'renovation': 'European renovation', 'house_type': 'Monolithic'
            }
        },
        {
            'name': 'Дешевая квартира',
            'features': {
                'area': 45.0, 'rooms': 1, 'floor': 2, 'total_floors': 9,
                'year_built': 1995, 'district': 'Suburban', 'metro_distance': 3.0,
                'parking': 0, 'elevator': 0, 'balcony': 0,
                'renovation': 'No renovation', 'house_type': 'Panel'
            }
        }
    ]
    
    print("   Test predictions:")
    for test_case in test_cases:
        test_df = pd.DataFrame([test_case['features']])
        test_df_processed = data_processor.preprocess_data(test_df)
        expected_features = model.feature_names
        for col in expected_features:
            if col not in test_df_processed.columns:
                test_df_processed[col] = 0
        test_df_processed = test_df_processed[expected_features]
        test_X, _, _ = data_processor.prepare_features(test_df_processed)
        
        prediction = model.predict(test_X)[0]
        print(f"   {test_case['name']}: {prediction:,.0f} ₽")
    
    test_features = {
        'area': 75.0,
        'rooms': 2,
        'floor': 5,
        'total_floors': 12,
        'year_built': 2010,
        'district': 'Central',
        'metro_distance': 1.5,
        'parking': 1,
        'elevator': 1,
        'balcony': 1,
        'renovation': 'European renovation',
        'house_type': 'Monolithic'
    }
    
    test_df = pd.DataFrame([test_features])
    test_df_processed = data_processor.preprocess_data(test_df)
    expected_features = model.feature_names
    for col in expected_features:
        if col not in test_df_processed.columns:
            test_df_processed[col] = 0
    test_df_processed = test_df_processed[expected_features]
    test_X, _, _ = data_processor.prepare_features(test_df_processed)
    
    prediction = model.predict(test_X)[0]
    print(f"   Main test prediction: {prediction:,.0f} ₽")
    
    explanation = explainer.explain_price_factors(test_X[0], test_features)
    print("   Prediction explanation:")
    print(explanation)
    
    print("\n✅ Enhanced XGBoost model training completed successfully!")
    print("\n📊 Results:")
    print(f"   - Model quality (R²): {metrics['r2']:.4f}")
    print(f"   - Average error: {metrics['mae']:.0f} ₽")
    print(f"   - Training data: {len(df)} records")
    print(f"   - Model saved: models/housing_price_model.pkl")
    print(f"   - Preprocessor saved: models/data_preprocessor.pkl")
    print(f"   - Data saved: data/housing_data.csv")
    print(f"   - Charts saved: models/")
    
    print("\n🚀 To run the web application, execute:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main() 