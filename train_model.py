
import sys
import os
import logging
from typing import List, Dict, Any

sys.path.insert(0, 'src')

from data_processing import DataProcessor
from model import HousingPriceModel
from visualization import DataVisualizer
from explainer import ModelExplainer
from exceptions import HousingPredictorError, ModelError, DataError
from config.config import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def setup_training_environment() -> None:
    logger.info("Setting up training environment")
    
    directories = [
        config.paths.data_path,
        config.paths.models_path,
        config.paths.reports_path
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory ensured: {directory}")

def generate_and_save_data(data_processor: DataProcessor) -> pd.DataFrame:
    logger.info(f"Generating {config.data.sample_size} synthetic samples")
    
    try:
        df = data_processor.generate_sample_data(n_samples=config.data.sample_size)
        
        data_path = config.get_data_path()
        df.to_csv(data_path, index=False)
        logger.info(f"Data saved to {data_path}")
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  - Records: {len(df)}")
        logger.info(f"  - Features: {list(df.columns)}")
        logger.info(f"  - Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        
        return df
        
    except Exception as e:
        raise DataError(f"Failed to generate data: {e}")

def train_and_evaluate_model(model: HousingPriceModel, X_train: np.ndarray, 
                           y_train: np.ndarray, X_test: np.ndarray, 
                           y_test: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    logger.info("Starting model training")
    
    try:
        model.train(X_train, y_train, feature_names)
        
        metrics, y_pred = model.evaluate(X_test, y_test)
        
        logger.info("Model evaluation results:")
        logger.info(f"  - R² Score: {metrics['r2']:.4f}")
        logger.info(f"  - RMSE: ${metrics['rmse']:.2f}")
        logger.info(f"  - MAE: ${metrics['mae']:.2f}")
        logger.info(f"  - MAPE: {metrics['mape']:.2f}%")
        
        return metrics
        
    except Exception as e:
        raise ModelError(f"Model training failed: {e}")

def analyze_feature_importance(model: HousingPriceModel) -> pd.DataFrame:
    logger.info("Analyzing feature importance")
    
    try:
        feature_importance = model.get_feature_importance()
        
        logger.info("Top-10 important features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
        
    except Exception as e:
        logger.warning(f"Feature importance analysis failed: {e}")
        return pd.DataFrame()

def create_model_explainer(model: HousingPriceModel, X_train: np.ndarray, 
                         feature_names: List[str]) -> ModelExplainer:
    logger.info("Creating model explainer")
    
    try:
        explainer = ModelExplainer(model.model, feature_names)
        explainer.create_explainer(X_train[:100], model_type='tree')
        logger.info("Model explainer created successfully")
        return explainer
        
    except Exception as e:
        logger.warning(f"Failed to create explainer: {e}")
        return None

def generate_visualizations(df: pd.DataFrame, feature_importance: pd.DataFrame, 
                          y_test: np.ndarray, y_pred: np.ndarray, 
                          explainer: ModelExplainer, X_test: np.ndarray) -> None:
    logger.info("Creating visualizations")
    
    try:
        visualizer = DataVisualizer()
        
        visualizations = [
            ('price_distribution.png', lambda: visualizer.plot_price_distribution(df)),
            ('price_vs_area.png', lambda: visualizer.plot_price_vs_area(df)),
            ('price_by_district.png', lambda: visualizer.plot_price_by_district(df)),
            ('model_performance.png', lambda: visualizer.plot_model_performance(y_test, y_pred))
        ]
        
        if not feature_importance.empty:
            visualizations.append(
                ('feature_importance.png', lambda: visualizer.plot_feature_importance(feature_importance))
            )
        
        if explainer is not None:
            visualizations.append(
                ('shap_summary.png', lambda: explainer.plot_summary(X_test[:100]))
            )
        
        for filename, plot_func in visualizations:
            try:
                fig = plot_func()
                filepath = os.path.join(config.paths.models_path, filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.debug(f"Saved visualization: {filename}")
            except Exception as e:
                logger.warning(f"Failed to create {filename}: {e}")
        
        logger.info(f"Visualizations saved to {config.paths.models_path}/")
        
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")

def run_test_predictions(model: HousingPriceModel, data_processor: DataProcessor) -> None:
    logger.info("Running test predictions")
    
    test_cases = [
        {
            'name': 'Average Apartment',
            'features': {
                'area': 75.0, 'rooms': 2, 'floor': 5, 'total_floors': 12,
                'year_built': 2010, 'district': 'Central', 'metro_distance': 1.5,
                'parking': 1, 'elevator': 1, 'balcony': 1,
                'renovation': 'European renovation', 'house_type': 'Monolithic'
            }
        },
        {
            'name': 'Expensive Apartment',
            'features': {
                'area': 120.0, 'rooms': 3, 'floor': 15, 'total_floors': 25,
                'year_built': 2020, 'district': 'Central', 'metro_distance': 0.5,
                'parking': 1, 'elevator': 1, 'balcony': 1,
                'renovation': 'European renovation', 'house_type': 'Monolithic'
            }
        },
        {
            'name': 'Budget Apartment',
            'features': {
                'area': 45.0, 'rooms': 1, 'floor': 2, 'total_floors': 9,
                'year_built': 1995, 'district': 'Eastern', 'metro_distance': 3.0,
                'parking': 0, 'elevator': 0, 'balcony': 0,
                'renovation': 'No renovation', 'house_type': 'Panel'
            }
        }
    ]
    
    logger.info("Test predictions:")
    for test_case in test_cases:
        try:
            test_df = pd.DataFrame([test_case['features']])
            test_df_processed = data_processor.preprocess_data(test_df, fit=False)
            
            expected_features = model.feature_names
            for col in expected_features:
                if col not in test_df_processed.columns:
                    test_df_processed[col] = 0
            
            test_df_processed = test_df_processed[expected_features]
            test_X, _, _ = data_processor.prepare_features(test_df_processed)
            
            prediction = model.predict(test_X)[0]
            logger.info(f"  {test_case['name']}: ${prediction:,.0f}")
            
        except Exception as e:
            logger.warning(f"Test prediction failed for {test_case['name']}: {e}")

def main():
    print("🏠 Housing Price Predictor - XGBoost Training")
    print("=" * 60)
    
    try:
        setup_training_environment()
        
        logger.info("Initializing components")
        data_processor = DataProcessor()
        model = HousingPriceModel(model_type=config.model.model_type)
        
        df = generate_and_save_data(data_processor)
        
        logger.info("Preprocessing data")
        df_processed = data_processor.preprocess_data(df, fit=True)
        X, y, feature_names = data_processor.prepare_features(df_processed)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Feature names: {feature_names}")
        
        logger.info("Splitting data")
        X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
        logger.info(f"Training set: {X_train.shape[0]} records")
        logger.info(f"Test set: {X_test.shape[0]} records")
        
        metrics = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, feature_names)
        
        feature_importance = analyze_feature_importance(model)
        
        explainer = create_model_explainer(model, X_train, feature_names)
        
        y_pred = model.predict(X_test)
        generate_visualizations(df, feature_importance, y_test, y_pred, explainer, X_test)
        
        logger.info("Saving model and preprocessor")
        model.save_model()
        data_processor.save_preprocessor()
        
        run_test_predictions(model, data_processor)
        
        print("\n✅ XGBoost model training completed successfully!")
        print("\n📊 Results:")
        print(f"   - Model quality (R²): {metrics['r2']:.4f}")
        print(f"   - Average error (MAE): ${metrics['mae']:.0f}")
        print(f"   - Training data: {len(df)} records")
        print(f"   - Model saved: {config.get_model_path()}")
        print(f"   - Preprocessor saved: {config.get_preprocessor_path()}")
        print(f"   - Data saved: {config.get_data_path()}")
        print(f"   - Charts saved: {config.paths.models_path}/")
        
        print("\n🚀 To run the web application, execute:")
        print("   streamlit run app.py")
        
    except HousingPredictorError as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 