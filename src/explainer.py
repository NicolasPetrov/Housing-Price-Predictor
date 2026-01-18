import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_background, model_type='tree'):
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif model_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_background)
        elif model_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict, X_background)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def explain_prediction(self, X_instance):
        if self.explainer is None:
            raise ValueError("Create explainer first using create_explainer()")
        
        if isinstance(X_instance, np.ndarray) and X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        elif isinstance(X_instance, list) and np.array(X_instance).ndim == 1:
            X_instance = np.array(X_instance).reshape(1, -1)
        
        shap_values = self.explainer.shap_values(X_instance)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0][0]
        else:
            shap_values = shap_values[0]
        
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(len(shap_values))]
        
        explanation_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values,
            'abs_shap': np.abs(shap_values)
        }).sort_values('abs_shap', ascending=False)
        
        return explanation_df
    
    def explain_multiple_predictions(self, X_instances):
        if self.explainer is None:
            raise ValueError("Create explainer first using create_explainer()")
        
        shap_values = self.explainer.shap_values(X_instances)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        return shap_values
    
    def plot_waterfall(self, X_instance, figsize=(10, 8)):
        if self.explainer is None:
            raise ValueError("Create explainer first using create_explainer()")
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[0]
        
        shap_values = self.explainer.shap_values(X_instance)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(shap_values))]
        sorted_idx = np.argsort(np.abs(shap_values))
        
        current_value = base_value
        colors = ['red' if val < 0 else 'blue' for val in shap_values]
        
        for i, idx in enumerate(sorted_idx):
            feature_name = feature_names[idx]
            shap_val = shap_values[idx]
            
            ax.arrow(i, current_value, 0, shap_val, 
                    head_width=0.3, head_length=abs(shap_val)*0.1, 
                    fc=colors[idx], ec=colors[idx], alpha=0.7)
            
            ax.text(i, current_value + shap_val/2, f'{feature_name}\n{shap_val:.2f}', 
                   ha='center', va='center', fontsize=8, rotation=45)
            
            current_value += shap_val
        
        ax.axhline(y=base_value, color='gray', linestyle='--', alpha=0.7, label='Base value')
        ax.axhline(y=current_value, color='red', linestyle='-', alpha=0.7, label='Prediction')
        
        ax.set_xlabel('Features')
        ax.set_ylabel('SHAP value')
        ax.set_title('SHAP Waterfall Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_summary(self, X_data, figsize=(12, 8)):
        if self.explainer is None:
            raise ValueError("Create explainer first using create_explainer()")
        
        shap_values = self.explainer.shap_values(X_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        plt.figure(figsize=figsize)
        feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(X_data.shape[1])]
        shap.summary_plot(shap_values, X_data, feature_names=feature_names, plot_type="bar", show=False)
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    
    def plot_force_plot(self, X_instance, figsize=(12, 6)):
        if self.explainer is None:
            raise ValueError("Create explainer first using create_explainer()")
        
        shap_values = self.explainer.shap_values(X_instance)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(shap_values))]
        
        shap.force_plot(self.explainer.expected_value, shap_values, X_instance,
                       feature_names=feature_names, show=False, matplotlib=True, ax=ax)
        
        ax.set_title('SHAP Force Plot')
        plt.tight_layout()
        return fig
    
    def create_interactive_explanation(self, X_instance):
        if self.explainer is None:
            raise ValueError("Create explainer first using create_explainer()")
        
        explanation_df = self.explain_prediction(X_instance)
        
        fig = go.Figure()
        
        colors = ['red' if val < 0 else 'blue' for val in explanation_df['shap_value']]
        
        fig.add_trace(go.Bar(
            x=explanation_df['feature'],
            y=explanation_df['shap_value'],
            marker_color=colors,
            name='SHAP values',
            text=[f'{val:.3f}' for val in explanation_df['shap_value']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Model Prediction Explanation (SHAP)',
            xaxis_title='Features',
            yaxis_title='SHAP value',
            showlegend=False,
            height=500
        )
        
        return fig
    
    def get_feature_impact_summary(self, X_instance):
        if self.explainer is None:
            raise ValueError("Create explainer first using create_explainer()")
        
        explanation_df = self.explain_prediction(X_instance)
        
        positive_impact = explanation_df[explanation_df['shap_value'] > 0]
        negative_impact = explanation_df[explanation_df['shap_value'] < 0]
        
        summary = {
            'top_positive': positive_impact.head(3)[['feature', 'shap_value']].to_dict('records'),
            'top_negative': negative_impact.head(3)[['feature', 'shap_value']].to_dict('records'),
            'total_positive': positive_impact['shap_value'].sum(),
            'total_negative': negative_impact['shap_value'].sum()
        }
        
        return summary
    
    def explain_price_factors(self, X_instance, original_features=None):
        if self.explainer is None:
            raise ValueError("Create explainer first using create_explainer()")
        
        explanation_df = self.explain_prediction(X_instance)
        
        explanation_text = "**Price prediction factors:**\n\n"
        
        for i, (_, row) in enumerate(explanation_df.head(5).iterrows()):
            feature = row['feature']
            shap_val = row['shap_value']
            
            if original_features and feature in original_features:
                feature_value = original_features[feature]
                explanation_text += f"**{i+1}. {feature}** (value: {feature_value})\n"
            else:
                explanation_text += f"**{i+1}. {feature}**\n"
            
            if shap_val > 0:
                explanation_text += f"   ➕ Increases price by ${shap_val:,.0f}\n"
            else:
                explanation_text += f"   ➖ Decreases price by ${abs(shap_val):,.0f}\n"
            explanation_text += "\n"
        
        return explanation_text 