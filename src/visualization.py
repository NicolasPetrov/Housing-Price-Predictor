import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class DataVisualizer:
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_price_distribution(self, df, figsize=(12, 6)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        ax1.hist(df['price'], bins=30, alpha=0.7, color=self.colors[0], edgecolor='black')
        ax1.set_title('Real Estate Price Distribution')
        ax1.set_xlabel('Price (rub.)')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        ax2.boxplot(df['price'], patch_artist=True, 
                   boxprops=dict(facecolor=self.colors[1], alpha=0.7))
        ax2.set_title('Price Boxplot')
        ax2.set_ylabel('Price (rub.)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_price_vs_area(self, df, figsize=(10, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(df['area'], df['price'], alpha=0.6, 
                           c=df['price'], cmap='viridis', s=50)
        ax.set_xlabel('Area (sq.m)')
        ax.set_ylabel('Price (rub.)')
        ax.set_title('Price vs Area Relationship')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Price (rub.)')
        
        plt.tight_layout()
        return fig
    
    def plot_price_by_district(self, df, figsize=(12, 6)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        district_data = [df[df['district'] == district]['price'] 
                        for district in df['district'].unique()]
        district_names = df['district'].unique()
        
        bp = ax1.boxplot(district_data, labels=district_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], self.colors[:len(district_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Price Distribution by District')
        ax1.set_ylabel('Price (rub.)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        avg_prices = df.groupby('district')['price'].mean().sort_values(ascending=False)
        bars = ax2.bar(avg_prices.index, avg_prices.values, 
                      color=self.colors[:len(avg_prices)], alpha=0.7)
        ax2.set_title('Average Prices by District')
        ax2.set_ylabel('Average Price (rub.)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, avg_prices.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'{value:,.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, df, figsize=(10, 8)):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, ax=ax)
        
        ax.set_title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance, figsize=(10, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        bars = ax.barh(range(len(feature_importance)), feature_importance['importance'],
                      color=self.colors[:len(feature_importance)], alpha=0.7)
        
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance in Model')
        ax.grid(True, alpha=0.3)
        
        for i, (bar, value) in enumerate(zip(bars, feature_importance['importance'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_model_performance(self, y_true, y_pred, figsize=(12, 5)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        ax1.scatter(y_true, y_pred, alpha=0.6, color=self.colors[0])
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Line')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predictions vs Actual Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        residuals = y_true - y_pred
        ax2.hist(residuals, bins=30, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_price_map(self, df):
        district_coords = {
            'Central': [55.7558, 37.6176],
            'Northern': [55.8354, 37.4856],
            'Southern': [55.6558, 37.6176],
            'Western': [55.7558, 37.4856],
            'Eastern': [55.7558, 37.7496]
        }
        
        df_map = df.copy()
        df_map['lat'] = df_map['district'].map(district_coords).apply(lambda x: x[0])
        df_map['lon'] = df_map['district'].map(district_coords).apply(lambda x: x[1])
        
        fig = px.scatter_mapbox(
            df_map, 
            lat='lat', 
            lon='lon',
            color='price',
            size='area',
            hover_data=['district', 'rooms', 'floor', 'price'],
            color_continuous_scale='viridis',
            zoom=10,
            title='Interactive Real Estate Price Map'
        )
        
        fig.update_layout(
            mapbox_style='open-street-map',
            margin={'r': 0, 't': 30, 'l': 0, 'b': 0}
        )
        
        return fig
    
    def create_price_analysis_dashboard(self, df):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Distribution', 'Price vs Area', 
                          'Average Prices by District', 'Prices by Number of Rooms'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Histogram(x=df['price'], nbinsx=30, name='Price Distribution',
                        marker_color=self.colors[0]),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['area'], y=df['price'], mode='markers',
                      marker=dict(color=df['price'], colorscale='viridis', size=8),
                      name='Price vs Area'),
            row=1, col=2
        )
        
        avg_prices_district = df.groupby('district')['price'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=avg_prices_district.index, y=avg_prices_district.values,
                  marker_color=self.colors[1], name='Average Prices by District'),
            row=2, col=1
        )
        
        avg_prices_rooms = df.groupby('rooms')['price'].mean()
        fig.add_trace(
            go.Bar(x=avg_prices_rooms.index, y=avg_prices_rooms.values,
                  marker_color=self.colors[2], name='Prices by Rooms'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Real Estate Price Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig 