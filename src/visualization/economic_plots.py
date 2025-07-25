# src/visualization/economic_plots.py
"""
Economic visualization plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EconomicVisualizer:
    """
    Visualization tools for economic time series analysis
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer
        
        Parameters:
        -----------
        style : str
            Matplotlib style to use
        figsize : tuple
            Default figure size
        """
        self.style = style
        self.figsize = figsize
        sns.set_theme(style="darkgrid")
        
    def plot_decomposition(self, components: Dict[str, pd.DataFrame], 
                          series_name: str, 
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series decomposition components
        
        Parameters:
        -----------
        components : Dict[str, pd.DataFrame]
            Dictionary with keys: 'trend', 'seasonal', 'cycle', 'irregular'
        series_name : str
            Name of the series being plotted
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure
        """
        fig, axes = plt.subplots(len(components), 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        if len(components) == 1:
            axes = [axes]
            
        for ax, (component_name, data) in zip(axes, components.items()):
            if series_name in data.columns:
                ax.plot(data.index, data[series_name])
                ax.set_title(f'{component_name.capitalize()} Component')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                
        axes[-1].set_xlabel('Date')
        plt.suptitle(f'Decomposition of {series_name}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_relationships(self, x: pd.Series, y: pd.Series,
                          xlabel: str = 'X', ylabel: str = 'Y',
                          title: str = 'Relationship Plot',
                          add_regression: bool = True,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot relationship between two variables
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6)
        
        # Add regression line if requested
        if add_regression:
            z = np.polyfit(x.dropna(), y.dropna(), 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.8, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
            ax.legend()
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Select top N features
        top_features = importance_df.nlargest(top_n, 'importance')
        
        # Create horizontal bar plot
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_forecast(self, actual: pd.Series, predicted: pd.Series,
                     lower_bound: Optional[pd.Series] = None,
                     upper_bound: Optional[pd.Series] = None,
                     title: str = 'Forecast vs Actual',
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot forecast with confidence intervals
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot actual values
        ax.plot(actual.index, actual.values, label='Actual', color='blue')
        
        # Plot predictions
        ax.plot(predicted.index, predicted.values, label='Predicted', 
                color='red', linestyle='--')
        
        # Add confidence intervals if provided
        if lower_bound is not None and upper_bound is not None:
            ax.fill_between(predicted.index, lower_bound, upper_bound,
                           alpha=0.2, color='red', label='Confidence Interval')
            
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
