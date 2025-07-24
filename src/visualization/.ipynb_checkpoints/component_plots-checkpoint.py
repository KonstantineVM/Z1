"""
Economic visualization functions
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


class EconomicVisualizer:
    """
    Visualization tools for economic time series analysis
    """
    
    # NBER recession dates
    RECESSION_DATES = [
        ('1953-07-01', '1954-05-01'),
        ('1957-08-01', '1958-04-01'),
        ('1960-04-01', '1961-02-01'),
        ('1969-12-01', '1970-11-01'),
        ('1973-11-01', '1975-03-01'),
        ('1980-01-01', '1980-07-01'),
        ('1981-07-01', '1982-11-01'),
        ('1990-07-01', '1991-03-01'),
        ('2001-03-01', '2001-11-01'),
        ('2007-12-01', '2009-06-01'),
        ('2020-02-01', '2020-04-01')
    ]
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size
        """
        self.figsize = figsize
        
    def plot_component_decomposition(self, series_name: str,
                                   components: Dict[str, pd.Series],
                                   original_series: Optional[pd.Series] = None):
        """
        Plot decomposition of a single series
        
        Parameters:
        -----------
        series_name : str
            Name of the series
        components : Dict[str, pd.Series]
            Component series (level, trend, cycle, seasonal)
        original_series : pd.Series, optional
            Original series for comparison
        """
        n_components = len([c for c in components.values() if c is not None])
        
        fig, axes = plt.subplots(n_components + (1 if original_series is not None else 0), 
                                1, figsize=(self.figsize[0], 3 * n_components))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
            
        ax_idx = 0
        
        # Plot original if provided
        if original_series is not None:
            axes[ax_idx].plot(original_series)
            axes[ax_idx].set_title(f'{series_name} - Original')
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1
            
        # Plot components
        for comp_name, comp_series in components.items():
            if comp_series is not None:
                axes[ax_idx].plot(comp_series)
                axes[ax_idx].set_title(f'{series_name} - {comp_name.capitalize()}')
                axes[ax_idx].grid(True, alpha=0.3)
                ax_idx += 1
                
        plt.tight_layout()
        return fig
    
    def plot_economic_relationships(self, data: pd.DataFrame,
                                  primary_vars: List[str],
                                  secondary_var: Optional[str] = None,
                                  title: str = "Economic Relationships",
                                  add_recessions: bool = True):
        """
        Plot multiple economic variables with optional secondary axis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to plot
        primary_vars : List[str]
            Variables for primary axis
        secondary_var : str, optional
            Variable for secondary axis
        title : str
            Plot title
        add_recessions : bool
            Whether to add recession bars
        """
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        # Plot primary variables
        colors = plt.cm.tab10(np.linspace(0, 1, len(primary_vars)))
        lines = []
        
        for i, var in enumerate(primary_vars):
            if var in data.columns:
                line, = ax1.plot(data.index, data[var], color=colors[i], 
                                 label=self._clean_label(var))
                lines.append(line)
                
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Primary Variables')
        
        # Add secondary axis if needed
        if secondary_var and secondary_var in data.columns:
            ax2 = ax1.twinx()
            line2, = ax2.plot(data.index, data[secondary_var], 'r--', 
                             label=self._clean_label(secondary_var))
            ax2.set_ylabel(self._clean_label(secondary_var), color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            lines.append(line2)
            
        # Add legend
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # Add recession bars
        if add_recessions:
            self._add_recession_bars(ax1, data.index)
            
        # Format dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_velocity_analysis(self, components: Dict[str, pd.DataFrame]):
        """
        Specialized plot for velocity of money analysis
        
        Parameters:
        -----------
        components : Dict[str, pd.DataFrame]
            Component DataFrames
        """
        if 'trend' not in components:
            print("No trend component available")
            return None
            
        trend_df = components['trend']
        
        # Find relevant series
        gdp_col = None
        m2_col = None
        
        for col in trend_df.columns:
            if 'FU086902005' in col or 'GDP' in col:
                gdp_col = col
            elif 'M2_N.M' in col or 'M2' in col:
                m2_col = col
                
        if not gdp_col or not m2_col:
            print("GDP or M2 series not found")
            return None
            
        # Create DataFrame for plotting
        plot_data = pd.DataFrame(index=trend_df.index)
        plot_data['GDP Growth'] = trend_df[gdp_col]
        plot_data['M2 Growth'] = trend_df[m2_col]
        plot_data['Velocity Change'] = plot_data['GDP Growth'] - plot_data['M2 Growth']
        
        # Convert index to datetime if needed
        if not isinstance(plot_data.index, pd.DatetimeIndex):
            plot_data.index = pd.to_datetime(plot_data.index)
            
        # Create plot
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        # Primary axis - GDP and M2
        ax1.plot(plot_data.index, plot_data['GDP Growth'], 'g-', label='GDP Growth')
        ax1.plot(plot_data.index, plot_data['M2 Growth'], 'b-', label='M2 Growth')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Growth Rate')
        ax1.legend(loc='upper left')
        
        # Secondary axis - Velocity
        ax2 = ax1.twinx()
        ax2.plot(plot_data.index, plot_data['Velocity Change'], 'r--', 
                label='Velocity Change')
        ax2.set_ylabel('Velocity Change', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')
        
        # Add recessions
        self._add_recession_bars(ax1, plot_data.index)
        
        # Format
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax1.set_title('Money Supply, GDP, and Velocity of Money')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                               top_n: int = 30,
                               title: str = "Feature Importance"):
        """
        Plot feature importance from model results
        
        Parameters:
        -----------
        feature_importance : pd.DataFrame
            DataFrame with 'Feature' and 'Importance' columns
        top_n : int
            Number of top features to show
        title : str
            Plot title
        """
        # Get top features
        top_features = feature_importance.nlargest(top_n, 'Importance')
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['Importance'].values)
        
        # Add feature names
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self._clean_label(f) for f in top_features['Feature']])
        ax.invert_yaxis()
        
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_results(self, actual: pd.Series,
                               predicted: np.ndarray,
                               uncertainty: Optional[np.ndarray] = None,
                               title: str = "Predictions vs Actual"):
        """
        Plot prediction results with optional uncertainty bands
        
        Parameters:
        -----------
        actual : pd.Series
            Actual values
        predicted : np.ndarray
            Predicted values
        uncertainty : np.ndarray, optional
            Uncertainty (standard deviation) for predictions
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create index for x-axis
        x = np.arange(len(actual))
        
        # Plot actual and predicted
        ax.plot(x, actual.values, 'b-', label='Actual', linewidth=2)
        ax.plot(x, predicted, 'r--', label='Predicted', linewidth=2)
        
        # Add uncertainty bands if provided
        if uncertainty is not None:
            ax.fill_between(x, 
                           predicted - uncertainty,
                           predicted + uncertainty,
                           color='red', alpha=0.2, 
                           label='Uncertainty')
        
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add metrics
        mse = np.mean((actual.values - predicted) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - mse / np.var(actual.values)
        
        metrics_text = f'RMSE: {rmse:.4f}\nR²: {r2:.4f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_cycle_with_recessions(self, cycle_component: pd.Series,
                                  series_name: str,
                                  normalize: bool = True):
        """
        Plot cyclical component with recession shading
        
        Parameters:
        -----------
        cycle_component : pd.Series
            Cyclical component
        series_name : str
            Name of the series
        normalize : bool
            Whether to normalize by level
        """
        # Ensure datetime index
        if not isinstance(cycle_component.index, pd.DatetimeIndex):
            cycle_component.index = pd.to_datetime(cycle_component.index)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot cycle
        ax.plot(cycle_component.index, cycle_component.values, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add recession bars
        self._add_recession_bars(ax, cycle_component.index)
        
        # Format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        title = f'Cyclical Component - {self._clean_label(series_name)}'
        if normalize:
            title += ' (Normalized)'
        ax.set_title(title)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Cycle Value')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, data: pd.DataFrame,
                                title: str = "Correlation Heatmap",
                                figsize: Optional[Tuple[int, int]] = None):
        """
        Plot correlation heatmap
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to compute correlations
        title : str
            Plot title
        figsize : Tuple[int, int], optional
            Figure size
        """
        if figsize is None:
            n_vars = len(data.columns)
            figsize = (min(20, max(10, n_vars * 0.5)), 
                      min(16, max(8, n_vars * 0.4)))
        
        # Compute correlation
        corr = data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    annot=False, ax=ax)
        
        ax.set_title(title)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        return fig
    
    def _add_recession_bars(self, ax, date_index):
        """Add recession shading to a plot"""
        for start, end in self.RECESSION_DATES:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            
            # Check if recession period overlaps with data
            if start_date <= date_index[-1] and end_date >= date_index[0]:
                ax.axvspan(start_date, end_date, alpha=0.2, color='gray')
    
    def _clean_label(self, label: str) -> str:
        """Clean up variable names for display"""
        # Remove common prefixes
        for prefix in ['Level_', 'Trend_', 'Cycle_', 'Seasonal_']:
            if label.startswith(prefix):
                label = label[len(prefix):]
                
        # Replace underscores with spaces
        label = label.replace('_', ' ')
        
        # Truncate if too long
        if len(label) > 40:
            label = label[:37] + '...'
            
        return label