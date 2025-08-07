"""
Visualization manager for Kalman filter analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import seaborn as sns
import logging
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from .config_manager import PlotConfig
from .results_manager import ResultsManager


class VisualizationManager:
    """Manages all visualization creation."""
    
    def __init__(self, config: PlotConfig, results_manager: ResultsManager):
        """
        Initialize visualization manager.
        
        Parameters
        ----------
        config : PlotConfig
            Plotting configuration
        results_manager : ResultsManager
            Results manager for saving plots
        """
        self.config = config
        self.results = results_manager
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib parameters
        plt.rcParams['figure.dpi'] = config.dpi
        plt.rcParams['savefig.dpi'] = config.dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def create_model_diagnostics(self, data: pd.DataFrame, model: Any,
                               fitted_results: Any, target_series: str):
        """Create comprehensive model diagnostic plots."""
        self.logger.info("Creating model diagnostic plots...")
        
        # Get filtered and smoothed series
        filtered_series = model.get_filtered_series(fitted_results)
        
        # Create multi-panel diagnostic plot
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Target series with filtered/smoothed
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_series_comparison(ax1, data, filtered_series, target_series)
        
        # Panel 2: Residuals
        ax2 = fig.add_subplot(gs[1, 0])
        residuals = self._plot_residuals(ax2, data, filtered_series, target_series)
        
        # Panel 3: Residual distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_residual_distribution(ax3, residuals)
        
        # Panel 4: Q-Q plot
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_qq(ax4, residuals)
        
        # Panel 5: ACF
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_acf(ax5, residuals)
        
        # Panel 6: Parameter estimates
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_parameters(ax6, model, fitted_results)
        
        plt.suptitle(f'Model Diagnostics - {target_series}', fontsize=16)
        self.results.add_plot('model_diagnostics_comprehensive', fig, subdir='diagnostics')
        
        # Create separate initialization plot
        self._create_initialization_plot(data, filtered_series, target_series)
        
    def _plot_series_comparison(self, ax, data, filtered_series, target_series):
        """Plot original vs filtered/smoothed series."""
        ax.plot(data.index, data[target_series], 'k-', alpha=0.5, 
                label='Original', linewidth=1)
        ax.plot(filtered_series['filtered'].index, 
                filtered_series['filtered'][target_series], 
                'b--', alpha=0.7, label='Filtered', linewidth=1.5)
        ax.plot(filtered_series['smoothed'].index, 
                filtered_series['smoothed'][target_series], 
                'r-', alpha=0.8, label='Smoothed', linewidth=1.5)
        
        ax.set_title(f'{target_series} - Kalman Filter Results')
        ax.set_ylabel('Value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add burn-in shading
        burn_in = 20
        if len(data) > burn_in:
            ax.axvspan(data.index[0], data.index[burn_in], 
                      alpha=0.2, color='red', label='Burn-in')
    
    def _plot_residuals(self, ax, data, filtered_series, target_series):
        """Plot residuals over time."""
        residuals = data[target_series] - filtered_series['smoothed'][target_series]
        
        ax.plot(residuals.index, residuals.values, 'b-', alpha=0.7, linewidth=1)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add ±2σ bands
        std = residuals.std()
        ax.axhline(y=2*std, color='r', linestyle=':', alpha=0.5)
        ax.axhline(y=-2*std, color='r', linestyle=':', alpha=0.5)
        
        ax.set_title('Residuals (Original - Smoothed)')
        ax.set_ylabel('Residual')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'μ={residuals.mean():.1f}\nσ={std:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        return residuals
    
    def _plot_residual_distribution(self, ax, residuals):
        """Plot residual distribution with normal overlay."""
        ax.hist(residuals.dropna(), bins=30, density=True, 
                alpha=0.7, color='blue', edgecolor='black')
        
        # Add normal distribution overlay
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 
                'r-', linewidth=2, label='Normal')
        
        ax.set_title('Residual Distribution')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add normality test
        _, p_value = stats.jarque_bera(residuals.dropna())
        ax.text(0.98, 0.98, f'JB p-value: {p_value:.3f}', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_qq(self, ax, residuals):
        """Create Q-Q plot."""
        stats.probplot(residuals.dropna(), dist="norm", plot=ax)
        ax.set_title('Normal Q-Q Plot')
        ax.grid(True, alpha=0.3)
    
    def _plot_acf(self, ax, residuals):
        """Plot autocorrelation function."""
        plot_acf(residuals.dropna(), lags=40, ax=ax, alpha=0.05)
        ax.set_title('Residual Autocorrelation')
        ax.grid(True, alpha=0.3)
    
    def _plot_parameters(self, ax, model, fitted_results):
        """Plot parameter estimates with confidence intervals."""
        params = fitted_results.params
        std_errors = np.sqrt(np.diag(fitted_results.cov_params())) if hasattr(fitted_results, 'cov_params') else np.zeros_like(params)
        
        # Create parameter names
        param_names = model.param_names[:len(params)]
        
        # Plot
        y_pos = np.arange(len(params))
        ax.barh(y_pos, params, xerr=1.96*std_errors, capsize=5, 
                color='steelblue', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names)
        ax.set_xlabel('Parameter Value')
        ax.set_title('Parameter Estimates (95% CI)')
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis to have first parameter at top
        ax.invert_yaxis()
    
    def _create_initialization_plot(self, data, filtered_series, target_series):
        """Create detailed initialization period plot."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Zoom on first 60 observations
        zoom_end = min(60, len(data))
        zoom_data = data.iloc[:zoom_end]
        
        # Panel 1: Series comparison
        ax = axes[0]
        ax.plot(zoom_data.index, zoom_data[target_series], 
                'ko-', alpha=0.5, label='Original', linewidth=1, markersize=4)
        ax.plot(zoom_data.index, 
                filtered_series['smoothed'][target_series].iloc[:zoom_end], 
                'rs-', alpha=0.8, label='Smoothed', linewidth=1.5, markersize=4)
        
        # Shade burn-in
        burn_in = 20
        if zoom_end > burn_in:
            ax.axvspan(zoom_data.index[0], zoom_data.index[burn_in], 
                      alpha=0.2, color='red', label='Burn-in')
        
        ax.set_title(f'Initialization Period - {target_series}')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Convergence metric
        ax = axes[1]
        diff = filtered_series['smoothed'][target_series] - filtered_series['filtered'][target_series]
        ax.plot(zoom_data.index, diff.iloc[:zoom_end].abs(), 
                'g-', linewidth=1.5)
        ax.set_title('Filter Convergence (|Smoothed - Filtered|)')
        ax.set_ylabel('Absolute Difference')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Model Initialization Analysis', fontsize=14)
        self.results.add_plot('model_initialization', fig, subdir='diagnostics')
    
    def create_forecast_evaluation_plots(self, eval_stats: Dict[str, Any],
                                       target_series: str):
        """Create comprehensive forecast evaluation plots."""
        self.logger.info("Creating forecast evaluation plots...")
        
        # Main evaluation figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.5], hspace=0.3, wspace=0.3)
        
        # Panel 1: Forecasts vs Actuals
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_forecast_vs_actual(ax1, eval_stats)
        
        # Panel 2: Forecast Errors
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_forecast_errors(ax2, eval_stats)
        
        # Panel 3: Error Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_error_distribution(ax3, eval_stats)
        
        # Panel 4: Rolling RMSE
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_rolling_rmse(ax4, eval_stats)
        
        # Panel 5: Error Autocorrelation
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_error_autocorrelation(ax5, eval_stats)
        
        plt.suptitle(f'Forecast Evaluation - {target_series}', fontsize=16)
        self.results.add_plot('forecast_evaluation_comprehensive', fig, subdir='evaluation')
        
        # Create accuracy metrics summary
        self._create_accuracy_summary(eval_stats, target_series)
    
    def _plot_forecast_vs_actual(self, ax, eval_stats):
        """Plot forecasts vs actuals with confidence bands."""
        dates = eval_stats['forecast_dates']
        forecasts = eval_stats['forecasts']
        actuals = eval_stats['actuals']
        
        # Calculate confidence bands (assuming normal errors)
        std_error = np.std(eval_stats['errors'])
        lower_68 = forecasts - std_error
        upper_68 = forecasts + std_error
        lower_95 = forecasts - 2 * std_error
        upper_95 = forecasts + 2 * std_error
        
        # Plot confidence bands
        ax.fill_between(dates, lower_95, upper_95, 
                       alpha=0.2, color='red', label='95% CI')
        ax.fill_between(dates, lower_68, upper_68, 
                       alpha=0.3, color='red', label='68% CI')
        
        # Plot forecasts and actuals
        ax.plot(dates, forecasts, 'r-', linewidth=1.5, 
                alpha=0.8, label='4Q Ahead Forecast')
        ax.plot(dates, actuals, 'k-', linewidth=1.5, 
                alpha=0.8, label='Actual')
        
        ax.set_title('4-Quarter Ahead Forecasts vs Actuals')
        ax.set_ylabel('Value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add performance metrics
        rmse = eval_stats['rmse']
        mae = eval_stats['mae']
        mape = eval_stats['mape']
        metrics_text = f'RMSE: {rmse:,.0f}\nMAE: {mae:,.0f}\nMAPE: {mape:.1f}%'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_forecast_errors(self, ax, eval_stats):
        """Plot forecast errors over time."""
        dates = eval_stats['forecast_dates']
        errors = eval_stats['errors']
        
        # Plot errors
        ax.plot(dates, errors, 'gray', alpha=0.5, linewidth=0.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Add moving average
        window = 4
        if len(errors) > window:
            ma_errors = pd.Series(errors).rolling(window).mean()
            ax.plot(dates, ma_errors, 'b-', linewidth=2, 
                    label=f'{window}Q Moving Average')
        
        # Add ±2σ bands
        error_std = np.std(errors)
        ax.axhline(y=2*error_std, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=-2*error_std, color='red', linestyle='--', alpha=0.5)
        
        ax.set_title('Forecast Errors (Actual - Forecast)')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.text(0.02, 0.95, f'Error σ: {error_std:,.0f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    def _plot_error_distribution(self, ax, eval_stats):
        """Plot error distribution."""
        errors = eval_stats['errors']
        
        # Histogram
        n, bins, patches = ax.hist(errors, bins=30, density=True, 
                                   alpha=0.7, color='blue', edgecolor='black')
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(errors)
        x = np.linspace(errors.min(), errors.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'Normal(μ={mu:.0f}, σ={sigma:.0f})')
        
        ax.set_title('Forecast Error Distribution')
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add skewness and kurtosis
        skew = stats.skew(errors)
        kurt = stats.kurtosis(errors)
        stats_text = f'Skew: {skew:.2f}\nKurtosis: {kurt:.2f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    def _plot_rolling_rmse(self, ax, eval_stats):
        """Plot rolling RMSE."""
        errors = eval_stats['errors']
        dates = eval_stats['forecast_dates']
        
        window = 20  # 5 years
        if len(errors) > window:
            rolling_rmse = []
            rolling_dates = []
            
            for i in range(window, len(errors)):
                window_errors = errors[i-window:i]
                rmse = np.sqrt(np.mean(window_errors**2))
                rolling_rmse.append(rmse)
                rolling_dates.append(dates[i])
            
            ax.plot(rolling_dates, rolling_rmse, 'g-', linewidth=2)
            ax.axhline(y=eval_stats['rmse'], color='red', linestyle='--', 
                      alpha=0.5, label=f"Overall RMSE: {eval_stats['rmse']:,.0f}")
            
            ax.set_title(f'Rolling RMSE ({window}-Quarter Window)')
            ax.set_ylabel('RMSE')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_error_autocorrelation(self, ax, eval_stats):
        """Plot error autocorrelation."""
        errors = pd.Series(eval_stats['errors'])
        plot_acf(errors.dropna(), lags=20, ax=ax, alpha=0.05)
        ax.set_title('Forecast Error Autocorrelation')
        ax.grid(True, alpha=0.3)
    
    def _create_accuracy_summary(self, eval_stats, target_series):
        """Create accuracy metrics summary card."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.axis('off')
        
        # Calculate additional metrics
        errors = eval_stats['errors']
        me = np.mean(errors)  # Mean error (bias)
        medae = np.median(np.abs(errors))  # Median absolute error
        
        # Directional accuracy
        if 'forecast_changes' in eval_stats and 'actual_changes' in eval_stats:
            same_direction = np.sign(eval_stats['forecast_changes']) == np.sign(eval_stats['actual_changes'])
            dir_accuracy = np.mean(same_direction) * 100
        else:
            dir_accuracy = None
        
        summary_text = f"""
        FORECAST ACCURACY SUMMARY
        Target Series: {target_series}
        
        ╔════════════════════════════════════════════════════════════╗
        ║                    ERROR METRICS                            ║
        ╠════════════════════════════════════════════════════════════╣
        ║ Root Mean Square Error (RMSE):     {eval_stats['rmse']:>24,.0f} ║
        ║ Mean Absolute Error (MAE):         {eval_stats['mae']:>24,.0f} ║
        ║ Mean Absolute Percentage Error:    {eval_stats['mape']:>23.1f}% ║
        ║ Mean Error (Bias):                 {me:>24,.0f} ║
        ║ Median Absolute Error:             {medae:>24,.0f} ║
        """
        
        if dir_accuracy is not None:
            summary_text += f"""║ Directional Accuracy:              {dir_accuracy:>23.1f}% ║
        """
        
        summary_text += """╚════════════════════════════════════════════════════════════╝
        
        INTERPRETATION:
        """
        
        # Add interpretation
        if abs(me) < eval_stats['mae'] * 0.1:
            summary_text += "• Forecasts are relatively unbiased\n"
        elif me > 0:
            summary_text += "• Forecasts tend to overestimate (positive bias)\n"
        else:
            summary_text += "• Forecasts tend to underestimate (negative bias)\n"
        
        if eval_stats['mape'] < 5:
            summary_text += "• Excellent forecast accuracy (MAPE < 5%)\n"
        elif eval_stats['mape'] < 10:
            summary_text += "• Good forecast accuracy (MAPE < 10%)\n"
        elif eval_stats['mape'] < 20:
            summary_text += "• Moderate forecast accuracy (MAPE < 20%)\n"
        else:
            summary_text += "• Poor forecast accuracy (MAPE > 20%)\n"
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, ha='center', va='center', family='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.3))
        
        plt.title('Forecast Accuracy Summary', fontsize=14, pad=20)
        self.results.add_plot('forecast_accuracy_summary', fig, subdir='evaluation')
    
    def create_risk_indicator_dashboard(self, risk_scores: Dict[str, pd.Series],
                                      data: pd.DataFrame):
        """Create comprehensive risk indicator dashboard."""
        self.logger.info("Creating risk indicator dashboard...")
        
        fig = plt.figure(figsize=self.config.figure_sizes['dashboard'])
        gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1.5, 2, 1], hspace=0.3, wspace=0.3)
        
        # Panel 1: Risk composites
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_risk_composites(ax1, risk_scores)
        
        # Panel 2: Dynamic risk score
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_dynamic_risk_score(ax2, risk_scores)
        
        # Panel 3: Net risk with regimes
        ax3 = fig.add_subplot(gs[2, :])
        self._plot_risk_regimes(ax3, risk_scores)
        
        # Panel 4: Current status
        ax4 = fig.add_subplot(gs[3, 0])
        self._plot_current_risk_status(ax4, risk_scores)
        
        # Panel 5: Risk gauge
        ax5 = fig.add_subplot(gs[3, 1])
        self._plot_risk_gauge(ax5, risk_scores)
        
        plt.suptitle('Risk Indicator Dashboard', fontsize=16)
        self.results.add_plot('risk_indicator_dashboard', fig, subdir='risk_analysis')
        
        # Create additional risk analysis plots
        self._create_risk_correlation_matrix(risk_scores)
        self._create_risk_signal_history(risk_scores)
    
    def _plot_risk_composites(self, ax, risk_scores):
        """Plot risk-on vs risk-off composites."""
        if 'risk_on_composite' in risk_scores and 'risk_off_composite' in risk_scores:
            risk_on = risk_scores['risk_on_composite']
            risk_off = risk_scores['risk_off_composite']
            
            ax.plot(risk_on.index, risk_on.values, 'g-', 
                    label='Risk-On Composite', alpha=0.8, linewidth=1.5)
            ax.plot(risk_off.index, risk_off.values, 'r-', 
                    label='Risk-Off Composite', alpha=0.8, linewidth=1.5)
            
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax.set_title('Risk-On vs Risk-Off Composites')
            ax.set_ylabel('Composite Score')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
    
    def _plot_dynamic_risk_score(self, ax, risk_scores):
        """Plot dynamic risk score with thresholds."""
        if 'dynamic_risk' in risk_scores:
            dynamic_risk = risk_scores['dynamic_risk']
            
            ax.plot(dynamic_risk.index, dynamic_risk.values, 'b-', 
                    linewidth=2, label='Dynamic Risk Score')
            
            if 'upper_threshold' in risk_scores and 'lower_threshold' in risk_scores:
                ax.plot(risk_scores['upper_threshold'].index, 
                       risk_scores['upper_threshold'].values, 
                       'g--', alpha=0.7, label='Upper Threshold (85th %ile)')
                ax.plot(risk_scores['lower_threshold'].index, 
                       risk_scores['lower_threshold'].values, 
                       'r--', alpha=0.7, label='Lower Threshold (15th %ile)')
            
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax.set_title('Dynamic Risk Score (Ridge-based)')
            ax.set_ylabel('Risk Score')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
    
    def _plot_risk_regimes(self, ax, risk_scores):
        """Plot risk regimes with colored backgrounds."""
        if 'dynamic_risk' in risk_scores:
            risk = risk_scores['dynamic_risk']
            
            # Define regimes
            strong_risk_on = risk > 1
            mild_risk_on = (risk > 0) & (risk <= 1)
            mild_risk_off = (risk >= -1) & (risk <= 0)
            strong_risk_off = risk < -1
            
            # Plot regime bands
            ax.fill_between(risk.index, 0, 1, where=strong_risk_on, 
                           alpha=0.3, color='darkgreen', 
                           transform=ax.get_xaxis_transform(),
                           label='Strong Risk-On')
            ax.fill_between(risk.index, 0, 1, where=mild_risk_on, 
                           alpha=0.2, color='lightgreen', 
                           transform=ax.get_xaxis_transform(),
                           label='Mild Risk-On')
            ax.fill_between(risk.index, 0, 1, where=mild_risk_off, 
                           alpha=0.2, color='orange', 
                           transform=ax.get_xaxis_transform(),
                           label='Mild Risk-Off')
            ax.fill_between(risk.index, 0, 1, where=strong_risk_off, 
                           alpha=0.3, color='darkred', 
                           transform=ax.get_xaxis_transform(),
                           label='Strong Risk-Off')
            
            # Plot risk score
            ax.plot(risk.index, risk.values, 'k-', linewidth=1, alpha=0.8)
            
            ax.set_title('Risk Regimes Over Time')
            ax.set_ylabel('Risk Score')
            ax.set_xlabel('Date')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    def _plot_current_risk_status(self, ax, risk_scores):
        """Plot current risk status text."""
        ax.axis('off')
        
        if 'dynamic_risk' in risk_scores and not risk_scores['dynamic_risk'].empty:
            current_risk = risk_scores['dynamic_risk'].iloc[-1]
            current_date = risk_scores['dynamic_risk'].index[-1]
            
            # Determine regime
            if current_risk > 1:
                regime = "STRONG RISK-ON"
                regime_color = 'darkgreen'
                forecast_bias = "Under-forecast likely"
            elif current_risk > 0:
                regime = "MILD RISK-ON"
                regime_color = 'green'
                forecast_bias = "Slight under-forecast"
            elif current_risk > -1:
                regime = "MILD RISK-OFF"
                regime_color = 'orange'
                forecast_bias = "Slight over-forecast"
            else:
                regime = "STRONG RISK-OFF"
                regime_color = 'darkred'
                forecast_bias = "Over-forecast likely"
            
            status_text = f"""
            CURRENT STATUS
            {current_date.strftime('%Y-%m-%d')}
            
            Risk Score: {current_risk:.2f}
            Regime: {regime}
            
            Forecast Implication:
            {forecast_bias}
            """
            
            ax.text(0.5, 0.5, status_text, transform=ax.transAxes,
                   fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=1', 
                            facecolor=regime_color, alpha=0.2))
    
    def _plot_risk_gauge(self, ax, risk_scores):
        """Create a risk gauge visualization."""
        if 'dynamic_risk' not in risk_scores or risk_scores['dynamic_risk'].empty:
            ax.axis('off')
            return
            
        current_risk = risk_scores['dynamic_risk'].iloc[-1]
        
        # Create gauge
        theta = np.linspace(np.pi, 0, 100)
        radius_inner = 0.7
        radius_outer = 1.0
        
        # Color segments
        colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
        boundaries = [-3, -2, -1, 0, 1, 2, 3]
        
        for i in range(len(boundaries)-1):
            mask = (theta >= np.pi * (1 - (boundaries[i+1]+3)/6)) & \
                   (theta <= np.pi * (1 - (boundaries[i]+3)/6))
            theta_segment = theta[mask]
            
            x_inner = radius_inner * np.cos(theta_segment)
            y_inner = radius_inner * np.sin(theta_segment)
            x_outer = radius_outer * np.cos(theta_segment)
            y_outer = radius_outer * np.sin(theta_segment)
            
            vertices = list(zip(x_outer, y_outer)) + \
                      list(zip(x_inner[::-1], y_inner[::-1]))
            
            from matplotlib.patches import Polygon
            poly = Polygon(vertices, facecolor=colors[i], alpha=0.6)
            ax.add_patch(poly)
        
        # Add needle
        angle = np.pi * (1 - (current_risk + 3) / 6)
        x_needle = [0, 0.9 * np.cos(angle)]
        y_needle = [0, 0.9 * np.sin(angle)]
        ax.plot(x_needle, y_needle, 'k-', linewidth=3)
        ax.plot(0, 0, 'ko', markersize=10)
        
        # Labels
        ax.text(0, -0.3, f'{current_risk:.2f}', ha='center', va='top', 
                fontsize=16, fontweight='bold')
        ax.text(0, -0.5, 'Risk Score', ha='center', va='top', fontsize=12)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _create_risk_correlation_matrix(self, risk_scores):
        """Create correlation matrix of risk indicators."""
        if 'z_panel' not in risk_scores:
            return
            
        z_panel = risk_scores['z_panel']
        
        # Calculate correlation matrix
        corr_matrix = z_panel.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        ax.set_title('Risk Indicator Correlation Matrix', fontsize=14)
        plt.tight_layout()
        
        self.results.add_plot('risk_correlation_matrix', fig, subdir='risk_analysis')
    
    def _create_risk_signal_history(self, risk_scores):
        """Create risk signal history with regime changes."""
        if 'dynamic_risk' not in risk_scores:
            return
            
        risk = risk_scores['dynamic_risk']
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Panel 1: Risk score with regime changes
        ax = axes[0]
        
        # Identify regime changes
        regime = pd.Series('Neutral', index=risk.index)
        regime[risk > 1] = 'Strong Risk-On'
        regime[(risk > 0) & (risk <= 1)] = 'Mild Risk-On'
        regime[(risk >= -1) & (risk <= 0)] = 'Mild Risk-Off'
        regime[risk < -1] = 'Strong Risk-Off'
        
        regime_changes = regime != regime.shift(1)
        change_dates = risk.index[regime_changes]
        
        # Plot risk score
        ax.plot(risk.index, risk.values, 'b-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax.axhline(y=1, color='g', linestyle='--', alpha=0.5)
        ax.axhline(y=-1, color='r', linestyle='--', alpha=0.5)
        
        # Mark regime changes
        for date in change_dates[1:]:
            ax.axvline(x=date, color='orange', linestyle=':', alpha=0.5)
        
        ax.set_title('Risk Score History with Regime Changes')
        ax.set_ylabel('Risk Score')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Regime duration
        ax = axes[1]
        
        # Calculate regime durations
        regime_groups = regime.groupby((regime != regime.shift()).cumsum())
        
        regime_stats = []
        for name, group in regime_groups:
            if len(group) > 0:
                regime_stats.append({
                    'regime': group.iloc[0],
                    'start': group.index[0],
                    'end': group.index[-1],
                    'duration': len(group)
                })
        
        # Plot regime bars
        regime_colors = {
            'Strong Risk-On': 'darkgreen',
            'Mild Risk-On': 'lightgreen',
            'Mild Risk-Off': 'orange',
            'Strong Risk-Off': 'darkred',
            'Neutral': 'gray'
        }
        
        for stat in regime_stats:
            color = regime_colors.get(stat['regime'], 'gray')
            ax.axvspan(stat['start'], stat['end'], 
                      alpha=0.5, color=color, 
                      label=stat['regime'] if stat == regime_stats[0] else "")
        
        ax.set_title('Risk Regime Timeline')
        ax.set_ylabel('Regime')
        ax.set_xlabel('Date')
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        
        # Add legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        plt.tight_layout()
        self.results.add_plot('risk_signal_history', fig, subdir='risk_analysis')
    
    def create_recession_analysis_plots(self, recession_results: Dict[str, Any],
                                      risk_scores: Dict[str, pd.Series]):
        """Create comprehensive recession analysis plots."""
        self.logger.info("Creating recession analysis plots...")
        
        # Main recession analysis figure
        fig = plt.figure(figsize=self.config.figure_sizes['large'])
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1.5, 1.5, 1], hspace=0.2)
        
        # Get recession dates
        recession_dates = recession_results.get('recession_dates', [])
        
        # Panel 1: Risk score with recession shading
        ax1 = fig.add_subplot(gs[0])
        self._plot_risk_vs_recessions(ax1, risk_scores, recession_dates)
        
        # Panel 2: Recession probability
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        self._plot_recession_probability(ax2, recession_results, recession_dates)
        
        # Panel 3: Lead-lag analysis
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        self._plot_recession_lead_lag(ax3, risk_scores, recession_dates)
        
        # Panel 4: Performance metrics
        ax4 = fig.add_subplot(gs[3])
        self._plot_recession_performance(ax4, recession_results)
        
        plt.suptitle('Recession Risk Analysis', fontsize=16)
        self.results.add_plot('recession_analysis_comprehensive', fig, subdir='recession')
        
        # Create recession scorecard
        self._create_recession_scorecard(recession_results)
    
    def _plot_risk_vs_recessions(self, ax, risk_scores, recession_dates):
        """Plot risk score with recession shading."""
        if 'dynamic_risk' in risk_scores:
            risk = risk_scores['dynamic_risk']
            
            # Add recession shading
            for start, end in recession_dates:
                if end >= risk.index[0] and start <= risk.index[-1]:
                    ax.axvspan(max(start, risk.index[0]), 
                             min(end, risk.index[-1]), 
                             alpha=0.3, color='gray', 
                             label='NBER Recession' if (start, end) == recession_dates[0] else "")
            
            # Plot risk score
            ax.plot(risk.index, risk.values, 'b-', linewidth=2, label='Risk Score')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax.axhline(y=-1, color='r', linestyle='--', alpha=0.5, 
                      label='Risk-Off Threshold')
            
            # Highlight pre-recession warnings
            for start, end in recession_dates:
                pre_start = start - pd.DateOffset(months=12)
                if pre_start >= risk.index[0] and start <= risk.index[-1]:
                    pre_risk = risk.loc[pre_start:start]
                    if len(pre_risk) > 0 and (pre_risk < -1).any():
                        ax.scatter(start, -2.5, marker='v', s=100, 
                                 color='green', edgecolor='black', zorder=5)
            
            ax.set_title('Risk Score vs NBER Recessions')
            ax.set_ylabel('Risk Score')
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
    
    def _plot_recession_probability(self, ax, recession_results, recession_dates):
        """Plot recession probability model."""
        if 'recession_probability' in recession_results:
            prob = recession_results['recession_probability']
            
            if isinstance(prob, pd.Series):
                # Add recession shading
                for start, end in recession_dates:
                    if end >= prob.index[0] and start <= prob.index[-1]:
                        ax.axvspan(max(start, prob.index[0]), 
                                 min(end, prob.index[-1]), 
                                 alpha=0.3, color='gray')
                
                # Plot probability
                ax.plot(prob.index, prob.values, 'purple', linewidth=2)
                ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
                ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, 
                          label='High Risk Threshold')
                
                # Highlight high-risk periods
                high_risk = prob > 0.7
                ax.fill_between(prob.index, 0, 1, where=high_risk, 
                               alpha=0.2, color='red', label='High Risk')
                
                ax.set_title('Recession Probability Model')
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    def _plot_recession_lead_lag(self, ax, risk_scores, recession_dates):
        """Plot lead-lag analysis."""
        if 'dynamic_risk' not in risk_scores:
            return
            
        risk = risk_scores['dynamic_risk']
        
        # Plot risk score at different lags
        lead_times = [0, 2, 4, 6]
        colors = ['black', 'blue', 'purple', 'orange']
        
        for lead, color in zip(lead_times, colors):
            shifted_risk = risk.shift(lead)
            ax.plot(risk.index, shifted_risk, color=color, 
                   alpha=0.7, linewidth=1.5, 
                   label=f'{lead}Q lead' if lead > 0 else 'Current')
        
        # Add recession shading
        for start, end in recession_dates:
            if end >= risk.index[0] and start <= risk.index[-1]:
                ax.axvspan(max(start, risk.index[0]), 
                         min(end, risk.index[-1]), 
                         alpha=0.3, color='gray')
        
        ax.axhline(y=-1, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Risk Score Lead-Lag Analysis')
        ax.set_ylabel('Risk Score')
        ax.set_xlabel('Date')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
    
    def _plot_recession_performance(self, ax, recession_results):
        """Plot recession prediction performance metrics."""
        ax.axis('off')
        
        if 'historical_accuracy' in recession_results:
            acc = recession_results['historical_accuracy']
            
            perf_text = f"""
            RECESSION PREDICTION PERFORMANCE
            
            Detection Rate: {acc.get('detection_rate', 0)*100:.0f}%
            False Positive Rate: {acc.get('false_positive_rate', 0)*100:.0f}%
            Average Lead Time: {acc.get('average_lead_time', 0):.1f} quarters
            
            Current Risk Level: {recession_results.get('risk_level', 'UNKNOWN')}
            """
            
            ax.text(0.5, 0.5, perf_text, transform=ax.transAxes,
                   fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=1', 
                            facecolor='lightblue', alpha=0.3))
    
    def _create_recession_scorecard(self, recession_results):
        """Create detailed recession prediction scorecard."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.axis('off')
        
        scorecard_text = """
        RECESSION PREDICTION SCORECARD
        
        ╔════════════════════════════════════════════════════════════╗
        ║                    MODEL PERFORMANCE                        ║
        ╠════════════════════════════════════════════════════════════╣
        """
        
        if 'historical_accuracy' in recession_results:
            acc = recession_results['historical_accuracy']
            scorecard_text += f"""║ Detection Rate:                    {acc.get('detection_rate', 0)*100:>23.0f}% ║
║ False Positive Rate:               {acc.get('false_positive_rate', 0)*100:>23.0f}% ║
║ Average Lead Time:                 {acc.get('average_lead_time', 0):>20.1f} Q ║
        """
        
        scorecard_text += """╚════════════════════════════════════════════════════════════╝
        
        CURRENT ASSESSMENT:
        """
        
        scorecard_text += f"""
        Recession Probability: {recession_results['current_prediction'].probability:.1%}
        Risk Level: {recession_results.get('risk_level', 'UNKNOWN')}
        
        SIGNAL INTERPRETATION:
        • Risk Score < -1.0 → Recession warning
        • Typical lead time: 2-4 quarters
        • Monitor risk-off persistence for confirmation
        """
        
        ax.text(0.5, 0.5, scorecard_text, transform=ax.transAxes,
               fontsize=11, ha='center', va='center', family='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.3))
        
        plt.title('Recession Prediction Scorecard', fontsize=14, pad=20)
        self.results.add_plot('recession_scorecard', fig, subdir='recession')
