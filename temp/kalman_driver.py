#!/usr/bin/env python3
"""
Driver script for Hierarchical Kalman Filter with FoF data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import logging
from hierarchical_kalman_filter import fit_hierarchical_kalman_filter
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

# Suppress the sparse efficiency warning from HP filter
warnings.filterwarnings('ignore', message='spsolve requires A be CSC or CSR matrix format')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Enable debug logging for hierarchical_kalman_filter module
logging.getLogger('hierarchical_kalman_filter').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def analyze_series_with_hp(data, series_name, save_plot=True):
    """Analyze a series with HP filter and create diagnostic plot."""
    if series_name not in data.columns:
        logger.warning(f"Series {series_name} not found in data")
        return None
        
    y = data[series_name].dropna()
    if len(y) < 20:
        logger.warning(f"Series {series_name} has too few observations for HP filter")
        return None
        
    # Apply HP filter
    cycle, trend = hpfilter(y, lamb=1600)  # 1600 for quarterly data
    
    # Calculate statistics
    trend_diff = np.diff(trend)
    stats = {
        'series_mean': y.mean(),
        'series_std': y.std(),
        'series_scale': np.median(np.abs(y[y != 0])),
        'cycle_var': np.var(cycle),
        'trend_var': np.var(trend_diff),
        'cycle_std': np.std(cycle),
        'trend_std': np.std(trend_diff)
    }
    
    if save_plot:
        # Create diagnostic plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        # Original and trend
        axes[0].plot(y.index, y.values, 'k-', alpha=0.7, label='Original', linewidth=1)
        axes[0].plot(y.index, trend, 'r-', linewidth=2, label='HP Trend')
        axes[0].set_title(f'{series_name} - HP Filter Decomposition')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylabel('Value')
        
        # Add scale annotation
        axes[0].text(0.02, 0.98, f'Scale: {stats["series_scale"]:.2e}', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Cycle (detrended)
        axes[1].plot(y.index, cycle, 'b-', alpha=0.7, linewidth=1)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_title(f'Cycle Component (std={stats["cycle_std"]:.2e})')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel('Cycle')
        
        # Trend changes
        axes[2].plot(y.index[1:], trend_diff, 'g-', alpha=0.7, linewidth=1)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[2].set_title(f'Trend Changes (std={stats["trend_std"]:.2e})')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylabel('Δ Trend')
        axes[2].set_xlabel('Date')
        
        plt.tight_layout()
        plt.savefig(f'hp_filter_{series_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    return stats

def create_forecast_evaluation_plot(model, fitted_results, data, target_series, 
                                   start_date=None, end_date=None):
    """
    Create plot comparing 4-quarter ahead forecasts with realized values.
    
    This generates forecasts at EVERY quarter and compares them with
    what actually happened 4 quarters later.
    """
    logger.info("\nGenerating 4-quarter ahead forecast evaluation (every quarter)...")
    
    # Determine evaluation period
    if start_date is None:
        # Start from 25% into the data (after model has stabilized)
        start_idx = len(data) // 4
        start_date = data.index[start_idx]
    else:
        start_idx = data.index.get_loc(start_date)
    
    if end_date is None:
        # End 4 quarters before the last observation (so we have actuals to compare)
        end_idx = len(data) - 4
        end_date = data.index[end_idx]
    else:
        end_idx = data.index.get_loc(end_date)
    
    # Storage for forecasts and actuals
    forecast_dates = []
    forecast_values = []
    forecast_lower_68 = []
    forecast_upper_68 = []
    forecast_lower_95 = []
    forecast_upper_95 = []
    actual_values = []
    
    # Generate forecasts at EVERY quarter (changed from every 4 quarters)
    logger.info(f"Generating {end_idx - start_idx} quarterly forecasts...")
    
    # Use tqdm for progress bar since this will take longer
    from tqdm import tqdm
    
    for t in tqdm(range(start_idx, end_idx, 1), desc="Generating forecasts"):  # Changed from 4 to 1
        base_date = data.index[t]
        
        # Get forecast 4 quarters ahead
        path_result = model.get_most_probable_path(fitted_results, base_date=base_date)
        
        # Extract 4-quarter ahead forecast (index 3, since 0-indexed)
        forecast_4q = path_result['path'][target_series].iloc[3]
        uncertainty_4q = path_result['uncertainty'][target_series].iloc[3]
        
        # Get actual value 4 quarters later
        actual_date = data.index[t + 4]
        actual_4q = data[target_series].iloc[t + 4]
        
        # Store results
        forecast_dates.append(base_date)
        forecast_values.append(forecast_4q)
        forecast_lower_68.append(forecast_4q - uncertainty_4q)
        forecast_upper_68.append(forecast_4q + uncertainty_4q)
        forecast_lower_95.append(forecast_4q - 2 * uncertainty_4q)
        forecast_upper_95.append(forecast_4q + 2 * uncertainty_4q)
        actual_values.append(actual_4q)
    
    # Convert to arrays
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    forecast_values = np.array(forecast_values)
    actual_values = np.array(actual_values)
    
    # Calculate forecast errors
    forecast_errors = actual_values - forecast_values
    rmse = np.sqrt(np.mean(forecast_errors**2))
    mae = np.mean(np.abs(forecast_errors))
    mape = np.mean(np.abs(forecast_errors / actual_values)) * 100
    
    logger.info(f"Computed {len(forecast_errors)} quarterly forecast errors")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Forecasts vs Actuals over time
    ax = axes[0]
    
    # For clearer visualization with many points, plot confidence bands as areas
    ax.fill_between(forecast_dates, forecast_lower_95, forecast_upper_95, 
                    alpha=0.1, color='red', label='95% Confidence')
    ax.fill_between(forecast_dates, forecast_lower_68, forecast_upper_68, 
                    alpha=0.2, color='red', label='68% Confidence')
    
    # Plot forecasts and actuals as lines (not scatter) for quarterly data
    ax.plot(forecast_dates, forecast_values, 'r-', linewidth=1.5, 
            alpha=0.8, label='4Q Ahead Forecast')
    ax.plot(forecast_dates, actual_values, 'k-', linewidth=1.5, 
            alpha=0.8, label='Actual (4Q Later)')
    
    # Add historical context
    hist_start = max(0, start_idx - 20)
    ax.plot(data.index[hist_start:end_idx+4], data[target_series].iloc[hist_start:end_idx+4], 
            'gray', alpha=0.3, linewidth=1, label='Full History')
    
    ax.set_title(f'{target_series} - 4-Quarter Ahead Forecast Evaluation (Quarterly)\n' + 
                 f'RMSE: {rmse:,.0f}, MAE: {mae:,.0f}, MAPE: {mape:.1f}%', 
                 fontsize=14)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Forecast Errors with moving average
    ax = axes[1]
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Plot individual errors as lighter line
    ax.plot(forecast_dates, forecast_errors, 'gray', alpha=0.5, linewidth=0.5)
    
    # Add moving average for trend
    window = 4  # 1-year moving average
    if len(forecast_errors) > window:
        from scipy.ndimage import uniform_filter1d
        ma_errors = uniform_filter1d(forecast_errors, size=window, mode='nearest')
        ax.plot(forecast_dates, ma_errors, 'b-', linewidth=2, 
                label=f'{window}Q Moving Average')
    
    ax.set_title('Forecast Errors (Actual - Forecast)', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add ±2σ bands
    error_std = np.std(forecast_errors)
    ax.axhline(y=2*error_std, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-2*error_std, color='red', linestyle='--', alpha=0.5)
    ax.text(0.02, 0.95, f'Error σ: {error_std:,.0f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Rolling accuracy metrics
    ax = axes[2]
    
    # Calculate rolling RMSE with 20-quarter window
    window = 20
    if len(forecast_errors) > window:
        rolling_rmse = []
        rolling_dates = []
        
        for i in range(window, len(forecast_errors)):
            window_errors = forecast_errors[i-window:i]
            rolling_rmse.append(np.sqrt(np.mean(window_errors**2)))
            rolling_dates.append(forecast_dates[i])
        
        ax.plot(rolling_dates, rolling_rmse, 'g-', linewidth=2)
        ax.set_title(f'Rolling RMSE ({window}-Quarter Window)', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for overall RMSE
        ax.axhline(y=rmse, color='red', linestyle='--', alpha=0.5, 
                   label=f'Overall RMSE: {rmse:,.0f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('forecast_evaluation_4q_quarterly.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot: forecast_evaluation_4q_quarterly.png")
    
    # Return statistics for further analysis
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'forecast_dates': forecast_dates,
        'forecasts': forecast_values,
        'actuals': actual_values,
        'errors': forecast_errors
    }    
def create_trend_and_forecast_comparison_plot(model, fitted_results, data, target_series,
                                             eval_stats=None, start_date=None):
    """
    Visualize the level and trend (slope) components along with 4-quarter forecast errors.
    
    For computed series, reconstruct components from source series.
    """
    logger.info("\nCreating level/trend and forecast comparison visualization...")
    
    # Determine if target is source or computed
    is_source = target_series in model.source_series
    
    # Get smoothed series
    smoothed_series = model.get_filtered_series(fitted_results)['smoothed']
    
    # Extract or reconstruct components
    if is_source:
        # Direct extraction for source series
        info = model.source_info[target_series]
        level_idx = info['start_idx']
        trend_idx = info['start_idx'] + 1
        
        levels = fitted_results.smoothed_state[level_idx, :]
        trends = fitted_results.smoothed_state[trend_idx, :]
        
        # Denormalize
        if model.normalize_data:
            levels = levels * model.scale_factor
            trends = trends * model.scale_factor
            
        level_component = pd.Series(levels, index=data.index)
        trend_component = pd.Series(trends, index=data.index)
        
    else:
        # Reconstruct for computed series
        formula_info = model.formulas.get(target_series, {})
        components = formula_info.get('derived_from', [])
        
        level_component = pd.Series(0.0, index=data.index)
        trend_component = pd.Series(0.0, index=data.index)
        
        # Sum components from all source series
        for comp in components:
            comp_code = comp.get('code')
            operator = comp.get('operator', '+')
            sign = 1 if operator == '+' else -1
            
            if comp_code in model.source_series:
                info = model.source_info[comp_code]
                level_idx = info['start_idx']
                trend_idx = info['start_idx'] + 1
                
                comp_levels = fitted_results.smoothed_state[level_idx, :]
                comp_trends = fitted_results.smoothed_state[trend_idx, :]
                
                if model.normalize_data:
                    comp_levels = comp_levels * model.scale_factor
                    comp_trends = comp_trends * model.scale_factor
                
                level_component += sign * comp_levels
                trend_component += sign * comp_trends
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # Main plot: Series with level component
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot actual data
    ax1.plot(data.index, data[target_series], 'k-', alpha=0.4, 
             linewidth=1, label='Actual Data')
    
    # Plot smoothed estimate
    ax1.plot(smoothed_series.index, smoothed_series[target_series], 'b-', 
             alpha=0.8, linewidth=1.5, label='Smoothed Estimate')
    
    # Plot level component (smooth underlying level)
    ax1.plot(level_component.index, level_component.values, 'r-', 
             linewidth=2.5, label='Level Component (Stochastic Trend)')
    
    ax1.set_title(f'{target_series} - Level Component and Smoothed Series', fontsize=14)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Trend (slope of level)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(trend_component.index, trend_component.values, 'g-', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_title('Trend Component (Slope of Level)', fontsize=12)
    ax2.set_ylabel('Growth Rate per Quarter', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    avg_growth = trend_component.mean()
    ax2.text(0.02, 0.95, f'Avg Growth: {avg_growth:.1f}/quarter', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Subplot 3: Cycle + Irregular (Actual - Level)
    ax3 = fig.add_subplot(gs[1, 1])
    cycle_irregular = data[target_series] - level_component
    ax3.plot(cycle_irregular.index, cycle_irregular.values, 'purple', alpha=0.7, linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_title('Cycle + Irregular (Actual - Level)', fontsize=12)
    ax3.set_ylabel('Deviation from Level', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: 4-Quarter Forecast Errors vs Level
    ax4 = fig.add_subplot(gs[2, :])
    
    if eval_stats is not None:
        # Plot level for reference
        ax4_twin = ax4.twinx()
        ax4_twin.plot(level_component.index, level_component.values, 
                      'lightgray', alpha=0.3, linewidth=2, label='Level')
        ax4_twin.set_ylabel('Level Value', fontsize=11, color='gray')
        ax4_twin.tick_params(axis='y', labelcolor='gray')
        
        # Plot forecast errors
        forecast_dates = eval_stats['forecast_dates']
        errors = eval_stats['errors']
        
        # Color by error sign
        colors = ['red' if e < 0 else 'green' for e in errors]
        ax4.scatter(forecast_dates, errors, c=colors, alpha=0.6, s=50)
        
        # Add zero line
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Add moving average of errors
        if len(errors) > 10:
            from scipy.ndimage import uniform_filter1d
            ma_errors = uniform_filter1d(errors, size=5, mode='nearest')
            ax4.plot(forecast_dates, ma_errors, 'b-', linewidth=2, 
                     alpha=0.8, label='MA(5) of Errors')
        
        ax4.set_title('4-Quarter Ahead Forecast Errors Over Time', fontsize=12)
        ax4.set_xlabel('Forecast Date', fontsize=11)
        ax4.set_ylabel('Forecast Error', fontsize=11)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        # Add error statistics by level regime
        level_median = np.median(level_component.values)
        high_level_mask = level_component.loc[forecast_dates].values > level_median
        
        high_level_rmse = np.sqrt(np.mean(errors[high_level_mask]**2))
        low_level_rmse = np.sqrt(np.mean(errors[~high_level_mask]**2))
        
        stats_text = (f'RMSE High Level: {high_level_rmse:,.0f}\n'
                     f'RMSE Low Level: {low_level_rmse:,.0f}')
        ax4.text(0.02, 0.95, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.suptitle(f'Level/Trend Analysis and Forecast Performance for {target_series}', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('trend_and_forecast_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved plot: trend_and_forecast_analysis.png")
    
    # Return components for further analysis
    return {
        'level': level_component,
        'trend': trend_component,  # This is the slope/growth rate
        'cycle_irregular': cycle_irregular
    }
    
def find_forecast_error_correlates(model, data, eval_stats, target_series, 
                                  min_lag=4, max_lag=8, top_n=40, min_obs=200):
    """
    Find series that correlate with 4-quarter ahead forecast errors.
    Only considers positive lags (series leads errors) and contemporaneous.
    
    Parameters:
    -----------
    model : HierarchicalKalmanFilter
        The fitted model
    data : pd.DataFrame
        Full Z1 dataset
    eval_stats : dict
        Evaluation statistics containing forecast errors
    target_series : str
        Target series name
    max_lag : int
        Maximum lag to consider (positive = series leads errors)
    top_n : int
        Number of top correlations to return
    min_obs : int
        Minimum observations required for correlation
    
    Returns:
    --------
    pd.DataFrame with correlation results
    """
    logger.info(f"\nFinding series that correlate with {target_series} forecast errors...")
    
    # Get forecast errors and dates
    error_dates = pd.DatetimeIndex(eval_stats['forecast_dates'])
    errors = pd.Series(eval_stats['errors'], index=error_dates, name='forecast_error')
    
    # Standardize errors
    errors_std = (errors - errors.mean()) / errors.std()
    
    logger.info(f"Analyzing {len(errors)} forecast errors from {error_dates[0]} to {error_dates[-1]}")
    
    # Store correlation results
    correlation_results = []
    
    # Get all series names except target
    all_series = [col for col in data.columns if col != target_series]
    logger.info(f"Checking correlations with {len(all_series)} series...")
    
    # Progress tracking
    from tqdm import tqdm
    
    for series_name in tqdm(all_series, desc="Computing correlations"):
        if series_name not in data.columns:
            continue
            
        series = data[series_name].dropna()
        
        # Skip if too few observations
        if len(series) < min_obs:
            continue
        
        # Standardize series
        if series.std() > 0:
            series_std = (series - series.mean()) / series.std()
        else:
            continue
        
        # Check correlations at different lags (0 to max_lag only)
        for lag in range(min_lag, max_lag + 1):
            # Series leads errors by 'lag' periods
            series_shifted = series_std.shift(lag)
            
            # Find common dates
            common_idx = errors_std.index.intersection(series_shifted.index)
            
            if len(common_idx) < min_obs:
                continue
            
            # Calculate correlation
            error_aligned = errors_std.loc[common_idx]
            series_aligned = series_shifted.loc[common_idx]
            
            if len(error_aligned) > 0 and series_aligned.std() > 0:
                corr = error_aligned.corr(series_aligned)
                
                if not np.isnan(corr):
                    correlation_results.append({
                        'series': series_name,
                        'lag': lag,
                        'correlation': corr,
                        'abs_correlation': abs(corr),
                        'n_obs': len(common_idx),
                        'series_mean': data[series_name].mean(),
                        'series_std': data[series_name].std()
                    })
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(correlation_results)
    
    if results_df.empty:
        logger.warning("No valid correlations found!")
        return pd.DataFrame()
    
    # Sort by absolute correlation
    results_df = results_df.sort_values('abs_correlation', ascending=False)
    
    # Get top correlations
    top_results = results_df.head(top_n * 3)  # Get more to show different lags
    
    # For each series, keep only the best lag
    best_by_series = []
    seen_series = set()
    
    for _, row in top_results.iterrows():
        if row['series'] not in seen_series:
            seen_series.add(row['series'])
            best_by_series.append(row)
            if len(best_by_series) >= top_n:
                break
    
    final_results = pd.DataFrame(best_by_series)
    
    # Add series descriptions if available
    logger.info(f"\nTop {len(final_results)} series correlating with forecast errors:")
    for idx, row in final_results.iterrows():
        lag_desc = f"leads by {row['lag']}Q" if row['lag'] > 0 else "contemporaneous"
        logger.info(f"  {row['series']}: r={row['correlation']:.3f} ({lag_desc}, n={row['n_obs']})")
    
    return final_results    

def plot_error_correlates(data, eval_stats, correlate_results, target_series, n_plots=6):
    """
    Plot the top correlating series with forecast errors.
    """
    # Get forecast errors
    error_dates = pd.DatetimeIndex(eval_stats['forecast_dates'])
    errors = pd.Series(eval_stats['errors'], index=error_dates, name='forecast_error')
    
    # Standardize for plotting
    errors_std = (errors - errors.mean()) / errors.std()
    
    # Create figure
    n_rows = min(n_plots, len(correlate_results))
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3*n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    
    fig.suptitle(f'Top Series Correlating with {target_series} 4Q-Ahead Forecast Errors', 
                 fontsize=16)
    
    for i, (idx, row) in enumerate(correlate_results.head(n_plots).iterrows()):
        ax = axes[i]
        
        # Get series data
        series = data[row['series']]
        series_std = (series - series.mean()) / series.std()
        
        # Apply lag
        lag = int(row['lag'])
        if lag != 0:
            series_shifted = series_std.shift(lag)
        else:
            series_shifted = series_std
        
        # Plot on twin axes for different scales
        ax2 = ax.twinx()
        
        # Plot errors
        ax.plot(errors.index, errors, 'r-', alpha=0.7, linewidth=1.5, label='Forecast Error')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('Forecast Error', color='red')
        ax.tick_params(axis='y', labelcolor='red')
        
        # Plot correlating series (shifted)
        common_dates = series_shifted.index.intersection(errors.index)
        ax2.plot(series_shifted.loc[common_dates].index, 
                series_shifted.loc[common_dates].values, 
                'b-', alpha=0.7, linewidth=1.5)
        
        # Add correlation info
        lag_desc = f"leads by {lag}Q" if lag > 0 else "contemporaneous"
        ax2.set_ylabel(f'{row["series"]} (standardized)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Title with correlation info
        ax.set_title(f'{row["series"]} ({lag_desc}) - Correlation: {row["correlation"]:.3f}', 
                    fontsize=11)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('forecast_error_correlates.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved plot: forecast_error_correlates.png")    

def create_corporate_monitor(data, eval_stats, target_series):
    """
    Create a monitoring dashboard for nonfinancial corporate indicators.
    """
    # Key corporate series from your results
    corporate_series = [
        'FU115114005',  # Nonfinancial corporate indicator 1
        'FA115114065',  # Nonfinancial corporate indicator 2
        'FU115114103',  # Related corporate series
        'FA115114103'   # Related corporate series
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Corporate indicators (7Q lead)
    ax = axes[0]
    for series in corporate_series[:2]:  # Top 2
        if series in data.columns:
            # Standardize for comparison
            standardized = (data[series] - data[series].mean()) / data[series].std()
            ax.plot(data.index, standardized, label=series, alpha=0.8)
    
    ax.set_title('Nonfinancial Corporate Indicators (7Q Lead)', fontsize=14)
    ax.set_ylabel('Standardized Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add shaded regions for high/low regimes
    threshold = 1.0  # 1 standard deviation
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-threshold, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(data.index, threshold, 3, alpha=0.1, color='red', label='High Risk')
    ax.fill_between(data.index, -threshold, -3, alpha=0.1, color='green', label='Low Risk')
    
    # Plot 2: 7Q Forward projection
    ax = axes[1]
    if 'FU115114005' in data.columns:
        series = data['FU115114005']
        # Shift by 7 quarters to show future impact
        shifted = series.shift(-7)
        ax.plot(data.index, series, 'b-', alpha=0.5, label='Current')
        ax.plot(data.index, shifted, 'r-', alpha=0.8, label='7Q Forward (Impact Period)')
    
    ax.set_title('Corporate Indicator Impact Timing', fontsize=12)
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Forecast errors
    ax = axes[2]
    if eval_stats is not None:
        error_dates = pd.DatetimeIndex(eval_stats['forecast_dates'])
        errors = pd.Series(eval_stats['errors'], index=error_dates)
        ax.plot(errors.index, errors, 'k-', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_title('4Q Ahead Forecast Errors', fontsize=12)
        ax.set_ylabel('Forecast Error')
        ax.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('corporate_indicator_monitor.png', dpi=150)
    plt.close()
    
    return corporate_series
    
def create_money_market_contrarian_signal(data, eval_stats):
    """
    Create contrarian signals from money market flows.
    """
    # Money market series (negative correlation)
    mm_series = ['FU513264103', 'FA513264103']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Money market flows
    ax = axes[0]
    for series in mm_series:
        if series in data.columns:
            # Calculate quarter-over-quarter change
            qoq_change = data[series].pct_change()
            ax.plot(data.index, qoq_change * 100, label=f'{series} QoQ %', alpha=0.8)
    
    ax.set_title('Money Market Fund Flows (Contrarian Indicator)', fontsize=14)
    ax.set_ylabel('QoQ Change (%)')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Contrarian signal
    ax = axes[1]
    if 'FU513264103' in data.columns:
        mm_data = data['FU513264103']
        
        # Create z-score
        z_score = (mm_data - mm_data.rolling(20).mean()) / mm_data.rolling(20).std()
        
        # Contrarian signal (inverted because of negative correlation)
        contrarian_signal = -z_score
        
        ax.plot(data.index, contrarian_signal, 'purple', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax.fill_between(data.index, 0, contrarian_signal.where(contrarian_signal > 0), 
                       alpha=0.3, color='green', label='Bullish Signal')
        ax.fill_between(data.index, 0, contrarian_signal.where(contrarian_signal < 0), 
                       alpha=0.3, color='red', label='Bearish Signal')
        
        ax.set_title('Contrarian Signal (6Q Lead)', fontsize=12)
        ax.set_ylabel('Signal Strength')
        ax.legend()
    
    # Plot 3: Signal vs Errors
    ax = axes[2]
    if eval_stats is not None and 'FU513264103' in data.columns:
        # Shift signal by 6 quarters
        signal_shifted = contrarian_signal.shift(6)
        
        error_dates = pd.DatetimeIndex(eval_stats['forecast_dates'])
        errors = pd.Series(eval_stats['errors'], index=error_dates)
        
        # Align data
        common_dates = signal_shifted.index.intersection(errors.index)
        
        ax2 = ax.twinx()
        ax.plot(common_dates, errors.loc[common_dates], 'k-', alpha=0.7, label='Forecast Errors')
        ax2.plot(common_dates, signal_shifted.loc[common_dates], 'purple', 
                alpha=0.7, label='Contrarian Signal (6Q earlier)')
        
        ax.set_ylabel('Forecast Error', color='black')
        ax2.set_ylabel('Contrarian Signal', color='purple')
        ax.set_xlabel('Date')
        ax.set_title('Contrarian Signal vs Realized Errors', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('money_market_contrarian.png', dpi=150)
    plt.close()
    
    return contrarian_signal if 'FU513264103' in data.columns else None
    
def create_risk_sentiment_indicator(data, correlate_results):
    """
    Build a composite risk sentiment indicator from multiple series.
    """
    # Group series by correlation sign
    positive_indicators = []
    negative_indicators = []
    
    for _, row in correlate_results.iterrows():
        if row['correlation'] > 0.6:
            positive_indicators.append((row['series'], row['lag'], row['correlation']))
        elif row['correlation'] < -0.6:
            negative_indicators.append((row['series'], row['lag'], row['correlation']))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Risk-on indicators
    ax = axes[0]
    risk_on_composite = pd.Series(0, index=data.index)
    
    for series, lag, corr in positive_indicators[:5]:  # Top 5
        if series in data.columns:
            # Standardize and weight by correlation
            std_series = (data[series] - data[series].mean()) / data[series].std()
            risk_on_composite += std_series * abs(corr)
    
    risk_on_composite = risk_on_composite / len(positive_indicators[:5])
    ax.plot(data.index, risk_on_composite, 'g-', linewidth=2, label='Risk-On Composite')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Risk-On Indicators (Positive Correlation)', fontsize=14)
    ax.set_ylabel('Composite Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Risk-off indicators
    ax = axes[1]
    risk_off_composite = pd.Series(0, index=data.index)
    
    for series, lag, corr in negative_indicators[:5]:  # Top 5
        if series in data.columns:
            # Standardize and weight by correlation
            std_series = (data[series] - data[series].mean()) / data[series].std()
            risk_off_composite += std_series * abs(corr)
    
    risk_off_composite = risk_off_composite / len(negative_indicators[:5])
    ax.plot(data.index, risk_off_composite, 'r-', linewidth=2, label='Risk-Off Composite')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Risk-Off Indicators (Negative Correlation)', fontsize=14)
    ax.set_ylabel('Composite Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Net Risk Sentiment
    ax = axes[2]
    net_sentiment = risk_on_composite - risk_off_composite
    
    ax.plot(data.index, net_sentiment, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.fill_between(data.index, 0, net_sentiment.where(net_sentiment > 0), 
                   alpha=0.3, color='green', label='Risk-On Regime')
    ax.fill_between(data.index, 0, net_sentiment.where(net_sentiment < 0), 
                   alpha=0.3, color='red', label='Risk-Off Regime')
    
    # Add regime change markers
    regime_changes = np.sign(net_sentiment).diff().fillna(0)
    change_dates = data.index[regime_changes != 0]
    for date in change_dates:
        ax.axvline(x=date, color='orange', linestyle=':', alpha=0.5)
    
    ax.set_title('Net Risk Sentiment Indicator', fontsize=14)
    ax.set_ylabel('Net Sentiment')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('risk_sentiment_indicator.png', dpi=150)
    plt.close()
    
    return {
        'risk_on': risk_on_composite,
        'risk_off': risk_off_composite,
        'net_sentiment': net_sentiment
    }
    
def create_adjusted_forecast(model, fitted_results, data, target_series, 
                           correlate_results, eval_stats):
    """
    Create an adjusted forecast using the leading indicators.
    """
    # Get base forecast
    base_forecast = model.predict_ahead(fitted_results, steps_ahead=4)
    
    # Get leading indicators
    corporate_signal = None
    mm_signal = None
    
    if 'FU115114005' in data.columns:
        # Corporate indicator (7Q lead)
        corp_data = data['FU115114005']
        corp_z = (corp_data - corp_data.rolling(20).mean()) / corp_data.rolling(20).std()
        corporate_signal = corp_z.iloc[-1]  # Current value
    
    if 'FU513264103' in data.columns:
        # Money market (6Q lead, inverted)
        mm_data = data['FU513264103']
        mm_z = (mm_data - mm_data.rolling(20).mean()) / mm_data.rolling(20).std()
        mm_signal = -mm_z.iloc[-1]  # Inverted
    
    # Build risk sentiment
    sentiment = create_risk_sentiment_indicator(data, correlate_results)
    # If dynamic_risk exists, use it; otherwise use sentiment
    if 'dynamic_risk' in globals() and dynamic_risk is not None:
        current_sentiment = dynamic_risk.iloc[-1]
    else:
        current_sentiment = sentiment['net_sentiment'].iloc[-1]
    
    # Create adjustment factor
    adjustment_factor = 0
    weights = {'corporate': 0.4, 'money_market': 0.3, 'dynamic_risk': 0.3}
    
    if corporate_signal is not None:
        adjustment_factor += weights['corporate']  * corporate_signal * 0.05
    if mm_signal is not None:
        adjustment_factor += weights['money_market'] * mm_signal      * 0.05
    adjustment_factor     += weights['dynamic_risk'] * current_sentiment * 0.04
    
    # Apply adjustment
    adjusted_forecast = base_forecast['point_forecast'].copy()
    for i in range(len(adjusted_forecast)):
        # Decay adjustment over horizon
        decay = 0.8 ** i
        adjusted_forecast.iloc[i] = adjusted_forecast.iloc[i] * (1 + adjustment_factor * decay)
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Forecasts comparison
    ax = axes[0]
    
    # Historical
    hist_data = data[target_series].iloc[-20:]
    ax.plot(hist_data.index, hist_data.values, 'k-', linewidth=2, label='Historical')
    
    # Base forecast
    ax.plot(base_forecast['point_forecast'].index, 
           base_forecast['point_forecast'][target_series], 
           'b--', linewidth=2, marker='o', label='Base Forecast')
    
    # Adjusted forecast
    ax.plot(adjusted_forecast.index, 
           adjusted_forecast[target_series], 
           'r-', linewidth=2, marker='s', label='Adjusted Forecast')
    
    # Confidence bands
    forecast_se = base_forecast['forecast_se'][target_series]
    ax.fill_between(adjusted_forecast.index,
                   adjusted_forecast[target_series] - 2*forecast_se,
                   adjusted_forecast[target_series] + 2*forecast_se,
                   alpha=0.2, color='red', label='95% CI')
    
    ax.set_title(f'Adjusted Forecast for {target_series}', fontsize=14)
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Adjustment factors
    ax = axes[1]
    
    # Create bar chart of current signals
    signals = {
        'Corporate\n(7Q lead)': corporate_signal if corporate_signal is not None else 0,
        'Money Market\n(6Q lead)': mm_signal if mm_signal is not None else 0,
        'Risk Sentiment': current_sentiment
    }
    
    colors = ['green' if v > 0 else 'red' for v in signals.values()]
    bars = ax.bar(signals.keys(), signals.values(), color=colors, alpha=0.7)
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.set_title('Current Leading Indicator Signals', fontsize=12)
    ax.set_ylabel('Signal Strength (Std Dev)')
    
    # Add adjustment percentage
    ax.text(0.98, 0.98, f'Net Adjustment: {adjustment_factor*100:.1f}%', 
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('adjusted_forecast.png', dpi=150)
    plt.close()
    
    # Return results
    return {
        'base_forecast': base_forecast['point_forecast'],
        'adjusted_forecast': adjusted_forecast,
        'adjustment_factor': adjustment_factor,
        'signals': signals
    }            

def get_recession_dates():
    """
    Get NBER recession dates for the US.
    Returns list of (start, end) date tuples.
    """
    # NBER recession dates (quarterly frequency)
    recessions = [
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
        ('2020-02-01', '2020-04-01')  # COVID recession
    ]
    
    return [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in recessions]


def plot_risk_indicators_with_recessions(risk_indicators, data, eval_stats=None):
    """
    Plot risk indicators with NBER recession shading and forecast error overlay.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1.5, 1.5, 1], hspace=0.15)
    
    # Get recession dates
    recession_dates = get_recession_dates()
    
    # Get the net risk score and its valid index
    net_risk = risk_indicators['net_risk_score']
    
    # Determine the valid range based on net_risk index
    if hasattr(net_risk, 'index'):
        valid_index = net_risk.index
        start_date = valid_index[0]
        end_date = valid_index[-1]
    else:
        valid_index = data.index
        start_date = data.index[0]
        end_date = data.index[-1]
    
    # Function to add recession bars
    def add_recession_bars(ax):
        for start, end in recession_dates:
            if end >= start_date and start <= end_date:
                ax.axvspan(max(start, start_date), min(end, end_date), 
                          alpha=0.3, color='gray', label='NBER Recession' if start == recession_dates[0][0] else "")
    
    # Plot 1: Net Risk Score with Recessions
    ax1 = fig.add_subplot(gs[0])
    
    # Add recession bars first (so they're in background)
    add_recession_bars(ax1)
    
    # Plot net risk score using its own index
    ax1.plot(valid_index, net_risk, 'b-', linewidth=2, label='Net Risk Score', zorder=3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5, zorder=2)
    ax1.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Risk-On Threshold', zorder=2)
    ax1.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Risk-Off Threshold', zorder=2)
    
    # Shade risk regimes using valid index
    ax1.fill_between(valid_index, 0, net_risk.where(net_risk > 0), 
                    alpha=0.2, color='green', label='Risk-On', zorder=1)
    ax1.fill_between(valid_index, 0, net_risk.where(net_risk < 0), 
                    alpha=0.2, color='red', label='Risk-Off', zorder=1)
    
    # Add early warning markers (when risk-off precedes recession)
    for start, end in recession_dates:
        if start > start_date and start < end_date:
            # Check if risk was negative before recession
            pre_recession_window = net_risk.loc[start - pd.DateOffset(months=12):start]
            if len(pre_recession_window) > 0 and pre_recession_window.mean() < -0.5:
                ax1.scatter(start, -2.5, marker='v', s=100, color='green', 
                          edgecolor='black', zorder=4)
    
    ax1.set_title('Net Risk Score vs NBER Recessions', fontsize=14)
    ax1.set_ylabel('Net Risk Score')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.set_xlim(start_date, end_date)
    
    # Plot 2: Risk Components with Recessions
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    add_recession_bars(ax2)
    
    # Plot risk components if they exist and align with net_risk
    if risk_indicators.get('risk_on_composite') is not None:
        risk_on = risk_indicators['risk_on_composite']
        # Align indices
        if hasattr(risk_on, 'index'):
            common_idx = risk_on.index.intersection(valid_index)
            ax2.plot(common_idx, risk_on.loc[common_idx], 'g-', 
                    linewidth=1.5, label='Risk-On Composite', alpha=0.8, zorder=3)
    
    if risk_indicators.get('risk_off_composite') is not None:
        risk_off = risk_indicators['risk_off_composite']
        # Align indices
        if hasattr(risk_off, 'index'):
            common_idx = risk_off.index.intersection(valid_index)
            ax2.plot(common_idx, risk_off.loc[common_idx], 'r-', 
                    linewidth=1.5, label='Risk-Off Composite', alpha=0.8, zorder=3)
    
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5, zorder=2)
    ax2.set_title('Risk-On vs Risk-Off Components', fontsize=12)
    ax2.set_ylabel('Composite Score')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3, zorder=0)
    
    # Plot 3: Lead-Lag Analysis
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    add_recession_bars(ax3)
    
    # Calculate shifted risk scores to show lead time
    lead_times = [0, 2, 4, 6]  # quarters
    colors = ['black', 'blue', 'purple', 'orange']
    
    for lead, color in zip(lead_times, colors):
        shifted_risk = net_risk.shift(lead)
        ax3.plot(valid_index, shifted_risk, color=color, 
                alpha=0.7, linewidth=1.5, 
                label=f'{lead}Q lead' if lead > 0 else 'Current', zorder=3)
    
    ax3.axhline(y=-1, color='r', linestyle='--', alpha=0.5, zorder=2)
    ax3.set_title('Risk Score Lead-Lag Analysis', fontsize=12)
    ax3.set_ylabel('Net Risk Score')
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, alpha=0.3, zorder=0)
    
    # Plot 4: Forecast Errors (if available)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    add_recession_bars(ax4)
    
    if eval_stats is not None:
        error_dates = pd.DatetimeIndex(eval_stats['forecast_dates'])
        errors = pd.Series(eval_stats['errors'], index=error_dates)
        
        # Plot errors
        ax4.scatter(errors.index, errors, c=errors, cmap='RdBu_r', 
                   alpha=0.6, s=30, zorder=3)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.5, zorder=2)
        
        # Add moving average
        if len(errors) > 4:
            ma_errors = errors.rolling(4).mean()
            ax4.plot(ma_errors.index, ma_errors, 'purple', 
                    linewidth=2, label='4Q MA', zorder=3)
        
        ax4.set_title('Forecast Errors vs Recessions', fontsize=12)
        ax4.set_ylabel('Forecast Error')
        ax4.legend(loc='lower left', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Forecast errors not available', 
                transform=ax4.transAxes, ha='center', va='center')
    
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3, zorder=0)
    
    # Format x-axis
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    
    plt.suptitle('Risk Indicators and NBER Recessions Analysis', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig('risk_indicators_with_recessions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create recession prediction scorecard
    create_recession_prediction_scorecard(net_risk, recession_dates, data.index)
    
    return analyze_recession_prediction_accuracy(net_risk, recession_dates, data.index)


def analyze_recession_prediction_accuracy(net_risk, recession_dates, date_index):
    """
    Analyze how well risk indicators predict recessions.
    """
    results = {
        'recessions_total': 0,
        'recessions_predicted': 0,
        'false_alarms': 0,
        'average_lead_time': [],
        'risk_score_at_start': []
    }
    
    # Analyze each recession
    for start, end in recession_dates:
        if start < date_index[0] or start > date_index[-1]:
            continue
            
        results['recessions_total'] += 1
        
        # Check risk score in the year before recession
        pre_recession_start = start - pd.DateOffset(months=12)
        pre_recession_end = start
        
        if pre_recession_start >= date_index[0]:
            pre_recession_scores = net_risk.loc[pre_recession_start:pre_recession_end]
            
            if len(pre_recession_scores) > 0:
                # Check if risk-off signal present
                if (pre_recession_scores < -1).any():
                    results['recessions_predicted'] += 1
                    
                    # Find first risk-off signal
                    first_signal = pre_recession_scores[pre_recession_scores < -1].index[0]
                    lead_time = (start - first_signal).days / 91.25  # Convert to quarters
                    results['average_lead_time'].append(lead_time)
                
                # Risk score at recession start
                if start in net_risk.index:
                    results['risk_score_at_start'].append(net_risk.loc[start])
    
    # Check for false alarms (risk-off not followed by recession)
    risk_off_periods = net_risk < -1
    risk_off_bool = risk_off_periods.astype(bool)
    risk_off_shifted = risk_off_periods.shift(1).fillna(False).astype(bool)
    risk_off_starts = risk_off_periods.loc[risk_off_bool & ~risk_off_shifted].index
    
    for signal_date in risk_off_starts:
        # Check if recession within 1 year
        future_date = signal_date + pd.DateOffset(months=12)
        recession_found = False
        
        for start, end in recession_dates:
            if start >= signal_date and start <= future_date:
                recession_found = True
                break
        
        if not recession_found:
            results['false_alarms'] += 1
    
    # Calculate statistics
    if results['recessions_total'] > 0:
        results['prediction_rate'] = results['recessions_predicted'] / results['recessions_total']
    else:
        results['prediction_rate'] = 0
    
    if results['average_lead_time']:
        results['avg_lead_time_quarters'] = np.mean(results['average_lead_time'])
    else:
        results['avg_lead_time_quarters'] = 0
    
    return results


def create_recession_prediction_scorecard(net_risk, recession_dates, date_index):
    """
    Create a visual scorecard of recession prediction performance.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.axis('off')
    
    # Analyze performance
    results = analyze_recession_prediction_accuracy(net_risk, recession_dates, date_index)
    
    # Count recessions in sample
    recessions_in_sample = sum(1 for start, end in recession_dates 
                              if start >= date_index[0] and start <= date_index[-1])
    
    scorecard_text = f"""
    RECESSION PREDICTION SCORECARD
    
    ╔════════════════════════════════════════════════════════════╗
    ║                    PERFORMANCE METRICS                      ║
    ╠════════════════════════════════════════════════════════════╣
    ║ Total Recessions in Sample:        {recessions_in_sample:>24} ║
    ║ Recessions with Warning Signal:    {results['recessions_predicted']:>24} ║
    ║ Detection Rate:                    {results['prediction_rate']*100:>23.1f}% ║
    ║ Average Lead Time:                 {results['avg_lead_time_quarters']:>20.1f} quarters ║
    ║ False Alarms:                      {results['false_alarms']:>24} ║
    ╚════════════════════════════════════════════════════════════╝
    
    SIGNAL INTERPRETATION:
    • Net Risk Score < -1.0 → Recession warning
    • Typical lead time: 2-4 quarters
    • Monitor risk-off persistence for confirmation
    
    KEY FINDINGS:
    """
    
    # Add specific recession analysis
    recession_analysis = []
    for start, end in recession_dates:
        if start >= date_index[0] and start <= date_index[-1]:
            # Check pre-recession signal
            pre_start = start - pd.DateOffset(months=12)
            if pre_start >= date_index[0]:
                pre_scores = net_risk.loc[pre_start:start]
                min_score = pre_scores.min() if len(pre_scores) > 0 else np.nan
                
                if not np.isnan(min_score):
                    if min_score < -1:
                        status = "✓ PREDICTED"
                    elif min_score < 0:
                        status = "~ Weak Signal"
                    else:
                        status = "✗ Missed"
                    
                    recession_analysis.append(
                        f"    {start.strftime('%Y-%m')}: {status} (min score: {min_score:.2f})"
                    )
    
    scorecard_text += "\n".join(recession_analysis[:5])  # Show first 5
    
    if len(recession_analysis) > 5:
        scorecard_text += f"\n    ... and {len(recession_analysis) - 5} more"
    
    # Display scorecard
    ax.text(0.5, 0.5, scorecard_text, transform=ax.transAxes,
           fontsize=11, ha='center', va='center', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.3))
    
    plt.title('Risk Indicator Recession Prediction Performance', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('recession_prediction_scorecard.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_clear_risk_indicators(data, correlate_results):
    """
    Create clear risk-on and risk-off indicators with defined thresholds and signals.
    """
    
    # Define clear risk-on indicators (positive correlation with errors)
    risk_on_indicators = {
        'Corporate Growth': {
            'series': ['FU115114005', 'FA115114065'],  # r = 0.837
            'weight': 0.30,
            'lag': 7,
            'description': 'Nonfinancial corporate expansion'
        },
        'Financial Expansion': {
            'series': ['FA223093043', 'FA223093005'],  # r = 0.679, 0.671
            'weight': 0.25,
            'lag': 4,
            'description': 'Finance company asset growth'
        },
        'Insurance Growth': {
            'series': ['FA593093005', 'FA593090015'],  # r = 0.655, 0.584
            'weight': 0.20,
            'lag': 4,
            'description': 'Insurance company expansion'
        },
        'Repo Activity': {
            'series': ['FR513264103'],  # r = 0.824
            'weight': 0.25,
            'lag': 6,
            'description': 'Repo market expansion'
        }
    }
    
    # Define clear risk-off indicators (negative correlation with errors)
    risk_off_indicators = {
        'Money Market Flight': {
            'series': ['FU513264103', 'FA513264103'],  # r = -0.828
            'weight': 0.35,
            'lag': 6,
            'description': 'Flight to money market safety'
        },
        'Corporate Bond Stress': {
            'series': ['FU313065473', 'FA313065473'],  # r = -0.811
            'weight': 0.25,
            'lag': 4,
            'description': 'Corporate bond market stress'
        },
        'Agency Securities': {
            'series': ['FA405013133', 'FU405013133'],  # r = -0.754
            'weight': 0.20,
            'lag': 6,
            'description': 'Flight to government agencies'
        },
        'Mortgage Market Stress': {
            'series': ['FA343064623', 'FU343064623'],  # r = -0.723
            'weight': 0.20,
            'lag': 6,
            'description': 'Mortgage market contraction'
        }
    }
    
    # Calculate standardized indicators
    risk_on_scores = {}
    risk_off_scores = {}
    
    for name, config in risk_on_indicators.items():
        score = pd.Series(0, index=data.index)
        count = 0
        for series in config['series']:
            if series in data.columns:
                # Calculate z-score
                z_score = (data[series] - data[series].rolling(20).mean()) / data[series].rolling(20).std()
                score += z_score
                count += 1
        if count > 0:
            risk_on_scores[name] = score / count
    
    for name, config in risk_off_indicators.items():
        score = pd.Series(0, index=data.index)
        count = 0
        for series in config['series']:
            if series in data.columns:
                # Calculate z-score
                z_score = (data[series] - data[series].rolling(20).mean()) / data[series].rolling(20).std()
                score += z_score
                count += 1
        if count > 0:
            risk_off_scores[name] = score / count
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Risk-On Indicators
    ax1 = fig.add_subplot(gs[0, :])
    for name, score in risk_on_scores.items():
        ax1.plot(data.index, score, label=name, linewidth=1.5, alpha=0.8)
    
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax1.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Risk-On Threshold')
    ax1.axhline(y=-1, color='r', linestyle='--', alpha=0.5)
    ax1.fill_between(data.index, 1, 3, alpha=0.1, color='green')
    ax1.set_title('Risk-On Indicators (Positive = Bullish for Growth)', fontsize=14)
    ax1.set_ylabel('Z-Score')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Risk-Off Indicators
    ax2 = fig.add_subplot(gs[1, :])
    for name, score in risk_off_scores.items():
        ax2.plot(data.index, score, label=name, linewidth=1.5, alpha=0.8)
    
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Risk-Off Threshold')
    ax2.axhline(y=-1, color='g', linestyle='--', alpha=0.5)
    ax2.fill_between(data.index, 1, 3, alpha=0.1, color='red')
    ax2.set_title('Risk-Off Indicators (Positive = Flight to Safety)', fontsize=14)
    ax2.set_ylabel('Z-Score')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Composite Risk Score
    ax3 = fig.add_subplot(gs[2, :])
    
    # Calculate weighted composite scores
    risk_on_composite = pd.Series(0, index=data.index)
    for name, config in risk_on_indicators.items():
        if name in risk_on_scores:
            risk_on_composite += risk_on_scores[name] * config['weight']
    
    risk_off_composite = pd.Series(0, index=data.index)
    for name, config in risk_off_indicators.items():
        if name in risk_off_scores:
            risk_off_composite += risk_off_scores[name] * config['weight']
    
    # Net risk score
    net_risk_score = risk_on_composite - risk_off_composite
    
    ax3.plot(data.index, net_risk_score, 'b-', linewidth=2.5)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax3.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Risk-On Signal')
    ax3.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Risk-Off Signal')
    
    # Color-coded background
    ax3.fill_between(data.index, 0, net_risk_score.where(net_risk_score > 0), 
                    alpha=0.3, color='green', label='Risk-On Regime')
    ax3.fill_between(data.index, 0, net_risk_score.where(net_risk_score < 0), 
                    alpha=0.3, color='red', label='Risk-Off Regime')
    
    ax3.set_title('Net Risk Score (Composite Indicator)', fontsize=14)
    ax3.set_ylabel('Net Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Current Status Dashboard
    ax4 = fig.add_subplot(gs[3, :])
    ax4.axis('off')
    
    # Get current values
    current_date = data.index[-1]
    current_risk_on = risk_on_composite.iloc[-1]
    current_risk_off = risk_off_composite.iloc[-1]
    current_net = net_risk_score.iloc[-1]
    
    # Determine regime
    if current_net > 1:
        regime = "STRONG RISK-ON"
        regime_color = 'darkgreen'
        forecast_bias = "Model likely to UNDER-forecast"
    elif current_net > 0:
        regime = "MILD RISK-ON"
        regime_color = 'green'
        forecast_bias = "Model may slightly under-forecast"
    elif current_net > -1:
        regime = "MILD RISK-OFF"
        regime_color = 'orange'
        forecast_bias = "Model may slightly over-forecast"
    else:
        regime = "STRONG RISK-OFF"
        regime_color = 'darkred'
        forecast_bias = "Model likely to OVER-forecast"
    
    # Create status text
    status_text = f"""
    CURRENT RISK ASSESSMENT ({current_date.strftime('%Y-%m-%d')})
    
    REGIME: {regime}
    
    Risk-On Score:  {current_risk_on:+.2f}
    Risk-Off Score: {current_risk_off:+.2f}
    Net Score:      {current_net:+.2f}
    
    FORECAST IMPLICATION:
    {forecast_bias}
    
    KEY SIGNALS:
    """
    
    # Add top signals
    all_scores = []
    for name, score in risk_on_scores.items():
        if not score.empty:
            all_scores.append((name, score.iloc[-1], 'Risk-On'))
    for name, score in risk_off_scores.items():
        if not score.empty:
            all_scores.append((name, score.iloc[-1], 'Risk-Off'))
    
    # Sort by absolute value
    all_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for i, (name, value, type_) in enumerate(all_scores[:4]):
        if value > 1:
            signal = "↑↑ STRONG"
        elif value > 0:
            signal = "↑ Positive"
        elif value > -1:
            signal = "↓ Negative"
        else:
            signal = "↓↓ STRONG"
        status_text += f"\n    • {name}: {value:+.2f} {signal}"
    
    # Display status
    ax4.text(0.5, 0.5, status_text, transform=ax4.transAxes,
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=regime_color, alpha=0.2))
    
    plt.suptitle('Risk Indicator Dashboard', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('risk_indicator_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create signal history
    create_risk_signal_history(net_risk_score, data.index)
    
    return {
        'risk_on_scores': risk_on_scores,
        'risk_off_scores': risk_off_scores,
        'risk_on_composite': risk_on_composite,
        'risk_off_composite': risk_off_composite,
        'net_risk_score': net_risk_score,
        'current_regime': regime,
        'forecast_bias': forecast_bias
    }


def create_risk_signal_history(net_risk_score, dates):
    """
    Create a detailed signal history showing regime changes.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: Risk regimes over time
    ax = axes[0]
    
    # Define regime thresholds
    strong_risk_on = net_risk_score > 1
    mild_risk_on = (net_risk_score > 0) & (net_risk_score <= 1)
    mild_risk_off = (net_risk_score >= -1) & (net_risk_score <= 0)
    strong_risk_off = net_risk_score < -1
    
    # Plot regime bands
    ax.fill_between(dates, 0, 1, where=strong_risk_on, alpha=0.8, 
                   color='darkgreen', label='Strong Risk-On', transform=ax.get_xaxis_transform())
    ax.fill_between(dates, 0, 1, where=mild_risk_on, alpha=0.5, 
                   color='lightgreen', label='Mild Risk-On', transform=ax.get_xaxis_transform())
    ax.fill_between(dates, 0, 1, where=mild_risk_off, alpha=0.5, 
                   color='orange', label='Mild Risk-Off', transform=ax.get_xaxis_transform())
    ax.fill_between(dates, 0, 1, where=strong_risk_off, alpha=0.8, 
                   color='darkred', label='Strong Risk-Off', transform=ax.get_xaxis_transform())
    
    ax.plot(dates, net_risk_score, 'k-', linewidth=1, alpha=0.8)
    ax.set_title('Risk Regime History', fontsize=14)
    ax.set_ylabel('Net Risk Score')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Regime duration analysis
    ax = axes[1]
    
    # Calculate regime durations
    regime_labels = pd.Series('Neutral', index=dates)
    regime_labels[strong_risk_on] = 'Strong Risk-On'
    regime_labels[mild_risk_on] = 'Mild Risk-On'
    regime_labels[mild_risk_off] = 'Mild Risk-Off'
    regime_labels[strong_risk_off] = 'Strong Risk-Off'
    
    # Find regime changes
    regime_changes = regime_labels != regime_labels.shift(1)
    change_dates = dates[regime_changes]
    
    # Mark regime changes
    for date in change_dates[1:]:  # Skip first
        ax.axvline(x=date, color='gray', linestyle=':', alpha=0.5)
    
    # Calculate average durations
    regime_groups = regime_labels.groupby((regime_labels != regime_labels.shift()).cumsum())
    
    durations = {}
    for regime in ['Strong Risk-On', 'Mild Risk-On', 'Mild Risk-Off', 'Strong Risk-Off']:
        regime_durations = []
        for name, group in regime_groups:
            if len(group) > 0 and group.iloc[0] == regime:
                regime_durations.append(len(group))
        if regime_durations:
            durations[regime] = np.mean(regime_durations)
    
    # Plot average durations
    if durations:
        regimes = list(durations.keys())
        avg_durations = list(durations.values())
        colors = ['darkgreen', 'lightgreen', 'orange', 'darkred']
        
        bars = ax.bar(range(len(regimes)), avg_durations, color=colors[:len(regimes)], alpha=0.7)
        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels(regimes)
        ax.set_ylabel('Average Duration (Quarters)')
        ax.set_title('Average Regime Duration', fontsize=12)
        
        # Add value labels
        for bar, val in zip(bars, avg_durations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('risk_regime_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_forecast_adjustment_rules(risk_indicators):
    """
    Create clear rules for forecast adjustment based on risk indicators.
    """
    rules = {
        'Strong Risk-On': {
            'threshold': 1.0,
            'adjustment': 0.05,  # 5% upward
            'confidence_expansion': 1.2,  # Widen confidence bands by 20%
            'description': 'Strong growth momentum, increase forecast'
        },
        'Mild Risk-On': {
            'threshold': 0.0,
            'adjustment': 0.02,  # 2% upward
            'confidence_expansion': 1.1,  # Widen confidence bands by 10%
            'description': 'Moderate growth bias, slight increase'
        },
        'Mild Risk-Off': {
            'threshold': -1.0,
            'adjustment': -0.02,  # 2% downward
            'confidence_expansion': 1.1,  # Widen confidence bands by 10%
            'description': 'Moderate caution, slight decrease'
        },
        'Strong Risk-Off': {
            'threshold': -np.inf,
            'adjustment': -0.05,  # 5% downward
            'confidence_expansion': 1.2,  # Widen confidence bands by 20%
            'description': 'Strong flight to safety, decrease forecast'
        }
    }
    
    # Create visual rule card
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    
    rule_text = """
    FORECAST ADJUSTMENT RULES
    
    Based on Net Risk Score:
    
    ╔═══════════════════╦═══════════════╦══════════════════╦═══════════════════╗
    ║ REGIME            ║ SCORE RANGE   ║ ADJUSTMENT       ║ CONFIDENCE BANDS  ║
    ╠═══════════════════╬═══════════════╬══════════════════╬═══════════════════╣
    ║ Strong Risk-On    ║ > +1.0        ║ +5% to forecast  ║ Widen by 20%      ║
    ║ Mild Risk-On      ║ 0 to +1.0     ║ +2% to forecast  ║ Widen by 10%      ║
    ║ Mild Risk-Off     ║ -1.0 to 0     ║ -2% to forecast  ║ Widen by 10%      ║
    ║ Strong Risk-Off   ║ < -1.0        ║ -5% to forecast  ║ Widen by 20%      ║
    ╚═══════════════════╩═══════════════╩══════════════════╩═══════════════════╝
    
    LEADING INDICATOR LAGS:
    • Corporate indicators: 7 quarters ahead
    • Money market flows: 6 quarters ahead  
    • Financial stress: 4-6 quarters ahead
    
    INTERPRETATION:
    • Risk-On → Model tends to under-forecast → Adjust UP
    • Risk-Off → Model tends to over-forecast → Adjust DOWN
    """
    
    ax.text(0.5, 0.5, rule_text, transform=ax.transAxes,
           fontsize=11, ha='center', va='center', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))
    
    plt.title('Forecast Adjustment Rule Card', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('forecast_adjustment_rules.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return rules
    
def diagnose_risk_indicator_nans(data, correlate_results):
    """
    Diagnose sources of NaN values in risk indicators.
    """
    logger.info("\n" + "="*60)
    logger.info("Diagnosing NaN Sources in Risk Indicators")
    logger.info("="*60)
    
    # Check each series used in risk indicators
    risk_series = {
        'Corporate Growth': ['FU115114005', 'FA115114065'],
        'Financial Expansion': ['FA223093043', 'FA223093005'],
        'Insurance Growth': ['FA593093005', 'FA593090015'],
        'Repo Activity': ['FR513264103'],
        'Money Market Flight': ['FU513264103', 'FA513264103'],
        'Corporate Bond Stress': ['FU313065473', 'FA313065473'],
        'Agency Securities': ['FA405013133', 'FU405013133'],
        'Mortgage Market Stress': ['FA343064623', 'FU343064623']
    }
    
    nan_report = []
    
    for indicator_name, series_list in risk_series.items():
        logger.info(f"\n{indicator_name}:")
        for series in series_list:
            if series in data.columns:
                series_data = data[series]
                nan_count = series_data.isna().sum()
                total_count = len(series_data)
                nan_pct = (nan_count / total_count) * 100
                
                # Check for leading/trailing NaNs
                first_valid = series_data.first_valid_index()
                last_valid = series_data.last_valid_index()
                
                logger.info(f"  {series}:")
                logger.info(f"    - Total NaNs: {nan_count} ({nan_pct:.1f}%)")
                logger.info(f"    - First valid data: {first_valid}")
                logger.info(f"    - Last valid data: {last_valid}")
                
                # Check rolling statistics
                rolling_mean = series_data.rolling(20).mean()
                rolling_std = series_data.rolling(20).std()
                
                # First 20 observations will have NaN in rolling stats
                rolling_nan_count = rolling_mean.isna().sum()
                logger.info(f"    - Rolling stats NaNs: {rolling_nan_count}")
                
                # Check for zero variance periods
                zero_std_count = (rolling_std == 0).sum()
                if zero_std_count > 0:
                    logger.info(f"    - WARNING: {zero_std_count} periods with zero variance!")
                
                nan_report.append({
                    'indicator': indicator_name,
                    'series': series,
                    'nan_count': nan_count,
                    'nan_pct': nan_pct,
                    'first_valid': first_valid,
                    'last_valid': last_valid,
                    'rolling_nans': rolling_nan_count,
                    'zero_variance_periods': zero_std_count
                })
            else:
                logger.info(f"  {series}: NOT FOUND IN DATA!")
                nan_report.append({
                    'indicator': indicator_name,
                    'series': series,
                    'nan_count': -1,
                    'nan_pct': -1,
                    'first_valid': None,
                    'last_valid': None,
                    'rolling_nans': -1,
                    'zero_variance_periods': -1
                })
    
    # Create summary
    nan_df = pd.DataFrame(nan_report)
    
    # Find problematic series
    missing_series = nan_df[nan_df['nan_count'] == -1]
    high_nan_series = nan_df[nan_df['nan_pct'] > 50]
    zero_var_series = nan_df[nan_df['zero_variance_periods'] > 0]
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    if not missing_series.empty:
        logger.info(f"\nMissing series: {len(missing_series)}")
        for _, row in missing_series.iterrows():
            logger.info(f"  - {row['series']} ({row['indicator']})")
    
    if not high_nan_series.empty:
        logger.info(f"\nHigh NaN series (>50%): {len(high_nan_series)}")
        for _, row in high_nan_series.iterrows():
            logger.info(f"  - {row['series']}: {row['nan_pct']:.1f}% NaN")
    
    if not zero_var_series.empty:
        logger.info(f"\nZero variance series: {len(zero_var_series)}")
        for _, row in zero_var_series.iterrows():
            logger.info(f"  - {row['series']}: {row['zero_variance_periods']} periods")
    
    return nan_df
    
def analyze_series_quality(data, series_list, min_years=30, min_variation_pct=5):
    """
    Analyze series quality for use in risk indicators.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full dataset
    series_list : list
        List of series to analyze
    min_years : int
        Minimum years of data required
    min_variation_pct : float
        Minimum percentage of time series should be varying
    
    Returns:
    --------
    pd.DataFrame with quality metrics
    """
    quality_report = []
    
    for series in series_list:
        if series not in data.columns:
            continue
            
        series_data = data[series]
        
        # Basic statistics
        first_valid = series_data.first_valid_index()
        last_valid = series_data.last_valid_index()
        
        if first_valid is None:
            continue
            
        # Calculate years of data
        years_of_data = (last_valid - first_valid).days / 365.25
        
        # Calculate variation metrics
        # Count quarters where value changes by more than 0.1%
        pct_change = series_data.pct_change().abs()
        meaningful_changes = (pct_change > 0.001).sum()
        variation_pct = (meaningful_changes / series_data.notna().sum()) * 100
        
        # Count unique values (to detect mostly constant series)
        unique_values = series_data.nunique()
        unique_pct = (unique_values / series_data.notna().sum()) * 100
        
        # Detect crisis-only series (large spikes only during known crises)
        crisis_periods = [
            ('2000-01-01', '2002-12-31'),  # Dot-com
            ('2007-01-01', '2009-12-31'),  # Financial crisis
            ('2020-01-01', '2021-12-31'),  # COVID
        ]
        
        # Calculate activity during crisis vs normal times
        crisis_mask = pd.Series(False, index=data.index)
        for start, end in crisis_periods:
            crisis_mask |= (data.index >= start) & (data.index <= end)
        
        crisis_variation = pct_change[crisis_mask].mean()
        normal_variation = pct_change[~crisis_mask].mean()
        crisis_ratio = crisis_variation / (normal_variation + 1e-10)
        
        quality_report.append({
            'series': series,
            'years_of_data': years_of_data,
            'first_valid': first_valid,
            'last_valid': last_valid,
            'variation_pct': variation_pct,
            'unique_pct': unique_pct,
            'unique_values': unique_values,
            'crisis_ratio': crisis_ratio,
            'mean': series_data.mean(),
            'std': series_data.std(),
            'cv': series_data.std() / (abs(series_data.mean()) + 1e-10)
        })
    
    return pd.DataFrame(quality_report)


def select_quality_risk_indicators(data, correlate_results, 
                                 min_years=30, 
                                 min_variation_pct=10,
                                 max_crisis_ratio=5,
                                 min_abs_correlation=0.5):
    """
    Select high-quality series for risk indicators based on:
    - Long history
    - Consistent variation (not just crisis spikes)
    - Strong correlation with forecast errors
    """
    logger.info("\n" + "="*60)
    logger.info("Selecting Quality Risk Indicators")
    logger.info("="*60)
    
    # Get all series from correlation results
    all_series = correlate_results['series'].unique()
    
    # Analyze quality
    quality_df = analyze_series_quality(data, all_series, min_years, min_variation_pct)
    
    # Apply quality filters
    quality_series = quality_df[
        (quality_df['years_of_data'] >= min_years) &
        (quality_df['variation_pct'] >= min_variation_pct) &
        (quality_df['crisis_ratio'] <= max_crisis_ratio) &
        (quality_df['unique_pct'] > 1)  # More than 1% unique values
    ]
    
    logger.info(f"\nQuality filters applied:")
    logger.info(f"  - Minimum {min_years} years of data")
    logger.info(f"  - Minimum {min_variation_pct}% variation")
    logger.info(f"  - Crisis ratio < {max_crisis_ratio}")
    logger.info(f"  - Started with {len(all_series)} series")
    logger.info(f"  - {len(quality_series)} series pass quality filters")
    
    # Merge with correlation results
    quality_correlations = correlate_results.merge(
        quality_series[['series', 'years_of_data', 'variation_pct', 'crisis_ratio']], 
        on='series'
    )
    
    # Separate into risk-on and risk-off
    risk_on_candidates = quality_correlations[
        (quality_correlations['correlation'] > min_abs_correlation)
    ].sort_values('abs_correlation', ascending=False)
    
    risk_off_candidates = quality_correlations[
        (quality_correlations['correlation'] < -min_abs_correlation)
    ].sort_values('abs_correlation', ascending=False)
    
    logger.info(f"\nRisk-On candidates: {len(risk_on_candidates)}")
    logger.info(f"Risk-Off candidates: {len(risk_off_candidates)}")
    
    # Select diverse indicators (different lags and types)
    def select_diverse_indicators(candidates, n_select=10):
        selected = []
        used_prefixes = set()
        used_lags = {}
        
        for _, row in candidates.iterrows():
            # Extract prefix (first 2-3 characters)
            prefix = row['series'][:3]
            lag = row['lag']
            
            # Skip if we already have this type with similar lag
            if prefix in used_lags and abs(used_lags[prefix] - lag) < 2:
                continue
                
            selected.append(row)
            used_prefixes.add(prefix)
            used_lags[prefix] = lag
            
            if len(selected) >= n_select:
                break
                
        return pd.DataFrame(selected)
    
    # Select final indicators
    final_risk_on = select_diverse_indicators(risk_on_candidates, 10)
    final_risk_off = select_diverse_indicators(risk_off_candidates, 10)
    
    # Log selected indicators
    logger.info("\n" + "="*60)
    logger.info("SELECTED RISK-ON INDICATORS:")
    for _, row in final_risk_on.iterrows():
        logger.info(f"  {row['series']}: r={row['correlation']:.3f}, "
                   f"lag={row['lag']}Q, {row['years_of_data']:.1f} years, "
                   f"{row['variation_pct']:.1f}% variation")
    
    logger.info("\nSELECTED RISK-OFF INDICATORS:")
    for _, row in final_risk_off.iterrows():
        logger.info(f"  {row['series']}: r={row['correlation']:.3f}, "
                   f"lag={row['lag']}Q, {row['years_of_data']:.1f} years, "
                   f"{row['variation_pct']:.1f}% variation")
    
    # Save quality analysis
    quality_df.to_csv('series_quality_analysis.csv', index=False)
    logger.info("\nSaved quality analysis to: series_quality_analysis.csv")
    
    return {
        'risk_on': final_risk_on,
        'risk_off': final_risk_off,
        'quality_df': quality_df
    }


def create_robust_risk_indicators(data, selected_indicators):
    """
    Create risk indicators using only high-quality series.
    """
    risk_on_df = selected_indicators['risk_on']
    risk_off_df = selected_indicators['risk_off']
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1.5, 2, 1], hspace=0.2)
    
    # Calculate risk scores
    risk_on_composite = pd.Series(0, index=data.index)
    risk_off_composite = pd.Series(0, index=data.index)
    
    # Plot 1: Risk-On indicators
    ax1 = fig.add_subplot(gs[0])
    
    for _, row in risk_on_df.head(5).iterrows():
        series = row['series']
        if series in data.columns:
            # Robust z-score with expanding window for early periods
            series_data = data[series]
            expanding_mean = series_data.expanding(min_periods=20).mean()
            expanding_std = series_data.expanding(min_periods=20).std()
            
            # Switch to rolling after enough data
            rolling_mean = series_data.rolling(40, min_periods=20).mean()
            rolling_std = series_data.rolling(40, min_periods=20).std()
            
            # Use expanding for first 60 periods, then rolling
            mean = expanding_mean.copy()
            std = expanding_std.copy()
            mean.iloc[60:] = rolling_mean.iloc[60:]
            std.iloc[60:] = rolling_std.iloc[60:]
            
            # Calculate z-score
            z_score = pd.Series(0.0, index=data.index, dtype=float)
            valid_mask = (std > 1e-6) & series_data.notna()
            z_score[valid_mask] = (series_data[valid_mask] - mean[valid_mask]) / std[valid_mask]
            z_score = z_score.clip(-3, 3)
            
            # Weight by correlation
            weight = abs(row['correlation'])
            risk_on_composite += z_score * weight
            
            # Plot
            ax1.plot(data.index, z_score, label=f"{series} (r={row['correlation']:.2f})", 
                    alpha=0.7, linewidth=1)
    
    risk_on_composite = risk_on_composite / risk_on_df['abs_correlation'].sum()
    
    ax1.set_title('Risk-On Indicators (Quality-Filtered)', fontsize=14)
    ax1.set_ylabel('Z-Score')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax1.axhline(y=1, color='g', linestyle='--', alpha=0.5)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Risk-Off indicators
    ax2 = fig.add_subplot(gs[1])
    
    for _, row in risk_off_df.head(5).iterrows():
        series = row['series']
        if series in data.columns:
            # Same robust z-score calculation
            series_data = data[series]
            expanding_mean = series_data.expanding(min_periods=20).mean()
            expanding_std = series_data.expanding(min_periods=20).std()
            rolling_mean = series_data.rolling(40, min_periods=20).mean()
            rolling_std = series_data.rolling(40, min_periods=20).std()
            
            mean = expanding_mean.copy()
            std = expanding_std.copy()
            mean.iloc[60:] = rolling_mean.iloc[60:]
            std.iloc[60:] = rolling_std.iloc[60:]
            
            z_score = pd.Series(0.0, index=data.index, dtype=float)
            valid_mask = (std > 1e-6) & series_data.notna()
            z_score[valid_mask] = (series_data[valid_mask] - mean[valid_mask]) / std[valid_mask]
            z_score = z_score.clip(-3, 3)
            
            weight = abs(row['correlation'])
            risk_off_composite += z_score * weight
            
            ax2.plot(data.index, z_score, label=f"{series} (r={row['correlation']:.2f})", 
                    alpha=0.7, linewidth=1)
    
    risk_off_composite = risk_off_composite / risk_off_df['abs_correlation'].sum()
    
    ax2.set_title('Risk-Off Indicators (Quality-Filtered)', fontsize=14)
    ax2.set_ylabel('Z-Score')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Net Risk Score
    ax3 = fig.add_subplot(gs[2])
    
    net_risk_score = risk_on_composite - risk_off_composite
    
    ax3.plot(data.index, net_risk_score, 'b-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax3.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Risk-On Signal')
    ax3.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Risk-Off Signal')
    
    ax3.fill_between(data.index, 0, net_risk_score.where(net_risk_score > 0), 
                    alpha=0.3, color='green', label='Risk-On Regime')
    ax3.fill_between(data.index, 0, net_risk_score.where(net_risk_score < 0), 
                    alpha=0.3, color='red', label='Risk-Off Regime')
    
    ax3.set_title('Net Risk Score (Robust Calculation)', fontsize=14)
    ax3.set_ylabel('Net Score')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Indicator summary
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')
    
    summary_text = "QUALITY INDICATORS USED:\n\n"
    summary_text += "Risk-On Series:\n"
    for i, (_, row) in enumerate(risk_on_df.head(5).iterrows()):
        summary_text += f"  • {row['series']}: {row['years_of_data']:.0f} years, "
        summary_text += f"{row['variation_pct']:.0f}% varying\n"
    
    summary_text += "\nRisk-Off Series:\n"
    for i, (_, row) in enumerate(risk_off_df.head(5).iterrows()):
        summary_text += f"  • {row['series']}: {row['years_of_data']:.0f} years, "
        summary_text += f"{row['variation_pct']:.0f}% varying\n"
    
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Robust Risk Indicators (Quality Series Only)', fontsize=16)
    plt.tight_layout()
    plt.savefig('robust_risk_indicators.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'risk_on_composite': risk_on_composite,
        'risk_off_composite': risk_off_composite,
        'net_risk_score': net_risk_score,
        'indicators_used': selected_indicators
    }        

def create_full_sample_risk_indicators(data, selected_indicators):
    """
    Create risk indicators that work across the full sample period.
    """
    risk_on_df = selected_indicators['risk_on']
    risk_off_df = selected_indicators['risk_off']
    
    # Initialize series
    risk_on_composite = pd.Series(0.0, index=data.index)
    risk_off_composite = pd.Series(0.0, index=data.index)
    
    # Process each indicator with appropriate normalization
    logger.info("Building risk composites with full-sample coverage...")
    
    # Risk-On indicators
    risk_on_weights = []
    for _, row in risk_on_df.iterrows():
        series_name = row['series']
        if series_name in data.columns:
            series_data = data[series_name]
            
            # Use expanding window for early periods, then rolling
            # This ensures we have values from the beginning
            expanding_mean = series_data.expanding(min_periods=4).mean()
            expanding_std = series_data.expanding(min_periods=4).std()
            
            # For recent periods, use rolling window
            rolling_mean = series_data.rolling(window=20, min_periods=4).mean()
            rolling_std = series_data.rolling(window=20, min_periods=4).std()
            
            # Combine: use expanding for first 40 obs, then rolling
            combined_mean = expanding_mean.copy()
            combined_std = expanding_std.copy()
            
            if len(series_data) > 40:
                combined_mean.iloc[40:] = rolling_mean.iloc[40:]
                combined_std.iloc[40:] = rolling_std.iloc[40:]
            
            # Calculate z-score
            z_score = pd.Series(0.0, index=data.index)
            valid_mask = (combined_std > 1e-8) & series_data.notna() & combined_mean.notna()
            
            if valid_mask.any():
                z_score[valid_mask] = (series_data[valid_mask] - combined_mean[valid_mask]) / combined_std[valid_mask]
                z_score = z_score.clip(-3, 3)
                
                # Add to composite with correlation weighting
                weight = abs(row['correlation'])
                risk_on_composite += z_score * weight
                risk_on_weights.append(weight)
                
                logger.info(f"  Added {series_name}: weight={weight:.3f}, coverage={valid_mask.sum()}/{len(data)}")
    
    # Normalize risk-on composite
    if risk_on_weights:
        risk_on_composite = risk_on_composite / sum(risk_on_weights)
    
    # Risk-Off indicators (same process)
    risk_off_weights = []
    for _, row in risk_off_df.iterrows():
        series_name = row['series']
        if series_name in data.columns:
            series_data = data[series_name]
            
            expanding_mean = series_data.expanding(min_periods=4).mean()
            expanding_std = series_data.expanding(min_periods=4).std()
            rolling_mean = series_data.rolling(window=20, min_periods=4).mean()
            rolling_std = series_data.rolling(window=20, min_periods=4).std()
            
            combined_mean = expanding_mean.copy()
            combined_std = expanding_std.copy()
            
            if len(series_data) > 40:
                combined_mean.iloc[40:] = rolling_mean.iloc[40:]
                combined_std.iloc[40:] = rolling_std.iloc[40:]
            
            z_score = pd.Series(0.0, index=data.index)
            valid_mask = (combined_std > 1e-8) & series_data.notna() & combined_mean.notna()
            
            if valid_mask.any():
                z_score[valid_mask] = (series_data[valid_mask] - combined_mean[valid_mask]) / combined_std[valid_mask]
                z_score = z_score.clip(-3, 3)
                
                weight = abs(row['correlation'])
                risk_off_composite += z_score * weight
                risk_off_weights.append(weight)
                
                logger.info(f"  Added {series_name}: weight={weight:.3f}, coverage={valid_mask.sum()}/{len(data)}")
    
    # Normalize risk-off composite
    if risk_off_weights:
        risk_off_composite = risk_off_composite / sum(risk_off_weights)
    
    # Calculate net risk score
    net_risk_score = risk_on_composite - risk_off_composite
    
    # Apply smoothing to reduce noise
    net_risk_smooth = net_risk_score.rolling(window=4, min_periods=1).mean()
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1.5, 2, 1], hspace=0.2)
    
    # Get recession dates
    recession_dates = get_recession_dates()
    
    # Plot 1: Risk composites
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, risk_on_composite, 'g-', label='Risk-On Composite', alpha=0.8)
    ax1.plot(data.index, risk_off_composite, 'r-', label='Risk-Off Composite', alpha=0.8)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Add recessions
    for start, end in recession_dates:
        if end >= data.index[0] and start <= data.index[-1]:
            ax1.axvspan(max(start, data.index[0]), min(end, data.index[-1]), 
                       alpha=0.3, color='gray', label='Recession' if start == recession_dates[0][0] else "")
    
    ax1.set_title('Risk-On vs Risk-Off Composites', fontsize=14)
    ax1.set_ylabel('Composite Score')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Net risk score with dynamic thresholds
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate percentile-based thresholds
    lookback = 40  # 10 years
    upper_threshold = net_risk_smooth.rolling(lookback, min_periods=20).quantile(0.85)
    lower_threshold = net_risk_smooth.rolling(lookback, min_periods=20).quantile(0.15)
    
    ax2.plot(data.index, net_risk_smooth, 'b-', linewidth=2, label='Net Risk Score (smoothed)')
    ax2.plot(data.index, upper_threshold, 'g--', alpha=0.7, label='85th percentile')
    ax2.plot(data.index, lower_threshold, 'r--', alpha=0.7, label='15th percentile')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Shade extreme periods
    risk_on_extreme = net_risk_smooth > upper_threshold
    risk_off_extreme = net_risk_smooth < lower_threshold
    
    ax2.fill_between(data.index, net_risk_smooth, upper_threshold,
                    where=risk_on_extreme, alpha=0.3, color='green', 
                    label='Extreme Risk-On', interpolate=True)
    ax2.fill_between(data.index, net_risk_smooth, lower_threshold,
                    where=risk_off_extreme, alpha=0.3, color='red', 
                    label='Extreme Risk-Off', interpolate=True)
    
    # Add recessions
    for start, end in recession_dates:
        if end >= data.index[0] and start <= data.index[-1]:
            ax2.axvspan(max(start, data.index[0]), min(end, data.index[-1]), 
                       alpha=0.3, color='gray')
    
    ax2.set_title('Net Risk Score with Percentile Thresholds', fontsize=14)
    ax2.set_ylabel('Net Score')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Recession probability model
    ax3 = fig.add_subplot(gs[2])
    
    # Create recession indicator (1 during recession, 0 otherwise)
    recession_indicator = pd.Series(0, index=data.index)
    for start, end in recession_dates:
        mask = (data.index >= start) & (data.index <= end)
        recession_indicator[mask] = 1
    
    # Calculate probability based on risk score percentile
    risk_percentile = net_risk_smooth.rolling(lookback, min_periods=20).apply(
        lambda x: (x.iloc[-1] < x).sum() / len(x) if len(x) > 0 else 0.5
    )
    
    # Convert to probability (lower percentile = higher recession probability)
    recession_probability = 1 - risk_percentile
    
    ax3.plot(data.index, recession_probability, 'purple', linewidth=2, label='Recession Probability')
    ax3.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='High Risk Threshold')
    
    # Shade actual recessions
    for start, end in recession_dates:
        if end >= data.index[0] and start <= data.index[-1]:
            ax3.axvspan(max(start, data.index[0]), min(end, data.index[-1]), 
                       alpha=0.3, color='gray')
    
    # Highlight high-risk periods
    high_risk = recession_probability > 0.7
    ax3.fill_between(data.index, 0, 1, where=high_risk, 
                    alpha=0.2, color='red', label='High Risk Signal')
    
    ax3.set_title('Recession Probability Model', fontsize=14)
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model performance
    ax4 = fig.add_subplot(gs[3])
    
    # Calculate hit rate for different thresholds
    thresholds = np.arange(0.5, 0.9, 0.05)
    hit_rates = []
    false_positive_rates = []
    
    for thresh in thresholds:
        # Signals
        signals = recession_probability > thresh
        
        # True positives: signal before/during recession
        true_positives = 0
        false_positives = 0
        
        for i in range(len(data.index)):
            if signals.iloc[i]:
                # Check if recession within next 4 quarters
                future_mask = (recession_indicator.index > data.index[i]) & \
                             (recession_indicator.index <= data.index[i] + pd.DateOffset(months=12))
                if recession_indicator[future_mask].any():
                    true_positives += 1
                else:
                    false_positives += 1
        
        total_signals = signals.sum()
        if total_signals > 0:
            hit_rate = true_positives / total_signals
            false_positive_rate = false_positives / total_signals
        else:
            hit_rate = 0
            false_positive_rate = 0
            
        hit_rates.append(hit_rate)
        false_positive_rates.append(false_positive_rate)
    
    ax4.plot(thresholds, hit_rates, 'g-', linewidth=2, label='Hit Rate')
    ax4.plot(thresholds, false_positive_rates, 'r-', linewidth=2, label='False Positive Rate')
    ax4.set_xlabel('Probability Threshold')
    ax4.set_ylabel('Rate')
    ax4.set_title('Model Performance by Threshold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Full-Sample Risk Indicators and Recession Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('full_sample_risk_indicators.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Analyze performance
    performance = analyze_full_performance(risk_off_extreme, recession_probability, 
                                         recession_indicator, data.index)
    
    logger.info("\nFull-Sample Risk Indicator Performance:")
    logger.info(f"  Recession periods covered: {performance['coverage']:.1f}%")
    logger.info(f"  Average lead time: {performance['avg_lead_time']:.1f} quarters")
    logger.info(f"  Best threshold: {performance['best_threshold']:.2f}")
    logger.info(f"  Best hit rate: {performance['best_hit_rate']:.1f}%")
    
    # Current assessment
    current_prob = recession_probability.iloc[-1] if not recession_probability.empty else 0.5
    current_risk = "HIGH" if current_prob > 0.7 else "MODERATE" if current_prob > 0.5 else "LOW"
    
    logger.info(f"\nCurrent Assessment:")
    logger.info(f"  Recession probability: {current_prob:.1f}")
    logger.info(f"  Risk level: {current_risk}")
    
    return {
        'risk_on_composite': risk_on_composite,
        'risk_off_composite': risk_off_composite,
        'net_risk_score': net_risk_score,
        'net_risk_smooth': net_risk_smooth,
        'recession_probability': recession_probability,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,
        'performance': performance
    }


def analyze_full_performance(risk_off_extreme, recession_probability, 
                           recession_indicator, date_index):
    """
    Analyze performance of the full-sample indicators.
    """
    # Calculate coverage of recession periods
    recession_dates = get_recession_dates()
    total_recession_quarters = 0
    covered_quarters = 0
    lead_times = []
    
    for start, end in recession_dates:
        if start >= date_index[0] and end <= date_index[-1]:
            recession_mask = (date_index >= start) & (date_index <= end)
            total_recession_quarters += recession_mask.sum()
            
            # Check if we had warning signals
            pre_recession = date_index[(date_index >= start - pd.DateOffset(months=12)) & 
                                      (date_index < start)]
            
            if len(pre_recession) > 0:
                # Check for risk-off signals
                pre_signals = risk_off_extreme[pre_recession]
                if pre_signals.any():
                    covered_quarters += recession_mask.sum()
                    # Find first signal
                    first_signal = pre_signals[pre_signals].index[0]
                    lead_time = (start - first_signal).days / 91.25
                    lead_times.append(lead_time)
    
    coverage = (covered_quarters / total_recession_quarters * 100) if total_recession_quarters > 0 else 0
    avg_lead_time = np.mean(lead_times) if lead_times else 0
    
    # Find best probability threshold
    best_threshold = 0.7
    best_hit_rate = 0
    
    for thresh in np.arange(0.5, 0.85, 0.05):
        signals = recession_probability > thresh
        hits = 0
        total = 0
        
        for i in range(len(signals) - 4):
            if signals.iloc[i]:
                total += 1
                # Check if recession within next 4 quarters
                if recession_indicator.iloc[i:i+4].any():
                    hits += 1
        
        if total > 0:
            hit_rate = hits / total * 100
            if hit_rate > best_hit_rate:
                best_hit_rate = hit_rate
                best_threshold = thresh
    
    return {
        'coverage': coverage,
        'avg_lead_time': avg_lead_time,
        'best_threshold': best_threshold,
        'best_hit_rate': best_hit_rate
    }
    
def compute_rolling_ridge_weights(
    zscore_df: pd.DataFrame,
    forecast_errors: pd.Series,
    window: int = 60,
    alphas=(0.1, 1.0, 10.0)
) -> pd.DataFrame:
    """
    Rolling 60-quarter ridge regression: 4Q-ahead errors ~ z-scores.
    Returns a DataFrame of coefficients (index = date, cols = indicators).
    """
    coef_hist = []
    dates = []
    
    # Get number of features
    n_features = zscore_df.shape[1]

    for t in range(window, len(zscore_df)):
        X_win = zscore_df.iloc[t-window:t]
        y_win = forecast_errors.shift(-4).iloc[t-window:t]  # align error with lead

        # Create mask for non-NaN rows
        mask = X_win.notna().all(axis=1) & y_win.notna()
        
        # Use .loc to avoid reindexing warning
        X_sub = X_win.loc[mask]
        y_sub = y_win.loc[mask]

        if len(X_sub) < window * 0.7:  # need 70% data
            # Append array of NaNs with correct shape
            coef_hist.append(np.full(n_features, np.nan))
            dates.append(zscore_df.index[t])
            continue

        try:
            model = RidgeCV(alphas=alphas, fit_intercept=False, cv=5)
            model.fit(X_sub, y_sub)
            coef_hist.append(model.coef_)
            dates.append(zscore_df.index[t])
        except Exception as e:
            # If model fitting fails, append NaNs
            logger.warning(f"Ridge regression failed at time {t}: {e}")
            coef_hist.append(np.full(n_features, np.nan))
            dates.append(zscore_df.index[t])

    coef_df = pd.DataFrame(coef_hist, index=dates, columns=zscore_df.columns)
    return coef_df

def build_dynamic_composites(zscore_df, coef_df):
    """Dot-product of time-varying weights and z-scores to get dynamic risk scores."""
    common = zscore_df.index.intersection(coef_df.index)
    beta = coef_df.loc[common]
    z    = zscore_df.loc[common]

    composite = (beta * z).sum(axis=1)
    return composite    
    
def main():
    """Main driver function."""
    
    # ---------------------------------
    # 1. Load and prepare data
    # ---------------------------------
    logger.info("Loading data...")
    data = pd.read_parquet(
        "/home/tesla/Z1/temp/data/z1_quarterly/z1_quarterly_data_filtered.parquet"
    )

    # Ensure proper frequency
    if data.index.freq is None:
        data.index = pd.date_range(
            start=data.index[0],
            periods=len(data),
            freq="QE"
        )

    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

    # ---------------------------------
    # 2. Set target series and analyze
    # ---------------------------------
    target_series = "FA086902005"
    logger.info(f"\nTarget series: {target_series}")

    # Analyze target series with HP filter
    logger.info("\nAnalyzing target series with HP filter...")
    hp_stats = analyze_series_with_hp(data, target_series)
    
    if hp_stats:
        logger.info("\nHP filter statistics:")
        for key, value in hp_stats.items():
            logger.info(f"  {key}: {value:.2e}")

    # ---------------------------------
    # 3. Fit Hierarchical Kalman Filter
    # ---------------------------------
    logger.info("\n" + "="*60)
    logger.info("Fitting Hierarchical Kalman Filter")
    logger.info("="*60)

    try:
        model, results = fit_hierarchical_kalman_filter(
            data=data,
            formulas_file="fof_formulas_extracted.json",
            series_list=[target_series],
            normalize=True,                # Use global normalization
            error_variance_ratio=0.01,      # 1% error for computed series
            loglikelihood_burn=20,          # Skip first 20 observations
            use_exact_diffuse=False,        # Approximate diffuse init
            transformation='square',        # Square transformation like statsmodels
            max_attempts=3                  # Try each method 3 times
        )
        
        logger.info("\nModel fitted successfully!")
        
    except Exception as e:
        logger.error(f"Error fitting model: {e}")
        raise

    # ---------------------------------
    # 4. Display Results
    # ---------------------------------
    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)

    # Model summary
    fitted_results = results['fitted_results']
    logger.info(f"\nLog-likelihood: {fitted_results.llf:.2f}")
    logger.info(f"AIC: {fitted_results.aic:.2f}")
    logger.info(f"Number of parameters: {len(fitted_results.params)}")

    # Parameter estimates
    logger.info("\nParameter estimates:")
    final_variances = model.transform_params(fitted_results.params)
    for i, (name, param, var) in enumerate(zip(model.param_names, fitted_results.params, final_variances)):
        if model.transformation == 'square':
            logger.info(f"  {name}: param={param:.6f} (sqrt), variance={var:.6f}")
        else:
            logger.info(f"  {name}: param={param:.6f} (log), variance={var:.6f}")
            
    # ---------------------------------
    # 5. Validate Formula Constraints
    # ---------------------------------
    logger.info("\n" + "="*60)
    logger.info("Formula Validation")
    logger.info("="*60)

    validation = results['validation']
    if not validation.empty:
        logger.info("\nValidation results for computed series:")
        
        # Summary statistics
        mean_error = validation['relative_error'].mean()
        max_error = validation['relative_error'].max()
        logger.info(f"\nOverall: Mean relative error = {mean_error:.2%}, Max = {max_error:.2%}")
        
        # Individual series
        for _, row in validation.iterrows():
            logger.info(f"\n{row['series']}: {row['formula']}")
            logger.info(f"  MAE: {row['mae']:.2f}")
            logger.info(f"  RMSE: {row['rmse']:.2f}")
            logger.info(f"  Relative error: {row['relative_error']:.2%}")

    # ---------------------------------
    # 6. Create Plots
    # ---------------------------------
    logger.info("\n" + "="*60)
    logger.info("Creating Plots")
    logger.info("="*60)

    filtered_series = results['filtered_series']
    series_list = results['series_list']
    
    # Get forecast for next 4 quarters
    logger.info("\n" + "="*60)
    logger.info("Forecasting")
    logger.info("="*60)

    forecast = model.predict_ahead(fitted_results, steps_ahead=4)
    logger.info("\nForecast for next 4 periods:")
    logger.info(forecast['point_forecast'][target_series])

    # Check differences between filtered and smoothed
    logger.info("\nDifferences between filtered and smoothed estimates:")
    diff_stats = filtered_series['filter_smooth_diff'].describe()
    logger.info(f"Mean absolute difference across all series: {diff_stats.loc['mean'].abs().mean():.2e}")
    logger.info(f"Max absolute difference across all series: {diff_stats.loc['max'].abs().max():.2e}")     

    # Define burn-in period for visualization
    burn_in = 20
    burn_in_end = data.index[burn_in] if len(data) > burn_in else data.index[-1]

    # Plot 1: Target series detailed analysis
    create_target_series_plot(data, filtered_series, target_series, burn_in, burn_in_end)
    
    # Plot 2: All series comparison
    create_all_series_plot(data, filtered_series, series_list, model, burn_in, burn_in_end)
    
    # Plot 3: Residual analysis
    create_residual_analysis_plot(data, filtered_series, target_series, burn_in)
    
    # Plot 4: Source vs Computed series
    create_hierarchy_plot(filtered_series, series_list, model, burn_in_end)

    # ---------------------------------
    # 7. Generate Most Probable Path
    # ---------------------------------
    logger.info("\n" + "="*60)
    logger.info("Generating Most Probable Path for Year Ahead")
    logger.info("="*60)

    # Generate forecast path
    path_result = model.get_most_probable_path(fitted_results)
    
    # Display path for target series
    logger.info(f"\nMost probable path for {target_series}:")
    logger.info(path_result['path'][target_series])
    logger.info(f"\nForecast uncertainty (std dev):")
    logger.info(path_result['uncertainty'][target_series])
    
    # Create path visualization
    create_forecast_path_plot(data, filtered_series, path_result, target_series)
    
    # Export forecast paths
    path_result['path'].to_csv('forecast_path_all_series.csv')
    path_result['uncertainty'].to_csv('forecast_uncertainty_all_series.csv')
    logger.info("\nExported forecast paths to: forecast_path_all_series.csv")
    logger.info("Exported uncertainties to: forecast_uncertainty_all_series.csv")

    # ---------------------------------
    # 7a. Evaluate 4-Quarter Ahead Forecasts
    # ---------------------------------
    logger.info("\n" + "="*60)
    logger.info("Evaluating 4-Quarter Ahead Forecast Accuracy")
    logger.info("="*60)
    
    # Create forecast evaluation plot
    eval_stats = create_forecast_evaluation_plot(
        model, fitted_results, data, target_series,
        start_date=None,  # Will start from 25% into data
        end_date=None     # Will end 4 quarters before last observation
    )
    
    # Display evaluation statistics
    logger.info(f"\n4-Quarter Ahead Forecast Performance:")
    logger.info(f"  RMSE: {eval_stats['rmse']:,.0f}")
    logger.info(f"  MAE: {eval_stats['mae']:,.0f}")
    logger.info(f"  MAPE: {eval_stats['mape']:.1f}%")
    logger.info(f"  Number of forecasts evaluated: {len(eval_stats['forecasts'])}")
    
    # Export evaluation results
    eval_df = pd.DataFrame({
        'forecast_date': eval_stats['forecast_dates'],
        'forecast': eval_stats['forecasts'],
        'actual': eval_stats['actuals'],
        'error': eval_stats['errors']
    })
    eval_df.to_csv('forecast_evaluation_4q.csv', index=False)
    logger.info("Exported forecast evaluation to: forecast_evaluation_4q.csv")
    
    # ---------------------------------
    # 7b. Trend and Forecast Analysis
    # ---------------------------------
    logger.info("\n" + "="*60)
    logger.info("Analyzing Trend and Forecast Performance")
    logger.info("="*60)
    
    # Create trend analysis plot
    trend_analysis = create_trend_and_forecast_comparison_plot(
        model, fitted_results, data, target_series,
        eval_stats=eval_stats  # Pass the evaluation stats from earlier
    )
    
    # Display trend statistics
    logger.info("\nTrend Analysis:")
    logger.info(f"  Average level: {trend_analysis['level'].mean():,.0f}")
    logger.info(f"  Average growth rate (trend): {trend_analysis['trend'].mean():.1f} per quarter")
    logger.info(f"  Trend volatility (std of growth): {trend_analysis['trend'].std():.1f}")
    
    # Analyze forecast errors by trend regime
    if eval_stats is not None:
        level_at_forecast = trend_analysis['level'].loc[eval_stats['forecast_dates']]
        errors = eval_stats['errors']
        
        # Correlation between level and forecast error
        correlation = np.corrcoef(level_at_forecast.values, errors)[0, 1]
        logger.info(f"  Correlation(level, forecast error): {correlation:.3f}")   

    # ---------------------------------
    # 7c. Find Forecast Error Correlates
    # ---------------------------------
    logger.info("\n" + "="*60)
    logger.info("Finding Series that Correlate with Forecast Errors")
    logger.info("="*60)
    
    # Find correlating series
    correlate_results = find_forecast_error_correlates(
        model=model,
        data=data,  # Full Z1 dataset
        eval_stats=eval_stats,
        target_series=target_series,
        max_lag=8,
        top_n=40,
        min_obs=200
    )
    
    if not correlate_results.empty:
        # Save results
        correlate_results.to_csv('forecast_error_correlates.csv', index=False)
        logger.info("\nSaved correlation results to: forecast_error_correlates.csv")
        
        # Create visualization
        plot_error_correlates(
            data=data,
            eval_stats=eval_stats,
            correlate_results=correlate_results,
            target_series=target_series,
            n_plots=6
        )
        
        # Display some insights
        logger.info("\nKey insights from correlation analysis:")
        
        # Check for leading indicators
        leading = correlate_results[correlate_results['lag'] > 0]
        if not leading.empty:
            logger.info(f"\nLeading indicators (predict errors {leading.iloc[0]['lag']}Q ahead):")
            for _, row in leading.head(3).iterrows():
                logger.info(f"  {row['series']}: r={row['correlation']:.3f}")
        
        # Check for contemporaneous
        contemp = correlate_results[correlate_results['lag'] == 0]
        if not contemp.empty:
            logger.info(f"\nContemporaneous indicators:")
            for _, row in contemp.head(3).iterrows():
                logger.info(f"  {row['series']}: r={row['correlation']:.3f}")
                
    # ---------------------------------
    # 7d. Implement Forecast Improvements
    # ---------------------------------
    logger.info("\n" + "="*60)
    logger.info("Implementing Forecast Improvements")
    logger.info("="*60)
    
    # Step 1: Corporate Monitor
    logger.info("\nStep 1: Creating corporate indicator monitor...")
    corporate_series = create_corporate_monitor(data, eval_stats, target_series)
    
    # Step 2: Money Market Contrarian
    logger.info("\nStep 2: Creating money market contrarian signals...")
    mm_signal = create_money_market_contrarian_signal(data, eval_stats)
    
    # Step 3: Risk Sentiment
    logger.info("\nStep 3: Building risk sentiment composite...")
    risk_sentiment = create_risk_sentiment_indicator(data, correlate_results)
    
    # Step 4: Adjusted Forecast
    logger.info("\nStep 4: Creating adjusted forecast...")
    adjusted_results = create_adjusted_forecast(
        model, fitted_results, data, target_series, 
        correlate_results, eval_stats
    )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("FORECAST ENHANCEMENT SUMMARY")
    logger.info("="*60)
    
    logger.info(f"\nCurrent Signals:")
    for signal_name, signal_value in adjusted_results['signals'].items():
        direction = "Positive" if signal_value > 0 else "Negative"
        logger.info(f"  {signal_name}: {signal_value:.2f} ({direction})")
    
    logger.info(f"\nForecast Adjustment: {adjusted_results['adjustment_factor']*100:.1f}%")
    
    logger.info(f"\nBase vs Adjusted Forecast (4Q ahead):")
    base_val = adjusted_results['base_forecast'][target_series].iloc[3]
    adj_val = adjusted_results['adjusted_forecast'][target_series].iloc[3]
    logger.info(f"  Base: {base_val:,.0f}")
    logger.info(f"  Adjusted: {adj_val:,.0f}")
    logger.info(f"  Difference: {adj_val - base_val:,.0f}")
    
    # Save adjusted forecast
    adjusted_results['adjusted_forecast'].to_csv('adjusted_forecast.csv')
    logger.info("\nSaved adjusted forecast to: adjusted_forecast.csv")   
    
    # ---------------------------------
    # 7e. Quality Risk Indicators
    # ---------------------------------
    logger.info("\n" + "="*60)
    logger.info("Creating Quality-Based Risk Indicators")
    logger.info("="*60)
    
    # First, get more correlation results (increase top_n)
    extended_correlate_results = find_forecast_error_correlates(
        model=model,
        data=data,
        eval_stats=eval_stats,
        target_series=target_series,
        max_lag=8,
        top_n=100,  # Get more candidates
        min_obs=200
    )
    
    # Select quality indicators
    selected_indicators = select_quality_risk_indicators(
        data=data,
        correlate_results=extended_correlate_results,
        min_years=30,
        min_variation_pct=10,
        max_crisis_ratio=5,
        min_abs_correlation=0.5
    )
    
    # Define build_z_panel function
    def build_z_panel(data, indicator_df):
        """Build panel of z-scores for all indicators."""
        z_mat = {}
        for _, row in indicator_df.iterrows():
            s = row['series']
            if s not in data.columns:
                continue
            lag = int(row['lag'])
            series = data[s]
            mu = series.rolling(20, min_periods=4).mean()
            sd = series.rolling(20, min_periods=4).std()
            z = pd.Series(0.0, index=series.index)
            valid_mask = sd > 1e-8
            z[valid_mask] = (series[valid_mask] - mu[valid_mask]) / sd[valid_mask]
            z = z.clip(-3, 3).shift(lag)
            z_mat[f"{s}_{lag}"] = z
        return pd.DataFrame(z_mat)    
    
    # ----------------  NEW ROLLING-RIDGE BLOCK  ----------------
    z_panel      = build_z_panel(data, extended_correlate_results)
    errors       = pd.Series(eval_stats['errors'], index=eval_stats['forecast_dates'])

    coef_df      = compute_rolling_ridge_weights(z_panel, errors, window=60)
    dynamic_risk = build_dynamic_composites(z_panel, coef_df)    
    
    # Create full-sample risk indicators
    logger.info("\nCreating full-sample risk indicators...")

    dynamic_score = build_dynamic_composites(z_panel, coef_df)

    # D.  Rolling 85 / 15 percentiles for thresholding
    thr_hi = dynamic_risk .rolling(60, min_periods=40).quantile(0.85)
    thr_lo = dynamic_risk .rolling(60, min_periods=40).quantile(0.15)
    risk_on_extreme  = dynamic_risk  >  thr_hi
    risk_off_extreme = dynamic_risk  <  thr_lo
    
    full_risk_indicators = create_full_sample_risk_indicators(data, selected_indicators)    
    
    # Persist for audit --------------------------------------------------
    Path("out").mkdir(exist_ok=True)
    pd.DataFrame({
        "net_risk"        : dynamic_risk,
        "thr_hi"          : thr_hi,
        "thr_lo"          : thr_lo,
        "risk_on_extreme" : risk_on_extreme.astype(int),
        "risk_off_extreme": risk_off_extreme.astype(int),
    }).to_csv("out/dynamic_risk_thresholds.csv")
    logger.info("Saved dynamic risk thresholds → out/dynamic_risk_thresholds.csv")       
    
    # Save results
    risk_df = pd.DataFrame({
        'date': data.index,
        'risk_on_composite': full_risk_indicators['risk_on_composite'],
        'risk_off_composite': full_risk_indicators['risk_off_composite'],
        'net_risk_score': full_risk_indicators['net_risk_score'],
        'net_risk_smooth': full_risk_indicators['net_risk_smooth'],
        'recession_probability': full_risk_indicators['recession_probability']
    })
    risk_df.to_csv('full_sample_risk_indicators.csv', index=False)
    logger.info("Saved risk indicators to: full_sample_risk_indicators.csv")
    
    # Determine current regime based on dynamic_risk
    current_net = dynamic_risk.iloc[-1]

    if current_net > 1:
        current_regime = "STRONG RISK-ON"
        forecast_bias = "Model likely to UNDER-forecast"
    elif current_net > 0:
        current_regime = "MILD RISK-ON"
        forecast_bias = "Model may slightly under-forecast"
    elif current_net > -1:
        current_regime = "MILD RISK-OFF"
        forecast_bias = "Model may slightly over-forecast"
    else:
        current_regime = "STRONG RISK-OFF"
        forecast_bias = "Model likely to OVER-forecast"
    # Prepare risk_indicators dict for recession analysis
    risk_indicators = {
        'risk_on_composite': full_risk_indicators['risk_on_composite'],
        'risk_off_composite': full_risk_indicators['risk_off_composite'],
        'net_risk_score': dynamic_risk,  # USE DYNAMIC_RISK
        'current_regime': current_regime,
        'forecast_bias': forecast_bias
    }
    
    # UPDATE FORECAST ADJUSTMENT WITH DYNAMIC RISK
    if 'adjusted_results' in locals():
        # Update adjustment with dynamic risk
        logger.info("\nUpdating forecast adjustment with dynamic risk...")
        
        # Recalculate adjustment factor
        adjustment_factor_dynamic = 0
        weights = {'corporate': 0.4, 'money_market': 0.3, 'dynamic_risk': 0.3}
        
        # Get the signals from the previous adjusted_results
        corporate_signal = adjusted_results['signals'].get('Corporate\n(7Q lead)', 0)
        mm_signal = adjusted_results['signals'].get('Money Market\n(6Q lead)', 0)
        
        if corporate_signal != 0:
            adjustment_factor_dynamic += weights['corporate'] * corporate_signal * 0.05
        if mm_signal != 0:
            adjustment_factor_dynamic += weights['money_market'] * mm_signal * 0.05
        adjustment_factor_dynamic += weights['dynamic_risk'] * current_net * 0.04
        
        logger.info(f"Original adjustment factor: {adjusted_results['adjustment_factor']*100:.1f}%")
        logger.info(f"Updated adjustment factor with dynamic risk: {adjustment_factor_dynamic*100:.1f}%")
        
        # Update the adjusted forecast
        adjusted_forecast_dynamic = adjusted_results['base_forecast'].copy()
        for i in range(len(adjusted_forecast_dynamic)):
            decay = 0.8 ** i
            adjusted_forecast_dynamic.iloc[i] = adjusted_forecast_dynamic.iloc[i] * (1 + adjustment_factor_dynamic * decay)
        
        # Save the updated forecast
        adjusted_forecast_dynamic.to_csv('adjusted_forecast_dynamic.csv')
        logger.info("Saved dynamic risk-adjusted forecast to: adjusted_forecast_dynamic.csv")

    
    # ---------------------------------
    # 7f. Recession Analysis
    # ---------------------------------
    # Add recession analysis
    
    # Create a compatible structure for recession analysis
    risk_indicators_for_recession = {
        'risk_on_composite': risk_indicators['risk_on_composite'],
        'risk_off_composite': risk_indicators['risk_off_composite'],
        'net_risk_score': risk_indicators['net_risk_score'],
        'current_regime': 'Unknown',  # Will be determined by the analysis
        'forecast_bias': 'To be determined'
    }
    
    recession_results = add_recession_analysis_to_main(
        risk_indicators_for_recession, data, eval_stats
    )                

    # ---------------------------------
    # 8. Export Results
    # ---------------------------------
    logger.info("\n" + "="*60)
    logger.info("Exporting Results")
    logger.info("="*60)

    # Export filtered and smoothed series separately
    filtered_output_file = f'kalman_filtered_series_{target_series}.csv'
    smoothed_output_file = f'kalman_smoothed_series_{target_series}.csv'

    filtered_df = filtered_series['filtered'][series_list]
    smoothed_df = filtered_series['smoothed'][series_list]

    filtered_df.to_csv(filtered_output_file)
    smoothed_df.to_csv(smoothed_output_file)

    logger.info(f"Exported filtered series to: {filtered_output_file}")
    logger.info(f"Exported smoothed series to: {smoothed_output_file}")

    # Export validation results
    if not validation.empty:
        validation.to_csv('kalman_validation_results.csv', index=False)
        logger.info("Exported validation results to: kalman_validation_results.csv")

    # Export parameter estimates
    param_df = pd.DataFrame({
        'parameter': model.param_names,
        'estimate': fitted_results.params,
        'variance': final_variances,
        'std_error': np.sqrt(np.diag(fitted_results.cov_params())) if hasattr(fitted_results, 'cov_params') else np.nan
    })
    param_df.to_csv('kalman_parameters.csv', index=False)
    logger.info("Exported parameter estimates to: kalman_parameters.csv")

    # Export diagnostics
    diagnostics = {
        'log_likelihood': fitted_results.llf,
        'aic': fitted_results.aic,
        'n_observations': len(data),
        'burn_in_periods': burn_in,
        'normalization_scale': model.scale_factor,
        'transformation': model.transformation,
        'mean_validation_error': validation['relative_error'].mean() if not validation.empty else np.nan
    }
    pd.Series(diagnostics).to_csv('kalman_diagnostics.csv')
    logger.info("Exported diagnostics to: kalman_diagnostics.csv")

    logger.info("\n" + "="*60)
    logger.info("Analysis Complete!")
    logger.info("="*60)


def create_target_series_plot(data, filtered_series, target_series, burn_in, burn_in_end):
    """Create detailed plot for target series."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Main plot with burn-in shading
    ax = axes[0]
    ax.axvspan(data.index[0], burn_in_end, alpha=0.2, color='red', label='Burn-in period')
    ax.plot(data.index, data[target_series], 'k-', alpha=0.5, label='Original', linewidth=1)
    ax.plot(data.index, filtered_series['filtered'][target_series], 'b--', alpha=0.7, label='Filtered', linewidth=1.5)
    ax.plot(data.index, filtered_series['smoothed'][target_series], 'r-', alpha=0.8, label='Smoothed', linewidth=1.5)
    ax.set_title(f'{target_series} - Kalman Filter Results')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Zoom on initialization
    ax = axes[1]
    zoom_end = min(60, len(data))
    ax.axvspan(data.index[0], burn_in_end, alpha=0.2, color='red')
    ax.plot(data.index[:zoom_end], data[target_series].iloc[:zoom_end], 'ko-', 
            alpha=0.5, label='Original', linewidth=1, markersize=3)
    ax.plot(data.index[:zoom_end], filtered_series['smoothed'][target_series].iloc[:zoom_end], 
            'rs-', alpha=0.8, label='Smoothed', linewidth=1.5, markersize=3)
    ax.set_title('Initialization Period (First 60 observations)')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Filtered residuals
    ax = axes[2]
    residuals = data[target_series] - filtered_series['filtered'][target_series]
    ax.axvspan(data.index[0], burn_in_end, alpha=0.2, color='red')
    ax.plot(data.index, residuals, 'b-', alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Residuals (Original - Filtered)')
    ax.set_ylabel('Residual')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    if burn_in < len(residuals):
        burn_stats = residuals.iloc[:burn_in]
        post_stats = residuals.iloc[burn_in:]
        stats_text = (f"Burn-in: μ={burn_stats.mean():.0f}, σ={burn_stats.std():.0f}\n"
                     f"Post: μ={post_stats.mean():.0f}, σ={post_stats.std():.0f}")
    else:
        stats_text = f"μ={residuals.mean():.0f}, σ={residuals.std():.0f}"
    
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Standardized residuals
    ax = axes[3]
    std_residuals = residuals / residuals.std()
    ax.axvspan(data.index[0], burn_in_end, alpha=0.2, color='red')
    ax.plot(data.index, std_residuals, 'g-', alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.axhline(y=2, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=-2, color='k', linestyle='--', alpha=0.3)
    ax.set_title('Standardized Residuals')
    ax.set_ylabel('Std. Residual')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'kalman_filter_{target_series}_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: kalman_filter_{target_series}_detailed.png")


def create_all_series_plot(data, filtered_series, series_list, model, burn_in, burn_in_end):
    """Create plot showing all series."""
    n_series = len(series_list)
    n_cols = 3
    n_rows = (n_series + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), sharex=True)
    axes = axes.flatten() if n_series > 1 else [axes]
    
    for idx, series in enumerate(series_list):
        ax = axes[idx]
        
        # Shade burn-in
        ax.axvspan(data.index[0], burn_in_end, alpha=0.1, color='red')
        
        # Plot series
        ax.plot(data.index, data[series], 'k-', alpha=0.4, label='Original', linewidth=0.8)
        ax.plot(data.index, filtered_series['smoothed'][series], 'r-', alpha=0.8, label='Smoothed', linewidth=1.2)
        
        ax.set_title(f'{series}', fontsize=10)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Add series type annotation
        if series in model.source_series:
            ax.text(0.02, 0.98, 'Source', transform=ax.transAxes, 
                    verticalalignment='top', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
        else:
            ax.text(0.02, 0.98, 'Computed', transform=ax.transAxes, 
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
        
        if idx == 0:
            ax.legend(fontsize=8, loc='best')
    
    # Hide empty subplots
    for idx in range(n_series, len(axes)):
        axes[idx].set_visible(False)
    
    fig.text(0.5, 0.02, 'Date', ha='center', fontsize=10)
    fig.suptitle('All Series - Kalman Filter Results', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, top=0.95)
    plt.savefig('kalman_all_series.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved plot: kalman_all_series.png")


def create_residual_analysis_plot(data, filtered_series, target_series, burn_in):
    """Create residual diagnostic plots."""
    residuals = data[target_series] - filtered_series['smoothed'][target_series]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Residual time series
    ax = axes[0, 0]
    ax.plot(residuals.index, residuals.values, 'b-', alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=residuals.index[burn_in], color='r', linestyle='--', alpha=0.5, label='End of burn-in')
    ax.set_title('Residuals Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Histogram
    ax = axes[0, 1]
    ax.hist(residuals.dropna(), bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Add normal distribution overlay
    from scipy import stats
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r-', linewidth=2, label='Normal')
    ax.set_title('Residual Distribution')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = axes[1, 0]
    stats.probplot(residuals.dropna(), dist="norm", plot=ax)
    ax.set_title('Normal Q-Q Plot')
    ax.grid(True, alpha=0.3)
    
    # ACF plot
    ax = axes[1, 1]
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals.dropna(), lags=40, ax=ax, alpha=0.05)
    ax.set_title('Residual Autocorrelation')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_residual_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved plot: kalman_residual_diagnostics.png")

def create_forecast_path_plot(data, filtered_series, path_result, target_series):
    """Create visualization of the most probable forecast path."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    
    # Plot 1: Forecast with confidence bands
    ax = axes[0]
    
    # Historical data (last 5 years)
    historical_start = -20  # 20 quarters = 5 years
    historical = data[target_series].iloc[historical_start:]
    ax.plot(historical.index, historical.values, 'k-', 
            label='Historical', linewidth=2, alpha=0.8)
    
    # Smoothed estimate (last few points for connection)
    smoothed_end = filtered_series['smoothed'][target_series].iloc[-4:]
    ax.plot(smoothed_end.index, smoothed_end.values, 'b--', 
            label='Smoothed', linewidth=1.5, alpha=0.6)
    
    # Forecast path
    path_values = path_result['path'][target_series]
    ax.plot(path_values.index, path_values.values, 'r-', 
            label='Forecast', linewidth=2.5, marker='o', markersize=6)
    
    # Confidence bands
    lower_68 = path_result['confidence_bands']['68%']['lower'][target_series]
    upper_68 = path_result['confidence_bands']['68%']['upper'][target_series]
    ax.fill_between(path_values.index, lower_68, upper_68, 
                    alpha=0.3, color='red', label='68% Confidence')
    
    lower_95 = path_result['confidence_bands']['95%']['lower'][target_series]
    upper_95 = path_result['confidence_bands']['95%']['upper'][target_series]
    ax.fill_between(path_values.index, lower_95, upper_95, 
                    alpha=0.15, color='red', label='95% Confidence')
    
    # Formatting
    ax.set_title(f'{target_series} - Most Probable Path Forecast', fontsize=14)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add vertical line at forecast start
    ax.axvline(x=data.index[-1], color='gray', linestyle=':', alpha=0.5)
    
    # Plot 2: Forecast uncertainty by horizon
    ax = axes[1]
    uncertainties = path_result['uncertainty'][target_series]
    horizons = range(1, len(uncertainties) + 1)
    
    bars = ax.bar(horizons, uncertainties.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Quarters Ahead', fontsize=12)
    ax.set_ylabel('Forecast Standard Deviation', fontsize=12)
    ax.set_title('Forecast Uncertainty by Horizon', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, uncertainties.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # Add relative uncertainty (as % of forecast)
    ax2 = ax.twinx()
    relative_uncertainty = 100 * uncertainties.values / path_values.values
    ax2.plot(horizons, relative_uncertainty, 'ro-', linewidth=2, markersize=6)
    ax2.set_ylabel('Relative Uncertainty (%)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    plt.savefig('forecast_path_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved plot: forecast_path_analysis.png")

def create_hierarchy_plot(filtered_series, series_list, model, burn_in_end):
    """Create plot comparing source and computed series."""
    source_series = [s for s in series_list if s in model.source_series]
    computed_series = [s for s in series_list if s in model.computed_series]
    
    if not (source_series and computed_series):
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Source series
    ax = axes[0]
    for i, series in enumerate(source_series[:5]):  # Limit to 5 for readability
        smoothed = filtered_series['smoothed'][series]
        ax.plot(smoothed.index, smoothed.values, label=series, linewidth=1.5, alpha=0.8)
    ax.axvline(x=burn_in_end, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Source Series (Smoothed)')
    ax.set_ylabel('Value')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Computed series
    ax = axes[1]
    for i, series in enumerate(computed_series[:5]):  # Limit to 5 for readability
        smoothed = filtered_series['smoothed'][series]
        ax.plot(smoothed.index, smoothed.values, label=series, linewidth=1.5, alpha=0.8)
    ax.axvline(x=burn_in_end, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Computed Series (Smoothed)')
    ax.set_ylabel('Value')
    ax.set_xlabel('Date')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_hierarchy.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved plot: kalman_hierarchy.png")
    
def add_recession_analysis_to_main(risk_indicators, data, eval_stats):
    """
    Add recession analysis to main function.
    """
    logger.info("\n" + "="*60)
    logger.info("Analyzing Risk Indicators vs NBER Recessions")
    logger.info("="*60)

    recession_results = plot_risk_indicators_with_recessions(
            risk_indicators, data, eval_stats
    )
    
    # Display results
    logger.info(f"\nRecession Prediction Performance:")
    logger.info(f"  Detection Rate: {recession_results['prediction_rate']*100:.1f}%")
    logger.info(f"  Average Lead Time: {recession_results['avg_lead_time_quarters']:.1f} quarters")
    logger.info(f"  False Alarms: {recession_results['false_alarms']}")
    
    # Analyze current recession risk
    current_risk = risk_indicators['net_risk_score'].iloc[-1]
    recent_trend = risk_indicators['net_risk_score'].iloc[-4:].mean()
    
    logger.info(f"\nCurrent Recession Risk Assessment:")
    logger.info(f"  Current Net Risk Score: {current_risk:.2f}")
    logger.info(f"  4Q Average: {recent_trend:.2f}")
    
    if current_risk < -1:
        logger.info("  ⚠️  WARNING: Strong risk-off signal - elevated recession risk")
    elif current_risk < -0.5:
        logger.info("  ⚠️  CAUTION: Moderate risk-off signal - monitor closely")
    else:
        logger.info("  ✓  No immediate recession signal")
    
    return recession_results    


if __name__ == "__main__":
    main()
