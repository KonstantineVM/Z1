#!/usr/bin/env python3
"""
Plot smoothed vs filtered series for all series from Kalman filter results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_results(target_series="FA086902005"):
    """Load the results from the Kalman filter analysis."""
    try:
        # Load filtered and smoothed series separately
        filtered_series = pd.read_csv(f'kalman_filtered_series_{target_series}.csv', 
                                     index_col=0, parse_dates=True)
        smoothed_series = pd.read_csv(f'kalman_smoothed_series_{target_series}.csv', 
                                     index_col=0, parse_dates=True)
        
        # Load original data for comparison
        original_data = pd.read_parquet(
            "/home/tesla/Z1/temp/data/z1_quarterly/z1_quarterly_data_filtered.parquet"
        )
        
        # Load formulas to identify source vs computed series
        with open('fof_formulas_extracted.json', 'r') as f:
            formulas = json.load(f).get('formulas', {})
        
        # Determine series types
        series_names = list(filtered_series.columns)
        computed_series = [s for s in series_names if s in formulas and 'derived_from' in formulas[s]]
        source_series = [s for s in series_names if s not in computed_series]
        
        logger.info(f"Loaded {len(series_names)} series")
        logger.info(f"Source series: {len(source_series)}")
        logger.info(f"Computed series: {len(computed_series)}")
        
        return filtered_series, smoothed_series, original_data, source_series, computed_series
        
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return None, None, None, None


def create_smoothed_vs_filtered_plot(smoothed_data, filtered_data, original_data, 
                                    series_list, series_type="All", burn_in=20):
    """
    Create comparison plots of smoothed vs filtered series.
    
    Note: In the current implementation, smoothed and filtered might be the same
    if the model only exports one type.
    """
    n_series = len(series_list)
    n_cols = 2
    n_rows = (n_series + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, series in enumerate(series_list):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Get the data
        if series in smoothed_data.columns:
            smoothed = smoothed_data[series]
            filtered = filtered_data[series] if series in filtered_data.columns else smoothed
            
            # Plot original data (faded)
            if series in original_data.columns:
                ax.plot(original_data.index, original_data[series], 'k-', 
                       alpha=0.2, linewidth=0.8, label='Original')
            
            # Plot filtered and smoothed
            ax.plot(filtered.index, filtered.values, 'b--', 
                   alpha=0.7, linewidth=1.5, label='Filtered')
            ax.plot(smoothed.index, smoothed.values, 'r-', 
                   alpha=0.8, linewidth=2, label='Smoothed')
            
            # Mark burn-in period
            if burn_in > 0 and len(smoothed) > burn_in:
                ax.axvline(x=smoothed.index[burn_in], color='gray', 
                          linestyle=':', alpha=0.5, label='End of burn-in')
            
            # Calculate difference statistics
            diff = (smoothed - filtered).dropna()
            rmse = np.sqrt((diff**2).mean())
            mae = diff.abs().mean()
            
            # Format title with statistics
            ax.set_title(f'{series}\nRMSE: {rmse:.2e}, MAE: {mae:.2e}', fontsize=10)
            ax.set_xlabel('Date', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # Legend only on first plot
            if idx == 0:
                ax.legend(fontsize=8, loc='best')
            
            # Add zoom inset for first 60 observations
            if idx < 4:  # Only for first 4 series to save space
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                axins = inset_axes(ax, width="40%", height="40%", loc='lower right')
                
                # Plot first 60 observations
                zoom_end = min(60, len(smoothed))
                axins.plot(filtered.index[:zoom_end], filtered.values[:zoom_end], 
                          'b--', alpha=0.7, linewidth=1)
                axins.plot(smoothed.index[:zoom_end], smoothed.values[:zoom_end], 
                          'r-', alpha=0.8, linewidth=1.2)
                axins.axvline(x=smoothed.index[burn_in], color='gray', 
                             linestyle=':', alpha=0.5)
                axins.set_xlim(smoothed.index[0], smoothed.index[zoom_end-1])
                axins.tick_params(labelsize=6)
                axins.grid(True, alpha=0.3)
                axins.set_title('First 60 obs', fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_series, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f'{series_type} Series: Smoothed vs Filtered Comparison', fontsize=14)
    plt.tight_layout()
    
    filename = f'kalman_smoothed_vs_filtered_{series_type.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: {filename}")
    
    return filename


def create_difference_analysis_plot(smoothed_data, filtered_data, series_list, series_type="All"):
    """Create plots analyzing the differences between smoothed and filtered series."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Calculate differences for all series
    all_differences = []
    all_relative_differences = []
    series_stats = []
    
    for series in series_list:
        if series in smoothed_data.columns and series in filtered_data.columns:
            diff = (smoothed_data[series] - filtered_data[series]).dropna()
            rel_diff = (diff / filtered_data[series].abs()).replace([np.inf, -np.inf], np.nan).dropna()
            
            all_differences.extend(diff.values)
            all_relative_differences.extend(rel_diff.values)
            
            series_stats.append({
                'series': series,
                'mean_diff': diff.mean(),
                'std_diff': diff.std(),
                'max_abs_diff': diff.abs().max(),
                'mean_rel_diff': rel_diff.mean(),
                'max_rel_diff': rel_diff.abs().max()
            })
    
    # Plot 1: Histogram of differences
    ax = axes[0, 0]
    ax.hist(all_differences, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title('Distribution of Smoothed - Filtered Differences')
    ax.set_xlabel('Difference')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.text(0.05, 0.95, f'Mean: {np.mean(all_differences):.2e}\nStd: {np.std(all_differences):.2e}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Histogram of relative differences
    ax = axes[0, 1]
    rel_diff_clipped = np.clip(all_relative_differences, -1, 1)  # Clip extreme values for visualization
    ax.hist(rel_diff_clipped, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    ax.set_title('Distribution of Relative Differences (clipped to [-1, 1])')
    ax.set_xlabel('Relative Difference')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Series-wise maximum absolute differences
    ax = axes[1, 0]
    stats_df = pd.DataFrame(series_stats)
    series_names = stats_df['series'].values
    max_diffs = stats_df['max_abs_diff'].values
    
    bars = ax.bar(range(len(series_names)), max_diffs, color='coral')
    ax.set_title('Maximum Absolute Difference by Series')
    ax.set_xlabel('Series')
    ax.set_ylabel('Max |Smoothed - Filtered|')
    ax.set_xticks(range(len(series_names)))
    ax.set_xticklabels(series_names, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Color bars by magnitude
    for i, bar in enumerate(bars):
        if max_diffs[i] > np.median(max_diffs) * 10:
            bar.set_color('red')
        elif max_diffs[i] > np.median(max_diffs) * 5:
            bar.set_color('orange')
    
    # Plot 4: Time series of average absolute difference
    ax = axes[1, 1]
    
    # Calculate average absolute difference over time
    time_index = smoothed_data.index
    avg_abs_diff_over_time = pd.Series(index=time_index, dtype=float)
    
    for t in range(len(time_index)):
        diffs_at_t = []
        for series in series_list:
            if series in smoothed_data.columns and series in filtered_data.columns:
                diff = abs(smoothed_data[series].iloc[t] - filtered_data[series].iloc[t])
                if not np.isnan(diff):
                    diffs_at_t.append(diff)
        avg_abs_diff_over_time.iloc[t] = np.mean(diffs_at_t) if diffs_at_t else np.nan
    
    ax.plot(avg_abs_diff_over_time.index, avg_abs_diff_over_time.values, 'purple', linewidth=1.5)
    ax.set_title('Average Absolute Difference Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Avg |Smoothed - Filtered|')
    ax.grid(True, alpha=0.3)
    
    # Mark burn-in period
    if len(time_index) > 20:
        ax.axvline(x=time_index[20], color='red', linestyle='--', alpha=0.5, label='End of burn-in')
        ax.legend()
    
    plt.suptitle(f'{series_type} Series: Smoothed vs Filtered Difference Analysis', fontsize=14)
    plt.tight_layout()
    
    filename = f'kalman_difference_analysis_{series_type.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: {filename}")
    
    # Save statistics to CSV
    stats_filename = f'kalman_difference_stats_{series_type.lower().replace(" ", "_")}.csv'
    stats_df.to_csv(stats_filename, index=False)
    logger.info(f"Saved statistics: {stats_filename}")
    
    return filename, stats_filename


def main():
    """Main function to create all plots."""
    logger.info("Starting smoothed vs filtered analysis...")
    
    # Load results
    filtered_series, smoothed_series, original_data, source_series, computed_series = load_results()

    if filtered_series is None or smoothed_series is None:
        logger.error("Failed to load results. Exiting.")
        return

    
    # Align data
    common_index = filtered_series.index
    if original_data is not None:
        original_data = original_data.reindex(common_index)
    
    # Create plots for all series
    logger.info("\nCreating smoothed vs filtered plots for all series...")
    all_series = source_series + computed_series
    create_smoothed_vs_filtered_plot(smoothed_series, filtered_series, original_data,
                                   all_series, "All", burn_in=20)
    
    # Create separate plots for source and computed series
    if source_series:
        logger.info("\nCreating plots for source series only...")
        create_smoothed_vs_filtered_plot(smoothed_series, filtered_series, original_data,
                                       source_series, "Source", burn_in=20)
    
    if computed_series:
        logger.info("\nCreating plots for computed series only...")
        create_smoothed_vs_filtered_plot(smoothed_series, filtered_series, original_data,
                                       computed_series, "Computed", burn_in=20)
    
    # Create difference analysis
    logger.info("\nCreating difference analysis plots...")
    create_difference_analysis_plot(smoothed_series, filtered_series, all_series, "All")
    
    if source_series:
        create_difference_analysis_plot(smoothed_series, filtered_series, source_series, "Source")
    
    if computed_series:
        create_difference_analysis_plot(smoothed_series, filtered_series, computed_series, "Computed")
    
    logger.info("\nAnalysis complete!")
    
    # Print summary statistics
    logger.info("\nSummary Statistics:")
    for series_type, series_list in [("Source", source_series), ("Computed", computed_series)]:
        if series_list:
            logger.info(f"\n{series_type} Series:")
            total_diff = 0
            for series in series_list[:5]:  # Show first 5
                if series in smoothed_series.columns:
                    diff = (smoothed_series[series] - filtered_series[series]).abs().mean()
                    logger.info(f"  {series}: Mean absolute difference = {diff:.2e}")
                    total_diff += diff
            
            if len(series_list) > 5:
                logger.info(f"  ... and {len(series_list) - 5} more series")


if __name__ == "__main__":
    main()
