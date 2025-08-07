#!/usr/bin/env python3
"""
Compare independent Kalman filters vs constrained hierarchical Kalman filter
Shows the value of respecting formula constraints
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents
import json
import logging
from hierarchical_kalman_filter import fit_hierarchical_kalman_filter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------
# 1. Load data and formulas
# ---------------------------------
logger.info("Loading data...")
data = pd.read_parquet(
    "/home/tesla/Z1/temp/data/z1_quarterly/z1_quarterly_data_filtered.parquet"
)

if data.index.freq is None:
    data.index = pd.date_range(start=data.index[0], periods=len(data), freq="QE")

# Load formulas
with open("fof_formulas_extracted.json", 'r') as f:
    fof_data = json.load(f)
formulas = fof_data.get('formulas', {})

# Target series and its dependencies
target_series = "FA086902005"

# ---------------------------------
# 2. Fit constrained hierarchical model
# ---------------------------------
logger.info("\n" + "="*60)
logger.info("Fitting Constrained Hierarchical Kalman Filter")
logger.info("="*60)

model, constrained_results = fit_hierarchical_kalman_filter(
    data=data,
    formulas_file="fof_formulas_extracted.json",
    series_list=[target_series],
    normalize=True,
    error_variance_ratio=0.001
)

constrained_smoothed = constrained_results['filtered_series']['smoothed']
series_list = constrained_results['series_list']

# ---------------------------------
# 3. Fit independent Kalman filters
# ---------------------------------
logger.info("\n" + "="*60)
logger.info("Fitting Independent Kalman Filters")
logger.info("="*60)

independent_smoothed = pd.DataFrame(index=data.index)
independent_filtered = pd.DataFrame(index=data.index)
independent_params = {}

# UCM configuration for individual models
ucm_config = {
    'level': 'local linear trend',
    'freq_seasonal': [{'period': 4, 'harmonics': 2}],
    'stochastic_level': False,
    'stochastic_trend': True,
    'stochastic_freq_seasonal': [True]
}

for series in series_list:
    logger.info(f"\nFitting independent model for {series}")
    try:
        # Normalize data for consistency
        series_mean = data[series].mean()
        series_std = data[series].std()
        normalized_data = (data[series] - series_mean) / series_std
        
        # Fit model
        model_indep = UnobservedComponents(normalized_data, **ucm_config)
        result_indep = model_indep.fit(method='powell', disp=False, maxiter=100)
        
        # Extract and denormalize
        smoothed = result_indep.level.smoothed + result_indep.trend.smoothed
        if hasattr(result_indep, 'freq_seasonal'):
            smoothed += result_indep.freq_seasonal.smoothed
        
        independent_smoothed[series] = smoothed * series_std + series_mean
        independent_filtered[series] = result_indep.fittedvalues * series_std + series_mean
        
        # Store parameters
        independent_params[series] = {
            'llf': result_indep.llf,
            'params': result_indep.params,
            'converged': True
        }
        
        logger.info(f"  Success! LLF: {result_indep.llf:.2f}")
        
    except Exception as e:
        logger.error(f"  Failed: {e}")
        # Use original series as fallback
        independent_smoothed[series] = data[series]
        independent_filtered[series] = data[series]
        independent_params[series] = {'converged': False}

# ---------------------------------
# 4. Check constraint violations
# ---------------------------------
logger.info("\n" + "="*60)
logger.info("Checking Constraint Violations")
logger.info("="*60)

constraint_violations = pd.DataFrame(index=data.index)

# Check each computed series
for series in series_list:
    if series in formulas and formulas[series].get('derived_from'):
        # Calculate what the series should be according to formula
        formula_value = pd.Series(0, index=data.index)
        
        for component in formulas[series]['derived_from']:
            comp_series = component['code']
            operator = component.get('operator', '+')
            
            if comp_series in independent_smoothed.columns:
                if operator == '+':
                    formula_value += independent_smoothed[comp_series]
                else:
                    formula_value -= independent_smoothed[comp_series]
        
        # Violation = independent estimate - formula value
        violation = independent_smoothed[series] - formula_value
        constraint_violations[series] = violation
        
        # Report statistics
        mae = np.abs(violation).mean()
        rmse = np.sqrt((violation**2).mean())
        rel_error = mae / np.abs(data[series]).mean() * 100
        
        logger.info(f"\n{series}:")
        logger.info(f"  Formula: {formulas[series].get('formula', 'Unknown')}")
        logger.info(f"  Mean Absolute Violation: {mae:,.0f}")
        logger.info(f"  RMSE Violation: {rmse:,.0f}")
        logger.info(f"  Relative Error: {rel_error:.1f}%")

# ---------------------------------
# 5. Create comparison plots
# ---------------------------------
logger.info("\n" + "="*60)
logger.info("Creating Comparison Plots")
logger.info("="*60)

# Get both filtered and smoothed from constrained model
constrained_filtered = constrained_results['filtered_series']['filtered']
constrained_smoothed = constrained_results['filtered_series']['smoothed']

# Plot 1: Filtered vs Filtered comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Filtered estimates comparison
ax = axes[0]
ax.plot(data.index, data[target_series], 'k-', alpha=0.4, label='Original', linewidth=0.8)
ax.plot(data.index, independent_filtered[target_series], 'b-', alpha=0.8, label='Independent Filtered', linewidth=1.5)
ax.plot(data.index, constrained_filtered[target_series], 'r--', alpha=0.8, label='Constrained Filtered', linewidth=1.5)
ax.set_title(f'{target_series}: Filtered Estimates - Independent vs Constrained')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, alpha=0.3)

# Smoothed estimates comparison
ax = axes[1]
ax.plot(data.index, data[target_series], 'k-', alpha=0.4, label='Original', linewidth=0.8)
ax.plot(data.index, independent_smoothed[target_series], 'b-', alpha=0.8, label='Independent Smoothed', linewidth=1.5)
ax.plot(data.index, constrained_smoothed[target_series], 'r--', alpha=0.8, label='Constrained Smoothed', linewidth=1.5)
ax.set_title(f'{target_series}: Smoothed Estimates - Independent vs Constrained')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, alpha=0.3)

# Difference between filtered estimates
ax = axes[2]
filtered_diff = independent_filtered[target_series] - constrained_filtered[target_series]
smoothed_diff = independent_smoothed[target_series] - constrained_smoothed[target_series]
ax.plot(data.index, filtered_diff, 'g-', alpha=0.8, label='Filtered Diff', linewidth=1.2)
ax.plot(data.index, smoothed_diff, 'm-', alpha=0.8, label='Smoothed Diff', linewidth=1.2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.set_title('Differences (Independent - Constrained)')
ax.set_ylabel('Difference')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)

# Add statistics
stats_text = f"Filtered - Mean: {filtered_diff.mean():,.0f}, Max Abs: {np.abs(filtered_diff).max():,.0f}\n"
stats_text += f"Smoothed - Mean: {smoothed_diff.mean():,.0f}, Max Abs: {np.abs(smoothed_diff).max():,.0f}"
ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
        verticalalignment='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('kalman_filtered_vs_filtered_comparison.png', dpi=150, bbox_inches='tight')
logger.info("Saved plot: kalman_filtered_vs_filtered_comparison.png")

# Plot 2: All series comparison in grid (Filtered estimates)
n_series = len(series_list)
n_cols = 3
n_rows = (n_series + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten() if n_series > 1 else [axes]

for idx, series in enumerate(series_list):
    ax = axes[idx]
    
    # Plot filtered estimates
    ax.plot(data.index, independent_filtered[series], 'b-', alpha=0.8, label='Indep. Filtered', linewidth=1.2)
    ax.plot(data.index, constrained_filtered[series], 'r--', alpha=0.8, label='Const. Filtered', linewidth=1.2)
    
    ax.set_title(f'{series} (Filtered)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    
    # Add difference indicator
    diff = independent_filtered[series] - constrained_filtered[series]
    rel_diff = (np.abs(diff).mean() / np.abs(data[series]).mean()) * 100
    ax.text(0.02, 0.98, f'Diff: {rel_diff:.1f}%', transform=ax.transAxes, 
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Show legend only on first plot
    if idx == 0:
        ax.legend(fontsize=8, loc='best')

# Hide empty subplots
for idx in range(n_series, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('kalman_all_series_filtered_comparison.png', dpi=150, bbox_inches='tight')
logger.info("Saved plot: kalman_all_series_filtered_comparison.png")

# Plot 3: Filtered vs Smoothed differences for both approaches
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Independent: Filtered vs Smoothed
ax = axes[0]
for series in series_list[:3]:  # Limit to 3 series for clarity
    diff = independent_smoothed[series] - independent_filtered[series]
    ax.plot(data.index, diff, label=series, linewidth=1.2, alpha=0.8)
ax.set_title('Independent Kalman: Smoothed - Filtered')
ax.set_ylabel('Difference')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Constrained: Filtered vs Smoothed  
ax = axes[1]
for series in series_list[:3]:  # Same 3 series
    diff = constrained_smoothed[series] - constrained_filtered[series]
    ax.plot(data.index, diff, label=series, linewidth=1.2, alpha=0.8)
ax.set_title('Constrained Kalman: Smoothed - Filtered')
ax.set_ylabel('Difference')
ax.set_xlabel('Date')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kalman_filtered_smoothed_differences.png', dpi=150, bbox_inches='tight')
logger.info("Saved plot: kalman_filtered_smoothed_differences.png")

# Plot 3: Constraint violation analysis
computed_series = [s for s in series_list if s in constraint_violations.columns]
if computed_series:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for series in computed_series:
        violation_pct = (constraint_violations[series] / data[series]) * 100
        ax.plot(data.index, violation_pct, label=series, linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Formula Constraint Violations (% of Series Value) - Independent Filters')
    ax.set_xlabel('Date')
    ax.set_ylabel('Violation (%)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_constraint_violations.png', dpi=150, bbox_inches='tight')
    logger.info("Saved plot: kalman_constraint_violations.png")

# ---------------------------------
# 6. Summary statistics
# ---------------------------------
logger.info("\n" + "="*60)
logger.info("Summary: Value of Constrained Filtering")
logger.info("="*60)

# Compare log-likelihoods
constrained_llf = constrained_results['fitted_results'].llf
independent_llf_sum = sum(p['llf'] for p in independent_params.values() if p.get('converged', False))

logger.info(f"\nLog-likelihood comparison:")
logger.info(f"  Constrained model: {constrained_llf:.2f}")
logger.info(f"  Sum of independent models: {independent_llf_sum:.2f}")

# Overall constraint violations
logger.info(f"\nConstraint violations (independent filtering):")
for series in computed_series:
    violation = constraint_violations[series]
    rel_viol = np.abs(violation).mean() / np.abs(data[series]).mean() * 100
    logger.info(f"  {series}: {rel_viol:.1f}% average relative violation")

# Difference statistics
logger.info(f"\nDifferences between approaches:")
for series in series_list:
    diff = independent_smoothed[series] - constrained_smoothed[series]
    rel_diff = np.abs(diff).mean() / np.abs(data[series]).mean() * 100
    logger.info(f"  {series}: {rel_diff:.1f}% average relative difference")

plt.show()

logger.info("\n" + "="*60)
logger.info("Analysis Complete!")
logger.info("="*60)
