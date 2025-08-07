#!/usr/bin/env python3
"""
Robust driver script for Constrained UCM with better numerical handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path

# Import the robust implementation
from robust_constrained_ucm import RobustConstrainedUCM, test_robust_ucm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------
# 0. Load and prepare the data
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
    logger.info("Set quarterly frequency")

logger.info(f"Data shape: {data.shape}")
logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

# ---------------------------------
# 1. Load formulas
# ---------------------------------
with open("fof_formulas_extracted.json", 'r') as f:
    fof_data = json.load(f)

formulas = fof_data.get('formulas', {})
identities = fof_data.get('identities', [])

# ---------------------------------
# 2. Select series and check dependencies
# ---------------------------------
target_series = "FL343073045"
series_list = [target_series]

# Find all dependencies
def get_all_dependencies(series_code, formulas, available_series):
    """Get all series needed for a given series."""
    dependencies = set([series_code])
    to_process = [series_code]
    
    while to_process:
        current = to_process.pop()
        if current in formulas and formulas[current].get('derived_from'):
            for comp in formulas[current]['derived_from']:
                dep_code = comp.get('code')
                if dep_code and dep_code in available_series and dep_code not in dependencies:
                    dependencies.add(dep_code)
                    to_process.append(dep_code)
    
    return dependencies

# Get complete series list
available_series = set(data.columns)
complete_series = get_all_dependencies(target_series, formulas, available_series)
series_list = list(complete_series)

logger.info(f"Target series: {target_series}")
logger.info(f"Total series needed: {len(series_list)}")

# Identify source vs computed
computed_series = [s for s in series_list if s in formulas and formulas[s].get('derived_from')]
source_series = [s for s in series_list if s not in computed_series]

logger.info(f"Source series: {len(source_series)}")
logger.info(f"Computed series: {len(computed_series)}")

# Check data statistics
logger.info("\nData statistics for source series:")
for series in source_series[:5]:  # Show first 5
    s_data = data[series]
    logger.info(f"  {series}: mean={s_data.mean():.2f}, std={s_data.std():.2f}, "
                f"min={s_data.min():.2f}, max={s_data.max():.2f}")

# ---------------------------------
# 3. Define UCM configuration
# ---------------------------------
ucm_config = {
    "level": True,
    "trend": True,
    "seasonal": False,
    "freq_seasonal": [{"period": 4, "harmonics": 2}],
    "cycle": True,
    "irregular": True,
    "stochastic_level": False,
    "stochastic_trend": True,
    "stochastic_freq_seasonal": [True],
    "stochastic_cycle": True,
    "damped_cycle": True,
    "cycle_period_bounds": None
}

# ---------------------------------
# 4. Fit the robust UCM model
# ---------------------------------
logger.info("\n" + "="*60)
logger.info("Fitting Robust Constrained UCM")
logger.info("="*60)

# Create series mapping
series_mapping = {col: i for i, col in enumerate(data[series_list].columns)}

# Create model with data normalization
model = RobustConstrainedUCM(
    data=data[series_list],
    formulas=formulas,
    series_mapping=series_mapping,
    normalize_data=True,  # This is key for numerical stability
    **ucm_config
)

# Try multiple optimization methods
methods = ['powell', 'nm', 'lbfgs']
results = None
best_llf = -np.inf

for method in methods:
    try:
        logger.info(f"\nTrying optimization method: {method}")
        
        # Get initial log-likelihood
        initial_llf = model.loglike(model.start_params)
        logger.info(f"Initial log-likelihood: {initial_llf:.2f}")
        
        # Fit model
        current_results = model.fit(
            method=method,
            disp=True,
            maxiter=100,
            cov_type='none'  # Skip covariance for speed
        )
        
        # Check if this is the best result
        if current_results.llf > best_llf and np.isfinite(current_results.llf):
            best_llf = current_results.llf
            results = current_results
            logger.info(f"New best result: llf={best_llf:.2f}")
            
            # Check convergence
            if hasattr(current_results, 'mle_retvals'):
                converged = current_results.mle_retvals.get('converged', False)
                logger.info(f"Converged: {converged}")
        
    except Exception as e:
        logger.error(f"Method {method} failed: {e}")
        continue

if results is None:
    raise RuntimeError("All optimization methods failed!")

# ---------------------------------
# 5. Extract and analyze components
# ---------------------------------
logger.info("\n" + "="*60)
logger.info("Extracting Components")
logger.info("="*60)

# Extract smoothed components
try:
    smoothed_components = model.get_components('smoothed', results)
    
    # Check component statistics
    logger.info("\nComponent statistics for target series:")
    for comp_name in ['level', 'trend', 'cycle', 'freq_seasonal']:
        if target_series in smoothed_components[comp_name].columns:
            comp_data = smoothed_components[comp_name][target_series]
            comp_mean = comp_data.mean()
            comp_std = comp_data.std()
            comp_range = comp_data.max() - comp_data.min()
            
            logger.info(f"  {comp_name}: mean={comp_mean:.2f}, std={comp_std:.2f}, range={comp_range:.2f}")
            
            # Check if component is essentially zero
            if np.abs(comp_mean) < 1e-6 and comp_std < 1e-6:
                logger.warning(f"    WARNING: {comp_name} appears to be zero!")

except Exception as e:
    logger.error(f"Error extracting components: {e}")
    smoothed_components = None

# Extract filtered components
try:
    filtered_components = model.get_components('filtered', results)
except Exception as e:
    logger.error(f"Error extracting filtered components: {e}")
    filtered_components = None

# ---------------------------------
# 6. Validate formula constraints
# ---------------------------------
if smoothed_components is not None:
    logger.info("\n" + "="*60)
    logger.info("Validating Formula Constraints")
    logger.info("="*60)
    
    for series in computed_series[:3]:  # Check first 3 computed series
        if series in smoothed_components['level'].columns:
            # Get formula
            formula_info = formulas[series]
            formula_str = formula_info.get('formula', 'Unknown')
            
            # Calculate from components
            calc_value = 0
            for comp in ['level', 'trend', 'cycle', 'freq_seasonal']:
                calc_value += smoothed_components[comp][series].values
            
            # Compare with actual
            actual_value = data[series].values
            error = np.abs(actual_value - calc_value).mean()
            rel_error = error / np.abs(actual_value).mean() if np.abs(actual_value).mean() > 0 else np.nan
            
            logger.info(f"\n{series}: {formula_str}")
            logger.info(f"  Mean absolute error: {error:.2f}")
            logger.info(f"  Relative error: {rel_error:.4%}")

# ---------------------------------
# 7. Visualize results
# ---------------------------------
def plot_robust_decomposition(series_code, data, components, title_suffix=""):
    """Plot decomposition with robust handling of missing components."""
    
    # Check which components exist
    available_components = []
    for comp in ['level', 'trend', 'cycle', 'freq_seasonal']:
        if comp in components and series_code in components[comp].columns:
            if not np.allclose(components[comp][series_code], 0):
                available_components.append(comp)
    
    if not available_components:
        logger.warning(f"No non-zero components found for {series_code}")
        return
    
    # Create figure
    n_plots = len(available_components) + 2  # +2 for original and fitted
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)
    
    # Original data
    axes[0].plot(data.index, data[series_code], 'k-', alpha=0.7, label='Original')
    axes[0].set_ylabel('Original')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Fitted (sum of components)
    fitted = sum(components[comp][series_code] for comp in available_components)
    axes[1].plot(data.index, fitted, 'b-', alpha=0.7, label='Fitted')
    axes[1].plot(data.index, data[series_code], 'k--', alpha=0.5, label='Original')
    axes[1].set_ylabel('Fitted vs Original')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Individual components
    for i, comp in enumerate(available_components):
        ax = axes[i + 2]
        comp_data = components[comp][series_code]
        ax.plot(data.index, comp_data, label=comp.capitalize())
        ax.set_ylabel(comp.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"mean={comp_data.mean():.1f}, std={comp_data.std():.1f}"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9)
    
    axes[0].set_title(f"{series_code} Decomposition{title_suffix}")
    axes[-1].set_xlabel('Date')
    
    plt.tight_layout()
    return fig

# Plot results if components were extracted
if smoothed_components is not None:
    try:
        fig = plot_robust_decomposition(
            target_series, 
            data[series_list], 
            smoothed_components,
            " (Robust UCM)"
        )
        if fig:
            plt.savefig(f"robust_decomposition_{target_series}.png", dpi=150, bbox_inches='tight')
            logger.info(f"\nSaved plot to robust_decomposition_{target_series}.png")
            plt.show()
    except Exception as e:
        logger.error(f"Error plotting: {e}")

# ---------------------------------
# 8. Summary diagnostics
# ---------------------------------
logger.info("\n" + "="*60)
logger.info("Summary Diagnostics")
logger.info("="*60)

logger.info(f"\nFinal log-likelihood: {results.llf:.2f}")
logger.info(f"Number of parameters: {len(results.params)}")
logger.info(f"Number of observations: {len(data[series_list]) * len(series_list)}")

# Parameter summary
params_finite = np.isfinite(results.params).sum()
logger.info(f"\nFinite parameters: {params_finite}/{len(results.params)}")

if hasattr(results, 'smoothed_state'):
    state_norm = np.linalg.norm(results.smoothed_state)
    logger.info(f"State vector norm: {state_norm:.2e}")

logger.info("\n" + "="*60)
logger.info("Analysis Complete")
logger.info("="*60)
