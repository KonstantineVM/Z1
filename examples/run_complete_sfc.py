# ==============================================================================
# FILE: examples/run_complete_sfc.py
# ==============================================================================
"""
Complete example running SFC Kalman filter with all data.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import yaml
from typing import Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/sfc_config.yaml',
               run_mode: Optional[str] = None) -> Dict:
    """Load and validate configuration."""
    
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        # Return minimal default config
        return {
            'sfc': {
                'enforcement': {
                    'enforce_formulas': True,
                    'enforce_fwtw': True,
                    'enforce_market_clearing': True
                },
                'constraints': {
                    'fwtw_weight': 0.3,
                    'market_clearing_weight': 0.1
                },
                'kalman': {
                    'error_variance_ratio': 0.01,
                    'normalize_data': True,
                    'transformation': 'square'
                },
                'output': {
                    'save_filtered': True,
                    'save_smoothed': True,
                    'output_dir': './output',
                    'denormalize': True
                }
            }
        }
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config: {e}")
    
    # Load run mode if specified
    if run_mode:
        run_mode_file = Path('config/run_modes.yaml')
        if run_mode_file.exists():
            with open(run_mode_file, 'r') as f:
                run_modes = yaml.safe_load(f)
                
            if run_mode in run_modes.get('run_modes', {}):
                mode_config = run_modes['run_modes'][run_mode]
                # Merge configs
                config = deep_merge(config, mode_config)
    
    return config


def deep_merge(base: Dict, update: Dict) -> Dict:
    """Recursively merge dictionaries."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def main(run_mode: Optional[str] = None):
    """Run complete SFC Kalman filter."""
    
    logger.info("="*60)
    logger.info("STOCK-FLOW CONSISTENT KALMAN FILTER")
    logger.info("="*60)
    
    # Load configuration
    config = load_config(run_mode=run_mode)
    
    # Load Z.1 data
    logger.info("Loading Z.1 data...")
    try:
        from src.data import CachedFedDataLoader
        fed_loader = CachedFedDataLoader()
        z1_data = fed_loader.load_single_source('Z1')
        logger.info(f"Loaded {len(z1_data.columns)} Z.1 series")
    except Exception as e:
        logger.error(f"Error loading Z.1 data: {e}")
        # Use sample data for demonstration
        logger.info("Using sample data for demonstration")
        dates = pd.date_range('2020-01-01', periods=20, freq='Q')
        z1_data = pd.DataFrame(
            np.random.randn(20, 10) * 100 + 1000,
            index=dates,
            columns=[f'FA{i:02d}30641005' for i in range(10, 20)]
        )
    
    # Load FWTW data
    logger.info("Loading FWTW data...")
    try:
        from src.network import FWTWDataLoader
        fwtw_loader = FWTWDataLoader()
        fwtw_data = fwtw_loader.load_fwtw_data()
        logger.info(f"Loaded {len(fwtw_data)} FWTW positions")
    except Exception as e:
        logger.error(f"Error loading FWTW data: {e}")
        fwtw_data = None
    
    # Load formulas
    logger.info("Loading formulas...")
    formula_path = Path('data/fof_formulas_extracted.json')
    if formula_path.exists():
        with open(formula_path, 'r') as f:
            formulas = json.load(f)
        logger.info(f"Loaded {len(formulas)} formulas")
    else:
        logger.warning("Formula file not found, using empty formulas")
        formulas = {}
    
    # Setup SFC configuration
    sfc_config = {
        'fwtw_weight': config['sfc']['constraints'].get('fwtw_weight', 0.3),
        'market_clearing_weight': config['sfc']['constraints'].get('market_clearing_weight', 0.1),
        'enforce_fwtw': config['sfc']['enforcement'].get('enforce_fwtw', True),
        'enforce_market_clearing': config['sfc']['enforcement'].get('enforce_market_clearing', True),
        'include_all': True  # Include all instruments and sectors
    }
    
    # Initialize unified filter
    logger.info("Initializing unified SFC Kalman filter...")
    
    from src.models.unified_sfc_kalman import UnifiedSFCKalmanFilter
    
    unified_filter = UnifiedSFCKalmanFilter(
        data=z1_data,
        formulas=formulas,
        fwtw_data=fwtw_data,
        sfc_config=sfc_config,
        # Kalman parameters from config
        error_variance_ratio=config['sfc']['kalman'].get('error_variance_ratio', 0.01),
        normalize_data=config['sfc']['kalman'].get('normalize_data', True),
        transformation=config['sfc']['kalman'].get('transformation', 'square')
    )
    
    # Run filter
    logger.info("Running Kalman filter with SFC constraints...")
    results = unified_filter.filter()
    
    # Get filtered series
    series_dict = unified_filter.get_filtered_series(results)
    filtered = series_dict['filtered']
    smoothed = series_dict['smoothed']
    
    # Denormalize if requested
    if config['sfc']['output'].get('denormalize', True) and unified_filter.normalize_data:
        filtered = filtered * unified_filter.scale_factor
        smoothed = smoothed * unified_filter.scale_factor
    
    # Save results
    output_dir = Path(config['sfc']['output'].get('output_dir', './output'))
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if config['sfc']['output'].get('save_filtered', True):
        output_file = output_dir / 'sfc_filtered.csv'
        filtered.to_csv(output_file)
        logger.info(f"Saved filtered series to {output_file}")
    
    if config['sfc']['output'].get('save_smoothed', True):
        output_file = output_dir / 'sfc_smoothed.csv'
        smoothed.to_csv(output_file)
        logger.info(f"Saved smoothed series to {output_file}")
    
    # Print summary
    logger.info("="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Filtered {len(filtered.columns)} series")
    logger.info(f"Time period: {filtered.index[0]} to {filtered.index[-1]}")
    
    if hasattr(unified_filter, 'network_discovery'):
        discovery = unified_filter.network_discovery
        logger.info(f"Network discovery:")
        logger.info(f"  Total required series: {len(discovery['all_series'])}")
        logger.info(f"  Available: {len(set(z1_data.columns) & discovery['all_series'])}")
    
    return results, filtered, smoothed


if __name__ == "__main__":
    import sys
    
    # Get run mode from command line
    run_mode = sys.argv[1] if len(sys.argv) > 1 else None
    
    if run_mode:
        logger.info(f"Running in {run_mode} mode")
    
    results, filtered, smoothed = main(run_mode)
    
    logger.info("SFC Kalman filter complete!")
