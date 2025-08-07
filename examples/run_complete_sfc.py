# ==============================================================================
# FILE: examples/run_complete_sfc.py
# ==============================================================================
"""
Complete example running SFC Kalman filter with full stock-flow consistency.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import yaml
from typing import Dict, Optional
import sys
import warnings
warnings.filterwarnings('ignore')

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
        return {
            'sfc': {
                'enforcement': {
                    'enforce_sfc': True,
                    'enforce_formulas': True,
                    'enforce_fwtw': True,
                    'enforce_market_clearing': True
                },
                'constraints': {
                    'fwtw_weight': 0.3,
                    'market_clearing_weight': 0.1,
                    'stock_flow_weight': 0.5,
                    'bilateral_weight': 0.3
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
                },
                'performance': {
                    'max_series': 500,  # Limit for memory management
                    'prioritize_pairs': True
                }
            }
        }
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load run mode specific config if provided
    if run_mode:
        run_mode_file = Path('config/run_modes.yaml')
        if run_mode_file.exists():
            with open(run_mode_file, 'r') as f:
                run_modes = yaml.safe_load(f)
                if run_mode in run_modes.get('run_modes', {}):
                    mode_config = run_modes['run_modes'][run_mode]
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


def select_series_for_sfc(data: pd.DataFrame, max_series: int = 500) -> pd.DataFrame:
    """
    Select series prioritizing stock-flow pairs.
    
    Parameters:
    -----------
    data : pd.DataFrame
        All available series
    max_series : int
        Maximum number of series to keep
        
    Returns:
    --------
    pd.DataFrame with selected series
    """
    if len(data.columns) <= max_series:
        return data
    
    logger.info(f"Selecting {max_series} series from {len(data.columns)} available")
    
    selected = []
    
    # Priority 1: Stock-flow pairs
    stock_flow_pairs = []
    for col in data.columns:
        if col.startswith('FA'):
            flow = 'FU' + col[2:]
            if flow in data.columns:
                stock_flow_pairs.append((col, flow))
        elif col.startswith('FL'):
            flow = 'FR' + col[2:]
            if flow in data.columns:
                stock_flow_pairs.append((col, flow))
    
    # Add pairs first
    for stock, flow in stock_flow_pairs:
        if stock not in selected:
            selected.append(stock)
        if flow not in selected:
            selected.append(flow)
        if len(selected) >= max_series:
            break
    
    logger.info(f"Selected {len(stock_flow_pairs)} stock-flow pairs")
    
    # Priority 2: Important sectors
    if len(selected) < max_series:
        important_sectors = ['10', '15', '26', '31', '40', '70', '89', '90']
        for col in data.columns:
            if col not in selected and len(col) >= 4:
                sector = col[2:4]
                if sector in important_sectors:
                    selected.append(col)
                    if len(selected) >= max_series:
                        break
    
    # Priority 3: Fill remaining
    if len(selected) < max_series:
        for col in data.columns:
            if col not in selected:
                selected.append(col)
                if len(selected) >= max_series:
                    break
    
    logger.info(f"Final selection: {len(selected)} series")
    logger.info(f"  Stocks (FA/FL): {len([s for s in selected if s[:2] in ['FA', 'FL']])}")
    logger.info(f"  Flows (FU/FR): {len([s for s in selected if s[:2] in ['FU', 'FR']])}")
    
    return data[selected]


def main(run_mode: Optional[str] = None):
    """Run complete SFC Kalman filter."""
    
    logger.info("="*60)
    logger.info("STOCK-FLOW CONSISTENT KALMAN FILTER")
    logger.info(f"Mode: {run_mode or 'default'}")
    logger.info("="*60)
    
    # Load configuration
    config = load_config(run_mode=run_mode)
    
    # Load Z.1 data
    logger.info("Loading Z.1 data...")
    try:
        from src.data import CachedFedDataLoader
        fed_loader = CachedFedDataLoader()
        z1_data_raw = fed_loader.load_single_source('Z1')
        
        # Filter for quarterly series only
        logger.info("Filtering for quarterly series...")
        quarterly_mask = z1_data_raw['SERIES_NAME'].str.endswith('.Q')
        z1_data_raw = z1_data_raw[quarterly_mask]
        logger.info(f"Found {len(z1_data_raw)} quarterly series")
        
        # Transform to time series format
        logger.info("Transforming to time series format...")
        metadata_cols = ['CURRENCY', 'FREQ', 'SERIES_INSTRUMENT', 'SERIES_NAME', 
                         'SERIES_PREFIX', 'SERIES_SECTOR', 'SERIES_TYPE', 'UNIT', 'UNIT_MULT']
        time_columns = [col for col in z1_data_raw.columns if col not in metadata_cols]
        
        # Transpose: dates as rows, series as columns
        z1_data = z1_data_raw.set_index('SERIES_NAME')[time_columns].T
        z1_data = z1_data.astype(float)
        z1_data.index = pd.to_datetime(z1_data.index)
        z1_data.index.name = 'Date'
        
        # Remove .Q suffix
        z1_data.columns = z1_data.columns.str.replace('.Q', '', regex=False)
        z1_data = z1_data.loc[:, ~z1_data.columns.duplicated()]
        
        # Remove empty columns
        z1_data = z1_data.dropna(axis=1, how='all')
        
        logger.info(f"Loaded {len(z1_data.columns)} Z.1 series with {len(z1_data)} time periods")
        
        # Select series for SFC (memory management)
        max_series = config['sfc'].get('performance', {}).get('max_series', 500)
        z1_data = select_series_for_sfc(z1_data, max_series)
        
    except Exception as e:
        logger.error(f"Error loading Z.1 data: {e}")
        logger.info("Using sample data for demonstration")
        
        # Generate sample data with stocks and flows
        dates = pd.date_range('2020-01-01', periods=20, freq='QE')
        n_series = 50
        
        # Create stock series
        stock_data = np.cumsum(np.random.randn(20, n_series//2), axis=0) * 100 + 1000
        stock_cols = [f'FA{i:02d}30641005' for i in range(10, 10 + n_series//4)]
        stock_cols += [f'FL{i:02d}30641005' for i in range(10, 10 + n_series//4)]
        
        # Create corresponding flow series
        flow_data = np.diff(stock_data, axis=0, prepend=stock_data[[0], :])
        flow_cols = [f'FU{i:02d}30641005' for i in range(10, 10 + n_series//4)]
        flow_cols += [f'FR{i:02d}30641005' for i in range(10, 10 + n_series//4)]
        
        # Combine
        all_data = np.hstack([stock_data, flow_data])
        all_cols = stock_cols + flow_cols
        
        z1_data = pd.DataFrame(all_data, index=dates, columns=all_cols)
    
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
        logger.warning("Formula file not found")
        formulas = {}
    
    # Setup SFC configuration
    sfc_config = {
        'enforce_sfc': config['sfc']['enforcement'].get('enforce_sfc', True),
        'enforce_fwtw': config['sfc']['enforcement'].get('enforce_fwtw', True),
        'enforce_market_clearing': config['sfc']['enforcement'].get('enforce_market_clearing', True),
        'fwtw_weight': config['sfc']['constraints'].get('fwtw_weight', 0.3),
        'market_clearing_weight': config['sfc']['constraints'].get('market_clearing_weight', 0.1),
        'stock_flow_weight': config['sfc']['constraints'].get('stock_flow_weight', 0.5),
        'bilateral_weight': config['sfc']['constraints'].get('bilateral_weight', 0.3),
        'include_all': True
    }
    
    # Initialize unified SFC filter
    logger.info("Initializing unified SFC Kalman filter...")
    
    try:
        from src.models.unified_sfc_kalman import UnifiedSFCKalmanFilter
        
        unified_filter = UnifiedSFCKalmanFilter(
            data=z1_data,
            formulas=formulas,
            fwtw_data=fwtw_data,
            sfc_config=sfc_config,
            error_variance_ratio=config['sfc']['kalman'].get('error_variance_ratio', 0.01),
            normalize_data=config['sfc']['kalman'].get('normalize_data', True),
            transformation=config['sfc']['kalman'].get('transformation', 'square')
        )
        
        # Run filter
        logger.info("Running Kalman filter with SFC constraints...")
        logger.info("This may take several minutes depending on data size...")
        
        results = unified_filter.filter()
        
        # Get filtered series
        series_dict = unified_filter.get_filtered_series(results)
        filtered = series_dict['filtered']
        smoothed = series_dict.get('smoothed', filtered)
        
        # Save results
        output_dir = Path(config['sfc']['output'].get('output_dir', './output'))
        if run_mode:
            output_dir = output_dir / run_mode
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if config['sfc']['output'].get('save_filtered', True):
            output_file = output_dir / 'sfc_filtered.csv'
            filtered.to_csv(output_file)
            logger.info(f"Saved filtered series to {output_file}")
        
        if config['sfc']['output'].get('save_smoothed', True) and smoothed is not None:
            output_file = output_dir / 'sfc_smoothed.csv'
            smoothed.to_csv(output_file)
            logger.info(f"Saved smoothed series to {output_file}")
        
        # Save bilateral flows if available
        bilateral_flows = unified_filter.get_bilateral_flows()
        if bilateral_flows is not None:
            output_file = output_dir / 'bilateral_flows.csv'
            bilateral_flows.to_csv(output_file, index=False)
            logger.info(f"Saved bilateral flows to {output_file}")
        
        # Print summary
        logger.info("="*60)
        logger.info("RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Filtered {len(filtered.columns)} series")
        logger.info(f"Time period: {filtered.index[0]} to {filtered.index[-1]}")
        
        # Data completeness
        completeness = unified_filter.get_data_completeness()
        logger.info("Data completeness:")
        for key, value in completeness.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2%}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("="*60)
        logger.info("SFC Kalman filter completed successfully!")
        logger.info("="*60)
        
        return results, filtered, smoothed
        
    except Exception as e:
        logger.error(f"Error during filtering: {e}")
        import traceback
        traceback.print_exc()
        
        logger.info("="*60)
        logger.info("SFC Kalman filter failed!")
        logger.info("="*60)
        
        return None, None, None


if __name__ == "__main__":
    # Get run mode from command line
    run_mode = sys.argv[1] if len(sys.argv) > 1 else None
    
    if run_mode:
        logger.info(f"Running in {run_mode} mode")
    
    results, filtered, smoothed = main(run_mode)
    
    if results is None:
        sys.exit(1)
    else:
        logger.info("Process completed successfully")
