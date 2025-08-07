import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(run_mode='production'):
    logger.info("="*60)
    logger.info("STOCK-FLOW CONSISTENT KALMAN FILTER")
    logger.info(f"Mode: {run_mode}")
    logger.info("="*60)
    
    # Load Z.1 data
    logger.info("Loading Z.1 data...")
    from src.data import CachedFedDataLoader
    fed_loader = CachedFedDataLoader()
    z1_data_raw = fed_loader.load_single_source('Z1')
    
    # Filter for quarterly series
    quarterly_mask = z1_data_raw['SERIES_NAME'].str.endswith('.Q')
    z1_data_raw = z1_data_raw[quarterly_mask]
    logger.info(f"Found {len(z1_data_raw)} quarterly series")
    
    # Transform to time series format
    metadata_cols = ['CURRENCY', 'FREQ', 'SERIES_INSTRUMENT', 'SERIES_NAME', 
                     'SERIES_PREFIX', 'SERIES_SECTOR', 'SERIES_TYPE', 'UNIT', 'UNIT_MULT']
    time_columns = [col for col in z1_data_raw.columns if col not in metadata_cols]
    
    z1_data = z1_data_raw.set_index('SERIES_NAME')[time_columns].T
    z1_data = z1_data.astype(float)
    z1_data.index = pd.to_datetime(z1_data.index)
    z1_data.index.name = 'Date'
    
    # Remove .Q suffix
    z1_data.columns = z1_data.columns.str.replace('.Q', '', regex=False)
    z1_data = z1_data.loc[:, ~z1_data.columns.duplicated()]
    
    # For testing, limit series but choose ones likely to match FWTW
    MAX_SERIES = 1000  # Increase for production
    
    # Prioritize financial sectors that appear in FWTW
    priority_sectors = ['10', '11', '15', '21', '26', '31', '40', '42', '47', 
                       '50', '51', '54', '59', '61', '64', '65', '66', '67',
                       '70', '73', '75', '76', '78', '79', '80', '81', '87', '89', '90']
    
    priority_series = []
    for col in z1_data.columns:
        if len(col) >= 4:
            sector = col[2:4]
            if sector in priority_sectors:
                priority_series.append(col)
    
    if len(priority_series) > MAX_SERIES:
        z1_data = z1_data[priority_series[:MAX_SERIES]]
    elif priority_series:
        z1_data = z1_data[priority_series]
    else:
        z1_data = z1_data.iloc[:, :MAX_SERIES]
    
    logger.info(f"Using {len(z1_data.columns)} series")
    
    # Load FWTW
    logger.info("Loading FWTW data...")
    from src.network import FWTWDataLoader
    fwtw_loader = FWTWDataLoader()
    fwtw_data = fwtw_loader.load_fwtw_data()
    logger.info(f"Loaded {len(fwtw_data)} FWTW positions")
    
    # Load formulas
    formula_path = Path('data/fof_formulas_extracted.json')
    formulas = {}
    if formula_path.exists():
        with open(formula_path, 'r') as f:
            formulas = json.load(f)
        logger.info(f"Loaded {len(formulas)} formulas")
    
    # Configure SFC
    sfc_config = {
        'fwtw_weight': 0.3,
        'market_clearing_weight': 0.1,
        'enforce_fwtw': True,
        'enforce_market_clearing': True,
        'include_all': False,
        'require_fwtw_overlap': False,
        'min_overlap_fraction': 0.0
    }
    
    # Initialize and run
    logger.info("Initializing SFC Kalman filter...")
    from src.models.unified_sfc_kalman import UnifiedSFCKalmanFilter
    
    try:
        unified_filter = UnifiedSFCKalmanFilter(
            data=z1_data,
            formulas=formulas,
            fwtw_data=fwtw_data,
            sfc_config=sfc_config,
            error_variance_ratio=0.01,
            normalize_data=True,
            transformation='square'
        )
        
        logger.info("Running Kalman filter (this may take a few minutes)...")
        results = unified_filter.filter()
        
        # Get results
        series_dict = unified_filter.get_filtered_series(results)
        filtered = series_dict['filtered']
        smoothed = series_dict['smoothed']
        
        # Denormalize
        if unified_filter.normalize_data and hasattr(unified_filter, 'scale_factor'):
            filtered = filtered * unified_filter.scale_factor
            smoothed = smoothed * unified_filter.scale_factor
        
        # Save
        output_dir = Path(f'./output/{run_mode}')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        output_file = output_dir / f'sfc_filtered_{len(filtered.columns)}_series.csv'
        filtered.to_csv(output_file)
        logger.info(f"Saved filtered results to {output_file}")
        
        output_file = output_dir / f'sfc_smoothed_{len(smoothed.columns)}_series.csv'
        smoothed.to_csv(output_file)
        logger.info(f"Saved smoothed results to {output_file}")
        
        # Summary
        logger.info("="*60)
        logger.info("SUCCESS - SFC Kalman filter complete!")
        logger.info("="*60)
        logger.info(f"Processed {len(filtered.columns)} series")
        logger.info(f"Time period: {filtered.index[0]} to {filtered.index[-1]}")
        logger.info(f"Output directory: {output_dir}")
        
        return results, filtered, smoothed
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    run_mode = sys.argv[1] if len(sys.argv) > 1 else 'production'
    results, filtered, smoothed = main(run_mode)
    
    if results is None:
        logger.error("Failed to complete SFC Kalman filter")
        sys.exit(1)
    else:
        logger.info("Process completed successfully")
