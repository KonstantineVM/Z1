# examples/run_with_config.py
import yaml
from pathlib import Path
from src.models.unified_sfc_kalman import UnifiedSFCKalmanFilter
from src.data import CachedFedDataLoader
from src.network import FWTWDataLoader

def load_config(config_path='config/sfc_config.yaml', 
                run_mode='production'):
    """Load configuration from YAML files."""
    
    # Load main config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load run mode if specified
    if run_mode:
        with open('config/run_modes.yaml', 'r') as f:
            run_modes = yaml.safe_load(f)
            mode_config = run_modes['run_modes'][run_mode]
            
        # Merge mode config with main config
        if 'sfc_config' in mode_config:
            config['sfc']['constraints'].update(mode_config['sfc_config'])
        if 'kalman_params' in mode_config:
            config['sfc']['kalman_params'].update(mode_config['kalman_params'])
    
    return config

def main(run_mode='production'):
    """Run SFC Kalman filter with configuration."""
    
    # Load configuration
    config = load_config(run_mode=run_mode)
    
    # Load data
    fed_loader = CachedFedDataLoader()
    z1_data = fed_loader.load_single_source(config['sfc']['data_sources']['z1_source'])
    
    fwtw_loader = FWTWDataLoader()
    fwtw_data = fwtw_loader.load_fwtw_data()
    
    # Load formulas
    with open(config['sfc']['data_sources']['formula_file'], 'r') as f:
        formulas = json.load(f)
    
    # Initialize filter with config
    unified_filter = UnifiedSFCKalmanFilter(
        data=z1_data,
        formulas=formulas,
        fwtw_data=fwtw_data,
        sfc_config=config['sfc']['constraints'],
        **config['sfc']['kalman_params']
    )
    
    # Run filter
    results = unified_filter.filter()
    
    # Save outputs based on config
    if config['sfc']['output']['save_filtered']:
        output_dir = Path(config['sfc']['output'].get('diagnostic_dir', './output'))
        output_dir.mkdir(exist_ok=True, parents=True)
        
        filtered = unified_filter.get_filtered_series(results)['filtered']
        
        if config['sfc']['output']['denormalize_output']:
            filtered = filtered * unified_filter.scale_factor
        
        if config['sfc']['output']['output_format'] == 'csv':
            filtered.to_csv(output_dir / 'filtered_series.csv')
        else:
            filtered.to_parquet(output_dir / 'filtered_series.parquet')
    
    return results

if __name__ == "__main__":
    import sys
    
    # Get run mode from command line
    run_mode = sys.argv[1] if len(sys.argv) > 1 else 'production'
    
    print(f"Running in {run_mode} mode...")
    results = main(run_mode)
