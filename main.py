"""
Main script for Economic Time Series Analysis
"""

import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.cached_fed_data_loader import CachedFedDataLoader
from src.data.external_data_loader import ExternalDataLoader
from src.data.data_processor import DataProcessor
from src.models.unobserved_components import UnobservedComponentsModel
from src.models.tree_models import TreeModelAnalyzer
from src.models.gaussian_process import GaussianProcessModel
from src.analysis.feature_engineering import FeatureEngineer
from src.analysis.economic_analysis import EconomicAnalysis
from src.visualization.economic_plots import EconomicVisualizer


def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    
    # Get the file configuration from the logging config
    file_config = log_config.get('file', {})
    
    # Extract the actual log file path from the file config
    if isinstance(file_config, dict):
        log_file = file_config.get('path', './logs/analysis.log')
    else:
        log_file = './logs/analysis.log'  # Default fallback
    
    # Create logs directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Get logging level
    if 'console' in log_config and 'level' in log_config['console']:
        log_level = log_config['console'].get('level', 'INFO')
    else:
        log_level = log_config.get('level', 'INFO')
    
    # Get format
    if 'console' in log_config and 'format' in log_config['console']:
        log_format = log_config['console'].get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_output_directories(config):
    """Create output directories"""
    output_config = config.get('output', {})
    
    for dir_key in ['figures_dir', 'models_dir', 'results_dir']:
        dir_path = Path(output_config.get(dir_key, f'./output/{dir_key.replace("_dir", "")}'))
        dir_path.mkdir(parents=True, exist_ok=True)


def load_data(config, logger):
    """Load all data sources"""
    logger.info("Loading data...")
    
    data_config = config.get('data', {})
    
    # Load Fed data with caching
    fed_loader = CachedFedDataLoader(
        base_directory=data_config.get('base_directory', './data/fed_data'),
        cache_directory=data_config.get('cache_directory', './data/cache'),
        start_year=data_config.get('start_year', 1959),
        end_year=data_config.get('end_year', 2024),
        cache_expiry_days=data_config.get('cache_expiry_days', 7)
    )
    
    # Check cache status
    cache_info = fed_loader.get_cache_info()
    if cache_info:
        logger.info("Cache status:")
        for source, info in cache_info.items():
            logger.info(f"  {source}: valid={info['valid']}, cached_at={info['cached_at']}")
    
    sources = data_config.get('sources', ['Z1'])
    force_download = data_config.get('force_download', False)
    fed_data = fed_loader.load_multiple_sources(sources, force_download=force_download)
    
    # Load external data with caching
    external_loader = ExternalDataLoader(
        cache_directory=data_config.get('external_cache_directory', './data/cache/external'),
        cache_expiry_days=data_config.get('cache_expiry_days', 7)
    )
    external_data = {}
    
    external_config = data_config.get('external', {})
    if 'sp500' in external_config:
        external_data['sp500'] = external_loader.load_shiller_data(
            external_config['sp500'].get('url'),
            force_download=force_download
        )
    
    if 'dallas_fed' in external_config:
        external_data['dallas_fed'] = external_loader.load_dallas_fed_debt(
            external_config['dallas_fed'].get('url'),
            force_download=force_download
        )
    
    if 'fred_series' in external_config:
        for series in external_config['fred_series']:
            external_data[series] = external_loader.load_fred_series(
                series, force_download=force_download
            )
    
    if 'gold' in external_config:
        external_data['gold'] = external_loader.load_gold_prices(
            force_download=force_download
        )
    
    logger.info(f"Loaded {len(fed_data)} Fed sources and {len(external_data)} external sources")
    
    return fed_data, external_data


def process_data(fed_data, external_data, config, logger):
    """Process and combine data"""
    logger.info("Processing data...")
    
    processor = DataProcessor()
    
    # Process Fed data
    processed_fed = {}
    for source, df in fed_data.items():
        processed_fed[source] = processor.process_fed_data(df, source)
    
    # Combine all data
    combined_data = processor.combine_data_sources(processed_fed, external_data)
    
    logger.info(f"Combined data shape: {combined_data.shape}")
    
    return combined_data


def decompose_series(data, config, logger):
    """Decompose time series"""
    logger.info("Decomposing time series...")
    
    model_config = config.get('models', {}).get('unobserved_components', {})
    
    uc_model = UnobservedComponentsModel(model_config)
    components = uc_model.decompose_parallel(
        data, 
        n_jobs=config.get('analysis', {}).get('n_jobs', -1)
    )
    
    # Identify zero-crossing series
    zero_crossing_cols = uc_model.identify_zero_crossing_series(data, components)
    
    # Normalize components for zero-crossing series
    if zero_crossing_cols:
        normalized_components = uc_model.normalize_components_by_amplitude(
            components, zero_crossing_cols
        )
        # Merge normalized components
        for comp_name in components:
            if comp_name in normalized_components:
                for col in zero_crossing_cols:
                    if col in normalized_components[comp_name].columns:
                        components[comp_name][col] = normalized_components[comp_name][col]
    
    logger.info(f"Decomposition complete. Found {len(zero_crossing_cols)} zero-crossing series")
    
    return components, zero_crossing_cols


def engineer_features(components, zero_crossing_cols, config, logger):
    """Create features from components"""
    logger.info("Engineering features...")
    
    feature_config = config.get('analysis', {}).get('features', {})
    
    engineer = FeatureEngineer()
    
    # Create component features
    features = engineer.create_component_features(components, zero_crossing_cols)
    
    # Add lagged features
    features_lagged = engineer.create_lagged_features(
        features,
        max_lags=feature_config.get('max_lags', 16),
        min_lag=feature_config.get('min_lag', 3)
    )
    
    # Add economic indicators
    indicators = engineer.create_economic_indicators(components)
    if not indicators.empty:
        features_lagged = pd.concat([features_lagged, indicators], axis=1)
    
    logger.info(f"Created {features_lagged.shape[1]} features")
    
    return features_lagged


def run_analysis(components, features, original_data, config, logger):
    """Run economic analysis"""
    logger.info("Running economic analysis...")
    
    results = {}
    
    # Economic analysis
    analyzer = EconomicAnalysis(components, original_data)
    
    # Velocity of money
    velocity_results = analyzer.analyze_velocity_of_money()
    if not velocity_results.empty:
        results['velocity'] = velocity_results
        logger.info("Completed velocity of money analysis")
    
    # Interest rate relationships
    rate_cols = [col for col in components.get('trend', pd.DataFrame()).columns if 'Rate' in col]
    if rate_cols:
        rate_results = analyzer.analyze_interest_rate_relationships(
            rate_cols[0],
            lasso_alpha=config.get('models', {}).get('lasso', {}).get('alpha', 0.001)
        )
        results['interest_rates'] = rate_results
        logger.info("Completed interest rate analysis")
    
    # Savings dynamics
    savings_results = analyzer.analyze_savings_dynamics()
    if not savings_results.empty:
        results['savings'] = savings_results
        logger.info("Completed savings dynamics analysis")
    
    return results


def create_visualizations(components, results, config, logger):
    """Create visualizations"""
    logger.info("Creating visualizations...")
    
    vis_config = config.get('visualization', {})
    output_config = config.get('output', {})
    
    visualizer = EconomicVisualizer(
        figsize=tuple(vis_config.get('figsize', [12, 6]))
    )
    
    figures_dir = Path(output_config.get('figures_dir', './output/figures'))
    
    # Velocity analysis plot
    if 'velocity' in results and not results['velocity'].empty:
        fig = visualizer.plot_velocity_analysis(components)
        if fig:
            fig.savefig(
                figures_dir / 'velocity_analysis.png',
                dpi=vis_config.get('dpi', 300),
                bbox_inches='tight'
            )
            logger.info("Saved velocity analysis plot")
    
    # Component decomposition for selected series
    if 'trend' in components:
        sample_series = components['trend'].columns[0]
        series_components = {
            name: df[sample_series] 
            for name, df in components.items() 
            if df is not None and sample_series in df.columns
        }
        
        fig = visualizer.plot_component_decomposition(
            sample_series,
            series_components
        )
        fig.savefig(
            figures_dir / f'decomposition_{sample_series}.png',
            dpi=vis_config.get('dpi', 300),
            bbox_inches='tight'
        )
        logger.info(f"Saved decomposition plot for {sample_series}")
    
    # Feature importance plots
    for analysis_name, analysis_results in results.items():
        if isinstance(analysis_results, dict) and 'feature_importance' in analysis_results:
            fig = visualizer.plot_feature_importance(
                analysis_results['feature_importance'],
                title=f"Feature Importance - {analysis_name.title()}"
            )
            fig.savefig(
                figures_dir / f'feature_importance_{analysis_name}.png',
                dpi=vis_config.get('dpi', 300),
                bbox_inches='tight'
            )
            logger.info(f"Saved feature importance plot for {analysis_name}")


def save_results(components, features, results, config, logger):
    """Save analysis results"""
    logger.info("Saving results...")
    
    output_config = config.get('output', {})
    results_dir = Path(output_config.get('results_dir', './output/results'))
    data_format = output_config.get('data_format', 'parquet')
    
    # Save components
    for comp_name, comp_df in components.items():
        if comp_df is not None:
            filename = results_dir / f'components_{comp_name}.{data_format}'
            if data_format == 'parquet':
                comp_df.to_parquet(filename)
            else:
                comp_df.to_csv(filename)
    
    # Save features
    filename = results_dir / f'features.{data_format}'
    if data_format == 'parquet':
        features.to_parquet(filename)
    else:
        features.to_csv(filename)
    
    # Save analysis results
    for name, result in results.items():
        if isinstance(result, pd.DataFrame):
            filename = results_dir / f'analysis_{name}.{data_format}'
            if data_format == 'parquet':
                result.to_parquet(filename)
            else:
                result.to_csv(filename)
    
    logger.info("Results saved successfully")


def main():
    """Main analysis pipeline"""
    parser = argparse.ArgumentParser(description='Economic Time Series Analysis')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--components', type=str, nargs='+',
                       choices=['data', 'decompose', 'features', 'analysis', 'visualize', 'all'],
                       default=['all'],
                       help='Components to run')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    logger = setup_logging(config)
    create_output_directories(config)
    
    logger.info("Starting Economic Time Series Analysis")
    logger.info(f"Configuration: {args.config}")
    
    # Run pipeline components
    run_all = 'all' in args.components
    
    # Load data
    if run_all or 'data' in args.components:
        fed_data, external_data = load_data(config, logger)
        processed_data = process_data(fed_data, external_data, config, logger)
        
        # Save processed data
        output_dir = Path(config['output']['results_dir'])
        data_format = config['output'].get('data_format', 'parquet')
        
        if data_format == 'parquet':
            processed_data.to_parquet(output_dir / f'processed_data.{data_format}')
        else:
            processed_data.to_csv(output_dir / f'processed_data.{data_format}')
    else:
        # Load existing processed data
        output_dir = Path(config['output']['results_dir'])
        data_format = config['output'].get('data_format', 'parquet')
        
        if data_format == 'parquet':
            processed_data = pd.read_parquet(output_dir / f'processed_data.{data_format}')
        else:
            processed_data = pd.read_csv(output_dir / f'processed_data.{data_format}', index_col=0)
    
    # Decompose series
    if run_all or 'decompose' in args.components:
        components, zero_crossing_cols = decompose_series(processed_data, config, logger)
    
    # Engineer features
    if run_all or 'features' in args.components:
        features = engineer_features(components, zero_crossing_cols, config, logger)
    
    # Run analysis
    if run_all or 'analysis' in args.components:
        results = run_analysis(components, features, processed_data, config, logger)
    
    # Create visualizations
    if run_all or 'visualize' in args.components:
        create_visualizations(components, results, config, logger)
    
    # Save results
    if run_all:
        save_results(components, features, results, config, logger)
    
    logger.info("Analysis complete!")
    logger.info(f"Results saved to {config['output']['results_dir']}")
    logger.info(f"Figures saved to {config['output']['figures_dir']}")


if __name__ == "__main__":
    main()
