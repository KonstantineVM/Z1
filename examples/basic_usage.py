"""
Basic usage example of the Economic Time Series Analysis framework
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.fed_data_loader import FedDataLoader
from src.models.unobserved_components import UnobservedComponentsModel
from src.analysis.feature_engineering import FeatureEngineer
from src.analysis.economic_analysis import EconomicAnalysis
from src.visualization.economic_plots import EconomicVisualizer


def main():
    """
    Main example demonstrating the analysis pipeline
    """
    print("Economic Time Series Analysis Example")
    print("=" * 50)
    
    # 1. Load Data
    print("\n1. Loading Federal Reserve Data...")
    
    # Initialize data loader
    fed_loader = FedDataLoader(
        base_directory="./data/fed_data",
        start_year=1959,
        end_year=2024
    )
    
    # Load multiple data sources
    sources = ['Z1', 'H6', 'H15']
    data_dict = fed_loader.load_multiple_sources(sources, download=True)
    
    # Combine and process data
    # For this example, we'll focus on Z1 data
    z1_data = data_dict.get('Z1', pd.DataFrame())
    
    if z1_data.empty:
        print("No Z1 data loaded. Please check data availability.")
        return
    
    # Filter for quarterly frequency and non-flow series
    z1_filtered = z1_data[
        (z1_data['FREQ'] == '162') & 
        (~z1_data['SERIES_PREFIX'].isin(['FA', 'PC', 'LA', 'FC', 'FG']))
    ].copy()
    
    # Process time series columns
    time_columns = [col for col in z1_filtered.columns if col.startswith('19') or col.startswith('20')]
    series_data = z1_filtered.set_index('SERIES_NAME')[time_columns].T
    series_data.index = pd.to_datetime(series_data.index)
    
    # Convert to numeric and handle missing values
    series_data = series_data.apply(pd.to_numeric, errors='coerce')
    series_data = series_data.dropna(axis=1, how='all')
    
    print(f"Loaded {len(series_data.columns)} series with {len(series_data)} time points")
    
    # 2. Decompose Series
    print("\n2. Decomposing Time Series...")
    
    # Initialize UC model
    uc_model = UnobservedComponentsModel()
    
    # Decompose series (using subset for speed)
    subset_columns = series_data.columns[:50]  # First 50 series for demo
    components = uc_model.decompose_parallel(series_data[subset_columns], n_jobs=4)
    
    print("Decomposition complete:")
    for comp_name, comp_df in components.items():
        if comp_df is not None:
            print(f"  - {comp_name}: {comp_df.shape}")
    
    # 3. Feature Engineering
    print("\n3. Creating Features...")
    
    # Identify zero-crossing series
    zero_crossing_cols = uc_model.identify_zero_crossing_series(
        series_data[subset_columns], 
        components
    )
    print(f"Found {len(zero_crossing_cols)} zero-crossing series")
    
    # Create features
    feature_engineer = FeatureEngineer()
    features = feature_engineer.create_component_features(components, zero_crossing_cols)
    
    # Add lagged features
    features_lagged = feature_engineer.create_lagged_features(features, max_lags=8, min_lag=3)
    print(f"Created {features_lagged.shape[1]} features including lags")
    
    # 4. Economic Analysis
    print("\n4. Performing Economic Analysis...")
    
    # Initialize analyzer
    analyzer = EconomicAnalysis(components, series_data[subset_columns])
    
    # Analyze velocity of money
    velocity_results = analyzer.analyze_velocity_of_money()
    if not velocity_results.empty:
        print("\nVelocity of Money Analysis:")
        print(velocity_results.describe())
    
    # Analyze interest rate relationships (if Rate series available)
    rate_cols = [col for col in components['trend'].columns if 'Rate' in col]
    if rate_cols:
        target_rate = rate_cols[0]
        rate_analysis = analyzer.analyze_interest_rate_relationships(
            target_rate, 
            lasso_alpha=0.001
        )
        
        if 'feature_importance' in rate_analysis:
            print(f"\nTop predictors for {target_rate}:")
            print(rate_analysis['feature_importance'].head(10))
    
    # 5. Visualization
    print("\n5. Creating Visualizations...")
    
    visualizer = EconomicVisualizer()
    
    # Plot component decomposition for first series
    first_series = subset_columns[0]
    series_components = {
        comp_name: comp_df[first_series] 
        for comp_name, comp_df in components.items() 
        if comp_df is not None and first_series in comp_df.columns
    }
    
    fig1 = visualizer.plot_component_decomposition(
        first_series,
        series_components,
        series_data[first_series]
    )
    fig1.savefig('./output/figures/decomposition_example.png', dpi=300, bbox_inches='tight')
    print("Saved decomposition plot")
    
    # Plot velocity analysis if available
    if not velocity_results.empty:
        plot_data = pd.DataFrame(index=components['trend'].index)
        plot_data['GDP Growth'] = velocity_results.get('gdp_growth', pd.Series())
        plot_data['M2 Growth'] = velocity_results.get('m2_growth', pd.Series())
        plot_data['Velocity Change'] = velocity_results.get('velocity_change', pd.Series())
        
        if not plot_data.dropna().empty:
            fig2 = visualizer.plot_economic_relationships(
                plot_data.dropna(),
                ['GDP Growth', 'M2 Growth'],
                'Velocity Change',
                title="Velocity of Money Analysis",
                add_recessions=True
            )
            fig2.savefig('./output/figures/velocity_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved velocity analysis plot")
    
    # Plot cyclical component with recessions
    if 'cycle' in components and components['cycle'] is not None:
        cycle_series = components['cycle'].iloc[:, 0]  # First series
        fig3 = visualizer.plot_cycle_with_recessions(
            cycle_series,
            cycle_series.name,
            normalize=True
        )
        fig3.savefig('./output/figures/cycle_with_recessions.png', dpi=300, bbox_inches='tight')
        print("Saved cycle plot with recessions")
    
    print("\n" + "=" * 50)
    print("Analysis complete! Check ./output/figures/ for visualizations.")


if __name__ == "__main__":
    # Create output directories
    Path("./output/figures").mkdir(parents=True, exist_ok=True)
    Path("./output/results").mkdir(parents=True, exist_ok=True)
    
    # Run example
    main()