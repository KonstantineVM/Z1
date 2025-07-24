#!/usr/bin/env python
"""
Complete Z1 Economic Analysis Example
Demonstrates the full pipeline from data loading to insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import framework components
from src.data import CachedFedDataLoader, DataProcessor
from src.models import UnobservedComponentsModel, TreeEnsembleModel
from src.analysis import FeatureEngineer, EconomicAnalysis
from src.visualization import EconomicVisualizer

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def main():
    """Run complete economic analysis pipeline"""
    
    print("=" * 80)
    print("Z1 Flow of Funds - Complete Economic Analysis")
    print("=" * 80)
    
    # 1. DATA LOADING
    print("\n1. Loading Federal Reserve Data...")
    print("-" * 50)
    
    # Initialize data loader with caching
    loader = CachedFedDataLoader(
        cache_directory="./data/cache",
        start_year=1980,
        end_year=2024
    )
    
    # Load Z1 data
    print("Loading Z1 Flow of Funds data...")
    z1_data = loader.load_source('Z1')
    print(f"✓ Loaded {len(z1_data)} series")
    
    # Load supplementary data
    print("\nLoading supplementary data...")
    h6_data = loader.load_source('H6')  # Money supply
    h15_data = loader.load_source('H15')  # Interest rates
    print(f"✓ Loaded H6: {len(h6_data)} series")
    print(f"✓ Loaded H15: {len(h15_data)} series")
    
    # 2. DATA PREPROCESSING
    print("\n2. Preprocessing Data...")
    print("-" * 50)
    
    # Filter for quarterly, non-flow series
    z1_filtered = z1_data[
        (z1_data['FREQ'] == '162') &  # Quarterly
        (~z1_data['SERIES_PREFIX'].isin(['FA', 'PC', 'LA', 'FC', 'FG']))
    ].copy()
    
    print(f"Filtered to {len(z1_filtered)} quarterly series")
    
    # Extract time series
    time_columns = [col for col in z1_filtered.columns 
                   if col.startswith(('19', '20'))]
    
    # Create time series dataframe
    ts_data = z1_filtered.set_index('SERIES_NAME')[time_columns].T
    ts_data.index = pd.to_datetime(ts_data.index)
    ts_data = ts_data.apply(pd.to_numeric, errors='coerce')
    
    # Clean data
    processor = DataProcessor()
    ts_clean = processor.clean_data(ts_data)
    ts_clean = processor.handle_missing_values(ts_clean, method='interpolate')
    
    # Remove series with too many missing values
    missing_pct = ts_clean.isna().sum() / len(ts_clean)
    valid_series = missing_pct[missing_pct < 0.2].index
    ts_clean = ts_clean[valid_series]
    
    print(f"✓ Cleaned data: {ts_clean.shape}")
    print(f"  Time range: {ts_clean.index[0]} to {ts_clean.index[-1]}")
    
    # 3. KEY ECONOMIC SERIES IDENTIFICATION
    print("\n3. Identifying Key Economic Series...")
    print("-" * 50)
    
    # Define key series patterns
    key_patterns = {
        'GDP': ['GDPA', 'GDP'],
        'Consumption': ['PCEA', 'PCE'],
        'Investment': ['GFCF', 'GPDI'],
        'Government': ['GCEA', 'GCE'],
        'Money_Supply': ['M1', 'M2', 'MBASE'],
        'Credit': ['TCMDO', 'CREDIT'],
        'Debt': ['DEBT', 'LIAB'],
        'Assets': ['ASSET', 'TA'],
        'Housing': ['RESL', 'HOME'],
        'Corporate': ['CORP', 'BUSINESS']
    }
    
    # Find matching series
    key_series = {}
    for category, patterns in key_patterns.items():
        matches = [col for col in ts_clean.columns 
                  if any(pattern in col.upper() for pattern in patterns)]
        if matches:
            # Take first match or aggregate
            key_series[category] = ts_clean[matches[0]]
            print(f"  {category}: {matches[0]}")
    
    # Create key series dataframe
    key_df = pd.DataFrame(key_series)
    
    # 4. TIME SERIES DECOMPOSITION
    print("\n4. Decomposing Time Series...")
    print("-" * 50)
    
    # Initialize UC model
    uc_model = UnobservedComponentsModel(
        cycle_period=32,  # 8 years for quarterly data
        damping_factor=0.98
    )
    
    # Decompose key series
    components = {}
    for series_name in ['GDP', 'Money_Supply', 'Credit']:
        if series_name in key_df.columns:
            print(f"  Decomposing {series_name}...")
            components[series_name] = uc_model.decompose(key_df[series_name])
    
    print("✓ Decomposition complete")
    
    # 5. FEATURE ENGINEERING
    print("\n5. Engineering Features...")
    print("-" * 50)
    
    engineer = FeatureEngineer()
    
    # Create cycle features for modeling
    cycle_data = pd.DataFrame({
        name: comp['cycle'] 
        for name, comp in components.items()
    })
    
    # Engineer features
    features = engineer.create_features(
        cycle_data,
        lags=[1, 2, 4, 8],
        rolling_windows=[4, 8],
        include_interactions=True
    )
    
    print(f"✓ Created {len(features.columns)} features")
    
    # 6. ECONOMIC ANALYSIS
    print("\n6. Performing Economic Analysis...")
    print("-" * 50)
    
    analyzer = EconomicAnalysis(key_df)
    
    # Velocity of money analysis
    if 'GDP' in key_df.columns and 'Money_Supply' in key_df.columns:
        print("\n  a) Velocity of Money Analysis")
        velocity_results = analyzer.analyze_velocity_of_money(
            money_supply_series='Money_Supply',
            gdp_series='GDP'
        )
        
        current_velocity = velocity_results['velocity'].iloc[-1]
        velocity_trend = velocity_results['velocity_trend'].iloc[-1]
        print(f"     Current velocity: {current_velocity:.2f}")
        print(f"     Velocity trend: {velocity_trend:+.2%} per quarter")
    
    # Credit cycle analysis
    if 'Credit' in key_df.columns and 'GDP' in key_df.columns:
        print("\n  b) Credit Cycle Analysis")
        credit_gdp_ratio = key_df['Credit'] / key_df['GDP']
        credit_gap = credit_gdp_ratio - credit_gdp_ratio.rolling(40).mean()
        
        current_gap = credit_gap.iloc[-1]
        gap_percentile = (credit_gap < current_gap).sum() / len(credit_gap) * 100
        
        print(f"     Credit/GDP gap: {current_gap:.2%}")
        print(f"     Historical percentile: {gap_percentile:.1f}%")
    
    # Find leading indicators
    print("\n  c) Leading Indicator Analysis")
    if 'GDP' in key_df.columns:
        leading_indicators = analyzer.find_leading_indicators(
            target='GDP',
            max_lag=8,
            significance_level=0.05
        )
        
        if not leading_indicators.empty:
            top_indicators = leading_indicators.nlargest(5, 'correlation')
            print("     Top 5 leading indicators:")
            for _, row in top_indicators.iterrows():
                print(f"     - {row['indicator']}: lag {row['lag']}, "
                      f"correlation {row['correlation']:.3f}")
    
    # 7. PREDICTIVE MODELING
    print("\n7. Building Predictive Models...")
    print("-" * 50)
    
    # Prepare data for modeling
    if 'GDP' in components:
        # Target: GDP growth (quarter-over-quarter)
        gdp_growth = key_df['GDP'].pct_change()
        
        # Align features and target
        model_data = features.copy()
        model_data['target'] = gdp_growth.shift(-1)  # Predict next quarter
        model_data = model_data.dropna()
        
        # Split data
        split_date = '2018-01-01'
        train_data = model_data[model_data.index < split_date]
        test_data = model_data[model_data.index >= split_date]
        
        X_train = train_data.drop('target', axis=1)
        y_train = train_data['target']
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        print(f"  Training set: {len(X_train)} observations")
        print(f"  Test set: {len(X_test)} observations")
        
        # Train XGBoost model
        print("\n  Training XGBoost model...")
        xgb_model = TreeEnsembleModel(
            model_type='xgboost',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05
        )
        
        xgb_model.fit(X_train, y_train)
        predictions = xgb_model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"\n  Model Performance:")
        print(f"    MSE: {mse:.6f}")
        print(f"    MAE: {mae:.4%}")
        print(f"    R²: {r2:.3f}")
        
        # Feature importance
        importance = xgb_model.get_feature_importance()
        print(f"\n  Top 10 Important Features:")
        for _, row in importance.head(10).iterrows():
            print(f"    - {row['feature']}: {row['importance']:.3f}")
    
    # 8. VISUALIZATION
    print("\n8. Creating Visualizations...")
    print("-" * 50)
    
    viz = EconomicVisualizer()
    
    # Create output directory
    output_dir = Path("output/z1_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Component decomposition
    if 'GDP' in components:
        print("  Creating GDP decomposition plot...")
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        comp = components['GDP']
        comp['observed'].plot(ax=axes[0], title='GDP (Observed)', color='black')
        comp['trend'].plot(ax=axes[1], title='Trend', color='blue')
        comp['cycle'].plot(ax=axes[2], title='Cycle', color='green')
        comp['irregular'].plot(ax=axes[3], title='Irregular', color='red')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gdp_decomposition.png', dpi=300)
        plt.close()
    
    # Plot 2: Velocity of money
    if 'velocity' in locals():
        print("  Creating velocity of money plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        velocity_results['velocity'].plot(ax=ax, label='Velocity', linewidth=2)
        velocity_results['velocity_trend'].plot(ax=ax, label='Trend', 
                                               linestyle='--', linewidth=2)
        
        ax.set_title('Velocity of Money (GDP/M2)', fontsize=14)
        ax.set_ylabel('Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add recession shading
        viz.add_recession_shading(ax)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'velocity_of_money.png', dpi=300)
        plt.close()
    
    # Plot 3: Credit cycle
    if 'credit_gap' in locals():
        print("  Creating credit cycle plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        credit_gap.plot(ax=ax, linewidth=2, color='darkblue')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.fill_between(credit_gap.index, 0, credit_gap, 
                       where=credit_gap > 0, alpha=0.3, color='red', 
                       label='Above trend')
        ax.fill_between(credit_gap.index, 0, credit_gap, 
                       where=credit_gap <= 0, alpha=0.3, color='green', 
                       label='Below trend')
        
        ax.set_title('Credit-to-GDP Gap', fontsize=14)
        ax.set_ylabel('Gap (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        viz.add_recession_shading(ax)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'credit_cycle.png', dpi=300)
        plt.close()
    
    # Plot 4: Forecast results
    if 'predictions' in locals():
        print("  Creating forecast plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual vs predicted
        ax.plot(y_test.index, y_test.values, label='Actual', 
                linewidth=2, color='black')
        ax.plot(y_test.index, predictions, label='Predicted', 
                linewidth=2, color='red', linestyle='--')
        
        # Add confidence interval (simplified)
        std_error = np.std(y_test - predictions)
        ax.fill_between(y_test.index, 
                       predictions - 2*std_error, 
                       predictions + 2*std_error,
                       alpha=0.2, color='red', label='95% CI')
        
        ax.set_title('GDP Growth Forecast', fontsize=14)
        ax.set_ylabel('Quarter-over-Quarter Growth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gdp_forecast.png', dpi=300)
        plt.close()
    
    # 9. GENERATE REPORT
    print("\n9. Generating Analysis Report...")
    print("-" * 50)
    
    report_path = output_dir / 'analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("Z1 FLOW OF FUNDS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        f.write("1. DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total series analyzed: {len(ts_clean.columns)}\n")
        f.write(f"Time period: {ts_clean.index[0]} to {ts_clean.index[-1]}\n")
        f.write(f"Frequency: Quarterly\n")
        f.write(f"Key series identified: {len(key_series)}\n\n")
        
        f.write("2. ECONOMIC INDICATORS\n")
        f.write("-" * 40 + "\n")
        
        if 'velocity_results' in locals():
            f.write(f"Velocity of Money:\n")
            f.write(f"  Current: {current_velocity:.2f}\n")
            f.write(f"  Trend: {velocity_trend:+.2%} per quarter\n")
            f.write(f"  Historical avg: {velocity_results['velocity'].mean():.2f}\n\n")
        
        if 'credit_gap' in locals():
            f.write(f"Credit Cycle:\n")
            f.write(f"  Credit/GDP gap: {current_gap:.2%}\n")
            f.write(f"  Percentile: {gap_percentile:.1f}%\n")
            f.write(f"  Status: {'Elevated' if current_gap > 0.02 else 'Normal'}\n\n")
        
        f.write("3. MODEL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        if 'r2' in locals():
            f.write(f"GDP Growth Forecast Model:\n")
            f.write(f"  R-squared: {r2:.3f}\n")
            f.write(f"  MAE: {mae:.4%}\n")
            f.write(f"  RMSE: {np.sqrt(mse):.4%}\n\n")
        
        f.write("4. KEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        
        # Add automated insights based on analysis
        insights = []
        
        if 'velocity_trend' in locals() and velocity_trend < -0.01:
            insights.append("- Velocity of money is declining, suggesting potential "
                          "deflationary pressures or increased money hoarding")
        
        if 'gap_percentile' in locals() and gap_percentile > 80:
            insights.append("- Credit cycle is in elevated territory, indicating "
                          "potential financial stability risks")
        
        if 'r2' in locals() and r2 > 0.5:
            insights.append("- Economic indicators show strong predictive power "
                          "for GDP growth")
        
        for insight in insights:
            f.write(insight + "\n")
    
    print(f"✓ Report saved to: {report_path}")
    
    # 10. SUMMARY
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    for file in output_dir.glob("*"):
        print(f"  - {file.name}")
    
    print("\nKey findings:")
    if 'current_velocity' in locals():
        print(f"  - Velocity of money: {current_velocity:.2f}")
    if 'gap_percentile' in locals():
        print(f"  - Credit gap percentile: {gap_percentile:.1f}%")
    if 'r2' in locals():
        print(f"  - Forecast model R²: {r2:.3f}")
    
    print("\nNext steps:")
    print("  1. Review the generated report for detailed insights")
    print("  2. Examine the visualizations in the output directory")
    print("  3. Fine-tune models based on specific use cases")
    print("  4. Extend analysis to other Fed data sources")
    
    return {
        'data': key_df,
        'components': components,
        'features': features,
        'model': xgb_model if 'xgb_model' in locals() else None,
        'results': {
            'velocity': velocity_results if 'velocity_results' in locals() else None,
            'credit_gap': credit_gap if 'credit_gap' in locals() else None,
            'predictions': predictions if 'predictions' in locals() else None
        }
    }


if __name__ == "__main__":
    # Run the complete analysis
    results = main()
    
    # Optional: Save results for further analysis
    print("\nSaving analysis results...")
    import pickle
    
    with open('output/z1_analysis/analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("✓ Results saved for future use")