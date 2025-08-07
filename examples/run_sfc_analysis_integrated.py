#!/usr/bin/env python3
"""
Integrated SFC Kalman Filter Analysis for Z1 Project.
This script runs SFC analysis using Z1's existing data infrastructure.
"""

import sys
import logging
from pathlib import Path
import json
import yaml
import argparse
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Z1 data infrastructure
from src.data import CachedFedDataLoader
from src.network import FWTWDataLoader

# Import utilities (from kalman_filter_analysis)
from src.utils.config_manager import ConfigManager
from src.utils.results_manager import ResultsManager
from src.utils.visualization import VisualizationManager

# Import models
from src.models.sfc_kalman_filter_extended import SFCKalmanFilter


class IntegratedSFCAnalysis:
    """Integrated SFC analysis using Z1 infrastructure."""
    
    def __init__(self, config_path: str = None, run_mode: str = 'development'):
        """Initialize integrated analysis."""
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config_path = config_path or 'config/sfc_config.yaml'
        self.run_mode = run_mode
        self.config = self._load_config()
        
        # Initialize results manager
        output_dir = Path(self.config.get('output_dir', 'output')) / run_mode
        self.results = ResultsManager(output_dir, timestamp_outputs=True)
        
        # Storage
        self.data = None
        self.fwtw_data = None
        self.formulas = None
        self.model = None
        
    def _load_config(self):
        """Load configuration with run mode overrides."""
        config = {
            'max_series': 500,
            'enforce_sfc': True,
            'enforce_market_clearing': True,
            'bilateral_weight': 0.3,
            'error_variance_ratio': 0.01,
            'normalize_data': True,
            'transformation': 'square',
            'output_dir': 'output'
        }
        
        # Load from file if exists
        config_path = Path(self.config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if 'sfc' in file_config:
                    config.update(file_config['sfc'])
        
        # Apply run mode settings
        if self.run_mode == 'test':
            config['max_series'] = 50
            config['enforce_market_clearing'] = False
        elif self.run_mode == 'development':
            config['max_series'] = 200
        elif self.run_mode == 'production':
            config['max_series'] = 500
            
        return config
    
    def run(self):
        """Run the complete SFC analysis."""
        self.logger.info("="*60)
        self.logger.info(f"Z1 INTEGRATED SFC KALMAN FILTER ANALYSIS")
        self.logger.info(f"Mode: {self.run_mode}")
        self.logger.info("="*60)
        
        try:
            # Load data using Z1 infrastructure
            self._load_z1_data()
            self._load_fwtw_data()
            self._load_formulas()
            
            # Initialize and fit model
            self._initialize_model()
            self._fit_model()
            
            # Run filtering and validation
            self._run_filtering()
            self._validate_results()
            
            # Create outputs
            self._create_visualizations()
            self._export_results()
            
            self.logger.info("\n✓ Analysis completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_z1_data(self):
        """Load Z.1 data using existing infrastructure."""
        self.logger.info("\nLoading Z.1 data...")
        
        try:
            # Use Z1's existing data loader
            fed_loader = CachedFedDataLoader()
            z1_raw = fed_loader.load_single_source('Z1')
            
            if z1_raw is None:
                raise ValueError("Fed loader returned None")
            
            # The data from CachedFedDataLoader is already in wide format:
            # - Index is dates
            # - Columns are series codes
            self.logger.info(f"Loaded data shape: {z1_raw.shape}")
            
            # Filter for Z.1 series patterns (FL, FU, FR, FV, FA)
            z1_patterns = ['FL', 'FU', 'FR', 'FV', 'FA']
            z1_cols = []
            
            for col in z1_raw.columns:
                # Convert to string and check if it starts with any Z1 pattern
                col_str = str(col)
                if any(col_str.startswith(pattern) for pattern in z1_patterns):
                    z1_cols.append(col)
            
            if z1_cols:
                self.data = z1_raw[z1_cols].copy()
                self.logger.info(f"Filtered to {len(z1_cols)} Z.1 series")
            else:
                # If no Z1 pattern columns found, use all data
                self.logger.warning("No standard Z.1 series patterns found, using all columns")
                self.data = z1_raw.copy()
            
            # Remove any columns with all NaN values
            self.data = self.data.dropna(axis=1, how='all')
            
            # Forward fill missing values
            self.data = self.data.ffill()
            
            # Select series based on configuration
            if len(self.data.columns) > self.config['max_series']:
                self.data = self._select_priority_series(self.data)
            
            self.logger.info(f"Final dataset: {len(self.data.columns)} series")
            self.logger.info(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            self._log_series_composition()
            
        except Exception as e:
            self.logger.warning(f"Could not load Z.1 data: {e}")
            self.logger.info("Using sample data for demonstration")
            self.data = self._generate_sample_data()
    
    def _load_fwtw_data(self):
        """Load FWTW data using existing infrastructure."""
        self.logger.info("\nLoading FWTW data...")
        
        try:
            fwtw_loader = FWTWDataLoader()
            self.fwtw_data = fwtw_loader.load_fwtw_data()
            self.logger.info(f"Loaded {len(self.fwtw_data)} FWTW positions")
        except Exception as e:
            self.logger.warning(f"Could not load FWTW data: {e}")
            self.fwtw_data = None
    
    def _load_formulas(self):
        """Load Z.1 formulas if available."""
        formula_path = Path('data/fof_formulas_extracted.json')
        if formula_path.exists():
            with open(formula_path, 'r') as f:
                formula_data = json.load(f)
                self.formulas = formula_data.get('formulas', {})
            self.logger.info(f"Loaded {len(self.formulas)} formulas")
        else:
            self.logger.warning("Formula file not found")
            self.formulas = {}
    
    def _process_to_timeseries(self, z1_raw: pd.DataFrame) -> pd.DataFrame:
        """Convert raw Z.1 data to time series format."""
        # Filter quarterly series
        quarterly = z1_raw[z1_raw['SERIES_NAME'].str.endswith('.Q')].copy()
        quarterly['SERIES_NAME'] = quarterly['SERIES_NAME'].str.replace('.Q', '', regex=False)
        
        # Pivot to wide format
        data = quarterly.pivot(
            index='date',
            columns='SERIES_NAME',
            values='value'
        )
        
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        data = data.ffill()
        
        return data
    
    def _select_priority_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select priority series based on Z.1 structure."""
        selected = []
        max_series = self.config['max_series']
        columns_set = set(data.columns)
        
        # Priority 1: Complete FL-FU pairs (stock-flow pairs)
        fl_series = [col for col in data.columns if str(col).startswith('FL')]
        
        for fl_col in fl_series:
            if len(selected) >= max_series:
                break
                
            fl_str = str(fl_col)
            # Extract sector and instrument from FL series
            if len(fl_str) >= 9:
                sector = fl_str[2:4]
                instrument = fl_str[4:9]
                
                # Look for matching FU (transaction flow)
                fu_pattern = f"FU{sector}{instrument}"
                fu_matches = [col for col in columns_set 
                             if str(col).startswith(fu_pattern)]
                
                if fu_matches:
                    # Add stock and flow
                    if fl_col not in selected:
                        selected.append(fl_col)
                    if fu_matches[0] not in selected:
                        selected.append(fu_matches[0])
                    
                    # Also look for FR (revaluation) and FV (other changes)
                    fr_pattern = f"FR{sector}{instrument}"
                    fv_pattern = f"FV{sector}{instrument}"
                    
                    fr_matches = [col for col in columns_set 
                                 if str(col).startswith(fr_pattern)]
                    fv_matches = [col for col in columns_set 
                                 if str(col).startswith(fv_pattern)]
                    
                    if fr_matches and fr_matches[0] not in selected:
                        selected.append(fr_matches[0])
                    if fv_matches and fv_matches[0] not in selected:
                        selected.append(fv_matches[0])
        
        # Priority 2: Important sectors (if room remains)
        important_sectors = ['10', '15', '26', '31', '70', '89', '90']
        
        for col in data.columns:
            if len(selected) >= max_series:
                break
                
            col_str = str(col)
            if col not in selected and len(col_str) >= 4:
                sector = col_str[2:4]
                if sector in important_sectors:
                    selected.append(col)
        
        # Priority 3: FA series (seasonally adjusted flows)
        for col in data.columns:
            if len(selected) >= max_series:
                break
                
            if str(col).startswith('FA') and col not in selected:
                selected.append(col)
        
        # Priority 4: Fill remaining with any series
        for col in data.columns:
            if len(selected) >= max_series:
                break
                
            if col not in selected:
                selected.append(col)
        
        # Remove duplicates while preserving order
        selected = list(dict.fromkeys(selected))[:max_series]
        
        self.logger.info(f"Selected {len(selected)} priority series")
        
        return data[selected]
    
    def _log_series_composition(self):
        """Log the composition of selected series."""
        composition = {
            'FL (stocks/levels)': 0,
            'FU (transactions)': 0,
            'FR (revaluations)': 0,
            'FV (other changes)': 0,
            'FA (seasonal flows)': 0,
            'Other': 0
        }
        
        for col in self.data.columns:
            col_str = str(col)
            if col_str.startswith('FL'):
                composition['FL (stocks/levels)'] += 1
            elif col_str.startswith('FU'):
                composition['FU (transactions)'] += 1
            elif col_str.startswith('FR'):
                composition['FR (revaluations)'] += 1
            elif col_str.startswith('FV'):
                composition['FV (other changes)'] += 1
            elif col_str.startswith('FA'):
                composition['FA (seasonal flows)'] += 1
            else:
                composition['Other'] += 1
        
        self.logger.info("Series composition:")
        for series_type, count in composition.items():
            if count > 0:
                self.logger.info(f"  {series_type}: {count}")
        
        # Log some stock-flow pairs found
        pairs_found = 0
        for col in self.data.columns:
            col_str = str(col)
            if col_str.startswith('FL') and len(col_str) >= 9:
                sector = col_str[2:4]
                instrument = col_str[4:9]
                fu_series = f"FU{sector}{instrument}"
                if any(str(c).startswith(fu_series) for c in self.data.columns):
                    pairs_found += 1
        
        self.logger.info(f"  Stock-flow pairs: {pairs_found}")
    
    def _generate_sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2010-01-01', periods=40, freq='QE')
        
        # Generate consistent FL-FU pairs
        n_pairs = 5
        columns = []
        data_arrays = []
        
        for i in range(n_pairs):
            sector = f"{10 + i:02d}"
            instrument = "30641"
            
            # Generate stock (FL)
            stock = np.cumsum(np.random.randn(40)) * 100 + 1000
            columns.append(f"FL{sector}{instrument}05")
            data_arrays.append(stock)
            
            # Generate flow (FU) consistent with stock
            flow = np.diff(stock, prepend=stock[0])
            flow += np.random.randn(40) * 5
            columns.append(f"FU{sector}{instrument}05")
            data_arrays.append(flow)
        
        return pd.DataFrame(
            np.column_stack(data_arrays),
            index=dates,
            columns=columns
        )
    
    def _initialize_model(self):
        """Initialize the SFC Kalman filter model."""
        self.logger.info("\nInitializing SFC Kalman filter...")
        
        self.model = SFCKalmanFilter(
            data=self.data,
            formulas=self.formulas,
            fwtw_data=self.fwtw_data,
            enforce_sfc=self.config['enforce_sfc'],
            enforce_market_clearing=self.config['enforce_market_clearing'],
            bilateral_weight=self.config['bilateral_weight'],
            error_variance_ratio=self.config['error_variance_ratio'],
            normalize_data=self.config['normalize_data'],
            transformation=self.config['transformation']
        )
        
        diagnostics = self.model.get_sfc_diagnostics()
        self.logger.info(f"Model initialized with {diagnostics['n_stock_flow_pairs']} stock-flow pairs")
    
    def _fit_model(self):
        """Fit the model parameters."""
        self.logger.info("\nFitting model parameters...")
        
        self.fitted_results = self.model.fit(
            start_params=self.model.start_params,
            method='lbfgs',
            maxiter=1000,
            disp=False
        )
        
        self.logger.info(f"Model fitted successfully:")
        self.logger.info(f"  Log-likelihood: {self.fitted_results.llf:.2f}")
        self.logger.info(f"  Parameters: {len(self.fitted_results.params)}")
    
    def _run_filtering(self):
        """Run the Kalman filter with SFC constraints."""
        self.logger.info("\nRunning Kalman filter with SFC constraints...")
        
        self.filter_results = self.model.filter(self.fitted_results.params)
        self.filtered_series = self.model.get_filtered_series(self.filter_results)
        
        # Save results
        self.results.add_dataframe(
            'filtered_series',
            self.filtered_series['filtered'],
            subdir='output'
        )
    
    def _validate_results(self):
        """Validate SFC constraints are satisfied."""
        self.logger.info("\nValidating SFC constraints...")
        
        # Check stock-flow consistency
        violations = []
        for pair in self.model.stock_flow_pairs:
            if pair.stock_series in self.filtered_series['smoothed'].columns:
                stock = self.filtered_series['smoothed'][pair.stock_series]
                if pair.flow_series in self.filtered_series['smoothed'].columns:
                    flow = self.filtered_series['smoothed'][pair.flow_series]
                    violation = (stock.diff() - flow).abs().mean()
                    violations.append(violation)
        
        if violations:
            mean_violation = np.mean(violations)
            max_violation = np.max(violations)
            self.logger.info(f"  Stock-flow violations: mean={mean_violation:.6f}, max={max_violation:.6f}")
    
    def _create_visualizations(self):
        """Create visualization plots."""
        self.logger.info("\nCreating visualizations...")
        
        # Implementation depends on your needs
        # Could use VisualizationManager from kalman_filter_analysis
        pass
    
    def _export_results(self):
        """Export results and create summary."""
        self.logger.info("\nExporting results...")
        
        summary = {
            'run_mode': self.run_mode,
            'n_series': len(self.data.columns),
            'n_observations': len(self.data),
            'n_stock_flow_pairs': len(self.model.stock_flow_pairs),
            'log_likelihood': float(self.fitted_results.llf)
        }
        
        self.results.export_key_results(summary)
        self.logger.info(f"Results saved to: {self.results.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Z1 Integrated SFC Analysis')
    parser.add_argument('mode', nargs='?', default='development',
                       choices=['test', 'development', 'production'],
                       help='Run mode')
    parser.add_argument('--config', type=str, help='Config file path')
    
    args = parser.parse_args()
    
    analysis = IntegratedSFCAnalysis(
        config_path=args.config,
        run_mode=args.mode
    )
    
    success = analysis.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
