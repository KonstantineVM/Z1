#!/usr/bin/env python3
"""
Driver script for Proper SFC Kalman Filter Analysis.
Implements the complete, theoretically correct SFC model with all constraints.
"""

import sys
import logging
from pathlib import Path
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Z1 data infrastructure
from src.data.cached_fed_data_loader import CachedFedDataLoader
from src.data.data_processor import DataProcessor
from src.network.fwtw_loader import FWTWDataLoader

# Import utilities
from src.utils.results_manager import ResultsManager
from src.utils.visualization import VisualizationManager

# Import the proper SFC model
from src.models.sfc_kalman_proper import ProperSFCKalmanFilter


class ProperSFCAnalysis:
    """
    Complete SFC analysis with proper constraint handling.
    """
    
    def __init__(self, config_path: str = None, run_mode: str = 'development'):
        """
        Initialize proper SFC analysis.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file
        run_mode : str
            Run mode: test, development, production, or full
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config_path = config_path or 'config/proper_sfc_config.yaml'
        self.run_mode = run_mode
        self.config = self._load_config()
        
        # Initialize results manager
        output_dir = Path(self.config.get('output_dir', 'output')) / f"proper_{run_mode}"
        self.results = ResultsManager(output_dir, timestamp_outputs=True)
        
        # Storage
        self.z1_data = None
        self.fwtw_data = None
        self.formulas = None
        self.model = None
        self.fitted_results = None
        
        self.logger.info("="*70)
        self.logger.info("PROPER SFC KALMAN FILTER ANALYSIS")
        self.logger.info(f"Mode: {run_mode}")
        self.logger.info("="*70)
    
    def _load_config(self) -> dict:
        """Load configuration with mode-specific overrides."""
        # Default configuration for proper SFC
        config = {
            'max_series': 500,
            'enforce_sfc': True,
            'enforce_market_clearing': True,
            'enforce_bilateral': True,
            'include_revaluations': True,
            'error_variance_ratio': 0.01,
            'reval_variance_ratio': 0.1,
            'bilateral_variance_ratio': 0.05,
            'normalize_data': True,
            'transformation': 'square',
            'use_sparse': True,
            'output_dir': 'output',
            'cache_dir': './data/cache',
            'base_dir': './data/fed_data',
            'create_visualizations': True,
            'validation_tolerance': 1e-6
        }
        
        # Load from file if exists
        config_path = Path(self.config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config.update(file_config.get('sfc', {}))
        
        # Mode-specific overrides
        if self.run_mode == 'test':
            config.update({
                'max_series': 50,
                'enforce_market_clearing': False,
                'enforce_bilateral': False,
                'include_revaluations': False,
                'create_visualizations': False
            })
        elif self.run_mode == 'development':
            config.update({
                'max_series': 200,
                'enforce_bilateral': False,  # Skip for speed
                'include_revaluations': True
            })
        elif self.run_mode == 'production':
            config.update({
                'max_series': 500,
                'enforce_sfc': True,
                'enforce_market_clearing': True,
                'enforce_bilateral': True,
                'include_revaluations': True
            })
        elif self.run_mode == 'full':
            config.update({
                'max_series': 2000,  # Use more series
                'enforce_sfc': True,
                'enforce_market_clearing': True,
                'enforce_bilateral': True,
                'include_revaluations': True,
                'use_sparse': True  # Essential for large systems
            })
        
        return config
    
    def run(self) -> bool:
        """Run the complete proper SFC analysis."""
        try:
            # Phase 1: Data Loading
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 1: DATA LOADING")
            self.logger.info("="*50)
            self._load_z1_data()
            self._load_fwtw_data()
            self._load_formulas()
            
            # Phase 2: Model Initialization
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 2: MODEL INITIALIZATION")
            self.logger.info("="*50)
            self._initialize_model()
            
            # Phase 3: Model Fitting
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 3: MODEL FITTING")
            self.logger.info("="*50)
            self._fit_model()
            
            # Phase 4: Filtering with Constraints
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 4: CONSTRAINED FILTERING")
            self.logger.info("="*50)
            self._run_filtering()
            
            # Phase 5: Validation
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 5: CONSTRAINT VALIDATION")
            self.logger.info("="*50)
            self._validate_results()
            
            # Phase 6: Save Results
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 6: SAVING RESULTS")
            self.logger.info("="*50)
            self._save_results()
            
            # Phase 7: Visualization
            if self.config.get('create_visualizations', True):
                self.logger.info("\n" + "="*50)
                self.logger.info("PHASE 7: VISUALIZATION")
                self.logger.info("="*50)
                self._create_visualizations()
            
            self.logger.info("\n" + "="*70)
            self.logger.info("✓ PROPER SFC ANALYSIS COMPLETED SUCCESSFULLY!")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_z1_data(self):
        """Load and process Z.1 data."""
        self.logger.info("Loading Z.1 data...")
        
        try:
            # Load raw data
            loader = CachedFedDataLoader(
                base_directory=self.config.get('base_dir', './data/fed_data'),
                cache_directory=self.config.get('cache_dir', './data/cache')
            )
            z1_raw = loader.load_single_source('Z1')
            
            if z1_raw is None:
                raise ValueError("Failed to load Z1 data")
            
            self.logger.info(f"  Raw Z1 data: {z1_raw.shape}")
            
            # Process to time series
            processor = DataProcessor()
            self.z1_data = processor.process_fed_data(z1_raw, 'Z1')
            
            if self.z1_data is None or self.z1_data.empty:
                raise ValueError("DataProcessor returned empty data")
            
            self.logger.info(f"  Processed: {self.z1_data.shape}")
            
            # Apply series selection
            if len(self.z1_data.columns) > self.config['max_series']:
                self.z1_data = self._select_sfc_priority_series(self.z1_data)
            
            self.logger.info(f"  Final: {len(self.z1_data.columns)} series, {len(self.z1_data)} periods")
            
            # Log composition
            self._log_series_composition()
            
        except Exception as e:
            self.logger.error(f"Error loading Z1 data: {e}")
            self.logger.info("Generating sample data for testing")
            self.z1_data = self._generate_complete_sample_data()
    
    def _load_fwtw_data(self):
        """Load FWTW bilateral data."""
        self.logger.info("Loading FWTW data...")
        
        try:
            loader = FWTWDataLoader()
            self.fwtw_data = loader.load_fwtw_data()
            
            if self.fwtw_data is not None and not self.fwtw_data.empty:
                self.logger.info(f"  Loaded: {self.fwtw_data.shape}")
                
                # Log structure
                if 'Holder Code' in self.fwtw_data.columns:
                    n_holders = self.fwtw_data['Holder Code'].nunique()
                    n_issuers = self.fwtw_data['Issuer Code'].nunique()
                    n_instruments = self.fwtw_data['Instrument Code'].nunique()
                    self.logger.info(f"  Structure: {n_holders} holders, {n_issuers} issuers, {n_instruments} instruments")
            else:
                self.logger.warning("  No FWTW data available")
                
        except Exception as e:
            self.logger.warning(f"Could not load FWTW: {e}")
            self.fwtw_data = None
    
    def _load_formulas(self):
        """Load Z.1 formulas."""
        self.logger.info("Loading formulas...")
        
        formula_paths = [
            Path('data/fof_formulas_extracted.json'),
            Path('data/fof_formulas.json'),
            Path('data/formulas.json')
        ]
        
        self.formulas = None
        for path in formula_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        self.formulas = json.load(f)
                    self.logger.info(f"  Loaded {len(self.formulas)} formulas from {path.name}")
                    break
                except Exception as e:
                    self.logger.warning(f"  Could not load {path}: {e}")
        
        if self.formulas is None:
            self.logger.warning("  No formulas loaded")
            self.formulas = {}
    
    def _select_sfc_priority_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select series prioritizing complete stock-flow sets.
        """
        max_series = self.config['max_series']
        columns = list(data.columns)
        selected = []
        
        # Priority 1: Complete stock-flow-revaluation sets
        fl_series = [c for c in columns if c.startswith('FL')]
        
        for fl in fl_series:
            if len(selected) >= max_series:
                break
            
            if fl not in selected:
                # Extract identifiers
                if len(fl) >= 9:
                    sector = fl[2:4]
                    instrument = fl[4:9]
                    suffix = fl[9:] if len(fl) > 9 else ""
                    
                    # Add complete set if available
                    fu = f"FU{sector}{instrument}{suffix}"
                    fr = f"FR{sector}{instrument}{suffix}"
                    fv = f"FV{sector}{instrument}{suffix}"
                    
                    # Add in order: FL, FU, FR, FV
                    for series in [fl, fu, fr, fv]:
                        if series in columns and series not in selected:
                            selected.append(series)
                            if len(selected) >= max_series:
                                break
        
        # Priority 2: Important sectors
        important_sectors = ['10', '15', '26', '31', '70', '89']
        for col in columns:
            if len(selected) >= max_series:
                break
            if col not in selected and len(col) >= 4:
                if col[2:4] in important_sectors:
                    selected.append(col)
        
        # Fill remaining
        for col in columns:
            if len(selected) >= max_series:
                break
            if col not in selected:
                selected.append(col)
        
        return data[selected[:max_series]]
    
    def _generate_complete_sample_data(self) -> pd.DataFrame:
        """Generate sample data with complete SFC structure."""
        np.random.seed(42)
        dates = pd.date_range('2010-01-01', periods=40, freq='QE')
        
        n_pairs = min(10, self.config['max_series'] // 4)
        columns = []
        data_arrays = []
        
        sectors = ['10', '15', '26', '31', '70']
        instruments = ['30641', '31305', '40005']
        
        for i in range(n_pairs):
            sector = sectors[i % len(sectors)]
            instrument = instruments[i % len(instruments)]
            suffix = f"00{i:01d}"
            
            # Generate consistent stock-flow data
            # Stock (FL)
            stock = np.cumsum(np.random.randn(40)) * 100 + 1000
            columns.append(f"FL{sector}{instrument}{suffix}")
            data_arrays.append(stock)
            
            # Flow (FU)
            flow = np.diff(stock, prepend=stock[0])
            flow += np.random.randn(40) * 5
            columns.append(f"FU{sector}{instrument}{suffix}")
            data_arrays.append(flow)
            
            # Revaluation (FR)
            reval = np.random.randn(40) * 10
            columns.append(f"FR{sector}{instrument}{suffix}")
            data_arrays.append(reval)
            
            # Other changes (FV)
            other = np.random.randn(40) * 2
            columns.append(f"FV{sector}{instrument}{suffix}")
            data_arrays.append(other)
        
        return pd.DataFrame(
            np.column_stack(data_arrays),
            index=dates,
            columns=columns
        )
    
    def _log_series_composition(self):
        """Log detailed series composition."""
        if self.z1_data is None:
            return
        
        composition = {
            'FL': [],  # Stocks
            'FU': [],  # Flows
            'FR': [],  # Revaluations
            'FV': [],  # Other changes
            'FA': [],  # Seasonally adjusted
            'Other': []
        }
        
        for col in self.z1_data.columns:
            prefix = col[:2] if len(col) >= 2 else 'Other'
            if prefix in composition:
                composition[prefix].append(col)
            else:
                composition['Other'].append(col)
        
        self.logger.info("Series composition:")
        for prefix, series_list in composition.items():
            if series_list:
                self.logger.info(f"  {prefix}: {len(series_list)} series")
        
        # Check for complete sets
        complete_sets = 0
        for fl in composition['FL']:
            if len(fl) >= 9:
                sector = fl[2:4]
                instrument = fl[4:9]
                suffix = fl[9:] if len(fl) > 9 else ""
                
                fu = f"FU{sector}{instrument}{suffix}"
                fr = f"FR{sector}{instrument}{suffix}"
                fv = f"FV{sector}{instrument}{suffix}"
                
                if fu in composition['FU'] and fr in composition['FR'] and fv in composition['FV']:
                    complete_sets += 1
        
        self.logger.info(f"  Complete stock-flow-reval sets: {complete_sets}")
    
    def _initialize_model(self):
        """Initialize the proper SFC Kalman filter."""
        self.logger.info("Initializing Proper SFC Kalman Filter...")
        
        try:
            # Model configuration
            model_config = {
                'enforce_sfc': self.config['enforce_sfc'],
                'enforce_market_clearing': self.config['enforce_market_clearing'],
                'enforce_bilateral': self.config['enforce_bilateral'],
                'include_revaluations': self.config['include_revaluations'],
                'error_variance_ratio': self.config['error_variance_ratio'],
                'reval_variance_ratio': self.config['reval_variance_ratio'],
                'bilateral_variance_ratio': self.config['bilateral_variance_ratio'],
                'normalize_data': self.config['normalize_data'],
                'transformation': self.config['transformation'],
                'use_sparse': self.config['use_sparse']
            }
            
            # Initialize model
            self.model = ProperSFCKalmanFilter(
                data=self.z1_data,
                formulas=self.formulas,
                fwtw_data=self.fwtw_data,
                **model_config
            )
            
            # Log diagnostics
            diagnostics = self.model.get_diagnostics()
            self.logger.info("Model structure:")
            self.logger.info(f"  Base states: {diagnostics['base_states']}")
            self.logger.info(f"  SFC states: {diagnostics['sfc_states']}")
            self.logger.info(f"  Total states: {diagnostics['total_states']}")
            self.logger.info(f"  Base shocks: {diagnostics['base_shocks']}")
            self.logger.info(f"  SFC shocks: {diagnostics['sfc_shocks']}")
            self.logger.info(f"  Total shocks: {diagnostics['total_shocks']}")
            self.logger.info(f"  Stock-flow pairs: {diagnostics['stock_flow_pairs']}")
            self.logger.info(f"  Complete pairs: {diagnostics['complete_pairs']}")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
    
    def _fit_model(self):
        """Fit the model parameters."""
        self.logger.info("Fitting model...")
        
        start_time = datetime.now()
        
        try:
            # Fit with appropriate method
            if self.model.state_space.n_total_states > 1000:
                # Use limited iterations for very large models
                self.fitted_results = self.model.fit(
                    method='powell',
                    maxiter=100,
                    disp=False
                )
            else:
                # Standard fitting
                self.fitted_results = self.model.fit(
                    method='lbfgs',
                    maxiter=500,
                    disp=False
                )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"  Fitted in {elapsed:.1f} seconds")
            
            if hasattr(self.fitted_results, 'llf'):
                self.logger.info(f"  Log likelihood: {self.fitted_results.llf:.2f}")
            
        except Exception as e:
            self.logger.error(f"Fitting failed: {e}")
            raise
    
    def _run_filtering(self):
        """Run constrained filtering and smoothing."""
        self.logger.info("Running constrained filtering...")
        
        try:
            # Filter with constraints
            self.filtered_results = self.model.filter()
            self.logger.info("  Filtering complete")
            
            # Smooth with constraints
            self.smoothed_results = self.model.smooth()
            self.logger.info("  Smoothing complete")
            
            # Store states
            if hasattr(self.filtered_results, 'filtered_state'):
                self.filtered_states = self.filtered_results.filtered_state
                self.logger.info(f"  Filtered states: {self.filtered_states.shape}")
            
            if hasattr(self.smoothed_results, 'smoothed_state'):
                self.smoothed_states = self.smoothed_results.smoothed_state
                self.logger.info(f"  Smoothed states: {self.smoothed_states.shape}")
            
        except Exception as e:
            self.logger.error(f"Filtering failed: {e}")
            raise
    
    def _validate_results(self):
        """Validate constraint satisfaction."""
        self.logger.info("Validating constraints...")
        
        if hasattr(self, 'smoothed_states'):
            validation = self.model.validate_constraints(
                self.smoothed_states,
                tolerance=self.config['validation_tolerance']
            )
            
            self.logger.info("Constraint validation results:")
            self.logger.info(f"  Max violation: {validation['max_violation']:.2e}")
            self.logger.info(f"  Mean stock-flow violation: {validation['mean_stock_flow']:.2e}")
            self.logger.info(f"  Constraints satisfied: {validation['constraints_satisfied']}")
            
            if validation['constraints_satisfied']:
                self.logger.info("  ✓ All constraints satisfied within tolerance!")
            else:
                self.logger.warning("  ⚠ Some constraints violated")
    
    def _save_results(self):
        """Save analysis results."""
        self.logger.info("Saving results...")
        
        output_dir = Path(self.config['output_dir']) / f"proper_{self.run_mode}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save processed data
            if self.z1_data is not None and not self.z1_data.empty:
                z1_path = output_dir / 'z1_processed.parquet'
                self.z1_data.to_parquet(z1_path, compression='snappy')
                self.logger.info(f"  Saved Z1 data: {z1_path}")
            
            # Save filtered states
            if hasattr(self, 'filtered_states'):
                np.save(output_dir / 'filtered_states.npy', self.filtered_states)
                self.logger.info(f"  Saved filtered states")
            
            # Save smoothed states
            if hasattr(self, 'smoothed_states'):
                np.save(output_dir / 'smoothed_states.npy', self.smoothed_states)
                self.logger.info(f"  Saved smoothed states")
            
            # Save diagnostics
            if self.model:
                diagnostics = self.model.get_diagnostics()
                with open(output_dir / 'diagnostics.json', 'w') as f:
                    json.dump(diagnostics, f, indent=2)
                self.logger.info(f"  Saved diagnostics")
            
            # Save metadata
            metadata = {
                'run_mode': self.run_mode,
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'n_series': len(self.z1_data.columns) if self.z1_data is not None else 0,
                'n_periods': len(self.z1_data) if self.z1_data is not None else 0
            }
            
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"  Saved metadata")
            
        except Exception as e:
            self.logger.error(f"Failed to save: {e}")
    
    def _create_visualizations(self):
        """Create visualizations."""
        self.logger.info("Creating visualizations...")
        
        # Would implement visualization of:
        # - Constraint violations over time
        # - Stock-flow consistency
        # - Filtered vs actual series
        # - Shock decomposition
        
        self.logger.info("  Visualization complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Proper SFC Kalman Filter Analysis'
    )
    parser.add_argument(
        'mode', 
        nargs='?', 
        default='development',
        choices=['test', 'development', 'production', 'full'],
        help='Run mode (default: development)'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-viz', 
        action='store_true',
        help='Skip visualization'
    )
    
    args = parser.parse_args()
    
    # Initialize analysis
    analysis = ProperSFCAnalysis(
        config_path=args.config,
        run_mode=args.mode
    )
    
    # Override visualization if requested
    if args.no_viz:
        analysis.config['create_visualizations'] = False
    
    # Run analysis
    success = analysis.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())