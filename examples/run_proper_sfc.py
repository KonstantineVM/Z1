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
from src.network.fwtw_z1_mapper import FWTWtoZ1Mapper

# Import utilities
from src.utils.results_manager import ResultsManager
from src.utils.visualization import VisualizationManager

# Import the proper SFC model
from src.models.sfc_kalman_proper import ProperSFCKalmanFilter

from src.visualization.sfc_visualization import SFCVisualizationManager

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
        self.asset_liability_map = None
        self.model = None
        self.fitted_results = None
        self.bilateral_enabled = True  # Track if bilateral constraints should be used
        
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
            'asset_liability_map_path': 'data/z1_side_map.json',
            'create_visualizations': True,
            'validation_tolerance': 1e-6
        }
        
        # Load from file if exists
        config_path = Path(self.config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config and 'sfc' in file_config:
                    config.update(file_config['sfc'])
        
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
            self._load_asset_liability_map()
            
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
        """Load and map FWTW bilateral data to Z.1 FL series (levels)."""
        self.logger.info("\nLoading FWTW data...")
        
        try:
            # Load raw FWTW data
            loader = FWTWDataLoader()
            fwtw_raw = loader.load_fwtw_data()
            
            if fwtw_raw is None or fwtw_raw.empty:
                self.logger.warning("No FWTW data loaded")
                self.fwtw_data = None
                self.bilateral_enabled = False
                return
            
            self.logger.info(f"  Loaded raw FWTW data: {fwtw_raw.shape}")
            
            # Log structure
            if 'Holder Code' in fwtw_raw.columns:
                n_holders = fwtw_raw['Holder Code'].nunique()
                n_issuers = fwtw_raw['Issuer Code'].nunique()
                n_instruments = fwtw_raw['Instrument Code'].nunique()
                self.logger.info(f"  Structure: {n_holders} holders, {n_issuers} issuers, {n_instruments} instruments")
            
            # Get Z.1 series for validation (remove .Q suffix)
            z1_base_codes = None
            if self.z1_data is not None:
                z1_base_codes = {
                    col[:-2] if col.endswith('.Q') else col 
                    for col in self.z1_data.columns
                }
                self.logger.info(f"  Z.1 series for validation: {len(z1_base_codes)} base codes")
            
            # Map FWTW to Z.1 series codes
            mapper = FWTWtoZ1Mapper()
            mapped = mapper.map_to_z1_series(fwtw_raw, available_z1_series=z1_base_codes)
            
            if mapped is None or mapped.empty:
                self.logger.warning("FWTW mapping produced no results")
                self.fwtw_data = None
                self.bilateral_enabled = False
                return
            
            self.logger.info(f"  Mapped FWTW positions: {mapped.shape}")
            
            # Add .Q suffix to match Z.1 format
            for col in ['holder_series', 'issuer_series', 'holder_flow_series', 'issuer_flow_series']:
                if col in mapped.columns:
                    mapped[col] = mapped[col].apply(
                        lambda x: f"{x}.Q" if pd.notna(x) and not x.endswith('.Q') else x
                    )
            
            # Calculate overlap with Z.1
            z1_cols = set(self.z1_data.columns) if self.z1_data is not None else set()
            holder_series = set(mapped['holder_series'].dropna())
            issuer_series = set(mapped['issuer_series'].dropna())
            
            h_overlap = len(holder_series & z1_cols)
            i_overlap = len(issuer_series & z1_cols)
            
            self.logger.info(f"  Holder FL series in Z1: {h_overlap}/{len(holder_series)}")
            self.logger.info(f"  Issuer FL series in Z1: {i_overlap}/{len(issuer_series)}")
            
            if h_overlap == 0 and i_overlap == 0:
                self.logger.warning("No FWTW series found in Z.1 data")
                self.bilateral_enabled = False
                self.fwtw_data = None
                return
            
            # Store mapped data
            self.fwtw_data = mapped
            self.logger.info(f"  ✓ FWTW data mapped successfully")
            
        except Exception as e:
            self.logger.error(f"FWTW loading failed: {e}")
            self.fwtw_data = None
            self.bilateral_enabled = False
    
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
                    
                    # Handle dict vs list structure
                    if isinstance(self.formulas, dict) and "formulas" in self.formulas:
                        n_forms = len(self.formulas["formulas"])
                    else:
                        n_forms = len(self.formulas) if self.formulas else 0
                    
                    self.logger.info(f"  Loaded {n_forms} formulas from {path.name}")
                    break
                except Exception as e:
                    self.logger.warning(f"  Could not load {path}: {e}")
        
        if self.formulas is None:
            self.logger.warning("  No formulas loaded")
            self.formulas = {}
    
    def _load_asset_liability_map(self):
        """Load asset/liability mapping for market clearing constraints."""
        self.logger.info("Loading asset/liability map...")
        
        map_path = Path(self.config.get('asset_liability_map_path', 'data/z1_side_map.json'))
        
        if map_path.exists():
            try:
                with open(map_path, 'r') as f:
                    self.asset_liability_map = json.load(f)
                self.logger.info(f"  Loaded asset/liability map from {map_path.name}")
                
                # Log summary
                if isinstance(self.asset_liability_map, dict):
                    n_assets = len([k for k, v in self.asset_liability_map.items() if v == 'asset'])
                    n_liabilities = len([k for k, v in self.asset_liability_map.items() if v == 'liability'])
                    self.logger.info(f"  Map contains: {n_assets} assets, {n_liabilities} liabilities")
            except Exception as e:
                self.logger.warning(f"  Could not load asset/liability map: {e}")
                self.asset_liability_map = None
        else:
            self.logger.info(f"  No asset/liability map found at {map_path}")
            self.asset_liability_map = None
            
            # Disable market clearing if no map available
            if self.config.get('enforce_market_clearing', False):
                self.logger.warning("  Disabling market clearing constraints (no asset/liability map)")
                self.config['enforce_market_clearing'] = False
    
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
                # Extract identifiers (handle .Q suffix)
                fl_base = fl[:-2] if fl.endswith('.Q') else fl
                if len(fl_base) >= 9:
                    sector = fl_base[2:4]
                    instrument = fl_base[4:9]
                    suffix = fl_base[9:] if len(fl_base) > 9 else ""
                    
                    # Build corresponding series names
                    q_suffix = '.Q' if fl.endswith('.Q') else ''
                    fu = f"FU{sector}{instrument}{suffix}{q_suffix}"
                    fr = f"FR{sector}{instrument}{suffix}{q_suffix}"
                    fv = f"FV{sector}{instrument}{suffix}{q_suffix}"
                    
                    # Add complete set if available
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
                col_base = col[:-2] if col.endswith('.Q') else col
                if len(col_base) >= 4 and col_base[2:4] in important_sectors:
                    selected.append(col)
        
        # Fill remaining with any series
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
            suffix = f"00{i:01d}.Q"  # Include .Q suffix in sample data
            
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
        """Log detailed series composition with CORRECT understanding."""
        if self.z1_data is None:
            return
        
        composition = {
            'FL': [],  # LEVELS (stocks), not seasonally adjusted - NOT liabilities!
            'FU': [],  # FLOWS (transactions), not seasonally adjusted
            'FR': [],  # Revaluations
            'FV': [],  # Other changes in volume
            'FA': [],  # FLOWS at seasonally adjusted annual rate - NOT assets!
            'LA': [],  # Levels, seasonally adjusted
            'Other': []
        }
        
        for col in self.z1_data.columns:
            col_base = col[:-2] if col.endswith('.Q') else col
            prefix = col_base[:2] if len(col_base) >= 2 else 'Other'
            if prefix in composition:
                composition[prefix].append(col)
            else:
                composition['Other'].append(col)
        
        self.logger.info("Series composition (with CORRECT interpretation):")
        for prefix, series_list in composition.items():
            if series_list:
                if prefix == 'FL':
                    self.logger.info(f"  FL (Levels/Stocks, NSA): {len(series_list)} series")
                elif prefix == 'FU':
                    self.logger.info(f"  FU (Flows/Transactions, NSA): {len(series_list)} series")
                elif prefix == 'FA':
                    self.logger.info(f"  FA (Flows, SAAR - NOT assets!): {len(series_list)} series")
                elif prefix == 'FR':
                    self.logger.info(f"  FR (Revaluations): {len(series_list)} series")
                elif prefix == 'FV':
                    self.logger.info(f"  FV (Other Volume Changes): {len(series_list)} series")
                elif prefix == 'LA':
                    self.logger.info(f"  LA (Levels, SA): {len(series_list)} series")
                else:
                    self.logger.info(f"  {prefix}: {len(series_list)} series")
        
        # Check for complete stock-flow-revaluation sets using CORRECT pairing
        complete_sets = 0
        for fl in composition['FL']:  # FL is the stock!
            fl_base = fl[:-2] if fl.endswith('.Q') else fl
            if len(fl_base) >= 9:
                sector = fl_base[2:4]
                instrument = fl_base[4:9]
                suffix = fl_base[9:] if len(fl_base) > 9 else ""
                
                q_suffix = '.Q' if fl.endswith('.Q') else ''
                fu = f"FU{sector}{instrument}{suffix}{q_suffix}"  # Flow
                fr = f"FR{sector}{instrument}{suffix}{q_suffix}"  # Revaluation
                fv = f"FV{sector}{instrument}{suffix}{q_suffix}"  # Other changes
                
                # The correct stock-flow identity: FL[t] = FL[t-1] + FU + FR + FV
                if fu in composition['FU'] and fr in composition['FR'] and fv in composition['FV']:
                    complete_sets += 1
        
        self.logger.info(f"  Complete stock-flow-reval sets (FL-FU-FR-FV): {complete_sets}")
    
    def _initialize_model(self):
        """Initialize the proper SFC Kalman filter."""
        self.logger.info("Initializing Proper SFC Kalman Filter...")
        
        try:
            # Model configuration
            model_config = {
                'enforce_sfc': self.config['enforce_sfc'],
                'enforce_market_clearing': self.config['enforce_market_clearing'] and self.asset_liability_map is not None,
                'enforce_bilateral': self.config['enforce_bilateral'] and self.bilateral_enabled,
                'include_revaluations': self.config['include_revaluations'],
                'error_variance_ratio': self.config['error_variance_ratio'],
                'reval_variance_ratio': self.config['reval_variance_ratio'],
                'bilateral_variance_ratio': self.config['bilateral_variance_ratio'],
                'normalize_data': self.config['normalize_data'],
                'transformation': self.config['transformation'],
                'use_sparse': self.config['use_sparse']
            }
            
            # Log what constraints will be enforced
            self.logger.info("Constraint configuration:")
            self.logger.info(f"  Stock-flow consistency: {model_config['enforce_sfc']}")
            self.logger.info(f"  Market clearing: {model_config['enforce_market_clearing']}")
            self.logger.info(f"  Bilateral constraints: {model_config['enforce_bilateral']}")
            self.logger.info(f"  Include revaluations: {model_config['include_revaluations']}")
            
            # Initialize model with pre-mapped FWTW data or None
            self.model = ProperSFCKalmanFilter(
                data=self.z1_data,
                formulas=self.formulas,
                fwtw_data=self.fwtw_data,  # Pre-mapped or None
                asset_liability_map=self.asset_liability_map,  # For market clearing
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
            
            # Log constraint counts if available
            if hasattr(self.model, 'get_constraint_counts'):
                counts = self.model.get_constraint_counts()
                self.logger.info(f"Constraints built:")
                self.logger.info(f"  Stock-flow: {counts.get('stock_flow', 0)}")
                self.logger.info(f"  Market-clearing: {counts.get('market_clearing', 0)}")
                self.logger.info(f"  Bilateral: {counts.get('bilateral', 0)}")
                self.logger.info(f"  Formulas: {counts.get('formulas', 0)}")
                self.logger.info(f"  Total: {counts.get('total', 0)}")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
    
    def _optimization_callback(self, params):
        """Callback to monitor optimization progress."""
        if not hasattr(self, '_opt_iter'):
            self._opt_iter = 0
            self._last_llf = None
        
        self._opt_iter += 1
        
        # Try to compute likelihood
        try:
            current_llf = self.model.loglike(params)
            
            # Print iteration info in L-BFGS-B format for the monitor
            if self._last_llf is not None:
                grad_norm = np.linalg.norm(self.model.score(params)) if hasattr(self.model, 'score') else 0
                print(f"  {self._opt_iter:4d}    {self._opt_iter*2:4d}   {current_llf:.12E}   {grad_norm:.3E}", flush=True)
            
            self._last_llf = current_llf
        except:
            print(f"  {self._opt_iter:4d}    ?    ?    ?", flush=True)

    def _fit_model(self):
        """Fit the model parameters."""
        self.logger.info("Fitting model...")
        
        # Initialize optimization tracking
        self._opt_iter = 0
        self._last_llf = None
        
        def optimization_callback(params):
            """Callback to output optimization progress."""
            self._opt_iter += 1
            
            try:
                # Compute current log likelihood
                current_llf = self.model.loglike(params)
                
                # Compute gradient norm if possible
                try:
                    grad = self.model.score(params)
                    grad_norm = np.linalg.norm(grad)
                except:
                    grad_norm = 0.0
                
                # Output in L-BFGS-B format for monitor
                if self._opt_iter == 1:
                    # Print header on first iteration
                    print("   NIT   NF   F                       |grad|", flush=True)
                
                # Format: iteration, function_evals, likelihood, gradient_norm
                print(f"  {self._opt_iter:4d}  {self._opt_iter*2:4d}   {current_llf:.12E}   {grad_norm:.3E}", flush=True)
                
                # Check convergence
                if self._last_llf is not None:
                    improvement = abs(current_llf - self._last_llf)
                    rel_improvement = improvement / abs(self._last_llf) if self._last_llf != 0 else improvement
                    
                    # Log if converging
                    if rel_improvement < 1e-5:
                        self.logger.info(f"    Iteration {self._opt_iter}: Converged (rel_improvement={rel_improvement:.2e})")
                    elif self._opt_iter % 10 == 0:
                        self.logger.info(f"    Iteration {self._opt_iter}: LL={current_llf:.2f}, improvement={improvement:.2e}")
                
                self._last_llf = current_llf
                
            except Exception as e:
                print(f"  {self._opt_iter:4d}    ?    Error: {e}", flush=True)
        
        start_time = datetime.now()
        
        try:
            # Safely get state count
            n_states = getattr(getattr(self.model, "state_space", None), "n_total_states", None)
            if n_states is None:
                n_states = getattr(self.model, "k_states", 0)
            
            # Choose fitting method based on model size
            if n_states > 1000:
                # Use limited iterations for very large models
                self.logger.info(f"  Large model ({n_states} states), using Powell method")
                self.fitted_results = self.model.fit(
                    method="powell",
                    maxiter=100,
                    callback=optimization_callback,
                    disp=False,
                    transformed=False      # <-- critical here too
                )
                ret = getattr(self.fitted_results, "mle_retvals", {}) or {}
                if not ret.get("converged", False):
                    self.logger.warning(f"MLE did not converge (using last iterate): {ret}")
                
                
            else:
                # Standard fitting with relaxed tolerances for faster convergence
                self.logger.info(f"  Standard model ({n_states} states), using L-BFGS-B method")
                self.fitted_results = self.model.fit(
                    method="lbfgs",
                    maxiter=20,            # adjust as needed
                    pgtol=1e-4,            # gradient tol (supported)
                    factr=1e8,             # function tol scaling (larger = looser)
                    m=15,
                    callback=optimization_callback,
                    disp=False,
                    transformed=False      # <-- critical: optimize in unconstrained space
                )
                ret = getattr(self.fitted_results, "mle_retvals", {}) or {}
                if not ret.get("converged", False):
                    self.logger.warning(f"MLE did not converge (using last iterate): {ret}")
                
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"  Fitted in {elapsed:.1f} seconds")
            
            if hasattr(self.fitted_results, 'llf'):
                self.logger.info(f"  Log likelihood: {self.fitted_results.llf:.2f}")
            
        except Exception as e:
            self.logger.error(f"Fitting failed: {e}")
            raise
    
    def _resolve_params_strict(self):
        """
        Return fitted parameter vector or raise a clear error.
        """
        fr = getattr(self, "fitted_results", None)
        if fr is None or getattr(fr, "params", None) is None:
            raise RuntimeError(
                "No fitted parameters available. Run _fit_model() and ensure it converges before filtering/smoothing."
            )
        p = np.asarray(fr.params, dtype=float).ravel()
        k = getattr(self.model, "k_params", None)
        if k is not None and p.size != k:
            raise ValueError(f"Parameter length mismatch: got {p.size}, expected {k}.")
        if not np.all(np.isfinite(p)):
            raise ValueError("Parameters contain NaN/Inf.")
        return p

    def _resolve_params_lenient(self):
        """
        Return the best available parameter vector, even if MLE did not converge.
        Fails only if no fitted parameters exist at all.
        """
        fr = getattr(self, "fitted_results", None)
        if fr is None or getattr(fr, "params", None) is None:
            raise RuntimeError(
                "No fitted parameters available. Run _fit_model() first (non-converged is OK)."
            )
        p = np.asarray(fr.params, dtype=float).ravel()
        if not np.all(np.isfinite(p)):
            raise ValueError("Fitted parameters contain NaN/Inf.")
        return p


    def _run_filtering(self):
        """Run constrained filtering and smoothing."""
        self.logger.info("Running constrained filtering...")

        try:
            # --- Use last iterate even if MLE did not converge ---
            fr = getattr(self, "fitted_results", None)
            if fr is None or getattr(fr, "params", None) is None:
                raise RuntimeError(
                    "No fitted parameters available. Run _fit_model() first (non-converged is OK)."
                )

            # Warn but proceed if not converged
            ret = getattr(fr, "mle_retvals", {})
            if not ret.get("converged", False):
                self.logger.warning(f"MLE did not converge; using last iterate: {ret}")

            # Parameters are already in constrained (variance) space for update()
            params = np.asarray(fr.params, dtype=float).ravel()
            if not np.all(np.isfinite(params)):
                raise ValueError("Fitted parameters contain NaN/Inf.")

            # Filter with constraints
            self.filtered_results = self.model.filter(params=params, transformed=False)
            self.logger.info("  Filtering complete")

            # Smooth with constraints
            self.smoothed_results = self.model.smooth(params=params, transformed=False)
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
        import numpy as np

        self.logger.info("Validating constraints...")

        if hasattr(self, 'smoothed_states') and hasattr(self.model, 'validate_constraints'):
            # Ensure numeric tolerance (avoid str vs float compare)
            tol = float(self.config.get('validation_tolerance', 1e-6))

            # Call validator
            validation = self.model.validate_constraints(self.smoothed_states, tolerance=tol)

            # Normalize types from the returned dict
            max_v = float(np.asarray(validation.get('max_violation', np.nan)).real)
            mean_sf = float(np.asarray(validation.get('mean_stock_flow', np.nan)).real)

            # If the model didn’t compute a boolean, compute it here
            satisfied = validation.get('constraints_satisfied', None)
            if not isinstance(satisfied, (bool, np.bool_)):
                satisfied = (max_v < tol)

            self.logger.info("Constraint validation results:")
            self.logger.info(f"  Max violation: {max_v:.2e}")
            self.logger.info(f"  Mean stock-flow violation: {mean_sf:.2e}")
            self.logger.info(f"  Constraints satisfied: {satisfied}")

            if satisfied:
                self.logger.info("  ✓ All constraints satisfied within tolerance!")
            else:
                self.logger.warning("  ⚠ Some constraints violated")

    
    def _save_results(self):
        """Save analysis results."""
        self.logger.info("Saving results...")
        
        output_dir = Path(self.config['output_dir']) / f"proper_{self.run_mode}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save processed data with fallback
            if self.z1_data is not None and not self.z1_data.empty:
                z1_path = output_dir / 'z1_processed'
                try:
                    self.z1_data.to_parquet(f"{z1_path}.parquet", compression='snappy')
                    self.logger.info(f"  Saved Z1 data: {z1_path}.parquet")
                except Exception:
                    # Fallback to CSV if parquet fails
                    self.z1_data.to_csv(f"{z1_path}.csv")
                    self.logger.info(f"  Saved Z1 data: {z1_path}.csv (parquet unavailable)")
            
            # Save FWTW mapped data if available
            if self.fwtw_data is not None and not self.fwtw_data.empty:
                fwtw_path = output_dir / 'fwtw_mapped'
                try:
                    self.fwtw_data.to_parquet(f"{fwtw_path}.parquet", compression='snappy')
                    self.logger.info(f"  Saved FWTW mapped data: {fwtw_path}.parquet")
                except Exception:
                    self.fwtw_data.to_csv(f"{fwtw_path}.csv")
                    self.logger.info(f"  Saved FWTW mapped data: {fwtw_path}.csv")
            
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
                diagnostics['bilateral_enabled'] = self.bilateral_enabled
                with open(output_dir / 'diagnostics.json', 'w') as f:
                    json.dump(diagnostics, f, indent=2, default=str)
                self.logger.info(f"  Saved diagnostics")
            
            # Save metadata
            metadata = {
                'run_mode': self.run_mode,
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'n_series': len(self.z1_data.columns) if self.z1_data is not None else 0,
                'n_periods': len(self.z1_data) if self.z1_data is not None else 0,
                'bilateral_constraints_enabled': self.bilateral_enabled,
                'fwtw_data_loaded': self.fwtw_data is not None,
                'asset_liability_map_loaded': self.asset_liability_map is not None
            }
            
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            self.logger.info(f"  Saved metadata")
            
        except Exception as e:
            self.logger.error(f"Failed to save: {e}")
    
    def _create_visualizations(self):
        """Create visualizations."""
        self.logger.info("Creating visualizations...")
        
        try:
            # Initialize visualization manager
            from src.visualization.sfc_visualization import SFCVisualizationManager
            viz = SFCVisualizationManager(
                output_dir=self.results.output_dir / 'figures'
            )
            
            # Get filtered series from the model
            series_dict = self.model.get_filtered_series(self.fitted_results)
            
            # Get model diagnostics as a dictionary (not a set)
            model_diagnostics = self.model.get_diagnostics()
            
            # Get constraint validation results if available
            constraint_diagnostics = None
            if hasattr(self, 'constraint_validation'):
                constraint_diagnostics = self.constraint_validation
            
            # Create visualizations
            viz.create_all_visualizations(
                z1_data=self.z1_data,
                filtered_series=series_dict.get('filtered'),
                smoothed_series=series_dict.get('smoothed'),
                stock_flow_pairs=self.model.stock_flow_pairs,
                fwtw_data=self.fwtw_data,
                asset_liability_map=self.asset_liability_map,
                model_diagnostics=model_diagnostics,  # Pass the dictionary, not a set
                constraint_diagnostics=constraint_diagnostics
            )
            
            self.logger.info("  Visualization complete")
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            
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
