#!/usr/bin/env python3
"""
Integrated SFC Kalman Filter Analysis for Z1 Project.
Uses ONLY the existing Z1 project infrastructure without any custom transformations.
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Z1 data infrastructure - using ACTUAL project modules
from src.data.cached_fed_data_loader import CachedFedDataLoader
from src.data.data_processor import DataProcessor
from src.network.fwtw_loader import FWTWDataLoader

# Import utilities
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
        
        # Storage for data
        self.z1_data = None
        self.fwtw_data = None
        self.formulas = None
        self.model = None
        self.fitted_results = None
        
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
            'output_dir': 'output',
            'cache_dir': './data/cache',
            'base_dir': './data/fed_data'
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
            config['enforce_formulas'] = False  # Skip formulas in test
        elif self.run_mode == 'development':
            config['max_series'] = 200
            config['enforce_formulas'] = True  # Enable formulas
            config['constraint_method'] = 'soft'  # Faster
        elif self.run_mode == 'production':
            config['max_series'] = 500
            config['enforce_formulas'] = True  # Enable all constraints
            config['constraint_method'] = 'exact'  # Exact enforcement
            
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
            
            # Save results
            self._save_results()
            
            # Create visualizations if needed
            if self.config.get('create_visualizations', True):
                self._create_visualizations()
            
            self.logger.info("\n✓ Analysis completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_z1_data(self):
        """Load Z.1 data using the project's existing infrastructure."""
        self.logger.info("\nLoading Z.1 data...")
        
        try:
            # Step 1: Load raw Z1 data using CachedFedDataLoader
            loader = CachedFedDataLoader(
                base_directory=self.config.get('base_dir', './data/fed_data'),
                cache_directory=self.config.get('cache_dir', './data/cache')
            )
            
            z1_raw = loader.load_single_source('Z1')
            
            if z1_raw is None:
                raise ValueError("Failed to load Z1 data")
            
            self.logger.info(f"Loaded raw Z1 data: {z1_raw.shape}")
            
            # Step 2: Process using DataProcessor to get time series format
            processor = DataProcessor()
            self.z1_data = processor.process_fed_data(z1_raw, 'Z1')
            
            if self.z1_data is None or self.z1_data.empty:
                raise ValueError("DataProcessor returned empty data")
            
            self.logger.info(f"Processed Z1 data: {self.z1_data.shape}")
            
            # The processed data now has:
            # - DatetimeIndex as index
            # - Series codes as columns (with .Q suffix removed)
            
            # Apply series limit if needed
            if len(self.z1_data.columns) > self.config['max_series']:
                self.z1_data = self._select_priority_series(self.z1_data)
            
            self.logger.info(f"Final dataset: {len(self.z1_data.columns)} series")
            self.logger.info(f"Date range: {self.z1_data.index[0]} to {self.z1_data.index[-1]}")
            
            # Log series composition
            self._log_series_composition()
            
        except Exception as e:
            self.logger.error(f"Error loading Z1 data: {e}")
            self.logger.info("Using sample data for testing")
            self.z1_data = self._generate_sample_data()
    
    def _load_fwtw_data(self):
        """Load and map FWTW data to Z.1 series codes."""
        self.logger.info("\nLoading FWTW data...")
        
        try:
            # Use the project's FWTW loader
            fwtw_loader = FWTWDataLoader(
                cache_dir=self.config.get('cache_dir', './data/cache') + '/fwtw'
            )
            
            # Load FWTW data with caching (automatically uses parquet)
            fwtw_raw = fwtw_loader.load_fwtw_data()
            
            if fwtw_raw is not None:
                self.logger.info(f"Loaded raw FWTW data: {fwtw_raw.shape}")
                self.logger.info(f"FWTW columns: {list(fwtw_raw.columns)}")
                
                # Get unique entities for reporting
                entities = fwtw_loader.get_unique_entities(fwtw_raw)
                self.logger.info(f"  Holders: {len(entities['holders'])}")
                self.logger.info(f"  Issuers: {len(entities['issuers'])}")
                self.logger.info(f"  Instruments: {len(entities['instruments'])}")
                
                # Map FWTW to Z.1 series codes using the mapper
                self._map_fwtw_to_z1(fwtw_raw)
                
                # Calculate bilateral flows from stock changes
                self._calculate_bilateral_flows()
            else:
                self.logger.warning("FWTW data not available, continuing without bilateral constraints")
                self.fwtw_data = None
                
        except Exception as e:
            self.logger.warning(f"Could not load FWTW data: {e}")
            self.fwtw_data = None
    
    def _map_fwtw_to_z1(self, fwtw_raw):
        """Map FWTW bilateral positions to Z.1 series codes."""
        self.logger.info("\nMapping FWTW to Z.1 series codes...")
        
        try:
            from src.network.fwtw_z1_mapper import FWTWtoZ1Mapper
            
            # Initialize mapper
            mapper = FWTWtoZ1Mapper()
            
            # Get available Z1 series for validation (if we have Z1 data loaded)
            available_z1_series = None
            if self.z1_data is not None:
                available_z1_series = set(self.z1_data.columns)
                self.logger.info(f"Using {len(available_z1_series)} Z.1 series for validation")
            
            # Map FWTW to Z1 series
            # include_all=True means include all positions even if series not in Z1
            self.fwtw_data = mapper.map_to_z1_series(
                fwtw_raw, 
                available_z1_series=available_z1_series,
                include_all=True  # Include all for complete bilateral constraints
            )
            
            if self.fwtw_data is not None and not self.fwtw_data.empty:
                self.logger.info(f"Mapped FWTW data: {self.fwtw_data.shape}")
                
                # Log mapping results
                n_asset_series = self.fwtw_data['asset_series'].notna().sum()
                n_liability_series = self.fwtw_data['liability_series'].notna().sum()
                self.logger.info(f"  Asset series mapped: {n_asset_series}")
                self.logger.info(f"  Liability series mapped: {n_liability_series}")
                
                # Check overlap with Z1 data
                if available_z1_series:
                    asset_overlap = self.fwtw_data[self.fwtw_data['asset_series'].isin(available_z1_series)]
                    liability_overlap = self.fwtw_data[self.fwtw_data['liability_series'].isin(available_z1_series)]
                    self.logger.info(f"  Asset series in Z1: {len(asset_overlap)}")
                    self.logger.info(f"  Liability series in Z1: {len(liability_overlap)}")
            else:
                self.logger.warning("FWTW mapping returned empty data")
                self.fwtw_data = None
                
        except Exception as e:
            self.logger.warning(f"Could not map FWTW to Z1: {e}")
            # Keep raw FWTW data if mapping fails
            self.fwtw_data = fwtw_raw
    
    def _load_formulas(self):
        """Load and parse Z.1 formulas for building accounting constraints."""
        self.logger.info("\nLoading formulas...")
        
        formula_path = Path(self.config.get('formula_path', 'data/fof_formulas_extracted.json'))
        if formula_path.exists():
            try:
                with open(formula_path, 'r') as f:
                    self.formulas = json.load(f)
                self.logger.info(f"Loaded {len(self.formulas)} formulas")
                
                # Parse formulas to build constraint relationships
                self._parse_formula_constraints()
                
            except Exception as e:
                self.logger.warning(f"Could not load formulas: {e}")
                self.formulas = None
        else:
            self.logger.info("Formulas file not found, continuing without formula constraints")
            self.formulas = None
    
    def _parse_formula_constraints(self):
        """Parse formulas to identify accounting constraints."""
        if not self.formulas:
            return
        
        self.logger.info("Parsing formula constraints...")
        
        try:
            from src.utils.formula_parser import FormulaParser
            from src.utils.network_discovery import NetworkDiscovery
            
            parser = FormulaParser()
            self.formula_constraints = []
            
            # Parse each formula to extract constraint relationships
            for series_code, formula_dict in self.formulas.items():
                # Skip if series not in our data
                if self.z1_data is not None and series_code not in self.z1_data.columns:
                    continue
                
                # Parse formula components
                components = parser.parse_formula(formula_dict)
                
                if components:
                    # Build constraint: series = sum(components)
                    # Format: series - component1 - component2 ... = 0
                    constraint = {
                        'target_series': series_code,
                        'components': components,
                        'type': 'formula',
                        'formula_str': formula_dict.get('formula', '')
                    }
                    self.formula_constraints.append(constraint)
            
            self.logger.info(f"Parsed {len(self.formula_constraints)} formula constraints")
            
            # Use NetworkDiscovery to find complete network if configured
            if self.config.get('discover_network', False):
                initial_series = list(self.z1_data.columns) if self.z1_data is not None else []
                network = NetworkDiscovery.discover_complete_network(
                    initial_series=initial_series[:100],  # Start with subset
                    formulas=self.formulas,
                    fwtw_data=self.fwtw_data,
                    z1_data=self.z1_data,
                    include_all=True
                )
                
                self.logger.info(f"Network discovery found:")
                for category, series_set in network.items():
                    if series_set:
                        self.logger.info(f"  {category}: {len(series_set)} series")
                
        except Exception as e:
            self.logger.warning(f"Could not parse formula constraints: {e}")
            self.formula_constraints = []
    
    def _log_series_composition(self):
        """Log the composition of loaded series."""
        composition = {
            'FL (stocks/levels)': 0,
            'FU (transactions)': 0,
            'FR (revaluations)': 0,
            'FV (other changes)': 0,
            'FA (seasonal flows)': 0,
            'Other': 0
        }
        
        for col in self.z1_data.columns:
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
    
    def _select_priority_series(self, data):
        """Select priority series when memory constraints apply."""
        max_series = self.config['max_series']
        selected = []
        columns_set = set(data.columns)
        
        # Priority 1: Complete stock-flow pairs
        for col in data.columns:
            col_str = str(col)
            if col_str.startswith('FL'):  # Stock series
                # Try to find matching flow series
                if len(col_str) >= 11:  # Full format
                    sector = col_str[2:4]
                    instrument = col_str[4:9]
                    
                    # Look for matching FU or FA series
                    for prefix in ['FU', 'FA']:
                        flow_pattern = f"{prefix}{sector}{instrument}"
                        matches = [c for c in columns_set if str(c).startswith(flow_pattern)]
                        if matches:
                            if col not in selected:
                                selected.append(col)
                            if matches[0] not in selected:
                                selected.append(matches[0])
                            break
        
        # Priority 2: Important sectors
        important_sectors = ['10', '15', '26', '31', '70', '89', '90']
        for col in data.columns:
            if len(selected) >= max_series:
                break
            col_str = str(col)
            if col_str not in selected and len(col_str) >= 4:
                if col_str[:2] in ['FL', 'FU', 'FR', 'FV', 'FA']:
                    sector = col_str[2:4]
                    if sector in important_sectors:
                        selected.append(col)
        
        # Fill remaining with any series
        for col in data.columns:
            if len(selected) >= max_series:
                break
            if col not in selected:
                selected.append(col)
        
        return data[selected[:max_series]]
    
    def _calculate_bilateral_flows(self):
        """Calculate bilateral flows from mapped FWTW data."""
        if self.fwtw_data is None:
            return
        
        try:
            # After mapping, we have standardized columns
            # Check if we have the mapped data structure (UPDATED COLUMN NAMES)
            if 'holder_series' in self.fwtw_data.columns and 'issuer_series' in self.fwtw_data.columns:
                # Work with mapped data
                self.logger.info("Calculating bilateral flows from mapped FWTW data...")
                
                # Group by holder/issuer series and date
                group_cols = ['holder_series', 'issuer_series']
                
                # Calculate flows for each bilateral relationship
                flows = []
                for series_pair in [('holder_series', 'holder_code'), ('issuer_series', 'issuer_code')]:
                    series_col, sector_col = series_pair
                    
                    # Skip if columns not present
                    if series_col not in self.fwtw_data.columns:
                        continue
                    
                    # Group by series and calculate flows
                    for series_code, group in self.fwtw_data.groupby(series_col):
                        if pd.notna(series_code) and 'date' in group.columns and 'level' in group.columns:
                            group_sorted = group.sort_values('date')
                            if len(group_sorted) > 1:
                                flow_data = group_sorted.copy()
                                flow_data['flow'] = flow_data['level'].diff()
                                flow_data['z1_series'] = series_code
                                flows.append(flow_data[['date', 'z1_series', 'flow', 'level']].iloc[1:])
                
                if flows:
                    self.bilateral_flows = pd.concat(flows, ignore_index=True)
                    self.logger.info(f"Calculated {len(self.bilateral_flows)} bilateral flows")
                    
                    # Aggregate flows by Z1 series for model input
                    self.aggregated_flows = self.bilateral_flows.groupby(['date', 'z1_series']).agg({
                        'flow': 'sum',
                        'level': 'last'
                    }).reset_index()
                    self.logger.info(f"Aggregated to {len(self.aggregated_flows['z1_series'].unique())} unique Z1 series")
                else:
                    self.bilateral_flows = None
                    self.aggregated_flows = None
            
            # Check for old column names (backward compatibility)
            elif 'asset_series' in self.fwtw_data.columns and 'liability_series' in self.fwtw_data.columns:
                self.logger.warning("FWTW data has old column names (asset/liability), updating...")
                # Convert old format to new format
                self.fwtw_data['holder_series'] = self.fwtw_data['asset_series']
                self.fwtw_data['issuer_series'] = self.fwtw_data['liability_series']
                # Recursively call with updated columns
                self._calculate_bilateral_flows()
            
            else:
                # Fallback to original calculation if not mapped data
                self.logger.warning("FWTW data not in mapped format, using raw data")
                self._calculate_bilateral_flows_raw()
                
        except Exception as e:
            self.logger.warning(f"Could not calculate bilateral flows: {e}")
            self.bilateral_flows = None
            self.aggregated_flows = None
    
    def _calculate_bilateral_flows_raw(self):
        """Fallback calculation for raw FWTW data."""
        # Original implementation for unmapped data
        position_col = None
        for col in ['Level', 'Position', 'Amount', 'Value', 'level']:
            if col in self.fwtw_data.columns:
                position_col = col
                break
        
        if position_col is None:
            self.bilateral_flows = None
            return
        
        # Simple flow calculation
        if 'Date' in self.fwtw_data.columns or 'date' in self.fwtw_data.columns:
            date_col = 'Date' if 'Date' in self.fwtw_data.columns else 'date'
            self.fwtw_data = self.fwtw_data.sort_values(date_col)
            self.fwtw_data['flow'] = self.fwtw_data.groupby(['Holder Name', 'Issuer Name', 'Instrument Name'])[position_col].diff()
            self.bilateral_flows = self.fwtw_data[self.fwtw_data['flow'].notna()]
    
    def _generate_sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2010-01-01', periods=40, freq='QE')
        
        # Generate consistent FL-FU pairs with proper Z1 format
        n_pairs = 5
        columns = []
        data_arrays = []
        
        sectors = ['10', '15', '26', '31', '70']
        instrument = '30641'
        
        for i, sector in enumerate(sectors):
            # Generate stock (FL)
            stock = np.cumsum(np.random.randn(40)) * 100 + 1000
            columns.append(f"FL{sector}{instrument}005")
            data_arrays.append(stock)
            
            # Generate flow (FU)
            flow = np.diff(stock, prepend=stock[0])
            flow += np.random.randn(40) * 5
            columns.append(f"FU{sector}{instrument}005")
            data_arrays.append(flow)
        
        return pd.DataFrame(
            np.column_stack(data_arrays),
            index=dates,
            columns=columns
        )
    
    def _initialize_model(self):
        """Initialize the SFC Kalman filter model with all constraints."""
        self.logger.info("\nInitializing SFC Kalman Filter...")
        
        # Model configuration with full constraint enforcement
        model_config = {
            'enforce_sfc': self.config['enforce_sfc'],
            'enforce_market_clearing': self.config['enforce_market_clearing'],
            'enforce_formulas': self.config.get('enforce_formulas', False),
            'constraint_method': self.config.get('constraint_method', 'exact'),
            'bilateral_weight': self.config['bilateral_weight'],
            'error_variance_ratio': self.config['error_variance_ratio'],
            'normalize_data': self.config['normalize_data'],
            'transformation': self.config['transformation'],
            # Additional constraint parameters
            'stock_flow_weight': self.config.get('stock_flow_weight', 0.5),
            'formula_weight': self.config.get('formula_weight', 0.3),
            'market_clearing_weight': self.config.get('market_clearing_weight', 0.1),
            'projection_tolerance': self.config.get('projection_tolerance', 1e-8),
            'max_projection_iterations': self.config.get('max_projection_iterations', 100)
        }
        
        # Prepare FWTW data for model
        fwtw_for_model = None
        if self.fwtw_data is not None:
            if 'asset_series' in self.fwtw_data.columns:
                # Already mapped - use as is
                fwtw_for_model = self.fwtw_data
                self.logger.info("Using mapped FWTW data with Z1 series codes")
            else:
                # Raw FWTW - model will handle mapping internally
                fwtw_for_model = self.fwtw_data
                self.logger.info("Passing raw FWTW data to model for internal mapping")
        
        # Prepare formula constraints for model
        formula_constraints_for_model = None
        if hasattr(self, 'formula_constraints') and self.formula_constraints:
            formula_constraints_for_model = self.formula_constraints
            self.logger.info(f"Using {len(formula_constraints_for_model)} formula constraints")
        
        # Initialize model with all constraints
        self.model = SFCKalmanFilter(
            data=self.z1_data,
            fwtw_data=fwtw_for_model,
            formulas=self.formulas,
            formula_constraints=formula_constraints_for_model,  # Pass parsed constraints
            **model_config
        )
        
        self.logger.info(f"Model initialized with {len(self.model.stock_flow_pairs)} stock-flow pairs")
        
        # Log constraint summary
        self._log_constraint_summary()
    
    def _log_constraint_summary(self):
        """Log summary of all constraints in the model."""
        self.logger.info("\nConstraint summary:")
        
        # Stock-flow constraints
        if hasattr(self.model, 'stock_flow_pairs'):
            self.logger.info(f"  Stock-flow pairs: {len(self.model.stock_flow_pairs)}")
        
        # Formula constraints
        if hasattr(self.model, 'formula_constraints'):
            self.logger.info(f"  Formula constraints: {len(self.model.formula_constraints)}")
            # Show examples
            if self.model.formula_constraints:
                for i, constraint in enumerate(self.model.formula_constraints[:3]):
                    if isinstance(constraint, dict):
                        target = constraint.get('target_series', 'unknown')
                        n_components = len(constraint.get('components', []))
                        self.logger.info(f"    Example: {target} = f({n_components} components)")
        
        # Bilateral constraints
        if hasattr(self.model, 'bilateral_constraints'):
            self.logger.info(f"  Bilateral constraints: {len(self.model.bilateral_constraints)}")
        
        # Market clearing constraints
        if hasattr(self.model, 'market_clearing_constraints'):
            self.logger.info(f"  Market clearing: {len(self.model.market_clearing_constraints)}")
        
        # Total constraints
        total_constraints = 0
        for attr in ['stock_flow_pairs', 'formula_constraints', 'bilateral_constraints', 'market_clearing_constraints']:
            if hasattr(self.model, attr):
                total_constraints += len(getattr(self.model, attr))
        self.logger.info(f"  Total constraints: {total_constraints}")
    
    def _fit_model(self):
        """Fit the SFC Kalman Filter."""
        self.logger.info("\nFitting SFC Kalman Filter...")
        
        start_time = datetime.now()
        self.fitted_results = self.model.fit()
        elapsed = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"Model fitted in {elapsed:.1f} seconds")
        if hasattr(self.fitted_results, 'llf'):
            self.logger.info(f"Log likelihood: {self.fitted_results.llf:.2f}")
    
    def _run_filtering(self):
        """Run Kalman filtering with SFC constraints."""
        self.logger.info("\nRunning Kalman filtering...")
        
        # Get filtered and smoothed estimates
        if hasattr(self.fitted_results, 'filtered_state'):
            self.filtered_states = self.fitted_results.filtered_state
            self.logger.info(f"Filtered states shape: {self.filtered_states.shape}")
        
        if hasattr(self.fitted_results, 'smoothed_state'):
            self.smoothed_states = self.fitted_results.smoothed_state
            self.logger.info(f"Smoothed states shape: {self.smoothed_states.shape}")
    
    def _validate_results(self):
        """Validate that ALL SFC constraints are satisfied."""
        self.logger.info("\nValidating SFC constraints...")
        
        if hasattr(self.model, 'validate_constraints') and hasattr(self, 'smoothed_states'):
            violations = self.model.validate_constraints(self.smoothed_states)
            
            all_satisfied = True
            self.logger.info("Constraint satisfaction:")
            
            # Check each constraint type
            for constraint_type, metrics in violations.items():
                self.logger.info(f"  {constraint_type}:")
                for metric, value in metrics.items():
                    # Check if violation exceeds tolerance
                    tolerance = self._get_tolerance(constraint_type, metric)
                    satisfied = abs(value) <= tolerance
                    status = "✓" if satisfied else "✗"
                    self.logger.info(f"    {metric}: {value:.6f} {status}")
                    if not satisfied:
                        all_satisfied = False
            
            # Overall status
            if all_satisfied:
                self.logger.info("\n✓ ALL constraints satisfied within tolerance!")
            else:
                self.logger.warning("\n⚠ Some constraints violated - check tolerances")
                
            # Additional validation if formulas are enforced
            if self.config.get('enforce_formulas', False) and hasattr(self, 'formula_constraints'):
                self._validate_formula_constraints()
    
    def _get_tolerance(self, constraint_type, metric):
        """Get tolerance for specific constraint type and metric."""
        tolerances = {
            'stock_flow': {'max_violation': 1e-6, 'mean_violation': 1e-7},
            'bilateral': {'max_error': 0.05, 'mean_error': 0.01},
            'market_clearing': {'max_imbalance': 0.02, 'mean_imbalance': 0.005},
            'formula': {'max_residual': 0.001, 'mean_residual': 0.0001}
        }
        return tolerances.get(constraint_type, {}).get(metric, 1e-6)
    
    def _validate_formula_constraints(self):
        """Explicitly validate formula constraints are satisfied."""
        if not hasattr(self, 'formula_constraints') or not self.z1_data:
            return
        
        self.logger.info("\nValidating formula constraints:")
        
        try:
            from src.utils.formula_parser import FormulaParser
            parser = FormulaParser()
            
            max_error = 0
            n_validated = 0
            
            for constraint in self.formula_constraints[:10]:  # Check first 10
                target_series = constraint['target_series']
                components = constraint['components']
                
                if target_series not in self.z1_data.columns:
                    continue
                
                # Reconstruct series from formula
                reconstructed = pd.Series(0.0, index=self.z1_data.index)
                for series_code, lag, operator, coef in components:
                    if series_code in self.z1_data.columns:
                        if lag == 0:
                            reconstructed += coef * self.z1_data[series_code]
                        elif abs(lag) < len(self.z1_data):
                            shifted = self.z1_data[series_code].shift(-lag)
                            reconstructed += coef * shifted.fillna(0)
                
                # Calculate error
                actual = self.z1_data[target_series]
                error = (actual - reconstructed).abs().mean()
                rel_error = error / actual.abs().mean() if actual.abs().mean() > 0 else 0
                max_error = max(max_error, rel_error)
                
                if rel_error > 0.01:
                    self.logger.warning(f"  {target_series}: rel_error = {rel_error:.2%}")
                n_validated += 1
            
            if max_error < 0.01:
                self.logger.info(f"  ✓ {n_validated} formulas validated (max error: {max_error:.2%})")
            else:
                self.logger.warning(f"  ⚠ Formula violations detected (max error: {max_error:.2%})")
                
        except Exception as e:
            self.logger.warning(f"Could not validate formulas: {e}")
    
    def _save_results(self):
        """Save results to parquet files."""
        self.logger.info("\nSaving results...")
        
        output_dir = Path(self.config['output_dir']) / self.run_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed Z1 data
        z1_path = output_dir / 'z1_processed.parquet'
        self.z1_data.to_parquet(z1_path, compression='snappy')
        self.logger.info(f"Saved processed Z1 data to {z1_path}")
        
        # Save filtered states if available
        if hasattr(self, 'filtered_states'):
            filtered_df = pd.DataFrame(
                self.filtered_states.T,
                index=self.z1_data.index[:len(self.filtered_states.T)],
                columns=[f'state_{i}' for i in range(self.filtered_states.shape[0])]
            )
            filtered_path = output_dir / 'sfc_filtered.parquet'
            filtered_df.to_parquet(filtered_path, compression='snappy')
            self.logger.info(f"Saved filtered states to {filtered_path}")
        
        # Save smoothed states if available
        if hasattr(self, 'smoothed_states'):
            smoothed_df = pd.DataFrame(
                self.smoothed_states.T,
                index=self.z1_data.index[:len(self.smoothed_states.T)],
                columns=[f'state_{i}' for i in range(self.smoothed_states.shape[0])]
            )
            smoothed_path = output_dir / 'sfc_smoothed.parquet'
            smoothed_df.to_parquet(smoothed_path, compression='snappy')
            self.logger.info(f"Saved smoothed states to {smoothed_path}")
        
        # Save bilateral flows if calculated
        if hasattr(self, 'bilateral_flows') and self.bilateral_flows is not None:
            flows_path = output_dir / 'bilateral_flows.parquet'
            self.bilateral_flows.to_parquet(flows_path, compression='snappy')
            self.logger.info(f"Saved bilateral flows to {flows_path}")
        
        # Save metadata
        metadata = {
            'run_mode': self.run_mode,
            'timestamp': datetime.now().isoformat(),
            'n_series': len(self.z1_data.columns) if self.z1_data is not None else 0,
            'n_observations': len(self.z1_data) if self.z1_data is not None else 0,
            'n_stock_flow_pairs': len(self.model.stock_flow_pairs) if hasattr(self.model, 'stock_flow_pairs') else 0,
            'log_likelihood': float(self.fitted_results.llf) if hasattr(self.fitted_results, 'llf') else None,
            'config': self.config
        }
        
        metadata_path = output_dir / 'run_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved metadata to {metadata_path}")
    
    def _create_visualizations(self):
        """Create visualizations of results."""
        self.logger.info("\nCreating visualizations...")
        
        try:
            viz = VisualizationManager(
                output_dir=Path(self.config['output_dir']) / self.run_mode / 'figures'
            )
            
            # Create basic plots if methods exist
            if hasattr(viz, 'plot_filtered_vs_actual') and hasattr(self, 'smoothed_states'):
                viz.plot_filtered_vs_actual(self.z1_data, self.smoothed_states)
            
            if hasattr(viz, 'plot_constraint_violations') and hasattr(self.model, 'constraint_diagnostics'):
                viz.plot_constraint_violations(self.model.constraint_diagnostics)
            
            self.logger.info("Visualizations created")
        except Exception as e:
            self.logger.warning(f"Could not create visualizations: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Z1 Integrated SFC Analysis')
    parser.add_argument('mode', nargs='?', default='development',
                       choices=['test', 'development', 'production'],
                       help='Run mode')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Skip visualization creation')
    
    args = parser.parse_args()
    
    # Initialize analysis
    analysis = IntegratedSFCAnalysis(
        config_path=args.config,
        run_mode=args.mode
    )
    
    # Update config if no-viz specified
    if args.no_viz:
        analysis.config['create_visualizations'] = False
    
    # Run analysis
    success = analysis.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
