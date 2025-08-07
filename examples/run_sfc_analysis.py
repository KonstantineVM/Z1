# ==============================================================================
# FILE: examples/run_sfc_analysis.py
# ==============================================================================
"""
Run Stock-Flow Consistent Kalman Filter Analysis using existing infrastructure.
Integrates with ConfigManager, ResultsManager, DataPipeline, and Visualization.
"""

import logging
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager
from src.results_manager import ResultsManager
from src.data_pipeline import DataPipeline
from src.visualization import VisualizationManager
from src.models.sfc_kalman_filter_extended import SFCKalmanFilter


class SFCAnalysis:
    """
    Orchestrates SFC Kalman filter analysis using existing infrastructure.
    """
    
    def __init__(self, config_path: Optional[str] = None, run_mode: str = 'production'):
        """
        Initialize SFC analysis.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        run_mode : str
            Run mode: 'test', 'development', 'production'
        """
        # Load configuration
        self.config = self._load_sfc_config(config_path, run_mode)
        self.run_mode = run_mode
        
        # Initialize results manager
        self.results = ResultsManager(
            output_dir=Path(self.config['output_dir']) / run_mode,
            timestamp_outputs=True
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self.data = None
        self.fwtw_data = None
        self.formulas = None
        self.model = None
        self.fitted_results = None
    
    def _load_sfc_config(self, config_path: Optional[str], run_mode: str) -> Dict:
        """Load SFC-specific configuration."""
        # Default configuration
        config = {
            'output_dir': './output',
            'max_series': 500,
            'enforce_sfc': True,
            'enforce_market_clearing': True,
            'bilateral_weight': 0.3,
            'error_variance_ratio': 0.01,
            'normalize_data': True,
            'transformation': 'square'
        }
        
        # Load from file if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                config.update(file_config.get('sfc', {}))
        
        # Apply run mode settings
        run_mode_settings = {
            'test': {'max_series': 50, 'enforce_market_clearing': False},
            'development': {'max_series': 200, 'enforce_sfc': True},
            'production': {'max_series': 500, 'enforce_sfc': True, 'enforce_market_clearing': True}
        }
        
        if run_mode in run_mode_settings:
            config.update(run_mode_settings[run_mode])
        
        return config
    
    def run_analysis(self):
        """Run complete SFC analysis pipeline."""
        self.logger.info("="*60)
        self.logger.info(f"STOCK-FLOW CONSISTENT KALMAN FILTER ANALYSIS")
        self.logger.info(f"Mode: {self.run_mode}")
        self.logger.info("="*60)
        
        try:
            # Step 1: Load data
            self._load_data()
            
            # Step 2: Initialize and fit model
            self._fit_model()
            
            # Step 3: Run filtering with constraints
            self._run_filtering()
            
            # Step 4: Validate results
            self._validate_results()
            
            # Step 5: Create visualizations
            self._create_visualizations()
            
            # Step 6: Export results
            self._export_results()
            
            # Create summary report
            report_path = self.results.create_summary_report()
            self.logger.info(f"\nAnalysis complete! Report: {report_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_data(self):
        """Load Z.1, FWTW, and formula data."""
        self.logger.info("\nStep 1: Loading data...")
        
        # Load Z.1 data
        try:
            from src.data import CachedFedDataLoader
            
            fed_loader = CachedFedDataLoader()
            z1_raw = fed_loader.load_single_source('Z1')
            
            # Convert to time series format
            self.data = self._process_z1_data(z1_raw)
            self.logger.info(f"Loaded {len(self.data.columns)} Z.1 series")
            
            # Select series based on max_series
            if len(self.data.columns) > self.config['max_series']:
                self.data = self._select_series(self.data, self.config['max_series'])
                self.logger.info(f"Selected {len(self.data.columns)} series")
            
        except Exception as e:
            self.logger.warning(f"Could not load Z.1 data: {e}")
            self.logger.info("Using sample data for demonstration")
            self.data = self._generate_sample_data()
        
        # Load FWTW data
        try:
            from src.network import FWTWDataLoader
            
            fwtw_loader = FWTWDataLoader()
            self.fwtw_data = fwtw_loader.load_fwtw_data()
            self.logger.info(f"Loaded {len(self.fwtw_data)} FWTW positions")
        except Exception as e:
            self.logger.warning(f"Could not load FWTW data: {e}")
            self.fwtw_data = None
        
        # Load formulas
        formula_path = Path('data/fof_formulas_extracted.json')
        if formula_path.exists():
            with open(formula_path, 'r') as f:
                formula_data = json.load(f)
                self.formulas = formula_data.get('formulas', {})
            self.logger.info(f"Loaded {len(self.formulas)} formulas")
        else:
            self.logger.warning("Formula file not found, using empty formulas")
            self.formulas = {}
        
        # Save data info
        data_info = {
            'n_series': len(self.data.columns),
            'n_observations': len(self.data),
            'date_range': f"{self.data.index[0]} to {self.data.index[-1]}",
            'has_fwtw': self.fwtw_data is not None,
            'n_formulas': len(self.formulas)
        }
        self.results.add_result('data_info', data_info, 'metadata')
    
    def _process_z1_data(self, z1_raw: pd.DataFrame) -> pd.DataFrame:
        """Process raw Z.1 data into time series format."""
        # Filter for quarterly series
        quarterly_mask = z1_raw['SERIES_NAME'].str.endswith('.Q')
        quarterly_data = z1_raw[quarterly_mask].copy()
        
        # Remove .Q suffix
        quarterly_data['SERIES_NAME'] = quarterly_data['SERIES_NAME'].str.replace('.Q', '', regex=False)
        
        # Pivot to wide format
        data = quarterly_data.pivot(
            index='date',
            columns='SERIES_NAME',
            values='value'
        )
        
        # Convert index to datetime
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        # Forward fill missing values
        data = data.ffill()
        
        return data
    
    def _select_series(self, data: pd.DataFrame, max_series: int) -> pd.DataFrame:
        """Select series prioritizing stock-flow pairs based on Z.1 structure."""
        selected = []
        columns_set = set(data.columns)
        
        # Priority 1: FL (stock) series with matching FU (flow) series
        # FL = Level (stock), FU = Transaction (flow)
        for col in data.columns:
            if len(selected) >= max_series:
                break
                
            if col.startswith('FL') and col not in selected:
                # Extract sector and instrument
                if len(col) >= 9:
                    sector = col[2:4]
                    instrument = col[4:9]
                    
                    # Look for matching flow (FU = transaction)
                    flow_pattern = f"FU{sector}{instrument}"
                    flow_matches = [c for c in columns_set if c.startswith(flow_pattern)]
                    
                    if flow_matches:
                        # Add both stock and flow
                        selected.append(col)
                        if flow_matches[0] not in selected:
                            selected.append(flow_matches[0])
                        
                        # Also look for FR (revaluation) and FV (other volume changes)
                        reval_pattern = f"FR{sector}{instrument}"
                        reval_matches = [c for c in columns_set if c.startswith(reval_pattern)]
                        if reval_matches and reval_matches[0] not in selected:
                            selected.append(reval_matches[0])
                        
                        other_pattern = f"FV{sector}{instrument}"
                        other_matches = [c for c in columns_set if c.startswith(other_pattern)]
                        if other_matches and other_matches[0] not in selected:
                            selected.append(other_matches[0])
        
        # Priority 2: Important sectors (from Z.1 documentation)
        important_sectors = [
            '10',  # Nonfinancial corporate business
            '15',  # Households and nonprofit organizations
            '26',  # Rest of the world
            '31',  # Federal government
            '70',  # Private depository institutions
            '89',  # All sectors (totals for market clearing)
            '90'   # Instrument discrepancies
        ]
        
        for col in data.columns:
            if len(selected) >= max_series:
                break
            if col not in selected and len(col) >= 4:
                sector = col[2:4]
                if sector in important_sectors:
                    selected.append(col)
        
        # Priority 3: FA series (seasonally adjusted flows)
        # These are derived from FU: FA = (FU + FS) * 4
        for col in data.columns:
            if len(selected) >= max_series:
                break
            if col.startswith('FA') and col not in selected:
                selected.append(col)
        
        # Priority 4: Fill remaining with any series
        for col in data.columns:
            if len(selected) >= max_series:
                break
            if col not in selected:
                selected.append(col)
        
        self.logger.info(f"Series selection summary:")
        self.logger.info(f"  FL (stocks/levels): {len([s for s in selected if s.startswith('FL')])}")
        self.logger.info(f"  FU (transactions): {len([s for s in selected if s.startswith('FU')])}")
        self.logger.info(f"  FR (revaluations): {len([s for s in selected if s.startswith('FR')])}")
        self.logger.info(f"  FV (other changes): {len([s for s in selected if s.startswith('FV')])}")
        self.logger.info(f"  FA (seasonal flows): {len([s for s in selected if s.startswith('FA')])}")
        
        return data[selected[:max_series]]
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for testing with correct Z.1 structure."""
        np.random.seed(42)
        dates = pd.date_range('2010-01-01', periods=40, freq='QE')
        
        # Generate data for a few sectors and instruments
        sectors = ['10', '15', '26']  # Corporate, Households, ROW
        instruments = ['30641', '30650']  # Equities, Mortgages
        
        all_series = []
        all_names = []
        
        for sector in sectors:
            for instrument in instruments:
                # Generate stock (FL = Level)
                stock = np.cumsum(np.random.randn(40)) * 100 + 1000
                
                # Generate flow components
                # FU = Transaction (main flow)
                flow_transaction = np.diff(stock, prepend=stock[0])
                flow_transaction += np.random.randn(40) * 5  # Add noise
                
                # FR = Revaluation (price changes)
                flow_reval = np.random.randn(40) * 10
                
                # FV = Other volume changes (rare)
                flow_other = np.zeros(40)
                flow_other[np.random.choice(40, 3)] = np.random.randn(3) * 20
                
                # Adjust stock to satisfy identity: FL[t] = FL[t-1] + FU + FR + FV
                for t in range(1, 40):
                    stock[t] = stock[t-1] + flow_transaction[t] + flow_reval[t] + flow_other[t]
                
                # FA = Seasonally adjusted flow at annual rate
                # FA = (FU + FS) * 4, we'll approximate as FU * 4
                flow_seasonal = flow_transaction * 4
                
                # Add series with proper naming
                all_series.append(stock)
                all_names.append(f"FL{sector}{instrument}05")  # Level
                
                all_series.append(flow_transaction)
                all_names.append(f"FU{sector}{instrument}05")  # Transaction
                
                all_series.append(flow_reval)
                all_names.append(f"FR{sector}{instrument}05")  # Revaluation
                
                all_series.append(flow_other)
                all_names.append(f"FV{sector}{instrument}05")  # Other changes
                
                all_series.append(flow_seasonal)
                all_names.append(f"FA{sector}{instrument}05")  # Seasonal flow
        
        # Add sector 89 (All sectors) for market clearing
        # This should be the sum of individual sectors
        for instrument in instruments:
            # Sum stocks across sectors
            stock_all = np.zeros(40)
            flow_all = np.zeros(40)
            
            for sector in sectors:
                stock_series = f"FL{sector}{instrument}05"
                flow_series = f"FU{sector}{instrument}05"
                
                idx_stock = all_names.index(stock_series)
                idx_flow = all_names.index(flow_series)
                
                stock_all += all_series[idx_stock]
                flow_all += all_series[idx_flow]
            
            all_series.append(stock_all)
            all_names.append(f"FL89{instrument}05")  # All sectors level
            
            all_series.append(flow_all)
            all_names.append(f"FU89{instrument}05")  # All sectors flow
        
        # Create DataFrame
        data = pd.DataFrame(
            np.column_stack(all_series),
            index=dates,
            columns=all_names
        )
        
        self.logger.info(f"Generated sample data with {len(all_names)} series")
        self.logger.info(f"  Sectors: {sectors}")
        self.logger.info(f"  Instruments: {instruments}")
        self.logger.info(f"  Series types: FL, FU, FR, FV, FA")
        
        return data
    
    def _fit_model(self):
        """Initialize and fit SFC Kalman filter."""
        self.logger.info("\nStep 2: Initializing SFC Kalman filter...")
        
        # Initialize model
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
        
        # Get diagnostics
        diagnostics = self.model.get_sfc_diagnostics()
        self.logger.info(f"Model initialized:")
        self.logger.info(f"  Stock-flow pairs: {diagnostics['n_stock_flow_pairs']}")
        self.logger.info(f"  Bilateral positions: {diagnostics['n_bilateral_positions']}")
        self.logger.info(f"  Total states: {diagnostics['state_dimensions']['total_states']}")
        
        # Save diagnostics
        self.results.add_result('sfc_diagnostics', diagnostics, 'model')
        
        # Fit model
        self.logger.info("\nFitting model parameters...")
        self.fitted_results = self.model.fit(
            start_params=self.model.start_params,
            method='lbfgs',
            maxiter=1000,
            disp=False
        )
        
        self.logger.info(f"Model fitted:")
        self.logger.info(f"  Log-likelihood: {self.fitted_results.llf:.2f}")
        self.logger.info(f"  AIC: {self.fitted_results.aic:.2f}")
        self.logger.info(f"  Parameters: {len(self.fitted_results.params)}")
        
        # Save model info
        model_info = {
            'log_likelihood': float(self.fitted_results.llf),
            'aic': float(self.fitted_results.aic),
            'n_parameters': len(self.fitted_results.params),
            'converged': self.fitted_results.mle_retvals['converged']
        }
        self.results.add_result('model_fit', model_info, 'model')
    
    def _run_filtering(self):
        """Run Kalman filter with SFC constraints."""
        self.logger.info("\nStep 3: Running Kalman filter with SFC constraints...")
        
        # Run filter (constraints are applied internally)
        filter_results = self.model.filter(self.fitted_results.params)
        
        # Get filtered series
        filtered_series = self.model.get_filtered_series(filter_results)
        
        # Save filtered series
        self.results.add_dataframe(
            'filtered_series',
            filtered_series['filtered'],
            subdir='output'
        )
        
        self.results.add_dataframe(
            'smoothed_series',
            filtered_series['smoothed'],
            subdir='output'
        )
        
        self.logger.info(f"Filtering complete:")
        self.logger.info(f"  Filtered {len(filtered_series['filtered'].columns)} series")
        self.logger.info(f"  Time periods: {len(filtered_series['filtered'])}")
        
        # Store for validation
        self.filter_results = filter_results
        self.filtered_series = filtered_series
    
    def _validate_results(self):
        """Validate SFC constraints are satisfied using Z.1 identity."""
        self.logger.info("\nStep 4: Validating SFC constraints...")
        
        validation_results = {}
        
        # Check stock-flow consistency: FL[t] - FL[t-1] = FU + FR + FV
        sf_violations = []
        for pair in self.model.stock_flow_pairs:
            if pair.stock_series in self.filtered_series['smoothed'].columns and \
               pair.flow_series in self.filtered_series['smoothed'].columns:
                
                stock = self.filtered_series['smoothed'][pair.stock_series]
                flow_fu = self.filtered_series['smoothed'][pair.flow_series]
                
                # Calculate stock change
                stock_change = stock.diff()
                
                # Total flow should include FU + FR + FV
                total_flow = flow_fu.copy()
                
                # Add FR (revaluation) if available
                if pair.reval_series and pair.reval_series in self.filtered_series['smoothed'].columns:
                    flow_fr = self.filtered_series['smoothed'][pair.reval_series]
                    total_flow += flow_fr
                
                # Add FV (other volume changes) if available
                if pair.other_series and pair.other_series in self.filtered_series['smoothed'].columns:
                    flow_fv = self.filtered_series['smoothed'][pair.other_series]
                    total_flow += flow_fv
                
                # Calculate violations
                violation = (stock_change - total_flow).abs().mean()
                sf_violations.append({
                    'pair': f"{pair.stock_series}-{pair.flow_series}",
                    'violation': violation,
                    'has_complete_flows': pair.has_complete_flows
                })
        
        validation_results['stock_flow'] = {
            'mean_violation': np.mean([v['violation'] for v in sf_violations]) if sf_violations else 0.0,
            'max_violation': np.max([v['violation'] for v in sf_violations]) if sf_violations else 0.0,
            'n_pairs_checked': len(sf_violations),
            'n_complete_pairs': len([v for v in sf_violations if v['has_complete_flows']])
        }
        
        # Check market clearing using sector 89 (All sectors)
        if self.config['enforce_market_clearing']:
            instruments = {}
            for col in self.filtered_series['smoothed'].columns:
                if len(col) >= 9:
                    prefix = col[:2]
                    sector = col[2:4]
                    instrument = col[4:9]
                    
                    if instrument not in instruments:
                        instruments[instrument] = {
                            'individual_sectors': [],
                            'all_sectors': None
                        }
                    
                    if sector == '89':  # All sectors total
                        instruments[instrument]['all_sectors'] = col
                    elif sector in ['10', '15', '26', '31', '70']:  # Individual sectors
                        instruments[instrument]['individual_sectors'].append(col)
            
            mc_violations = []
            for instrument, series_dict in instruments.items():
                if series_dict['all_sectors'] and series_dict['individual_sectors']:
                    # Check if sum of individuals equals total
                    total_series = self.filtered_series['smoothed'][series_dict['all_sectors']]
                    individual_sum = self.filtered_series['smoothed'][series_dict['individual_sectors']].sum(axis=1)
                    
                    violation = (total_series - individual_sum).abs().mean()
                    mc_violations.append({
                        'instrument': instrument,
                        'violation': violation
                    })
            
            validation_results['market_clearing'] = {
                'mean_violation': np.mean([v['violation'] for v in mc_violations]) if mc_violations else 0.0,
                'max_violation': np.max([v['violation'] for v in mc_violations]) if mc_violations else 0.0,
                'n_instruments_checked': len(mc_violations)
            }
        
        # Log results
        self.logger.info("Validation results:")
        self.logger.info(f"  Stock-flow consistency (FL = FL[-1] + FU + FR + FV):")
        self.logger.info(f"    Mean violation: {validation_results['stock_flow']['mean_violation']:.6f}")
        self.logger.info(f"    Max violation: {validation_results['stock_flow']['max_violation']:.6f}")
        self.logger.info(f"    Pairs with complete flows: {validation_results['stock_flow']['n_complete_pairs']}/{validation_results['stock_flow']['n_pairs_checked']}")
        
        if 'market_clearing' in validation_results:
            self.logger.info(f"  Market clearing (sector 89 = sum of individuals):")
            self.logger.info(f"    Mean violation: {validation_results['market_clearing']['mean_violation']:.6f}")
            self.logger.info(f"    Max violation: {validation_results['market_clearing']['max_violation']:.6f}")
        
        # Save validation results
        self.results.add_result('validation', validation_results, 'validation')
    
    def _create_visualizations(self):
        """Create diagnostic visualizations."""
        self.logger.info("\nStep 5: Creating visualizations...")
        
        # Create plots directory
        plot_dir = Path(self.results.output_dir) / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # Select example series for visualization
        example_pairs = self.model.stock_flow_pairs[:3]  # First 3 pairs
        
        if example_pairs:
            import matplotlib.pyplot as plt
            
            # Create stock-flow consistency plot
            fig, axes = plt.subplots(len(example_pairs), 2, figsize=(14, 4*len(example_pairs)))
            if len(example_pairs) == 1:
                axes = axes.reshape(1, -1)
            
            for i, pair in enumerate(example_pairs):
                if pair.stock_series in self.filtered_series['smoothed'].columns and \
                   pair.flow_series in self.filtered_series['smoothed'].columns:
                    
                    # Stock and flow series
                    stock = self.filtered_series['smoothed'][pair.stock_series]
                    flow = self.filtered_series['smoothed'][pair.flow_series]
                    
                    # Plot stock level
                    ax = axes[i, 0]
                    ax.plot(self.data.index, self.data[pair.stock_series], 'k-', alpha=0.3, label='Original')
                    ax.plot(stock.index, stock, 'b-', label='Filtered')
                    ax.set_title(f'{pair.stock_series} (Stock)')
                    ax.set_ylabel('Level')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Plot flow and implied flow
                    ax = axes[i, 1]
                    implied_flow = stock.diff()
                    ax.plot(flow.index, flow, 'g-', label='Filtered Flow', alpha=0.7)
                    ax.plot(implied_flow.index, implied_flow, 'r--', label='Implied Flow (ΔStock)', alpha=0.7)
                    ax.set_title(f'{pair.flow_series} (Flow)')
                    ax.set_ylabel('Flow')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.suptitle('Stock-Flow Consistency Check', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            plot_path = plot_dir / 'stock_flow_consistency.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Created stock-flow consistency plot: {plot_path}")
        
        # Create summary statistics plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Series coverage
        ax = axes[0, 0]
        diagnostics = self.model.get_sfc_diagnostics()
        coverage = diagnostics['series_coverage']
        bars = ax.bar(range(len(coverage)), list(coverage.values()))
        ax.set_xticks(range(len(coverage)))
        ax.set_xticklabels(list(coverage.keys()), rotation=45, ha='right')
        ax.set_title('Series Coverage')
        ax.set_ylabel('Count')
        
        # Plot 2: State composition
        ax = axes[0, 1]
        state_dims = diagnostics['state_dimensions']
        labels = ['Base', 'Flow', 'Bilateral']
        sizes = [state_dims['base_states'], state_dims['flow_states'], state_dims['bilateral_states']]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('State Space Composition')
        
        # Plot 3: Sample filtered series
        ax = axes[1, 0]
        sample_series = self.filtered_series['filtered'].columns[:3]
        for series in sample_series:
            ax.plot(self.filtered_series['filtered'].index, 
                   self.filtered_series['filtered'][series], 
                   label=series, alpha=0.7)
        ax.set_title('Sample Filtered Series')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Information summary
        ax = axes[1, 1]
        info_text = f"""
SFC Kalman Filter Results
{'='*30}
Mode: {self.run_mode}
Series: {len(self.data.columns)}
Observations: {len(self.data)}
Stock-Flow Pairs: {diagnostics['n_stock_flow_pairs']}
Bilateral Positions: {diagnostics['n_bilateral_positions']}
Total States: {diagnostics['state_dimensions']['total_states']}

Model Fit:
  Log-Likelihood: {self.fitted_results.llf:.2f}
  AIC: {self.fitted_results.aic:.2f}
  Parameters: {len(self.fitted_results.params)}
"""
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')
        ax.axis('off')
        
        plt.suptitle('SFC Kalman Filter Analysis Summary', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = plot_dir / 'analysis_summary.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created analysis summary plot: {plot_path}")
    
    def _export_results(self):
        """Export results and create final report."""
        self.logger.info("\nStep 6: Exporting results...")
        
        # Create summary
        summary = {
            'run_mode': self.run_mode,
            'timestamp': pd.Timestamp.now().isoformat(),
            'data': {
                'n_series': len(self.data.columns),
                'n_observations': len(self.data),
                'date_range': f"{self.data.index[0]} to {self.data.index[-1]}"
            },
            'model': {
                'n_stock_flow_pairs': len(self.model.stock_flow_pairs),
                'n_bilateral_positions': len(self.model.bilateral_positions) if self.model.bilateral_positions else 0,
                'total_states': self.model.k_states,
                'log_likelihood': float(self.fitted_results.llf),
                'aic': float(self.fitted_results.aic)
            },
            'configuration': self.config
        }
        
        # Save summary
        summary_path = Path(self.results.output_dir) / 'sfc_analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Saved analysis summary: {summary_path}")
        
        # List all outputs
        self.logger.info("\nGenerated outputs:")
        output_dir = Path(self.results.output_dir)
        for subdir in ['output', 'plots', 'validation']:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob('*'))
                if files:
                    self.logger.info(f"\n{subdir}:")
                    for file in files[:5]:
                        self.logger.info(f"  - {file.name}")
                    if len(files) > 5:
                        self.logger.info(f"  ... and {len(files)-5} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run SFC Kalman Filter Analysis')
    parser.add_argument('mode', nargs='?', default='production',
                       choices=['test', 'development', 'production'],
                       help='Run mode (default: production)')
    parser.add_argument('--config', type=str, default='config/sfc_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run analysis
    analysis = SFCAnalysis(config_path=args.config, run_mode=args.mode)
    success = analysis.run_analysis()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())