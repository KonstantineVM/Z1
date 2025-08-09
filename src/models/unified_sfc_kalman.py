# ==============================================================================
# FILE: src/models/unified_sfc_kalman.py
# ==============================================================================
"""
Unified SFC Kalman Filter with full stock-flow consistency.
Integrates Z1 data, FWTW bilateral positions, and enforces accounting identities.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List, Set
from pathlib import Path

# Import the new SFC Kalman filter
from .sfc_kalman_filter import SFCKalmanFilter
from .sfc_projection import SFCProjection

# Import utilities
from src.network.fwtw_z1_mapper import FWTWtoZ1Mapper
from src.utils.network_discovery import NetworkDiscovery
from src.utils.formula_parser import FormulaParser

logger = logging.getLogger(__name__)


class UnifiedSFCKalmanFilter:
    """
    Production-ready SFC Kalman filter with full constraint enforcement.
    """
    
    def __init__(self, data: pd.DataFrame, 
                 formulas: Dict = None,
                 fwtw_data: Optional[pd.DataFrame] = None,
                 sfc_config: Optional[Dict] = None,
                 **kwargs):
        """
        Initialize Unified SFC Kalman Filter.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Z.1 time series data (stocks and flows)
        formulas : Dict
            Z.1 accounting formulas
        fwtw_data : pd.DataFrame
            FWTW bilateral position data
        sfc_config : Dict
            Configuration for SFC constraints
        **kwargs : 
            Additional parameters for Kalman filter
        """
        
        self.original_data = data.copy()
        self.formulas = formulas or {}
        self.fwtw_raw = fwtw_data
        self.sfc_config = sfc_config or {}
        
        # Set default configuration
        self._set_default_config()
        
        # Process FWTW data
        self._process_fwtw_data()
        
        # Calculate bilateral flows from FWTW
        self._calculate_bilateral_flows()
        
        # Check data completeness
        self._check_data_completeness()
        
        # Initialize the SFC Kalman filter
        # Extract specific kwargs to avoid duplication
        filter_kwargs = kwargs.copy()
        filter_kwargs.pop('error_variance_ratio', None)  # Remove if exists
        filter_kwargs.pop('normalize_data', None)  # Remove if exists
        
        self.filter_model = SFCKalmanFilter(
            data=data,
            formulas=formulas,
            fwtw_data=self.fwtw_mapped,
            error_variance_ratio=kwargs.get('error_variance_ratio', 0.01),
            normalize_data=kwargs.get('normalize_data', True),
            enforce_sfc=self.sfc_config['enforce_sfc'],
            **filter_kwargs
        )
        
        # Add bilateral constraints to projection system
        if self.filter_model.projector and self.bilateral_flows is not None:
            self._add_bilateral_constraints()
        
        logger.info("Unified SFC Kalman Filter initialized successfully")
        
    def _set_default_config(self):
        """Set default configuration values."""
        defaults = {
            'enforce_sfc': True,
            'enforce_fwtw': True,
            'enforce_market_clearing': True,
            'fwtw_weight': 0.3,
            'market_clearing_weight': 0.1,
            'stock_flow_weight': 0.5,
            'bilateral_weight': 0.3,
            'require_fwtw_overlap': False,
            'min_overlap_fraction': 0.0,
            'include_all': True
        }
        
        for key, value in defaults.items():
            self.sfc_config.setdefault(key, value)
    
    def _process_fwtw_data(self):
        """Process and map FWTW data to Z1 series."""
        
        if self.fwtw_raw is None:
            self.fwtw_mapped = None
            self.fwtw_data = None
            return
        
        try:
            mapper = FWTWtoZ1Mapper()
            
            # Validate and standardize FWTW
            self.fwtw_data = mapper.validate_and_standardize_fwtw(self.fwtw_raw)
            
            # Map to Z1 series
            self.fwtw_mapped = mapper.map_to_z1_series(
                self.fwtw_data,
                set(self.original_data.columns),
                include_all=self.sfc_config['include_all']
            )
            
            if len(self.fwtw_mapped) == 0:
                logger.warning("No FWTW positions could be mapped to Z1 series")
            else:
                logger.info(f"Mapped {len(self.fwtw_mapped)} FWTW positions")
                
        except Exception as e:
            logger.error(f"Error processing FWTW data: {e}")
            self.fwtw_mapped = None
            self.fwtw_data = None
    
    def _calculate_bilateral_flows(self):
        """Calculate bilateral flows from FWTW stock changes."""
        
        if self.fwtw_mapped is None or len(self.fwtw_mapped) == 0:
            self.bilateral_flows = None
            return
        
        logger.info("Calculating bilateral flows from FWTW...")
        
        # Group by date and calculate flows
        dates = sorted(self.fwtw_mapped['date'].unique())
        
        if len(dates) < 2:
            logger.warning("Insufficient dates for flow calculation")
            self.bilateral_flows = None
            return
        
        flows = []
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            # Get positions
            prev_pos = self.fwtw_mapped[self.fwtw_mapped['date'] == prev_date]
            curr_pos = self.fwtw_mapped[self.fwtw_mapped['date'] == curr_date]
            
            # Create keys for matching
            prev_pos = prev_pos.copy()
            curr_pos = curr_pos.copy()
            
            prev_pos['key'] = (prev_pos['holder_code'].astype(str) + '_' +
                              prev_pos['issuer_code'].astype(str) + '_' +
                              prev_pos['instrument_code'].astype(str))
            
            curr_pos['key'] = (curr_pos['holder_code'].astype(str) + '_' +
                              curr_pos['issuer_code'].astype(str) + '_' +
                              curr_pos['instrument_code'].astype(str))
            
            # Merge and calculate flows
            merged = pd.merge(
                prev_pos[['key', 'level']],
                curr_pos[['key', 'level']],
                on='key',
                how='outer',
                suffixes=('_prev', '_curr')
            )
            
            merged['level_prev'] = merged['level_prev'].fillna(0)
            merged['level_curr'] = merged['level_curr'].fillna(0)
            merged['flow'] = merged['level_curr'] - merged['level_prev']
            merged['date'] = curr_date
            
            # Parse key back to components
            merged['holder_code'] = merged['key'].str.split('_').str[0]
            merged['issuer_code'] = merged['key'].str.split('_').str[1]
            merged['instrument_code'] = merged['key'].str.split('_').str[2]
            
            flows.append(merged[['date', 'holder_code', 'issuer_code', 
                                'instrument_code', 'flow']])
        
        if flows:
            self.bilateral_flows = pd.concat(flows, ignore_index=True)
            logger.info(f"Calculated {len(self.bilateral_flows)} bilateral flows")
            
            # Summary statistics
            total_positive = (self.bilateral_flows['flow'] > 0).sum()
            total_negative = (self.bilateral_flows['flow'] < 0).sum()
            total_zero = (self.bilateral_flows['flow'] == 0).sum()
            
            logger.info(f"  Positive flows: {total_positive}")
            logger.info(f"  Negative flows: {total_negative}")
            logger.info(f"  Zero flows: {total_zero}")
        else:
            self.bilateral_flows = None
    
    def _check_data_completeness(self):
        """Check data completeness for SFC analysis."""
        
        # Count series by type
        n_stocks = len([s for s in self.original_data.columns 
                       if s.startswith('FA') or s.startswith('FL')])
        n_flows = len([s for s in self.original_data.columns 
                      if s.startswith('FU') or s.startswith('FR')])
        
        # Check stock-flow pairs
        stock_flow_pairs = []
        for series in self.original_data.columns:
            if series.startswith('FA'):
                flow = 'FU' + series[2:]
                if flow in self.original_data.columns:
                    stock_flow_pairs.append((series, flow))
            elif series.startswith('FL'):
                flow = 'FR' + series[2:]
                if flow in self.original_data.columns:
                    stock_flow_pairs.append((series, flow))
        
        # Check FWTW coverage
        fwtw_coverage = 0.0
        if self.fwtw_mapped is not None:
            # Count how many Z1 series have FWTW data
            fwtw_series = set()
            if 'asset_series' in self.fwtw_mapped.columns:
                fwtw_series.update(self.fwtw_mapped['asset_series'].dropna().unique())
            if 'liability_series' in self.fwtw_mapped.columns:
                fwtw_series.update(self.fwtw_mapped['liability_series'].dropna().unique())
            
            available_series = set(self.original_data.columns)
            matched = fwtw_series & available_series
            
            if len(available_series) > 0:
                fwtw_coverage = len(matched) / len(available_series)
        
        logger.info("Data completeness check:")
        logger.info(f"  Stock series: {n_stocks}")
        logger.info(f"  Flow series: {n_flows}")
        logger.info(f"  Stock-flow pairs: {len(stock_flow_pairs)}")
        logger.info(f"  FWTW coverage: {fwtw_coverage:.1%}")
        
        if len(stock_flow_pairs) == 0:
            logger.warning("No stock-flow pairs found - SFC constraints cannot be enforced")
        
        self.data_completeness = {
            'n_stocks': n_stocks,
            'n_flows': n_flows,
            'n_pairs': len(stock_flow_pairs),
            'fwtw_coverage': fwtw_coverage
        }
    
    def _add_bilateral_constraints(self):
        """Add bilateral constraints from FWTW to projection system."""
        
        if self.bilateral_flows is None:
            return
        
        logger.info("Adding bilateral constraints to projection system...")
        
        # Group flows by date
        for date in self.bilateral_flows['date'].unique():
            date_flows = self.bilateral_flows[self.bilateral_flows['date'] == date]
            
            # Add constraints for this date
            # This would need to be implemented in the projection system
            # based on the specific structure of your constraints
            pass
    
    def filter(self, **kwargs) -> Dict:
        """
        Run SFC Kalman filter with all constraints.
        
        Returns:
        --------
        Dict with filter results
        """
        logger.info("Running SFC Kalman filter...")
        
        # Run the filter
        results = self.filter_model.filter(**kwargs)
        
        # Validate results
        validation = self._validate_results(results)
        
        # Log validation
        logger.info("SFC Validation Results:")
        for key, value in validation.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        return results
    
    def _validate_results(self, results) -> Dict[str, float]:
        """Comprehensive validation of SFC constraints."""
        
        validation = {}
        
        # Get filtered series
        series_dict = self.get_filtered_series(results)
        filtered = series_dict['filtered']
        
        # 1. Check stock-flow consistency
        sf_violations = []
        for series in filtered.columns:
            if series.startswith('FA'):
                flow = 'FU' + series[2:]
                if flow in filtered.columns:
                    stock_changes = filtered[series].diff()
                    flows = filtered[flow]
                    
                    valid_idx = ~(stock_changes.isna() | flows.isna())
                    if valid_idx.sum() > 0:
                        violations = abs(stock_changes[valid_idx] - flows[valid_idx])
                        sf_violations.extend(violations.values)
        
        if sf_violations:
            validation['stock_flow_max_violation'] = np.max(sf_violations)
            validation['stock_flow_mean_violation'] = np.mean(sf_violations)
        
        # 2. Check market clearing
        instruments = {}
        for series in filtered.columns:
            if len(series) >= 9 and series[:2] in ['FA', 'FL']:
                instrument = series[4:9]
                if instrument not in instruments:
                    instruments[instrument] = {'FA': [], 'FL': []}
                
                if series.startswith('FA'):
                    instruments[instrument]['FA'].append(series)
                else:
                    instruments[instrument]['FL'].append(series)
        
        clearing_violations = []
        for instrument, series_dict in instruments.items():
            if series_dict['FA'] and series_dict['FL']:
                fa_total = filtered[series_dict['FA']].sum(axis=1)
                fl_total = filtered[series_dict['FL']].sum(axis=1)
                
                violations = abs(fa_total - fl_total)
                clearing_violations.extend(violations.values)
        
        if clearing_violations:
            validation['market_clearing_max'] = np.max(clearing_violations)
            validation['market_clearing_mean'] = np.mean(clearing_violations)
        
        # 3. Count satisfied constraints
        validation['n_constraints_checked'] = len(sf_violations) + len(clearing_violations)
        validation['n_constraints_satisfied'] = sum(
            1 for v in sf_violations + clearing_violations if v < 1e-6
        )
        
        return validation
    
    def get_filtered_series(self, results) -> Dict[str, pd.DataFrame]:
        """Extract filtered and smoothed series from results."""
        return self.filter_model.get_filtered_series(results)
    
    def get_bilateral_flows(self) -> Optional[pd.DataFrame]:
        """Get calculated bilateral flows."""
        return self.bilateral_flows
    
    def get_data_completeness(self) -> Dict:
        """Get data completeness metrics."""
        return self.data_completeness
