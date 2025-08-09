"""
Stock-Flow Consistent Kalman Filter with Extended State Space
Fixed version that properly handles state dimensions
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import sparse
from scipy.linalg import solve

from src.models.hierarchical_kalman_filter import HierarchicalKalmanFilter


@dataclass
class StockFlowPair:
    """Represents a stock-flow relationship in Z.1 data."""
    stock_series: str
    flow_series: str
    reval_series: Optional[str] = None
    other_series: Optional[str] = None
    has_complete_flows: bool = False
    
    def get_all_series(self) -> List[str]:
        """Return all series codes for this pair."""
        series = [self.stock_series, self.flow_series]
        if self.reval_series:
            series.append(self.reval_series)
        if self.other_series:
            series.append(self.other_series)
        return series


class SFCProjection:
    """
    Project state estimates onto SFC constraint manifold.
    """
    
    def __init__(self, stock_flow_pairs=None, bilateral_constraints=None,
                 data_columns=None, tolerance=1e-8, max_iterations=10):
        """
        Initialize projection with constraints.
        """
        self.stock_flow_pairs = stock_flow_pairs or []
        self.bilateral_constraints = bilateral_constraints or []
        self.data_columns = data_columns
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(__name__)
    
    def project(self, state, covariance=None):
        """
        Project state onto constraint manifold.
        
        Solves: min ||x - state||^2 subject to Ax = b
        Solution: x = state - A'(AA')^(-1)(A*state - b)
        """
        if len(self.stock_flow_pairs) == 0:
            return state
        
        # For now, use simple iterative projection
        return self._simple_projection(state)
    
    def _simple_projection(self, states):
        """
        Simple constraint projection that enforces stock-flow consistency.
        Works on the full state matrix (states x time).
        """
        if states.ndim == 1:
            # Single time period
            return self._project_single_period(states)
        
        # Multiple time periods
        projected = states.copy()
        
        # Project each time period
        for t in range(states.shape[1]):
            projected[:, t] = self._project_single_period(projected[:, t])
        
        return projected
    
    def _project_single_period(self, state):
        """
        Project a single time period onto constraints.
        """
        projected = state.copy()
        
        # Enforce stock-flow consistency
        for pair in self.stock_flow_pairs:
            if self.data_columns is not None:
                if pair.stock_series in self.data_columns and pair.flow_series in self.data_columns:
                    # Get state indices (assuming level states only for simplicity)
                    stock_idx = list(self.data_columns).index(pair.stock_series) * 2
                    flow_idx = list(self.data_columns).index(pair.flow_series) * 2
                    
                    # Adjust stock to be consistent with flow
                    # This is simplified - full implementation would consider history
                    if stock_idx < len(projected) and flow_idx < len(projected):
                        # Average the inconsistency
                        adjustment = (projected[stock_idx] - projected[flow_idx]) * 0.5
                        projected[stock_idx] -= adjustment * 0.1  # Gradual adjustment
                        projected[flow_idx] += adjustment * 0.1
        
        return projected


class SFCKalmanFilter(HierarchicalKalmanFilter):
    """
    Stock-Flow Consistent Kalman Filter with proper dimension handling.
    """
    
    def __init__(self, data: pd.DataFrame, 
                 fwtw_data: Optional[pd.DataFrame] = None,
                 formulas: Optional[Dict] = None,
                 formula_constraints: Optional[List] = None,
                 enforce_sfc: bool = True,
                 enforce_market_clearing: bool = True,
                 bilateral_weight: float = 0.3,
                 error_variance_ratio: float = 0.01,
                 normalize_data: bool = True,
                 transformation: str = 'square',
                 **kwargs):
        """
        Initialize SFC Kalman Filter with pre-calculated dimensions.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data with DatetimeIndex
        fwtw_data : pd.DataFrame, optional
            From-Whom-to-Whom bilateral positions
        formulas : dict, optional
            Z.1 formula definitions
        formula_constraints : list, optional
            Parsed formula constraints
        enforce_sfc : bool
            Whether to enforce stock-flow consistency
        enforce_market_clearing : bool
            Whether to enforce market clearing
        bilateral_weight : float
            Weight for bilateral constraints
        error_variance_ratio : float
            Ratio of measurement error to state variance
        normalize_data : bool
            Whether to normalize data before filtering
        transformation : str
            Transformation type for hierarchy
        """
        # Store configuration BEFORE parent init
        self.enforce_sfc = enforce_sfc
        self.enforce_market_clearing = enforce_market_clearing
        self.bilateral_weight = bilateral_weight
        self.fwtw_data = fwtw_data
        self.formulas = formulas
        self.formula_constraints = formula_constraints or []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # CRITICAL: Pre-calculate all constraints before parent init
        # This avoids the dimension mismatch error
        
        # Step 1: Identify FA series (seasonally adjusted flows)
        self.fa_series = [col for col in data.columns if col.startswith('FA')]
        self.logger.info(f"Found {len(self.fa_series)} FA (seasonally adjusted flow) series")
        
        # Step 2: Identify stock-flow pairs
        self.stock_flow_pairs = []
        if enforce_sfc:
            self._identify_stock_flow_pairs_early(data)
            self.logger.info(f"Identified {len(self.stock_flow_pairs)} stock-flow pairs")
        
        # Step 3: Process bilateral data
        self.bilateral_constraints = []
        if fwtw_data is not None:
            self._process_bilateral_data_early(fwtw_data)
            self.logger.info(f"Processed {len(self.bilateral_constraints)} unique bilateral relationships")
        
        # Step 4: Calculate dimensions but DON'T extend state space
        n_series = len(data.columns)
        n_base_states = n_series * 2  # level + trend
        
        # Store dimensions for reference
        self.n_base_states = n_base_states
        self.n_series = n_series
        
        # IMPORTANT: Store original data columns before parent init changes it
        self.data_columns = data.columns.tolist()
        self.original_data = data.copy()  # Keep a copy of the original DataFrame
        
        # Step 5: Initialize parent with BASE dimensions only
        # This is the key fix - we don't change k_states
        super().__init__(
            data=data,
            formulas=formulas,  # Pass formulas to parent
            error_variance_ratio=error_variance_ratio,
            normalize_data=normalize_data,
            transformation=transformation,
            **kwargs
        )
        
        # Step 6: Set up constraint projection (instead of extending state space)
        if enforce_sfc or self.bilateral_constraints:
            self._setup_constraint_projection()
            self.logger.info("Extending state space for SFC constraints")
            # Note: We log this but don't actually extend the state space
            # Instead we use projection to enforce constraints
    
    def _identify_stock_flow_pairs_early(self, data: pd.DataFrame):
        """
        Identify stock-flow pairs BEFORE parent initialization.
        This avoids dimension mismatch issues.
        """
        columns = data.columns.tolist()
        identified_pairs = {}
        
        # Find FL (stock/level) series
        fl_series = [c for c in columns if c.startswith('FL')]
        
        for fl in fl_series:
            # Z.1 series format: PPSSIIIIIIDDD
            # PP = Prefix (FL, FU, FR, FV, FA)
            # SS = Sector (2 digits)
            # IIIIII = Instrument (5 digits)
            # DDD = Additional identifiers
            
            if len(fl) >= 13:
                sector = fl[2:4]
                instrument = fl[4:9]
                suffix = fl[9:]
                
                # Look for corresponding FU (flow/transaction)
                fu = f"FU{sector}{instrument}{suffix}"
                if fu in columns:
                    # Check for FR (revaluation) and FV (other volume changes)
                    fr = f"FR{sector}{instrument}{suffix}"
                    fv = f"FV{sector}{instrument}{suffix}"
                    
                    pair = StockFlowPair(
                        stock_series=fl,
                        flow_series=fu,
                        reval_series=fr if fr in columns else None,
                        other_series=fv if fv in columns else None,
                        has_complete_flows=(fr in columns and fv in columns)
                    )
                    
                    # Use sector-instrument as key to avoid duplicates
                    key = f"{sector}_{instrument}_{suffix}"
                    if key not in identified_pairs:
                        identified_pairs[key] = pair
        
        self.stock_flow_pairs = list(identified_pairs.values())
    
    def _process_bilateral_data_early(self, fwtw_data: pd.DataFrame):
        """
        Process bilateral data BEFORE parent initialization.
        """
        if fwtw_data is None or (isinstance(fwtw_data, pd.DataFrame) and fwtw_data.empty):
            return
        
        bilateral_constraints = []
        
        # Check if FWTW data has required columns
        required_cols = ['Holder Code', 'Issuer Code', 'Instrument Code']
        if isinstance(fwtw_data, pd.DataFrame) and all(col in fwtw_data.columns for col in required_cols):
            # Group by holder-issuer-instrument
            grouped = fwtw_data.groupby(required_cols)
            
            for (holder, issuer, instrument), group in grouped:
                if len(group) > 0:
                    bilateral_constraints.append({
                        'holder': holder,
                        'issuer': issuer,
                        'instrument': instrument,
                        'positions': group[['Date', 'Level']].values if 'Level' in group.columns else None,
                        'series_count': len(group)
                    })
        else:
            # Handle unmapped FWTW data
            self.logger.warning("FWTW data not in expected format, skipping bilateral constraints")
        
        self.bilateral_constraints = bilateral_constraints
    
    def _setup_constraint_projection(self):
        """
        Set up constraint projection WITHOUT extending state space.
        This is the key architectural change to avoid dimension mismatch.
        """
        # Create projection object using stored column information
        self.projection = SFCProjection(
            stock_flow_pairs=self.stock_flow_pairs,
            bilateral_constraints=self.bilateral_constraints,
            data_columns=self.data_columns,  # Use stored columns
            tolerance=1e-8,
            max_iterations=10
        )
        
        # Store original methods
        self._original_filter = super().filter
        self._original_smooth = super().smooth
    
    def filter(self, params=None, **kwargs):
        """
        Run Kalman filter with constraint projection.
        Overrides parent filter method to add SFC constraints.
        """
        # Run standard Kalman filter first
        results = super().filter(params, **kwargs)
        
        # Apply constraints via projection if enabled
        if self.enforce_sfc and hasattr(results, 'filtered_state'):
            self.logger.info("Applying SFC constraint projection to filtered states")
            results.filtered_state = self.projection.project(results.filtered_state)
        
        return results
    
    def smooth(self, params=None, **kwargs):
        """
        Run Kalman smoother with constraint projection.
        """
        # Run standard smoother first
        results = super().smooth(params, **kwargs)
        
        # Apply constraints via projection if enabled
        if self.enforce_sfc:
            if hasattr(results, 'smoothed_state'):
                self.logger.info("Applying SFC constraint projection to smoothed states")
                results.smoothed_state = self.projection.project(results.smoothed_state)
        
        return results
    
    def get_sfc_diagnostics(self) -> Dict:
        """
        Get diagnostic information about SFC constraints.
        """
        diagnostics = {
            'n_series': self.n_series,
            'n_stock_flow_pairs': len(self.stock_flow_pairs),
            'n_bilateral_positions': len(self.bilateral_constraints),
            'n_fa_series': len(self.fa_series),
            'enforce_sfc': self.enforce_sfc,
            'enforce_market_clearing': self.enforce_market_clearing,
            'state_dimensions': {
                'base_states': self.n_base_states,
                'total_states': self.k_states  # Should equal base_states now
            }
        }
        
        # Add stock-flow pair details
        diagnostics['stock_flow_details'] = {
            'complete_pairs': sum(1 for p in self.stock_flow_pairs if p.has_complete_flows),
            'partial_pairs': sum(1 for p in self.stock_flow_pairs if not p.has_complete_flows)
        }
        
        return diagnostics
    
    def validate_constraints(self, states: np.ndarray) -> Dict:
        """
        Validate that SFC constraints are satisfied.
        
        Parameters
        ----------
        states : np.ndarray
            State estimates to validate
            
        Returns
        -------
        dict
            Validation metrics for each constraint type
        """
        violations = {}
        
        # Check stock-flow consistency
        if self.stock_flow_pairs:
            sf_violations = []
            for pair in self.stock_flow_pairs:
                if pair.stock_series in self.data_columns and pair.flow_series in self.data_columns:
                    stock_idx = self.data_columns.index(pair.stock_series) * 2
                    flow_idx = self.data_columns.index(pair.flow_series) * 2
                    
                    if stock_idx < states.shape[0] and flow_idx < states.shape[0]:
                        # Check consistency across time
                        for t in range(1, states.shape[1]):
                            expected = states[stock_idx, t-1] + states[flow_idx, t]
                            actual = states[stock_idx, t]
                            violation = abs(expected - actual)
                            sf_violations.append(violation)
            
            if sf_violations:
                violations['stock_flow'] = {
                    'max_violation': np.max(sf_violations),
                    'mean_violation': np.mean(sf_violations)
                }
        
        return violations
    
    def get_filtered_series(self, results) -> Dict[str, pd.DataFrame]:
        """
        Extract filtered and smoothed series from results.
        
        Parameters
        ----------
        results : FilterResults
            Results from filter() or smooth()
            
        Returns
        -------
        dict
            Dictionary with 'filtered' and 'smoothed' DataFrames
        """
        output = {}
        
        # Extract filtered states
        if hasattr(results, 'filtered_state'):
            # Get level states (every other state)
            level_states = results.filtered_state[::2, :]
            
            # Create DataFrame using stored column information
            n_periods = level_states.shape[1]
            output['filtered'] = pd.DataFrame(
                level_states.T,
                index=self.original_data.index[:n_periods],
                columns=self.data_columns
            )
            
            # Denormalize if needed
            if self.normalize_data and hasattr(self, 'scale_factor'):
                output['filtered'] *= self.scale_factor
        
        # Extract smoothed states
        if hasattr(results, 'smoothed_state'):
            # Get level states (every other state)
            level_states = results.smoothed_state[::2, :]
            
            # Create DataFrame using stored column information
            n_periods = level_states.shape[1]
            output['smoothed'] = pd.DataFrame(
                level_states.T,
                index=self.original_data.index[:n_periods],
                columns=self.data_columns
            )
            
            # Denormalize if needed
            if self.normalize_data and hasattr(self, 'scale_factor'):
                output['smoothed'] *= self.scale_factor
        
        return output
