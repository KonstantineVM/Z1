# ==============================================================================
# FILE: src/models/sfc_kalman_filter.py
# ==============================================================================
"""
Stock-Flow Consistent Kalman Filter with extended state space.
Integrates stocks, flows, and trends with exact constraint enforcement.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.initialization import Initialization
import logging
from typing import Dict, List, Tuple, Optional
from .sfc_projection import SFCProjection

logger = logging.getLogger(__name__)


class SFCKalmanFilter(MLEModel):
    """
    Extended Kalman filter with stock-flow consistent state space.
    """
    
    def __init__(self, data: pd.DataFrame, 
                 formulas: Dict = None,
                 fwtw_data: Optional[pd.DataFrame] = None,
                 error_variance_ratio: float = 0.01,
                 normalize_data: bool = True,
                 enforce_sfc: bool = True,
                 **kwargs):
        """
        Initialize SFC Kalman Filter.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with stocks (FA/FL) and flows (FU/FR)
        formulas : Dict
            Z1 formulas (optional)
        fwtw_data : pd.DataFrame
            FWTW bilateral positions
        error_variance_ratio : float
            Ratio of observation error for computed vs source series
        normalize_data : bool
            Whether to normalize data
        enforce_sfc : bool
            Whether to enforce SFC constraints
        """
        self.original_data = data.copy()
        self.formulas = formulas or {}
        self.fwtw_data = fwtw_data
        self.error_variance_ratio = error_variance_ratio
        self.normalize_data = normalize_data
        self.enforce_sfc = enforce_sfc
        
        # Identify series types
        self.series_names = list(data.columns)
        self._categorize_series()
        
        # Setup normalization
        if normalize_data:
            self._setup_normalization()
            endog = self.original_data / self.scale_factor
        else:
            self.scale_factor = 1.0
            endog = self.original_data
        
        # Build extended state space
        self._build_extended_state_space()
        
        # Initialize parent class
        super().__init__(
            endog=endog,
            k_states=self.k_states,
            k_posdef=self.k_posdef,
            initialization='diffuse',
            loglikelihood_burn=10,
            **kwargs
        )
        
        # Setup state space matrices
        self._setup_state_space_matrices()
        
        # Initialize projection system if enforcing SFC
        if self.enforce_sfc:
            self.projector = SFCProjection(
                state_dim=self.k_states,
                series_names=self.series_names,
                state_mapping=self.state_mapping,
                normalize_scale=self.scale_factor
            )
        else:
            self.projector = None
        
        logger.info(f"SFC Kalman Filter initialized:")
        logger.info(f"  Series: {len(self.series_names)}")
        logger.info(f"  Stocks: {len(self.stock_series)}")
        logger.info(f"  Flows: {len(self.flow_series)}")
        logger.info(f"  State dimension: {self.k_states}")
        logger.info(f"  SFC enforcement: {self.enforce_sfc}")
    
    def _categorize_series(self):
        """Categorize series into stocks, flows, and others."""
        self.stock_series = []  # FA, FL series
        self.flow_series = []   # FU, FR series
        self.other_series = []  # Everything else
        self.stock_flow_pairs = []
        
        for series in self.series_names:
            if series.startswith('FA') or series.startswith('FL'):
                self.stock_series.append(series)
            elif series.startswith('FU') or series.startswith('FR'):
                self.flow_series.append(series)
            else:
                self.other_series.append(series)
        
        # Identify stock-flow pairs
        for stock in self.stock_series:
            if stock.startswith('FA'):
                flow = 'FU' + stock[2:]
            else:  # FL
                flow = 'FR' + stock[2:]
            
            if flow in self.flow_series:
                self.stock_flow_pairs.append((stock, flow))
        
        logger.info(f"Series categorization:")
        logger.info(f"  Stocks: {len(self.stock_series)}")
        logger.info(f"  Flows: {len(self.flow_series)}")
        logger.info(f"  Others: {len(self.other_series)}")
        logger.info(f"  Stock-flow pairs: {len(self.stock_flow_pairs)}")
    
    def _setup_normalization(self):
        """Setup global normalization to preserve accounting identities."""
        all_values = self.original_data.values.flatten()
        all_values = all_values[~np.isnan(all_values) & (all_values != 0)]
        
        if len(all_values) > 0:
            # Use robust scale estimator
            self.scale_factor = np.median(np.abs(all_values))
            if self.scale_factor == 0:
                self.scale_factor = 1.0
        else:
            self.scale_factor = 1.0
        
        logger.info(f"Normalization scale: {self.scale_factor:.2e}")
    
    def _build_extended_state_space(self):
        """Build extended state space with stocks, flows, and trends."""
        
        self.state_mapping = {}
        state_idx = 0
        
        # For each stock series: [Stock_Level, Stock_Trend, Flow_Level]
        for stock in self.stock_series:
            # Find corresponding flow
            if stock.startswith('FA'):
                flow = 'FU' + stock[2:]
            else:  # FL
                flow = 'FR' + stock[2:]
            
            has_flow = flow in self.flow_series
            
            self.state_mapping[stock] = {
                'level_idx': state_idx,
                'trend_idx': state_idx + 1,
                'flow_idx': state_idx + 2 if has_flow else None,
                'has_flow': has_flow
            }
            
            state_idx += 3 if has_flow else 2
        
        # For flow series without corresponding stocks: [Flow_Level, Flow_Trend]
        for flow in self.flow_series:
            if flow not in [pair[1] for pair in self.stock_flow_pairs]:
                self.state_mapping[flow] = {
                    'level_idx': state_idx,
                    'trend_idx': state_idx + 1,
                    'flow_idx': None,
                    'has_flow': False
                }
                state_idx += 2
        
        # For other series: [Level, Trend]
        for series in self.other_series:
            self.state_mapping[series] = {
                'level_idx': state_idx,
                'trend_idx': state_idx + 1,
                'flow_idx': None,
                'has_flow': False
            }
            state_idx += 2
        
        self.k_states = state_idx
        self.k_posdef = len(self.stock_series) + len(self.flow_series) + len(self.other_series)
        
        logger.info(f"Extended state space built: {self.k_states} states")
    
    def _setup_state_space_matrices(self):
        """Setup the state space system matrices."""
        
        # Design matrix (observation equation)
        self._setup_design_matrix()
        
        # Transition matrix (state evolution)
        self._setup_transition_matrix()
        
        # Selection matrix (state noise)
        self._setup_selection_matrix()
        
        # Covariance matrices
        self._setup_covariance_matrices()
    
    def _setup_design_matrix(self):
        """Setup observation matrix Z."""
        self['design'] = np.zeros((len(self.series_names), self.k_states))
        
        for i, series in enumerate(self.series_names):
            if series in self.state_mapping:
                # Observe the level component
                level_idx = self.state_mapping[series]['level_idx']
                self['design'][i, level_idx] = 1.0
    
    def _setup_transition_matrix(self):
        """Setup state transition matrix T with SFC constraints."""
        T = np.zeros((self.k_states, self.k_states))
        
        for series, mapping in self.state_mapping.items():
            level_idx = mapping['level_idx']
            trend_idx = mapping['trend_idx']
            flow_idx = mapping['flow_idx']
            
            if series in self.stock_series and flow_idx is not None:
                # Stock-flow consistent evolution
                # Stock[t] = Stock[t-1] + Flow[t]
                T[level_idx, level_idx] = 1.0  # Stock[t-1]
                T[level_idx, flow_idx] = 1.0   # + Flow[t]
                
                # Flow evolves with trend
                # Flow[t] = Flow[t-1] + Trend[t-1]
                T[flow_idx, flow_idx] = 1.0    # Flow[t-1]
                T[flow_idx, trend_idx] = 1.0   # + Trend[t-1]
                
                # Trend is random walk
                T[trend_idx, trend_idx] = 1.0
                
            else:
                # Standard local linear trend
                # Level[t] = Level[t-1] + Trend[t-1]
                T[level_idx, level_idx] = 1.0
                T[level_idx, trend_idx] = 1.0
                
                # Trend[t] = Trend[t-1]
                T[trend_idx, trend_idx] = 1.0
        
        self['transition'] = T
    
    def _setup_selection_matrix(self):
        """Setup selection matrix R for state noise."""
        R = np.zeros((self.k_states, self.k_posdef))
        noise_idx = 0
        
        for series, mapping in self.state_mapping.items():
            trend_idx = mapping['trend_idx']
            flow_idx = mapping['flow_idx']
            
            # Trend gets noise
            R[trend_idx, noise_idx] = 1.0
            noise_idx += 1
            
            # Flow gets noise if it exists and is independent
            if flow_idx is not None and series in self.flow_series:
                R[flow_idx, noise_idx] = 1.0
                noise_idx += 1
        
        self['selection'] = R
    
    def _setup_covariance_matrices(self):
        """Setup initial covariance matrices Q and H."""
        # State noise covariance
        self['state_cov'] = np.eye(self.k_posdef) * 0.01
        
        # Observation noise covariance
        self['obs_cov'] = np.eye(len(self.series_names)) * 1.0
    
    @property
    def param_names(self):
        """Names of parameters to estimate."""
        return [f'var_{i}' for i in range(self.k_posdef)] + ['obs_var']
    
    @property
    def start_params(self):
        """Starting values for parameters."""
        return np.ones(self.k_posdef + 1) * 0.1
    
    def transform_params(self, unconstrained):
        """Transform unconstrained parameters to variances."""
        return unconstrained ** 2
    
    def untransform_params(self, constrained):
        """Transform variances to unconstrained parameters."""
        return np.sqrt(np.maximum(constrained, 1e-10))
    
    def update(self, params, **kwargs):
        """Update system matrices with new parameters."""
        variances = self.transform_params(params)
        
        # Update state covariance
        self['state_cov'] = np.diag(variances[:-1])
        
        # Update observation covariance
        obs_var = variances[-1]
        self['obs_cov'] = np.eye(len(self.series_names)) * obs_var
    
    def filter(self, **kwargs):
        """Run filter with SFC constraint projection."""
        
        # Store previous states for constraints
        self.prev_states = []
        
        # Run standard Kalman filter
        results = super().filter(**kwargs)
        
        if self.enforce_sfc and self.projector is not None:
            logger.info("Applying SFC constraint projection...")
            
            # Apply projection at each time step
            filtered_states = results.filtered_state.copy()
            filtered_cov = results.filtered_state_cov.copy()
            
            for t in range(filtered_states.shape[1]):
                # Get previous state for dynamic constraints
                prev_state = filtered_states[:, t-1] if t > 0 else None
                
                # Build constraint matrices
                A, b = self.projector.build_constraint_matrices(t, prev_state)
                
                if A.shape[0] > 0:
                    # Project onto constraints
                    state_t = filtered_states[:, t]
                    cov_t = filtered_cov[:, :, t]
                    
                    state_proj, cov_proj = self.projector.project_onto_constraints(
                        state_t, cov_t, A, b, method='exact'
                    )
                    
                    # Update with projected values
                    filtered_states[:, t] = state_proj
                    filtered_cov[:, :, t] = cov_proj
            
            # Update results with constrained estimates
            results.filtered_state = filtered_states
            results.filtered_state_cov = filtered_cov
            
            # Validate constraints
            validation = self._validate_sfc_constraints(results)
            logger.info(f"SFC validation: {validation}")
        
        return results
    
    def _validate_sfc_constraints(self, results) -> Dict[str, float]:
        """Validate SFC constraints are satisfied."""
        
        violations = []
        states = results.filtered_state
        
        for t in range(1, states.shape[1]):
            prev_state = states[:, t-1]
            curr_state = states[:, t]
            
            # Check stock-flow consistency
            for stock, flow in self.stock_flow_pairs:
                if stock in self.state_mapping and flow in self.state_mapping:
                    stock_idx = self.state_mapping[stock]['level_idx']
                    flow_idx = self.state_mapping[flow]['level_idx']
                    
                    stock_change = curr_state[stock_idx] - prev_state[stock_idx]
                    flow_value = curr_state[flow_idx]
                    
                    violation = abs(stock_change - flow_value)
                    violations.append(violation)
        
        if violations:
            return {
                'max_violation': np.max(violations),
                'mean_violation': np.mean(violations),
                'n_constraints_checked': len(violations)
            }
        else:
            return {'max_violation': 0.0, 'mean_violation': 0.0}
    
    def get_filtered_series(self, results) -> Dict[str, pd.DataFrame]:
        """Extract filtered series from results."""
        
        # Get states
        filtered_states = results.filtered_state
        smoothed_states = results.smoothed_state if hasattr(results, 'smoothed_state') else None
        
        # Extract series
        filtered_data = np.zeros((len(self.original_data), len(self.series_names)))
        smoothed_data = np.zeros_like(filtered_data) if smoothed_states is not None else None
        
        Z = self['design']
        
        for t in range(len(self.original_data)):
            filtered_data[t, :] = Z @ filtered_states[:, t]
            if smoothed_data is not None:
                smoothed_data[t, :] = Z @ smoothed_states[:, t]
        
        # Denormalize
        if self.normalize_data:
            filtered_data *= self.scale_factor
            if smoothed_data is not None:
                smoothed_data *= self.scale_factor
        
        # Create DataFrames
        filtered_df = pd.DataFrame(
            filtered_data,
            index=self.original_data.index,
            columns=self.original_data.columns
        )
        
        result = {'filtered': filtered_df}
        
        if smoothed_data is not None:
            smoothed_df = pd.DataFrame(
                smoothed_data,
                index=self.original_data.index,
                columns=self.original_data.columns
            )
            result['smoothed'] = smoothed_df
        
        return result
