"""
Robust Constrained UCM with improved numerical stability
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate, unconstrain_stationary_univariate
)
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx
from scipy.optimize import minimize
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustConstrainedUCM(MLEModel):
    """
    Robust Constrained Unobserved Components Model with better numerical stability
    """
    
    def __init__(self, data: pd.DataFrame, formulas: dict, series_mapping: dict,
                 level='local level', trend=True, seasonal=None, 
                 freq_seasonal=[{'period': 4, 'harmonics': 2}],
                 cycle=True, autoregressive=None, exog=None, 
                 irregular=True, stochastic_level=False, stochastic_trend=True,
                 stochastic_seasonal=True, stochastic_freq_seasonal=[True],
                 stochastic_cycle=True, damped_cycle=True,
                 cycle_period_bounds=None, mle_regression=True, 
                 use_exact_diffuse=False, normalize_data=True):
        
        # --- Basic Setup ---
        self.formulas = formulas
        self.series_mapping = series_mapping
        self.series_names = list(series_mapping.keys())
        self.n_series = len(self.series_names)
        self.normalize_data = normalize_data
        
        # --- Data Normalization for Numerical Stability ---
        self.original_data = data.copy()
        if normalize_data:
            self.data_means = data.mean()
            self.data_stds = data.std()
            # Avoid division by zero
            self.data_stds[self.data_stds < 1e-8] = 1.0
            data = (data - self.data_means) / self.data_stds
            logger.info("Data normalized for numerical stability")
        
        # --- Identify Source vs. Computed Series ---
        self.source_series = [
            s for s in self.series_names
            if not (formulas.get(s, {}).get("data_type") == "Computed" or
                    bool(formulas.get(s, {}).get("derived_from")))
        ]
        self.computed_series = [s for s in self.series_names if s not in self.source_series]
        
        logger.info(f"Source series ({len(self.source_series)}): {self.source_series[:5]}...")
        logger.info(f"Computed series ({len(self.computed_series)}): {self.computed_series[:5]}...")
        
        # --- UCM Configuration ---
        self.ucm_config = {
            'level': level, 'trend': trend, 'seasonal': seasonal, 'freq_seasonal': freq_seasonal,
            'cycle': cycle, 'autoregressive': autoregressive, 'irregular': irregular,
            'stochastic_level': stochastic_level, 'stochastic_trend': stochastic_trend,
            'stochastic_seasonal': stochastic_seasonal, 'stochastic_freq_seasonal': stochastic_freq_seasonal,
            'stochastic_cycle': stochastic_cycle, 'damped_cycle': damped_cycle,
            'cycle_period_bounds': cycle_period_bounds
        }
        
        # --- Build State Space from SOURCE series ONLY ---
        self._setup_individual_models(data)
        self._setup_combined_state_space()
        
        # --- Initialize the statsmodels MLEModel ---
        super().__init__(
            endog=data.values, k_states=self.k_states_total, exog=exog,
            dates=data.index, freq=data.index.freq
        )
        self.data = data
        self._endog_names = data.columns.tolist()
        
        self._setup_parameters()
        self.initialize_state_space()
        
        # Initialize parameters from individual models
        self._initialize_params_from_individual_models()

    def _setup_individual_models(self, data):
        """Create individual UCM models for SOURCE series only."""
        self.individual_models = {}
        self.individual_results = {}  # Store individual fit results
        self.state_mappings = {}
        self.shock_mappings = {}
        
        cumulative_states = 0
        cumulative_shocks = 0

        for series in self.source_series:
            try:
                series_data = data[series]
                logger.info(f"Setting up model for {series}: mean={series_data.mean():.4f}, std={series_data.std():.4f}")
                
                # Create individual model
                model = UnobservedComponents(endog=series_data, **self.ucm_config)
                self.individual_models[series] = model
                
                # Try to fit individual model
                try:
                    individual_result = model.fit(disp=False, maxiter=50, method='powell')
                    self.individual_results[series] = individual_result
                    logger.info(f"  {series} individual fit: llf={individual_result.llf:.2f}")
                except Exception as e:
                    logger.warning(f"  {series} individual fit failed: {e}")
                    self.individual_results[series] = None
                
                # Setup state mappings
                self.state_mappings[series] = {
                    'start_idx': cumulative_states,
                    'end_idx': cumulative_states + model.k_states,
                    'k_states': model.k_states
                }
                
                # Store component indices if they exist
                for comp in ['level', 'trend', 'cycle', 'freq_seasonal']:
                    idx_name = f'_idx_state_{comp}'
                    if hasattr(model, idx_name) and getattr(model, idx_name) is not None:
                        idx = getattr(model, idx_name)
                        if isinstance(idx, list) and len(idx) > 0:
                            self.state_mappings[series][f'{comp}_idx'] = cumulative_states + idx[0]
                
                # Setup shock mappings
                n_shocks = model.k_posdef if hasattr(model, 'k_posdef') else model.k_states
                self.shock_mappings[series] = {
                    'start_idx': cumulative_shocks,
                    'end_idx': cumulative_shocks + n_shocks,
                    'n_shocks': n_shocks
                }
                
                cumulative_states += model.k_states
                cumulative_shocks += n_shocks
                
            except Exception as e:
                logger.error(f"Error setting up model for {series}: {e}")
                raise
                
        self.k_states_total = cumulative_states
        self.k_shocks_total = cumulative_shocks
        logger.info(f"Total states: {self.k_states_total}, Total shocks: {self.k_shocks_total}")

    def _setup_combined_state_space(self):
        """Setup combined state space dimensions based on source models."""
        self.k_params_total = 0
        self.param_mappings = {}
        for series, model in self.individual_models.items():
            self.param_mappings[series] = {
                'start_idx': self.k_params_total,
                'end_idx': self.k_params_total + model.k_params,
                'k_params': model.k_params
            }
            self.k_params_total += model.k_params

    def _setup_parameters(self):
        """Setup combined parameter names."""
        self.k_params = self.k_params_total
        self._param_names = []
        for s, m in self.individual_models.items():
            self._param_names.extend([f"{s}.{p}" for p in m.param_names])

    @property
    def param_names(self):
        return self._param_names

    def initialize_state_space(self):
        """Initialize combined state space matrices with proper scaling."""
        # Initialize transition matrix
        self['transition'] = np.zeros((self.k_states_total, self.k_states_total))
        
        # Initialize selection matrix
        self['selection'] = np.zeros((self.k_states_total, self.k_shocks_total))
        
        # Initialize state covariance with small positive values
        self['state_cov'] = np.eye(self.k_shocks_total) * 0.01
        
        # Build design matrix
        self._build_design_matrix()
        
        # Initialize observation covariance
        self['obs_cov'] = np.eye(self.n_series) * 0.1
        
        # Use finite initialization instead of diffuse
        initial_variance = 100.0  # Moderate initial uncertainty
        self.initialize_known(
            initial_state=np.zeros(self.k_states_total),
            initial_state_cov=np.eye(self.k_states_total) * initial_variance
        )

    def _build_design_matrix(self):
        """Build design (Z) matrix with recursive formula resolution."""
        self['design'] = np.zeros((self.n_series, self.k_states_total))

        def get_source_contributions(series_code, sign=1.0):
            """Recursively find all source series that contribute to a series."""
            contributions = []
            if series_code in self.source_series:
                contributions.append({'code': series_code, 'coef': sign})
            elif series_code in self.computed_series:
                formula_info = self.formulas.get(series_code, {})
                for comp in formula_info.get('derived_from', []):
                    comp_code = comp.get('code')
                    operator = comp.get('operator', '+')
                    nested_sign = sign if operator == '+' else -sign
                    contributions.extend(get_source_contributions(comp_code, nested_sign))
            return contributions

        # Build design matrix row by row
        for i, series in enumerate(self.series_names):
            source_contributions = get_source_contributions(series)
            
            for item in source_contributions:
                s_code = item['code']
                coef = item['coef']
                
                if s_code in self.individual_models:
                    s_model = self.individual_models[s_code]
                    s_map = self.state_mappings[s_code]
                    start = s_map['start_idx']
                    end = s_map['end_idx']
                    
                    # Copy the design row from individual model
                    if hasattr(s_model, 'design') and s_model['design'] is not None:
                        self['design'][i, start:end] += coef * s_model['design'][0, :]

    def _initialize_params_from_individual_models(self):
        """Get initial parameters from individual model fits or defaults."""
        initial_params = []
        
        for series in self.source_series:
            if series in self.individual_results and self.individual_results[series] is not None:
                # Use fitted parameters
                params = self.individual_results[series].params
                initial_params.extend(params)
            else:
                # Use default starting parameters
                model = self.individual_models[series]
                initial_params.extend(model.start_params)
        
        self._initial_params = np.array(initial_params)
        
        # Ensure parameters are finite
        if not np.all(np.isfinite(self._initial_params)):
            logger.warning("Some initial parameters are not finite, using defaults")
            self._initial_params = np.concatenate([
                model.start_params for model in self.individual_models.values()
            ])

    def update(self, params, **kwargs):
        """Update state space representation with new parameters."""
        params = np.asarray(params)
        
        # Check for invalid parameters
        if not np.all(np.isfinite(params)):
            return
        
        # Update each individual model
        for series, model in self.individual_models.items():
            param_map = self.param_mappings[series]
            series_params = params[param_map['start_idx']:param_map['end_idx']]
            
            # Update individual model
            try:
                model.update(series_params, **kwargs)
            except Exception as e:
                logger.debug(f"Error updating {series}: {e}")
                continue
            
            # Update transition matrix
            state_map = self.state_mappings[series]
            start = state_map['start_idx']
            end = state_map['end_idx']
            
            if hasattr(model, 'transition') and model['transition'] is not None:
                trans_block = model['transition']
                if trans_block.shape == (end-start, end-start):
                    self['transition'][start:end, start:end] = trans_block.real
            
            # Update selection matrix
            shock_map = self.shock_mappings[series]
            if shock_map['n_shocks'] > 0:
                s_start = shock_map['start_idx']
                s_end = shock_map['end_idx']
                
                if hasattr(model, 'selection') and model['selection'] is not None:
                    sel_block = model['selection']
                    state_dim = end - start
                    shock_dim = s_end - s_start
                    
                    if sel_block.shape[0] == state_dim and sel_block.shape[1] <= shock_dim:
                        self['selection'][start:end, s_start:s_start+sel_block.shape[1]] = sel_block.real
                
                # Update state covariance
                if hasattr(model, 'state_cov') and model['state_cov'] is not None:
                    cov_block = model['state_cov']
                    cov_size = min(cov_block.shape[0], shock_dim)
                    if cov_size > 0:
                        self['state_cov'][s_start:s_start+cov_size, s_start:s_start+cov_size] = (
                            cov_block[:cov_size, :cov_size].real + np.eye(cov_size) * 1e-6
                        )

        # Update observation covariance
        for i, series in enumerate(self.series_names):
            if series in self.computed_series:
                # Very small variance for computed series
                self['obs_cov'][i, i] = 1e-8
            elif series in self.individual_models:
                model = self.individual_models[series]
                if 'sigma2.irregular' in model.param_names:
                    idx = model.param_names.index('sigma2.irregular')
                    param_map = self.param_mappings[series]
                    param_idx = param_map['start_idx'] + idx
                    if param_idx < len(params):
                        obs_var = max(params[param_idx].real, 1e-6)
                        self['obs_cov'][i, i] = obs_var

    @property  
    def start_params(self):
        """Return initial parameters."""
        if hasattr(self, '_initial_params'):
            return self._initial_params
        else:
            return np.concatenate([
                model.start_params for model in self.individual_models.values()
            ])

    def transform_params(self, unconstrained):
        """Transform parameters to constrained space."""
        constrained = np.zeros_like(unconstrained)
        for series, model in self.individual_models.items():
            param_map = self.param_mappings[series]
            start = param_map['start_idx']
            end = param_map['end_idx']
            try:
                constrained[start:end] = model.transform_params(unconstrained[start:end])
            except:
                constrained[start:end] = unconstrained[start:end]
        return constrained

    def untransform_params(self, constrained):
        """Transform parameters to unconstrained space."""
        unconstrained = np.zeros_like(constrained)
        for series, model in self.individual_models.items():
            param_map = self.param_mappings[series]
            start = param_map['start_idx']
            end = param_map['end_idx']
            try:
                unconstrained[start:end] = model.untransform_params(constrained[start:end])
            except:
                unconstrained[start:end] = constrained[start:end]
        return unconstrained
    
    def loglike(self, params, **kwargs):
        """Compute log-likelihood with numerical checks."""
        try:
            llf = super().loglike(params, **kwargs)
            
            # Check for numerical issues
            if not np.isfinite(llf) or np.abs(llf) > 1e20:
                logger.debug(f"Numerical issue in likelihood: {llf}")
                return -1e20  # Return large negative number instead
            
            return llf
        except Exception as e:
            logger.debug(f"Error in loglike: {e}")
            return -1e20
    
    def get_components(self, which='smoothed', results=None):
        """Extract components with denormalization if needed."""
        if results is None: 
            raise ValueError("Pass the fitted `results` object.")
            
        # Get states
        if which == 'smoothed':
            states = results.smoothed_state
        elif which == 'filtered':
            states = results.filtered_state
        else:
            raise ValueError("which must be 'smoothed' or 'filtered'")
            
        # Check for zero states
        state_norm = np.linalg.norm(states)
        if state_norm < 1e-10:
            logger.warning(f"States have very small norm: {state_norm}")
            
        # Initialize component DataFrames
        components = {
            'level': pd.DataFrame(index=self.data.index),
            'trend': pd.DataFrame(index=self.data.index),
            'cycle': pd.DataFrame(index=self.data.index),
            'freq_seasonal': pd.DataFrame(index=self.data.index),
            'irregular': pd.DataFrame(index=self.data.index)
        }
        
        # Extract components for source series
        for series in self.source_series:
            state_map = self.state_mappings[series]
            model = self.individual_models[series]
            
            # Get series states
            start = state_map['start_idx']
            end = state_map['end_idx']
            series_states = states[start:end, :]
            
            # Extract each component
            for comp_name in ['level', 'trend', 'cycle', 'freq_seasonal']:
                comp_idx_key = f'{comp_name}_idx'
                if comp_idx_key in state_map:
                    state_idx = state_map[comp_idx_key] - start  # Relative index
                    if 0 <= state_idx < series_states.shape[0]:
                        comp_values = series_states[state_idx, :].real
                        
                        # Denormalize if data was normalized
                        if self.normalize_data:
                            comp_values = comp_values * self.data_stds[series]
                            
                        components[comp_name][series] = comp_values
                    else:
                        components[comp_name][series] = 0.0
                else:
                    components[comp_name][series] = 0.0
            
            # Calculate irregular component
            fitted_val = sum(components[c][series] for c in ['level', 'trend', 'cycle', 'freq_seasonal'])
            actual_val = self.original_data[series].values if self.normalize_data else self.data[series].values
            components['irregular'][series] = actual_val - fitted_val

        # Augment with computed series
        if self.computed_series:
            components = self._augment_with_computed(components)
            
        return components

    def _augment_with_computed(self, comp_dict):
        """Add computed series to components."""
        out = {k: df.copy() for k, df in comp_dict.items()}
        
        # Build dependency graph
        G = nx.DiGraph()
        computed_set = set(self.computed_series)
        
        for target in self.computed_series:
            G.add_node(target)
            formula_info = self.formulas.get(target, {})
            for comp in formula_info.get("derived_from", []):
                source = comp.get("code")
                if source in computed_set:
                    G.add_edge(source, target)
        
        # Get calculation order
        try:
            calculation_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            logger.error("Circular dependency detected in formulas")
            calculation_order = self.computed_series
        
        # Calculate components for computed series
        for target in calculation_order:
            formula_info = self.formulas.get(target, {})
            derived_from = formula_info.get("derived_from", [])
            
            for comp_name, df in out.items():
                df[target] = 0.0
                for item in derived_from:
                    source = item.get("code")
                    operator = item.get("operator", "+")
                    coef = 1.0 if operator == "+" else -1.0
                    
                    if source in df.columns:
                        df[target] += coef * df[source]
        
        # Calculate irregulars for computed series
        for target in self.computed_series:
            fitted = sum(out[c][target] for c in ['level', 'trend', 'cycle', 'freq_seasonal'])
            actual = self.original_data[target].values if self.normalize_data else self.data[target].values
            out['irregular'][target] = actual - fitted
            
        return out


def test_robust_ucm(data, formulas, series_list, ucm_config=None):
    """Test the robust UCM implementation."""
    
    # Create series mapping
    series_mapping = {col: i for i, col in enumerate(data[series_list].columns)}
    
    # Default configuration
    if ucm_config is None:
        ucm_config = {
            'level': True,
            'trend': True,
            'seasonal': False,
            'freq_seasonal': [{'period': 4, 'harmonics': 2}],
            'cycle': True,
            'irregular': True,
            'stochastic_level': False,
            'stochastic_trend': True,
            'stochastic_freq_seasonal': [True],
            'stochastic_cycle': True,
            'damped_cycle': True
        }
    
    # Create model with normalization
    model = RobustConstrainedUCM(
        data=data[series_list],
        formulas=formulas,
        series_mapping=series_mapping,
        normalize_data=True,  # Enable normalization
        **ucm_config
    )
    
    # Try different optimization methods
    methods = ['powell', 'lbfgs', 'nm']
    best_result = None
    best_llf = -np.inf
    
    for method in methods:
        try:
            logger.info(f"\nTrying method: {method}")
            result = model.fit(
                method=method,
                disp=True,
                maxiter=100,
                cov_type='none'  # Skip covariance calculation for speed
            )
            
            if result.llf > best_llf and np.isfinite(result.llf):
                best_llf = result.llf
                best_result = result
                logger.info(f"New best: {method} with llf={best_llf:.2f}")
                
        except Exception as e:
            logger.warning(f"Method {method} failed: {e}")
            continue
    
    if best_result is None:
        raise RuntimeError("All optimization methods failed")
    
    return model, best_result