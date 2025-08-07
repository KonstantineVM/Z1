"""
Constrained UCM using statsmodels UnobservedComponents
This leverages the robust statsmodels implementation while adding formula constraints
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
from dataclasses import dataclass
import networkx as nx
from scipy.optimize import minimize
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import ray
from ray import remote
import multiprocessing


@ray.remote
def fit_individual_ucm_remote(series_data, series_name, ucm_config):
    """Remote function to fit individual UCM model"""
    try:
        model = UnobservedComponents(
            endog=series_data,
            **ucm_config
        )
        
        # Get model structure info without fitting
        return {
            'series_name': series_name,
            'k_states': model.k_states,
            'k_params': model.k_params,
            'param_names': model.param_names,
            'success': True
        }
    except Exception as e:
        logger.error(f"Error creating UCM for {series_name}: {e}")
        return {
            'series_name': series_name,
            'error': str(e),
            'success': False
        }


class ConstrainedUnobservedComponents(MLEModel):
    """
    Constrained Unobserved Components Model with parallel initialization
    """
    
    def __init__(self, data: pd.DataFrame, formulas: dict, series_mapping: dict,
                 level='local level', trend=True, seasonal=None, 
                 freq_seasonal=[{'period': 4, 'harmonics': 2}],
                 cycle=True, autoregressive=None, exog=None, 
                 irregular=True, stochastic_level=False, stochastic_trend=True,
                 stochastic_seasonal=True, stochastic_freq_seasonal=[True],
                 stochastic_cycle=True, damped_cycle=True,
                 cycle_period_bounds=None, mle_regression=True, 
                 use_exact_diffuse=False, use_parallel=True):
        
        # --- Basic Setup ---
        self.formulas = formulas
        self.series_mapping = series_mapping
        self.series_names = list(series_mapping.keys())
        self.n_series = len(self.series_names)
        self.use_parallel = use_parallel and ray.is_initialized()
        
        # --- Identify Source vs. Computed Series ---
        self.source_series = [
            s for s in self.series_names
            if not (formulas.get(s, {}).get("data_type") == "Computed" or
                    bool(formulas.get(s, {}).get("derived_from")))
        ]
        self.computed_series = [s for s in self.series_names if s not in self.source_series]
        
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
        
        # Initialize with better starting values
        self._initialize_params_from_individual_models()

    def _setup_individual_models(self, data):
        """Create individual UCM models for SOURCE series only."""
        self.individual_models = {}
        self.state_mappings = {}
        self.shock_mappings = {}
        
        cumulative_states = 0
        cumulative_shocks = 0

        # Iterate over source_series, not all series_names
        for series in self.source_series:
            try:
                # Log data statistics
                series_data = data[series]
                logger.info(f"Setting up model for {series}: mean={series_data.mean():.4f}, std={series_data.std():.4f}")
                
                model = UnobservedComponents(endog=series_data, **self.ucm_config)
                self.individual_models[series] = model
                
                self.state_mappings[series] = {
                    'start_idx': cumulative_states,
                    'end_idx': cumulative_states + model.k_states,
                    'level_idx': cumulative_states + model._idx_state_level[0] if hasattr(model, '_idx_state_level') and model._idx_state_level is not None else None,
                    'trend_idx': cumulative_states + model._idx_state_trend[0] if hasattr(model, '_idx_state_trend') and model._idx_state_trend is not None else None,
                    'cycle_idx': cumulative_states + model._idx_state_cycle[0] if hasattr(model, '_idx_state_cycle') and model._idx_state_cycle is not None else None,
                    'freq_seasonal_idx': cumulative_states + model._idx_state_freq_seasonal[0] if hasattr(model, '_idx_state_freq_seasonal') and model._idx_state_freq_seasonal is not None else None
                }
                
                n_shocks = model.k_posdef if hasattr(model, 'k_posdef') else model.k_states
                self.shock_mappings[series] = {
                    'start_idx': cumulative_shocks,
                    'end_idx': cumulative_shocks + n_shocks, 'n_shocks': n_shocks
                }
                cumulative_states += model.k_states
                cumulative_shocks += n_shocks
                
            except Exception as e:
                logger.error(f"Error setting up model for {series}: {e}")
                raise
                
        self.k_states_total = cumulative_states
        self.k_shocks_total = cumulative_shocks

    def _setup_combined_state_space(self):
        """Setup combined state space dimensions based on source models."""
        self.k_params_total = 0
        self.param_mappings = {}
        for series, model in self.individual_models.items():
            self.param_mappings[series] = {
                'start_idx': self.k_params_total,
                'end_idx': self.k_params_total + model.k_params
            }
            self.k_params_total += model.k_params

    def _setup_parameters(self):
        """Setup combined parameter names."""
        self.k_params = self.k_params_total
        self._param_names = [f"{s}.{p}" for s, m in self.individual_models.items() for p in m.param_names]

    @property
    def param_names(self):
        return self._param_names

    def initialize_state_space(self):
        """Initialize combined state space matrices."""
        self['transition'] = np.zeros((self.k_states_total, self.k_states_total))
        self['selection'] = np.zeros((self.k_states_total, self.k_shocks_total))
        self['state_cov'] = np.eye(self.k_shocks_total)
        self._build_design_matrix()
        self['obs_cov'] = np.eye(self.n_series) * 0.01
        
        # Use known initialization instead of approximate diffuse for better stability
        self.initialize_known(
            initial_state=np.zeros(self.k_states_total),
            initial_state_cov=np.eye(self.k_states_total) * 1e6
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
                formula_info = self.formulas[series_code]
                for comp in formula_info.get('derived_from', []):
                    nested_sign = sign if comp.get('operator', '+') == '+' else -sign
                    contributions.extend(get_source_contributions(comp.get('code'), nested_sign))
            return contributions

        for i, series in enumerate(self.series_names):
            source_contributions = get_source_contributions(series)
            for item in source_contributions:
                s_code, coef = item['code'], item['coef']
                if s_code in self.individual_models:
                    s_model = self.individual_models[s_code]
                    s_map = self.state_mappings[s_code]
                    start, end = s_map['start_idx'], s_map['end_idx']
                    if hasattr(s_model, 'design'):
                        self['design'][i, start:end] += coef * s_model['design'][0, :]

    def _initialize_params_from_individual_models(self):
        """Get better initial parameters by fitting individual models first."""
        initial_params = []
        
        for series, model in self.individual_models.items():
            try:
                # Fit individual model to get good starting values
                logger.info(f"Getting initial parameters for {series}")
                individual_result = model.fit(disp=False, maxiter=50, method='powell')
                initial_params.extend(individual_result.params)
                logger.info(f"  {series} individual fit successful, llf={individual_result.llf:.2f}")
            except Exception as e:
                # If individual fit fails, use default start params
                logger.warning(f"Could not fit individual model for {series}: {e}, using defaults")
                initial_params.extend(model.start_params)
        
        self._initial_params = np.array(initial_params)

    def update(self, params, **kwargs):
        """Update state space representation with new parameters."""
        params = np.asarray(params)
        
        # Add bounds checking
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            logger.warning("Invalid parameters detected, skipping update")
            return
            
        for series, model in self.individual_models.items():
            param_map = self.param_mappings[series]
            model.update(params[param_map['start_idx']:param_map['end_idx']], **kwargs)
            
            state_map = self.state_mappings[series]
            start, end = state_map['start_idx'], state_map['end_idx']
            self['transition'][start:end, start:end] = model['transition'].real
            
            shock_map = self.shock_mappings[series]
            if shock_map['n_shocks'] > 0:
                s_start, s_end = shock_map['start_idx'], shock_map['end_idx']
                
                model_selection = model['selection']
                shocks_in_model = model_selection.shape[1]
                shocks_allocated = s_end - s_start
                k_shocks = min(shocks_in_model, shocks_allocated)
                self['selection'][start:end, s_start:s_start + k_shocks] = model_selection[:, :k_shocks]
                
                model_state_cov = model['state_cov']
                cov_size = min(model_state_cov.shape[0], s_end - s_start)
                self['state_cov'][s_start:s_start+cov_size, s_start:s_start+cov_size] = model_state_cov[:cov_size, :cov_size].real

                # Add regularization to prevent singular matrices
                self['state_cov'][s_start:s_start+cov_size, s_start:s_start+cov_size] += 1e-6 * np.eye(cov_size)

        for i, series in enumerate(self.series_names):
            if series in self.computed_series:
                self['obs_cov'][i, i] = 1e-10
            elif series in self.individual_models:
                model = self.individual_models[series]
                if 'sigma2.irregular' in model.param_names:
                    idx = model.param_names.index('sigma2.irregular')
                    param_map = self.param_mappings[series]
                    obs_var = params[param_map['start_idx']:param_map['end_idx']][idx]
                    self['obs_cov'][i, i] = max(obs_var.real, 1e-6)

    @property
    def start_params(self):
        """Use better initial parameters if available."""
        if hasattr(self, '_initial_params'):
            return self._initial_params
        else:
            return np.concatenate([model.start_params for model in self.individual_models.values()])

    def transform_params(self, unconstrained):
        constrained = np.zeros_like(unconstrained)
        for series, model in self.individual_models.items():
            param_map = self.param_mappings[series]
            start, end = param_map['start_idx'], param_map['end_idx']
            constrained[start:end] = model.transform_params(unconstrained[start:end])
        return constrained

    def untransform_params(self, constrained):
        unconstrained = np.zeros_like(constrained)
        for series, model in self.individual_models.items():
            param_map = self.param_mappings[series]
            start, end = param_map['start_idx'], param_map['end_idx']
            unconstrained[start:end] = model.untransform_params(constrained[start:end])
        return unconstrained
    
    def fit(self, start_params=None, transformed=True, includes_fixed=False,
            cov_type='opg', cov_kwds=None, method='lbfgs', maxiter=50,
            full_output=1, disp=5, callback=None, return_params=False,
            optim_score=None, optim_complex_step=None,
            optim_hessian=None, **kwargs):
        """
        Fit the model with better optimization settings.
        """
        # Use better initial parameters if not provided
        if start_params is None:
            start_params = self.start_params
            
        # Log initial likelihood
        self.update(start_params)
        initial_llf = self.loglike(start_params)
        logger.info(f"Initial log-likelihood: {initial_llf}")
        
        # Set optimization options for better convergence
        if method == 'lbfgs':
            kwargs.setdefault('options', {})
            kwargs['options'].setdefault('ftol', 1e-8)
            kwargs['options'].setdefault('gtol', 1e-5)
            kwargs['options'].setdefault('maxls', 40)
        
        # Call parent fit method
        return super().fit(
            start_params=start_params, transformed=transformed,
            includes_fixed=includes_fixed, cov_type=cov_type,
            cov_kwds=cov_kwds, method=method, maxiter=maxiter,
            full_output=full_output, disp=disp, callback=callback,
            return_params=return_params, optim_score=optim_score,
            optim_complex_step=optim_complex_step,
            optim_hessian=optim_hessian, **kwargs
        )
    
    def get_components(self, which='smoothed', results=None):
        if results is None: 
            raise ValueError("Pass the fitted `results` object.")
            
        # Get states
        states = results.smoothed_state if which == 'smoothed' else results.filtered_state
        
        # Check if states are all zero
        if np.allclose(states, 0):
            logger.warning("All states are zero - model may not have converged properly")
            
        components = {
            'level': pd.DataFrame(index=self.data.index),
            'trend': pd.DataFrame(index=self.data.index),
            'cycle': pd.DataFrame(index=self.data.index),
            'freq_seasonal': pd.DataFrame(index=self.data.index),
            'irregular': pd.DataFrame(index=self.data.index)
        }
        
        # Stage 1: Robustly extract components for all SOURCE series
        for series in self.source_series:
            state_map = self.state_mappings[series]
            model = self.individual_models[series]
            series_states = states[state_map['start_idx']:state_map['end_idx'], :]
            
            # Log state statistics
            logger.debug(f"{series} states - mean: {np.mean(series_states):.6f}, std: {np.std(series_states):.6f}")
            
            # For each component type, extract if it exists, otherwise fill with zero
            for comp_name in ['level', 'trend', 'cycle', 'freq_seasonal']:
                idx_name = f'_idx_state_{comp_name}'
                if hasattr(model, idx_name) and getattr(model, idx_name) is not None:
                    idx = getattr(model, idx_name)
                    # Handle multiple indices for freq_seasonal
                    if comp_name == 'freq_seasonal' and isinstance(idx, list):
                        components[comp_name][series] = np.sum([series_states[i, :].real for i in idx], axis=0)
                    else:
                        comp_idx = idx[0] if isinstance(idx, list) else idx
                        components[comp_name][series] = series_states[comp_idx, :].real
                else:
                    components[comp_name][series] = 0.0

            # Calculate irregular component
            fitted_val = sum(components[c][series] for c in ['level', 'trend', 'cycle', 'freq_seasonal'])
            components['irregular'][series] = self.data[series].values - fitted_val

        # Stage 2: Augment with COMPUTED series
        if self.computed_series:
            components = self._augment_with_computed(components)
        return components

    def _augment_with_computed(self, comp_dict):
        out = {k: df.copy() for k, df in comp_dict.items()}
        computed_series_set = set(self.computed_series)
        
        G = nx.DiGraph()
        G.add_nodes_from(self.computed_series)
        for target in self.computed_series:
            for comp in self.formulas.get(target, {}).get("derived_from", []):
                if comp.get("code") in computed_series_set:
                    G.add_edge(comp.get("code"), target)
        
        try:
            calculation_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            raise RuntimeError("A circular dependency was detected in formulas.")
        
        for target in calculation_order:
            pairs = [(1 if d["operator"] == "+" else -1, d["code"]) for d in self.formulas[target].get("derived_from", [])]
            # Calculate components for the target series
            for cname, df in out.items():
                df[target] = sum(coef * df.get(src, 0.0) for coef, src in pairs)

        # Calculate irregulars for the computed series
        for target in self.computed_series:
            fitted_val = sum(out[c].get(target, 0.0) for c in ['level', 'trend', 'cycle', 'freq_seasonal'])
            out['irregular'][target] = self.data[target].values - fitted_val
        return out

def build_complete_dependency_groups(data: pd.DataFrame, formulas: Dict, identities: List[Dict], 
                                   series_list: Optional[List[str]] = None) -> List[Set[str]]:
    """
    Build complete dependency groups ensuring all formula components are included
    and filter out tautological constraints
    """
    # Filter to available series
    available_series = set(data.columns)
    if series_list:
        working_series = set(series_list) & available_series
    else:
        working_series = available_series
    
    # Build dependency graph
    G = nx.DiGraph()
    
    # Add nodes
    for series in working_series:
        G.add_node(series)
    
    # Add edges from formulas
    for series, formula_info in formulas.items():
        if series in working_series and formula_info.get('derived_from'):
            for component in formula_info['derived_from']:
                comp_code = component.get('code')
                if comp_code and comp_code in working_series:
                    G.add_edge(comp_code, series)  # component -> computed
    
    # Complete each group
    complete_groups = []
    processed = set()
    
    for series in working_series:
        if series not in processed:
            # Get complete dependencies
            group = set([series])
            
            # If series has a formula, add ALL its components
            if series in formulas and formulas[series].get('derived_from'):
                for component in formulas[series]['derived_from']:
                    comp_code = component.get('code')
                    if comp_code and comp_code in available_series:
                        group.add(comp_code)
                        # Recursively add dependencies
                        if comp_code in formulas:
                            for sub_comp in formulas[comp_code].get('derived_from', []):
                                if sub_comp['code'] in available_series:
                                    group.add(sub_comp['code'])
            
            # If series is used in formulas, consider including those
            for other_series, formula_info in formulas.items():
                if other_series in working_series:
                    components = [c['code'] for c in formula_info.get('derived_from', [])]
                    if series in components:
                        # This series is used in other_series formula
                        # Add if it completes the group
                        if all(c in group or c in available_series for c in components):
                            group.add(other_series)
                            group.update(c for c in components if c in available_series)
            
            complete_groups.append(group)
            processed.update(group)
    
    # Merge overlapping groups
    merged_groups = []
    for group in complete_groups:
        merged = False
        for i, existing_group in enumerate(merged_groups):
            if group & existing_group:  # If groups share any series
                merged_groups[i] = existing_group | group
                merged = True
                break
        if not merged:
            merged_groups.append(group)
    
    return merged_groups


def filter_valid_identities(identities: List[Dict], series_set: Set[str]) -> List[Dict]:
    """Filter out tautological identities and those with missing series"""
    valid_identities = []
    
    for identity in identities:
        # Check if it's tautological
        left_set = set(s.lstrip('Δ') for s in identity.get('left_side', []))
        right_set = set(s.lstrip('Δ') for s in identity.get('right_side', []))
        
        if left_set == right_set:
            logger.debug(f"Skipping tautological identity: {identity.get('identity_name')}")
            continue
        
        # Check if all series are available
        all_series = left_set | right_set
        if all_series.issubset(series_set):
            valid_identities.append(identity)
        else:
            missing = all_series - series_set
            logger.debug(f"Skipping identity {identity.get('identity_name')} - missing series: {missing}")
    
    return valid_identities


def fit_constrained_ucm(data: pd.DataFrame, formulas_file: str, 
                       series_list: Optional[List[str]] = None,
                       ucm_config: Optional[Dict] = None,
                       ensure_complete_groups: bool = True,
                       extract_filter: bool = True,
                       extract_smoother: bool = True) -> Tuple[ConstrainedUnobservedComponents, Dict]:
    """
    Fit constrained UCM model using statsmodels implementation
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    formulas_file : str
        Path to JSON file with formulas
    series_list : List[str], optional
        Specific series to process
    ucm_config : Dict, optional
        UCM configuration parameters
    ensure_complete_groups : bool
        If True, automatically include all dependencies
    extract_filter : bool
        If True, extract filtered (forward-pass only) components
    extract_smoother : bool
        If True, extract smoothed (forward+backward pass) components
        
    Returns:
    --------
    model : ConstrainedUnobservedComponents
        Fitted model
    results : Dict
        Results dictionary with components
    """
    
    # Load formulas and identities
    with open(formulas_file, 'r') as f:
        fof_data = json.load(f)
    
    formulas = fof_data.get('formulas', {})
    identities = fof_data.get('identities', [])
    
    # Get complete groups if requested
    if ensure_complete_groups:
        groups = build_complete_dependency_groups(data, formulas, identities, series_list)
        
        # Find group containing our series
        if series_list:
            relevant_groups = [g for g in groups if any(s in g for s in series_list)]
            if relevant_groups:
                # Merge all relevant groups
                complete_series = set()
                for g in relevant_groups:
                    complete_series.update(g)
                series_list = list(complete_series)
                logger.info(f"Expanded series list from {len(series_list)} to {len(complete_series)} to ensure complete dependencies")
            else:
                logger.warning("No complete groups found for specified series")
    
    # Select series
    if series_list:
        available = [s for s in series_list if s in data.columns]
        missing = set(series_list) - set(available)
        if missing:
            logger.warning(f"Missing series in data: {missing}")
        data_subset = data[available].copy()
    else:
        data_subset = data.copy()
    
    # Ensure data has proper frequency
    if data_subset.index.freq is None:
        # Try to infer quarterly frequency
        data_subset.index.freq = pd.infer_freq(data_subset.index)
        if data_subset.index.freq is None:
            # Force quarterly frequency
            data_subset.index = pd.date_range(
                start=data_subset.index[0], 
                periods=len(data_subset), 
                freq='QE'  # Use QE instead of Q
            )
            logger.info("Set quarterly frequency for time series data")
    
    # Filter valid identities
    valid_identities = filter_valid_identities(identities, set(data_subset.columns))
    logger.info(f"Using {len(valid_identities)} valid identities out of {len(identities)} total")
    
    # Log formula relationships
    computed_series = [s for s in data_subset.columns if s in formulas and formulas[s].get('derived_from')]
    source_series = [s for s in data_subset.columns if s not in computed_series]
    
    logger.info(f"Series breakdown: {len(source_series)} source, {len(computed_series)} computed")
    
    # Add a check to ensure there is at least one source series to model.
    if not source_series:
        raise ValueError(
            "The selected series group does not contain any source series."
            " A model cannot be built on computed series alone."
        )    
    
    # Check for lagged formulas
    lagged_formulas = []
    for series in computed_series:
        formula = formulas[series]
        formula_str = formula.get('formula', '')
        if '(-' in formula_str:
            lagged_formulas.append(series)
            components = [f"{c['operator']}{c['code']}" for c in formula.get('derived_from', [])]
            logger.info(f"  {series} = {' '.join(components)} [HAS LAGS: {formula_str}]")
        else:
            components = [f"{c['operator']}{c['code']}" for c in formula.get('derived_from', [])]
            logger.info(f"  {series} = {' '.join(components)}")
    
    if lagged_formulas:
        logger.warning(f"Found {len(lagged_formulas)} formulas with lags. Filter results may differ from smoother results.")
    
    # Create series mapping
    series_mapping = {col: i for i, col in enumerate(data_subset.columns)}
    
    # Default UCM configuration - exact specification requested
    if ucm_config is None:
        ucm_config = {
            'level': True,
            'trend': True,
            'seasonal': False,
            'freq_seasonal': [{'period': 4, 'harmonics': 2}],
            'cycle': True,
            'autoregressive': None,
            'exog': None,
            'irregular': True,
            'stochastic_level': False,
            'stochastic_trend': True,
            'stochastic_seasonal': True,
            'stochastic_freq_seasonal': [True],
            'stochastic_cycle': True,
            'damped_cycle': True,
            'cycle_period_bounds': None,
            'mle_regression': True,
            'use_exact_diffuse': False
        }
    
    logger.info(f"Fitting constrained UCM for {len(data_subset.columns)} series")
    
    # Create and fit model
    model = ConstrainedUnobservedComponents(
        data=data_subset,
        formulas=formulas,
        series_mapping=series_mapping,
        **ucm_config
    )
    
    # Try multiple optimization methods
    results = None
    for method in ['lbfgs', 'powell', 'nm']:
        try:
            logger.info(f"Trying optimization method: {method}")
            results = model.fit(disp=True, method=method, maxiter=100)
            if results.mle_retvals['converged']:
                logger.info(f"Converged using {method} method")
                break
        except Exception as e:
            logger.warning(f"Method {method} failed: {e}")
            continue
    
    if results is None:
        raise RuntimeError("All optimization methods failed")
    
    # Prepare output dictionary
    output_dict = {
        'fitted_model': results,
        'valid_identities': valid_identities,
        'complete_series_list': list(data_subset.columns),
        'lagged_formulas': lagged_formulas
    }
    
    # Extract filtered components
    if extract_filter:
        logger.info("Extracting filtered (forward-pass) components...")
        filtered_components = model.get_components('filtered', results)
        output_dict['filtered_components'] = filtered_components
        
        # Validate with filtered components
        filtered_validation = validate_formula_constraints(data_subset, filtered_components, formulas)
        output_dict['filtered_validation'] = filtered_validation
        
        # Check consistency with filtered
        filtered_consistency = check_level_trend_consistency(filtered_components)
        output_dict['filtered_consistency'] = filtered_consistency
    
    # Extract smoothed components
    if extract_smoother:
        logger.info("Extracting smoothed (forward+backward pass) components...")
        smoothed_components = model.get_components('smoothed', results)
        output_dict['smoothed_components'] = smoothed_components
        
        # Validate with smoothed components
        smoothed_validation = validate_formula_constraints(data_subset, smoothed_components, formulas)
        output_dict['smoothed_validation'] = smoothed_validation
        
        # Check consistency with smoothed
        smoothed_consistency = check_level_trend_consistency(smoothed_components)
        output_dict['smoothed_consistency'] = smoothed_consistency
    
    # Compare filter vs smoother if both extracted
    if extract_filter and extract_smoother:
        comparison = compare_filter_smoother(filtered_components, smoothed_components)
        output_dict['filter_smoother_comparison'] = comparison
    
    return model, output_dict


def compare_filter_smoother(filtered: Dict[str, pd.DataFrame], 
                           smoothed: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare filtered and smoothed components"""
    comparison_results = []
    
    for component in ['level', 'trend', 'cycle', 'freq_seasonal']:
        if component in filtered and component in smoothed:
            for series in filtered[component].columns:
                if series in smoothed[component].columns:
                    filt = filtered[component][series]
                    smth = smoothed[component][series]
                    
                    # Compute differences
                    mae = np.abs(filt - smth).mean()
                    rmse = np.sqrt(((filt - smth)**2).mean())
                    max_diff = np.abs(filt - smth).max()
                    
                    comparison_results.append({
                        'series': series,
                        'component': component,
                        'mae': mae,
                        'rmse': rmse,
                        'max_diff': max_diff,
                        'relative_mae': mae / np.abs(smth).mean() if np.abs(smth).mean() > 0 else np.nan
                    })
    
    return pd.DataFrame(comparison_results)


def validate_formula_constraints(data: pd.DataFrame, components: Dict[str, pd.DataFrame], 
                                formulas: Dict) -> pd.DataFrame:
    """Validate that formula constraints are satisfied"""
    validation_results = []
    
    for series in data.columns:
        if series in formulas and formulas[series].get('derived_from'):
            formula_info = formulas[series]
            
            # Compute formula result from components
            formula_result = pd.Series(0, index=data.index)
            
            for item in formula_info['derived_from']:
                comp_code = item.get('code')
                operator = item.get('operator', '+')
                
                if comp_code in components['level'].columns:
                    comp_sum = (components['level'][comp_code] + 
                               components['cycle'][comp_code] + 
                               components['freq_seasonal'][comp_code])
                    
                    if operator == '+':
                        formula_result += comp_sum
                    else:
                        formula_result -= comp_sum
            
            # Compare with actual
            actual = data[series]
            error = (actual - formula_result).abs().mean()
            
            validation_results.append({
                'series': series,
                'formula': formula_info.get('formula', ''),
                'mean_absolute_error': error,
                'relative_error': error / actual.abs().mean() if actual.abs().mean() > 0 else np.nan
            })
    
    return pd.DataFrame(validation_results)


def check_level_trend_consistency(components: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check if level = integrated trend"""
    consistency_results = []
    
    for series in components['level'].columns:
        if series in components['trend'].columns:
            level = components['level'][series].values
            trend = components['trend'][series].values
            
            # Check if diff(level) ≈ trend
            level_diff = np.diff(level)
            trend_vals = trend[:-1]
            
            mae = np.abs(level_diff - trend_vals).mean()
            correlation = np.corrcoef(level_diff, trend_vals)[0, 1] if len(level_diff) > 1 else np.nan
            
            consistency_results.append({
                'series': series,
                'mean_absolute_error': mae,
                'correlation': correlation,
                'is_consistent': mae < 1.0 and correlation > 0.95
            })
    
    return pd.DataFrame(consistency_results)


@ray.remote
def process_group_parallel(group_series, data, formulas, identities, ucm_config):
    """Process a group of series in parallel"""
    try:
        # Get data for this group
        group_data = data[list(group_series)].copy()
        
        # Create series mapping
        series_mapping = {col: i for i, col in enumerate(group_data.columns)}
        
        # Log group info
        computed = [s for s in group_series if s in formulas and formulas[s].get('derived_from')]
        source = [s for s in group_series if s not in computed]
        
        logger.info(f"Processing group with {len(group_series)} series ({len(source)} source, {len(computed)} computed)")
        
        # Create and fit model
        model = ConstrainedUnobservedComponents(
            data=group_data,
            formulas=formulas,
            series_mapping=series_mapping,
            use_parallel=False,  # Don't nest parallelism
            **ucm_config
        )
        
        # Fit the model
        results = model.fit(disp=False)
        
        # Extract components
        smoothed_components = model.get_components('smoothed', results)
        filtered_components = model.get_components('filtered', results)
        
        return {
            'group_series': list(group_series),
            'success': True,
            'smoothed_components': smoothed_components,
            'filtered_components': filtered_components,
            'loglikelihood': results.llf,
            'params': results.params,
            'param_names': model.param_names
        }
        
    except Exception as e:
        logger.error(f"Error processing group: {e}")
        return {
            'group_series': list(group_series),
            'success': False,
            'error': str(e)
        }


def fit_constrained_ucm_parallel(data: pd.DataFrame, formulas_file: str, 
                                series_list: Optional[List[str]] = None,
                                ucm_config: Optional[Dict] = None,
                                ensure_complete_groups: bool = True,
                                extract_filter: bool = True,
                                extract_smoother: bool = True,
                                n_cores: Optional[int] = None) -> Dict:
    """
    Fit constrained UCM model using parallel processing for multiple groups
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    formulas_file : str
        Path to JSON file with formulas
    series_list : List[str], optional
        Specific series to process
    ucm_config : Dict, optional
        UCM configuration parameters
    ensure_complete_groups : bool
        If True, automatically include all dependencies
    extract_filter : bool
        If True, extract filtered components
    extract_smoother : bool
        If True, extract smoothed components
    n_cores : int, optional
        Number of cores to use (default: all available)
        
    Returns:
    --------
    results : Dict
        Combined results from all groups
    """
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        n_cores = n_cores or multiprocessing.cpu_count()
        ray.init(num_cpus=n_cores, ignore_reinit_error=True)
        logger.info(f"Initialized Ray with {n_cores} cores")
    
    # Load formulas and identities
    with open(formulas_file, 'r') as f:
        fof_data = json.load(f)
    
    formulas = fof_data.get('formulas', {})
    identities = fof_data.get('identities', [])
    
    # Get complete groups
    groups = build_complete_dependency_groups(data, formulas, identities, series_list)
    logger.info(f"Found {len(groups)} complete dependency groups")
    
    # Filter groups to those we want to process
    if series_list:
        relevant_groups = [g for g in groups if any(s in g for s in series_list)]
    else:
        relevant_groups = groups
    
    logger.info(f"Processing {len(relevant_groups)} groups in parallel")
    
    # Default UCM configuration
    if ucm_config is None:
        ucm_config = {
            'level': 'local level',
            'trend': True,
            'seasonal': None,
            'freq_seasonal': [{'period': 4, 'harmonics': 2}],
            'cycle': True,
            'irregular': True,
            'stochastic_level': False,
            'stochastic_trend': True,
            'stochastic_freq_seasonal': [True],
            'stochastic_cycle': True,
            'damped_cycle': True
        }
    
    # Submit parallel tasks
    futures = []
    for i, group in enumerate(relevant_groups):
        logger.info(f"Submitting group {i+1}/{len(relevant_groups)} with {len(group)} series")
        future = process_group_parallel.remote(
            group, data, formulas, identities, ucm_config
        )
        futures.append(future)
    
    # Collect results
    logger.info("Waiting for parallel processing to complete...")
    group_results = ray.get(futures)
    
    # Combine results
    combined_results = {
        'all_series': [],
        'successful_series': [],
        'failed_groups': [],
        'group_results': group_results
    }
    
    if extract_smoother:
        combined_results['smoothed_components'] = {
            'level': pd.DataFrame(index=data.index),
            'trend': pd.DataFrame(index=data.index),
            'cycle': pd.DataFrame(index=data.index),
            'freq_seasonal': pd.DataFrame(index=data.index),
            'irregular': pd.DataFrame(index=data.index)
        }
    
    if extract_filter:
        combined_results['filtered_components'] = {
            'level': pd.DataFrame(index=data.index),
            'trend': pd.DataFrame(index=data.index),
            'cycle': pd.DataFrame(index=data.index),
            'freq_seasonal': pd.DataFrame(index=data.index),
            'irregular': pd.DataFrame(index=data.index)
        }
    
    # Merge results from all groups
    for result in group_results:
        combined_results['all_series'].extend(result['group_series'])
        
        if result['success']:
            combined_results['successful_series'].extend(result['group_series'])
            
            # Merge components
            if extract_smoother:
                for comp_name, comp_df in result['smoothed_components'].items():
                    for col in comp_df.columns:
                        combined_results['smoothed_components'][comp_name][col] = comp_df[col]
            
            if extract_filter:
                for comp_name, comp_df in result['filtered_components'].items():
                    for col in comp_df.columns:
                        combined_results['filtered_components'][comp_name][col] = comp_df[col]
        else:
            combined_results['failed_groups'].append({
                'series': result['group_series'],
                'error': result.get('error', 'Unknown error')
            })
    
    logger.info(f"Successfully processed {len(combined_results['successful_series'])} series")
    if combined_results['failed_groups']:
        logger.warning(f"Failed to process {len(combined_results['failed_groups'])} groups")
    
    # Validate results if we have smoothed components
    if extract_smoother and combined_results['successful_series']:
        validation = validate_formula_constraints(
            data[combined_results['successful_series']], 
            combined_results['smoothed_components'], 
            formulas
        )
        combined_results['validation'] = validation
        
        consistency = check_level_trend_consistency(combined_results['smoothed_components'])
        combined_results['consistency'] = consistency
    
    return combined_results