"""
Hierarchical Kalman Filter with Global Normalization for FoF Data

This implementation preserves accounting identities while providing numerical stability.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.filters.hp_filter import hpfilter
import json
import logging
import re
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DependencyGraph:
    """
    Handles dependency logic with a specific rule for FR/FV cycles.
    """
    def __init__(self, formulas: Dict, all_series_names: List[str]):
        self.formulas = formulas
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(all_series_names)
        
        self._build_graph_from_formulas()
        self._resolve_fr_fv_cycles()
        
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"Could not resolve all circular dependencies. Remaining cycles: {cycles}")

        self._identify_series_types()
        self._precompute_source_contributions()

    def _parse_formula_string(self, formula_str: str) -> Dict[str, int]:
        """
        Parses a formula string and returns a mapping
        {series_code: lag}.  Accepts either  CODE[t-1]  or  CODE(-1).
        """
        components: Dict[str, int] = {}

        pattern = re.compile(
            r'([A-Z0-9_]+)'                     # series code
            r'(?:'                              # --- optional lag group ---
            r'\[\s*t\s*([+-]?\d+)\s*\]'         #   [t-1]  or [t+1]
            r'|'                                #   or
            r'\(\s*([+-]?\d+)\s*\)'             #   (-1)
            r')?'                               # --------------------------------
        )

        for match in pattern.finditer(formula_str):
            series_code, lag1, lag2 = match.groups()
            lag = lag1 or lag2 or "0"
            components[series_code] = int(lag)

        return components

    def _build_graph_from_formulas(self):
        """Builds the graph based on all components found in formula strings."""
        for series, details in self.formulas.items():
            if series not in self.graph: continue
            
            formula_str = details.get("formula", "")
            if not formula_str: continue

            parsed_components = self._parse_formula_string(formula_str)
            
            for comp_code in parsed_components.keys():
                if comp_code in self.graph and comp_code != series:
                    self.graph.add_edge(comp_code, series)
    
    def _resolve_fr_fv_cycles(self):
        """
        Detects and resolves 2-node cycles based on the rule 'FR is primary'.
        """
        while not nx.is_directed_acyclic_graph(self.graph):
            try:
                cycle_edges = nx.find_cycle(self.graph, orientation='original')
                
                if len(cycle_edges) == 2:
                    u, v = cycle_edges[0][0], cycle_edges[0][1]
                    
                    u_is_fr, u_is_fv = u.startswith('FR'), u.startswith('FV')
                    v_is_fr, v_is_fv = v.startswith('FR'), v.startswith('FV')

                    if (u_is_fr and v_is_fv):
                        logger.info(f"Resolving cycle based on FR/FV rule: '{u}' is primary. Removing edge {v} -> {u}.")
                        self.graph.remove_edge(v, u)
                    elif (v_is_fr and u_is_fv):
                        logger.info(f"Resolving cycle based on FR/FV rule: '{v}' is primary. Removing edge {u} -> {v}.")
                        self.graph.remove_edge(u, v)
                    else:
                        logger.error(f"Detected an unresolvable cycle between {u} and {v}.")
                        break 
                else:
                    logger.error(f"Detected a complex cycle of length {len(cycle_edges)}. Cannot resolve automatically.")
                    break
            except nx.NetworkXNoCycle:
                break
                
    def _identify_series_types(self):
        self.source_series = sorted([n for n, d in self.graph.in_degree() if d == 0])
        self.computed_series = sorted(list(set(self.graph.nodes()) - set(self.source_series)))

    def _precompute_source_contributions(self):
        """Computes contributions from source series, respecting operators."""
        self.source_contributions = {}
        for series in nx.topological_sort(self.graph):
            if series in self.source_series:
                self.source_contributions[series] = {0: {series: 1.0}}
            else:
                aggregated_contribs = {}
                formula_info = self.formulas.get(series, {})
                
                # Use the structured `derived_from` to get operators correctly
                for component in formula_info.get("derived_from", []):
                    comp_code = component.get("code")
                    operator = component.get("operator", "+")
                    sign = -1.0 if operator == "-" else 1.0
                    
                    formula_str = formula_info.get("formula", "")
                    parsed_lags = self._parse_formula_string(formula_str)
                    lag = parsed_lags.get(comp_code, 0)
                    
                    comp_contribs = self.source_contributions.get(comp_code, {})
                    for comp_lag, sources in comp_contribs.items():
                        new_lag = lag + comp_lag
                        if new_lag not in aggregated_contribs: aggregated_contribs[new_lag] = {}
                        
                        for source, coef in sources.items():
                            # Apply the correct sign here
                            aggregated_contribs[new_lag][source] = aggregated_contribs[new_lag].get(source, 0) + (sign * coef)
                            
                self.source_contributions[series] = aggregated_contribs

    def get_all_dependencies(self, series_list: List[str]) -> Set[str]:
        required_series = set(series_list)
        for series in series_list:
            if series in self.graph: required_series.update(nx.ancestors(self.graph, series))
        return required_series


class HierarchicalKalmanFilter(MLEModel):
    """
    Hierarchical Kalman Filter with global normalization for FoF data.
    """
    
    def __init__(self, data: pd.DataFrame, formulas: dict, 
                 error_variance_ratio: float = 0.01,
                 normalize_data: bool = True,
                 loglikelihood_burn: int = 10,
                 use_exact_diffuse: bool = False,
                 transformation: str = 'square',
                 **kwargs):
        self.original_data = data.copy()
        self.formulas = formulas
        self.error_variance_ratio = error_variance_ratio
        self.normalize_data = normalize_data
        self.use_exact_diffuse = use_exact_diffuse
        self.transformation = transformation
        
        self.series_names = list(data.columns)
        self.dep_graph = DependencyGraph(self.formulas, self.series_names)
        
        self.source_series = self.dep_graph.source_series
        self.computed_series = self.dep_graph.computed_series
        self.n_series = len(self.series_names)
        self.n_source = len(self.source_series)
        
        self.source_info = {}
        k_states_total = 0
        for series in self.source_series:
            max_lag = 0
            for s_name in self.series_names:
                for lag, sources in self.dep_graph.source_contributions.get(s_name, {}).items():
                    if series in sources: max_lag = max(max_lag, abs(lag))
            
            k_states_series = 2 + max_lag
            self.source_info[series] = {'start_idx': k_states_total, 'k_states': k_states_series, 'max_lag': max_lag}
            k_states_total += k_states_series
            
        self.k_states = k_states_total
        self.k_posdef = self.n_source
        
        logger.info(f"State dimension augmented for lags: {self.k_states}")
        
        # GLOBAL NORMALIZATION to preserve FoF identities
        if normalize_data:
            # Find robust global scale
            all_values = data.values.flatten()
            all_values = all_values[~np.isnan(all_values) & (all_values != 0)]
            
            # Use median absolute value as scale
            self.scale_factor = np.median(np.abs(all_values))
            if self.scale_factor == 0:
                self.scale_factor = 1.0
            
            # Normalize all series by same factor
            data = data / self.scale_factor
            
            logger.info(f"Global normalization scale: {self.scale_factor:.2e}")
            
            # Verify identities are preserved
            self._verify_identities(data, "after normalization")
        else:
            self.scale_factor = 1.0
            data = data.copy()
        
        # Set loglikelihood_burn
        if 'loglikelihood_burn' not in kwargs:
            kwargs['loglikelihood_burn'] = loglikelihood_burn
            
        super().__init__(endog=data.values, k_states=self.k_states, k_posdef=self.k_posdef, **kwargs)
        self._setup_parameters()
        self._build_state_space()
        self.initialize_state_space()
    
    def _verify_identities(self, data, stage=""):
        """Verify that FoF identities hold within tolerance."""
        logger.info(f"\nVerifying FoF identities {stage}...")
        max_error = 0
        
        for series in self.computed_series[:3]:  # Check first few
            formula_info = self.formulas.get(series, {})
            derived_from = formula_info.get("derived_from", [])
            if not derived_from:
                continue
                
            reconstructed = pd.Series(0.0, index=data.index)
            for component in derived_from:
                comp_code = component.get("code")
                if comp_code in data.columns:
                    operator = component.get("operator", "+")
                    sign = -1.0 if operator == "-" else 1.0
                    reconstructed += sign * data[comp_code]
            
            error = (data[series] - reconstructed).abs().mean()
            rel_error = error / data[series].abs().mean() if data[series].abs().mean() > 0 else 0
            max_error = max(max_error, rel_error)
            
            if rel_error > 0.01:
                logger.warning(f"  {series}: relative error = {rel_error:.2%}")
        
        if max_error < 0.01:
            logger.info(f"  All identities preserved (max error: {max_error:.2%})")
    
    def _setup_parameters(self):
        self.k_params = self.n_source + 1
        self._param_names = [f'sigma2.trend.{s}' for s in self.source_series] + ['sigma2.obs']
    
    @property
    def param_names(self): 
        return self._param_names
    
    def _build_state_space(self):
        self._build_design_matrix()
        self._build_transition_matrix()
        self._build_selection_matrix()
    
    def _build_design_matrix(self):
        self['design'] = np.zeros((self.n_series, self.k_states))
        series_idx = {name: i for i, name in enumerate(self.series_names)}
        for series, i in series_idx.items():
            contributions = self.dep_graph.source_contributions.get(series, {})
            for lag, sources in contributions.items():
                for source, coef in sources.items():
                    if source in self.source_info:
                        info = self.source_info[source]
                        state_idx = info['start_idx'] + (0 if lag == 0 else 2 + abs(lag) - 1)
                        self['design'][i, state_idx] += coef
    
    def _build_transition_matrix(self):
        self['transition'] = np.zeros((self.k_states, self.k_states))
        for info in self.source_info.values():
            start, max_l = info['start_idx'], info['max_lag']
            self['transition'][start, start] = 1
            self['transition'][start, start+1] = 1
            self['transition'][start+1, start+1] = 1
            if max_l > 0:
                self['transition'][start+2, start] = 1 
                for lag in range(1, max_l): 
                    self['transition'][start+2+lag, start+2+lag-1] = 1
    
    def _build_selection_matrix(self):
        self['selection'] = np.zeros((self.k_states, self.k_posdef))
        for i, series in enumerate(self.source_series):
            info = self.source_info[series]
            self['selection'][info['start_idx'] + 1, i] = 1
    
    def initialize_state_space(self):
        """Initialize state space with HP filter or diffuse initialization."""
        # Initial covariances
        self['state_cov'] = np.eye(self.k_posdef) * 0.01
        self['obs_cov'] = np.eye(self.n_series) * 1.0
        
        # Try HP-based initialization first
        try:
            self._hp_based_initialization()
        except Exception as e:
            logger.warning(f"HP initialization failed: {e}, using diffuse initialization")
            if self.use_exact_diffuse:
                init = Initialization(self.k_states, 'diffuse')
            else:
                init = Initialization(self.k_states, 'approximate_diffuse', 
                                    approximate_diffuse_variance=1e4)
            self.ssm.initialize(init)
    
    def _hp_based_initialization(self, lambda_hp=1600):
        """Initialize using HP filter on normalized data."""
        logger.info("Using HP filter for state initialization...")
        
        initial_state_mean = np.zeros(self.k_states)
        initial_state_cov = np.eye(self.k_states) * 0.1
        
        for i, series in enumerate(self.source_series):
            if series in self.original_data.columns:
                # Use normalized data
                y_normalized = self.original_data[series] / self.scale_factor
                y_normalized = y_normalized.dropna()
                
                if len(y_normalized) > 10:
                    try:
                        cycle, trend = hpfilter(y_normalized, lamb=lambda_hp)
                        
                        # Set initial level
                        info = self.source_info[series]
                        initial_state_mean[info['start_idx']] = trend.iloc[0]
                        
                        # Set initial trend
                        if len(trend) > 5:
                            initial_slope = (trend.iloc[5] - trend.iloc[0]) / 5
                            initial_state_mean[info['start_idx'] + 1] = initial_slope
                        
                        # Set covariances based on HP decomposition
                        level_var = np.var(cycle) * 0.1
                        trend_var = np.var(np.diff(trend)) * 0.1
                        
                        initial_state_cov[info['start_idx'], info['start_idx']] = level_var
                        initial_state_cov[info['start_idx'] + 1, info['start_idx'] + 1] = trend_var
                        
                    except Exception as e:
                        logger.warning(f"HP filter failed for {series}: {e}")
        
        self.ssm.initialize_known(initial_state_mean, initial_state_cov)
        logger.info("HP-based initialization complete")
    
    @property
    def start_params(self):
        """Starting parameters for optimization."""
        if self.transformation == 'square':
            # Square root of small variances
            return np.ones(self.k_params) * 0.1
        else:  # log
            # Log of small variances
            return np.ones(self.k_params) * np.log(0.01)
    
    def transform_params(self, unconstrained):
        """Transform optimizer-space params to variance space (complex-step safe)."""
        if unconstrained is None:
            raise TypeError("transform_params received None.")

        # Preserve complex dtype so complex-step gradients are not destroyed
        u = np.asarray(unconstrained).ravel()
        if not np.isfinite(u).all():
            raise ValueError("transform_params: parameters contain NaN/Inf (real or imag).")

        if self.transformation == 'square':
            v = u ** 2

        elif self.transformation == 'log':
            # Check overflow on the real part only; complex-step adds tiny imag
            LOG_MAX = float(np.log(np.finfo(np.float64).max))  # ~709.78
            if np.any(u.real > LOG_MAX):
                raise OverflowError(
                    f"transform_params(log): component(s) exceed exp overflow threshold ({LOG_MAX:.2f})."
                )
            v = np.exp(u)

        else:
            raise ValueError(f"Unknown transformation '{self.transformation}'.")

        # Variances must be positive in the real part (allow tiny numerical noise)
        if np.any(v.real <= -1e-12):
            bad = np.where(v.real <= -1e-12)[0].tolist()
            raise ValueError(f"Non-positive variance (real part) at indices {bad}.")

        return v

    
    def untransform_params(self, constrained):
        """Transform from variance space to optimizer space."""
        constrained = np.asarray(constrained)
        if self.transformation == 'square':
            return np.sqrt(np.maximum(constrained, 1e-10))
        else:  # log
            return np.log(np.maximum(constrained, 1e-10))
    
    def update(self, params, **kwargs):
        """
        Update state-space matrices from a parameter vector.

        Semantics:
          - If kwargs.get('transformed', False) is True, `params` are already variances (constrained space).
          - If False, `params` are unconstrained and will be transformed via `transform_params`.

        Expected layout:
          len(params) == self.n_source + 1
            * first self.n_source entries: state shock variances (Q diagonal, one per shock)
            * last entry: scalar observation variance (H diagonal)
        """
        import numpy as np

        # ---------- strict input checks ----------
        if params is None:
            raise ValueError("update(params=...) received None. Pass a parameter vector.")

        p = np.asarray(params).ravel()  # PRESERVE complex dtype (complex-step safe)
        expected_len = int(self.n_source) + 1
        if p.size != expected_len:
            raise ValueError(f"Parameter length {p.size} != expected {expected_len} (n_source + 1).")
        if not np.isfinite(p).all():
            raise ValueError("Parameters contain NaN/Inf (real or imag).")

        # Default to *untransformed* (optimizer space) unless explicitly told otherwise
        transformed_flag = bool(kwargs.get('transformed', False))

        # Safety override: if caller claims "transformed" but entries are nonpositive,
        # treat as unconstrained and transform here.
        if transformed_flag and np.any(p.real <= 0.0):
            # Optional: self.logger.debug("update: overriding transformed=True due to nonpositive entries.")
            transformed_flag = False

        variances = p if transformed_flag else self.transform_params(p)

        # Hard checks on transformed variances
        if not (np.isfinite(variances.real).all() and np.isfinite(variances.imag).all()):
            raise ValueError("update: variances contain NaN/Inf (real or imag).")

        if np.any(variances.real <= 0.0):
            bad = np.where(variances.real <= 0.0)[0].tolist()
            raise ValueError(f"Non-positive variance (real part) at indices {bad}.")

        # Split into state (Q) and observation (H) parts
        trend_variances = variances[:self.n_source]
        obs_variance = variances[self.n_source]

        k_posdef = int(self.ssm.k_posdef)
        if trend_variances.size != k_posdef:
            raise ValueError(
                f"Length of trend variances ({trend_variances.size}) != k_posdef ({k_posdef})."
            )

        # Build Q (state covariance) and H (observation covariance) — keep dtype for complex-step
        Q = np.diag(trend_variances)
        self['state_cov'] = Q

        H = np.eye(self.n_series, dtype=variances.dtype) * obs_variance
        if getattr(self, 'computed_series', None):
            ratio = float(getattr(self, 'error_variance_ratio', 1.0))
            if ratio <= 0.0:
                raise ValueError("error_variance_ratio must be positive.")
            series_idx = {name: i for i, name in enumerate(self.series_names)}
            for s in self.computed_series:
                i = series_idx.get(s)
                if i is not None:
                    H[i, i] *= ratio
        self['obs_cov'] = H


    
    def loglike(self, params, *args, **kwargs):
        """Override to add debugging and monitoring."""
        if not hasattr(self, '_loglike_count'):
            self._loglike_count = 0
        self._loglike_count += 1
        
        # Get base log likelihood
        try:
            llf = super().loglike(params, *args, **kwargs)
        except Exception as e:
            logger.debug(f"Loglike failed with params {params}: {e}")
            return -np.inf
        
        # Periodic debugging
        if self._loglike_count <= 5 or self._loglike_count % 100 == 0:
            variances = self.transform_params(params)
            logger.debug(f"Loglike call {self._loglike_count}: LLF={llf:.2f}, "
                        f"var range=[{variances.min():.2e}, {variances.max():.2e}]")
        
        return llf
    
    def get_filtered_series(self, results):
        """
        Extract filtered, smoothed, and predicted series.
        
        Filtered: Uses all information up to and including time t to estimate time t
        Smoothed: Uses all information from the entire sample to estimate time t
        Predicted: Uses information up to time t to predict time t+1 (one-step-ahead)
        """
        # Get the states from Kalman filter results
        filtered_states = results.filtered_state  # State at t using info up to t
        smoothed_states = results.smoothed_state  # State at t using all info
        predicted_states = results.predicted_state  # State at t+1 using info up to t
        
        # Design matrix to convert states to observations
        Z = self['design']  # Shape: (n_series, k_states)
        
        # Initialize output arrays
        n_obs = len(self.original_data)
        filtered_obs = np.zeros((n_obs, self.n_series))
        smoothed_obs = np.zeros((n_obs, self.n_series))
        predicted_obs = np.zeros((n_obs, self.n_series))
        
        # Reconstruct observations from states
        for t in range(n_obs):
            # Filtered: estimate at t using data up to t
            filtered_obs[t, :] = Z @ filtered_states[:, t]
            
            # Smoothed: estimate at t using all data
            smoothed_obs[t, :] = Z @ smoothed_states[:, t]
            
            # Predicted: forecast for t using data up to t-1
            # Note: predicted_states[:,t] is the prediction FOR time t made at time t-1
            if t < predicted_states.shape[1]:
                predicted_obs[t, :] = Z @ predicted_states[:, t]
        
        # Create DataFrames
        filtered_df = pd.DataFrame(
            filtered_obs,
            index=self.original_data.index,
            columns=self.original_data.columns
        )
        
        smoothed_df = pd.DataFrame(
            smoothed_obs,
            index=self.original_data.index,
            columns=self.original_data.columns
        )
        
        predicted_df = pd.DataFrame(
            predicted_obs,
            index=self.original_data.index,
            columns=self.original_data.columns
        )
        
        # Denormalize if needed
        if self.normalize_data:
            filtered_df = filtered_df * self.scale_factor
            smoothed_df = smoothed_df * self.scale_factor
            predicted_df = predicted_df * self.scale_factor
        
        # Calculate differences for diagnostics
        filter_smooth_diff = filtered_df - smoothed_df
        
        # Get forecast errors (actual - predicted)
        forecast_errors = self.original_data[self.series_names] - predicted_df
        
        return {
            'filtered': filtered_df,      # Estimate at t using info up to t
            'smoothed': smoothed_df,       # Estimate at t using all info
            'predicted': predicted_df,     # Forecast for t using info up to t-1
            'filter_smooth_diff': filter_smooth_diff,
            'forecast_errors': forecast_errors
        }


    def predict_ahead(self, results, steps_ahead=4):
        """
        Predict ahead based on the filtered state at the last time point.
        
        Parameters:
        -----------
        results : FilterResults
            The fitted Kalman filter results
        steps_ahead : int
            Number of periods to forecast ahead
            
        Returns:
        --------
        dict with 'point_forecast' and 'forecast_variance'
        """
        # Get the last filtered state and covariance
        last_filtered_state = results.filtered_state[:, -1]
        last_filtered_cov = results.filtered_state_cov[:, :, -1]
        
        # Get system matrices
        T = self['transition']  # State transition matrix
        Z = self['design']      # Observation matrix
        R = self['selection']   # Selection matrix
        Q = self['state_cov']   # State noise covariance
        H = self['obs_cov']     # Observation noise covariance
        
        # Initialize forecast arrays
        forecast_states = np.zeros((self.k_states, steps_ahead))
        forecast_obs = np.zeros((self.n_series, steps_ahead))
        forecast_state_cov = np.zeros((self.k_states, self.k_states, steps_ahead))
        forecast_obs_var = np.zeros((self.n_series, self.n_series, steps_ahead))
        
        # Current state and covariance
        state = last_filtered_state.copy()
        state_cov = last_filtered_cov.copy()
        
        for h in range(steps_ahead):
            # Predict state h steps ahead
            state = T @ state
            state_cov = T @ state_cov @ T.T + R @ Q @ R.T
            
            # Store state forecast
            forecast_states[:, h] = state
            forecast_state_cov[:, :, h] = state_cov
            
            # Predict observations
            forecast_obs[:, h] = Z @ state
            forecast_obs_var[:, :, h] = Z @ state_cov @ Z.T + H
        
        # Create forecast DataFrame
        last_date = self.original_data.index[-1]
        forecast_index = pd.date_range(
            start=last_date, 
            periods=steps_ahead + 1, 
            freq=self.original_data.index.freq
        )[1:]  # Skip first date (it's the last observation)
        
        forecast_df = pd.DataFrame(
            forecast_obs.T,
            index=forecast_index,
            columns=self.original_data.columns
        )
        
        # Denormalize if needed
        if self.normalize_data:
            forecast_df = forecast_df * self.scale_factor
            # Also scale variances
            forecast_obs_var = forecast_obs_var * (self.scale_factor ** 2)
        
        # Extract standard errors (diagonal of covariance)
        forecast_se = np.zeros((steps_ahead, self.n_series))
        for h in range(steps_ahead):
            forecast_se[h, :] = np.sqrt(np.diag(forecast_obs_var[:, :, h]))
        
        forecast_se_df = pd.DataFrame(
            forecast_se,
            index=forecast_index,
            columns=self.original_data.columns
        )
        
        return {
            'point_forecast': forecast_df,
            'forecast_se': forecast_se_df,
            'forecast_variance': forecast_obs_var
        }

    def get_most_probable_path(self, results, base_date=None):
        """
        Generate the most probable path for the year ahead.
        
        Returns:
        --------
        dict with:
            - 'path': DataFrame with most probable values
            - 'uncertainty': DataFrame with standard deviations
            - 'confidence_bands': Dict with 68% and 95% bands
        """
        if base_date is None:
            base_date = self.original_data.index[-1]
        
        # Find the position of base_date
        base_idx = self.original_data.index.get_loc(base_date)
        
        # Get filtered state and covariance at base date
        state = results.filtered_state[:, base_idx].copy()
        state_cov = results.filtered_state_cov[:, :, base_idx].copy()
        
        # System matrices
        T = self['transition']
        Z = self['design']
        R = self['selection']
        Q = self['state_cov']
        H = self['obs_cov']
        
        # Generate path
        path_values = []
        path_std = []
        
        for h in range(1, 5):  # 4 quarters ahead
            # Propagate state
            state = T @ state
            state_cov = T @ state_cov @ T.T + R @ Q @ R.T
            
            # Get observation prediction
            obs_mean = Z @ state
            obs_var = Z @ state_cov @ Z.T + H
            obs_std = np.sqrt(np.diag(obs_var))
            
            path_values.append(obs_mean)
            path_std.append(obs_std)
        
        # Create DataFrames
        future_dates = pd.date_range(
            start=base_date, 
            periods=5, 
            freq=self.original_data.index.freq
        )[1:]
        
        path_df = pd.DataFrame(
            np.array(path_values),
            index=future_dates,
            columns=self.original_data.columns
        )
        
        uncertainty_df = pd.DataFrame(
            np.array(path_std),
            index=future_dates,
            columns=self.original_data.columns
        )
        
        # Denormalize if needed
        if self.normalize_data:
            path_df = path_df * self.scale_factor
            uncertainty_df = uncertainty_df * self.scale_factor
        
        # Calculate confidence bands
        confidence_bands = {
            '68%': {
                'lower': path_df - uncertainty_df,
                'upper': path_df + uncertainty_df
            },
            '95%': {
                'lower': path_df - 2 * uncertainty_df,
                'upper': path_df + 2 * uncertainty_df
            }
        }
        
        return {
            'path': path_df,
            'uncertainty': uncertainty_df,
            'confidence_bands': confidence_bands
        }
    
    def validate_formulas(self, filtered_series_dict):
        """Validate that formulas hold for filtered series."""
        validation_results = []
        smoothed_series = filtered_series_dict['smoothed']

        for series_code in self.computed_series:
            formula_info = self.formulas.get(series_code, {})
            formula_str = formula_info.get('formula', 'N/A')
            derived_from = formula_info.get('derived_from')
            if not derived_from:
                continue

            reconstructed = pd.Series(0.0, index=smoothed_series.index)
            parsed_lags = self.dep_graph._parse_formula_string(formula_str)

            for component in derived_from:
                comp_code = component.get("code")
                operator = component.get("operator", "+")
                sign = -1.0 if operator == "-" else 1.0
                lag = parsed_lags.get(comp_code, 0)

                if comp_code in smoothed_series.columns:
                    component_series = smoothed_series[comp_code].shift(lag)
                    reconstructed += sign * component_series

            error_series = (smoothed_series[series_code] - reconstructed).dropna()
            mae = error_series.abs().mean()
            rmse = np.sqrt((error_series ** 2).mean())
            mean_abs_actual = smoothed_series[series_code].abs().mean()
            relative_error = mae / mean_abs_actual if mean_abs_actual > 0 else np.nan

            validation_results.append({
                'series': series_code,
                'formula': formula_str,
                'mae': mae,
                'rmse': rmse,
                'relative_error': relative_error
            })

        return pd.DataFrame(validation_results)


def fit_hierarchical_kalman_filter(data: pd.DataFrame, formulas_file: str, 
                                   series_list: List[str], 
                                   normalize: bool = True,
                                   error_variance_ratio: float = 0.01,
                                   loglikelihood_burn: int = 20,
                                   use_exact_diffuse: bool = False,
                                   transformation: str = 'square',
                                   max_attempts: int = 3):
    """Fit hierarchical Kalman filter with multiple optimization attempts."""
    
    with open(formulas_file, 'r') as f:
        formulas = json.load(f).get('formulas', {})
    
    graph = DependencyGraph(formulas, list(data.columns))
    all_required_series = graph.get_all_dependencies(series_list)
    
    final_series_list = sorted(list(all_required_series))
    logger.info(f"Processing {len(final_series_list)} series: {final_series_list}")
    
    data_subset = data[final_series_list].copy()
    
    # Check data statistics
    logger.info("\nData statistics:")
    for series in final_series_list[:5]:
        logger.info(f"  {series}: mean={data_subset[series].mean():.2e}, "
                   f"std={data_subset[series].std():.2e}, "
                   f"min={data_subset[series].min():.2e}, "
                   f"max={data_subset[series].max():.2e}")
    
    model = HierarchicalKalmanFilter(
        data=data_subset, 
        formulas=formulas, 
        error_variance_ratio=error_variance_ratio,
        normalize_data=normalize,
        loglikelihood_burn=loglikelihood_burn,
        use_exact_diffuse=use_exact_diffuse,
        transformation=transformation
    )
    
    # Try different optimization approaches
    optimization_configs = [
        {'method': 'lbfgs', 'maxiter': 1000},
        {'method': 'powell', 'maxiter': 500},
        {'method': 'nm', 'maxiter': 1000}
    ]
    
    best_result = None
    best_llf = -np.inf
    
    for config in optimization_configs:
        method = config['method']
        logger.info(f"\n--- Trying optimization method: {method} ---")
        
        for attempt in range(max_attempts):
            try:
                # Vary starting points slightly
                if attempt == 0:
                    start_params = model.start_params
                else:
                    # Add small random perturbation
                    start_params = model.start_params + np.random.normal(0, 0.1, model.k_params)
                
                result = model.fit(
                    start_params=start_params,
                    method=method,
                    maxiter=config['maxiter'],
                    disp=True
                )
                
                if result.llf > best_llf:
                    best_llf = result.llf
                    best_result = result
                    logger.info(f"New best LLF: {best_llf:.2f}")
                
                # If converged well, stop trying
                if hasattr(result, 'converged') and result.converged:
                    break
                    
            except Exception as e:
                logger.error(f"Attempt {attempt+1} with {method} failed: {e}")
    
    if best_result is None:
        raise RuntimeError("All optimization attempts failed")
    
    # Final diagnostics
    logger.info(f"\n=== FINAL RESULTS ===")
    logger.info(f"Best log-likelihood: {best_result.llf:.2f}")
    logger.info(f"AIC: {best_result.aic:.2f}")
    logger.info(f"Final parameters (variances):")
    final_variances = model.transform_params(best_result.params)
    for name, var in zip(model.param_names, final_variances):
        logger.info(f"  {name}: {var:.6f}")
    
    filtered_series = model.get_filtered_series(best_result)
    validation = model.validate_formulas(filtered_series)
    
    return model, {
        'fitted_results': best_result, 
        'filtered_series': filtered_series,
        'validation': validation,
        'series_list': final_series_list
    }
