"""
Comprehensive Aligned Sum-Constrained Unobserved Components System
Utilizes ALL information from the extracted FOF formulas and identities
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.formula.api import ols
from statsmodels.tsa.seasonal import seasonal_decompose
import ray
from typing import Dict, List, Tuple, Optional, Set, Union
import logging
from dataclasses import dataclass, field
from scipy import stats
import arviz as az
import networkx as nx
import json
import os
import warnings
from collections import defaultdict
import hashlib
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

# --- Ray Remote Functions (defined at module level) ---
@ray.remote
def decompose_single_series(series_df, metadata, series_name):
    """Remote function to decompose a single independent series"""
    try:
        model = UnobservedComponents(
            series_df[series_name].dropna(), 
            level='lltrend', 
            cycle=True, 
            seasonal=4, 
            stochastic_cycle=True, 
            damped_cycle=True
        )
        fit = model.fit(disp=False)
        index = series_df.index
        
        return {
            series_name: {
                'level': pd.Series(fit.level.smoothed, index=fit.level.smoothed.index).reindex(index).interpolate(),
                'trend': pd.Series(fit.trend.smoothed, index=fit.trend.smoothed.index).reindex(index).interpolate(),
                'cycle': pd.Series(fit.cycle.smoothed, index=fit.cycle.smoothed.index).reindex(index).interpolate(),
                'seasonal': pd.Series(fit.seasonal.smoothed, index=fit.seasonal.smoothed.index).reindex(index).interpolate()
            }
        }
    except Exception as e:
        logging.warning(f"Statsmodels failed for {series_name}: {e}")
        # Fallback to simple decomposition
        series = series_df[series_name]
        if len(series.dropna()) < 8:
            return {
                series_name: {
                    'level': series, 
                    'trend': pd.Series(0, index=series.index), 
                    'cycle': pd.Series(0, index=series.index), 
                    'seasonal': pd.Series(0, index=series.index)
                }
            }
        
        decomp = seasonal_decompose(series.dropna(), model='additive', period=4)
        return {
            series_name: {
                'level': decomp.trend.reindex(series.index).interpolate(), 
                'trend': pd.Series(0, index=series.index), 
                'cycle': decomp.resid.reindex(series.index).interpolate(), 
                'seasonal': decomp.seasonal.reindex(series.index).interpolate()
            }
        }

@ray.remote
def decompose_constrained_group(system_state, df_values, df_index, df_columns, metadata_dict, group_name, previous_results):
    """Remote function to decompose a constrained group"""
    # Reconstruct DataFrame
    df = pd.DataFrame(df_values, index=df_index, columns=df_columns)
    
    # Reconstruct necessary system components
    identities = system_state['identities']
    series_metadata = system_state['series_metadata']
    
    T = len(df)
    series_names = df.columns.tolist()
    
    try:
        with pm.Model() as model:
            components = {}
            
            # Create components for each series
            for series_name in series_names:
                model_hash = hashlib.md5(f"{group_name}_{series_name}".encode()).hexdigest()[:8]
                components[series_name] = create_components_for_series(
                    series_name, df[series_name], metadata_dict[series_name], T, df, model_hash
                )
            
            # Apply constraints
            applicable_constraints = []
            for identity_name, identity_data in identities.items():
                affected_series = {s.lstrip('Δ') for s in identity_data['left_side'] + identity_data['right_side']}
                if affected_series.intersection(series_names):
                    applicable_constraints.append((identity_name, identity_data))
            
            for i, (constraint_name, constraint_data) in enumerate(applicable_constraints):
                apply_constraint_to_model(constraint_data, constraint_name, components, T, i)
            
            # Observation equations
            for series_name in series_names:
                comp = components[series_name]
                mu = comp['level'] + comp['cycle'] + comp['seasonal']
                pm.Normal(f'obs_{series_name}', mu=mu, sigma=comp['sigma_obs'], observed=df[series_name].values)
            
            # Fit model
            approx = pm.fit(n=20000, method='advi', obj_optimizer=pm.adam(learning_rate=0.005))
            trace = approx.sample(1000)
        
        # Extract results
        return extract_results_from_trace(trace, df)
        
    except Exception as e:
        logging.error(f"Error in constrained decomposition for group {group_name}: {e}")
        # Return simple decomposition as fallback
        results = {}
        for series_name in series_names:
            results[series_name] = {
                'level': df[series_name],
                'trend': pd.Series(0, index=df.index),
                'cycle': pd.Series(0, index=df.index),
                'seasonal': pd.Series(0, index=df.index)
            }
        return results

# Helper functions for Ray remote tasks
def create_components_for_series(name, series, metadata, T, df, unique_id):
    """Create PyMC components for a series"""
    components = {}
    vname = f"{name}_{unique_id}"
    dtype = 'float64'
    
    series_values = series.dropna().values
    if len(series_values) < 8:
        return {
            'level': pt.zeros(T, dtype=dtype), 
            'trend': pt.zeros(T, dtype=dtype), 
            'cycle': pt.zeros(T, dtype=dtype), 
            'seasonal': pt.zeros(T, dtype=dtype), 
            'sigma_obs': pm.HalfNormal(f'{vname}_sigma_obs', 1., dtype=dtype)
        }
    
    # Get variance estimates
    cycle1, trend1 = hpfilter(series_values, lamb=1600)
    var_resid = np.var(cycle1, ddof=1) if len(cycle1) > 1 else 1.0
    _, trend2 = hpfilter(trend1, lamb=1600)
    trend_var_estimate = np.var(np.diff(trend2), ddof=1) if len(trend2) > 1 else 1.0
    
    var_resid = max(var_resid, 1e-6)
    trend_var_estimate = max(trend_var_estimate, 1e-9)
    
    init_level = float(series_values[0])
    init_trend = float(np.mean(np.diff(series_values[:5]))) if len(series_values) > 5 else 0.0
    
    # Level and trend components
    level_init = pm.Normal(f'{vname}_level_init', mu=init_level, sigma=float(np.sqrt(var_resid)), dtype=dtype)
    trend_init = pm.Normal(f'{vname}_trend_init', mu=init_trend, sigma=float(np.sqrt(trend_var_estimate)), dtype=dtype)
    sigma_trend = pm.HalfNormal(f'{vname}_sigma_trend', sigma=float(np.sqrt(trend_var_estimate)), dtype=dtype)
    trend_innovations = pm.Normal(f'{vname}_trend_innovations', mu=0, sigma=sigma_trend, shape=T-1, dtype=dtype)
    
    trend = pt.concatenate([pt.atleast_1d(trend_init), trend_init + pt.cumsum(trend_innovations)])
    components['trend'] = pm.Deterministic(f'{vname}_trend', trend)
    
    def level_step(beta_prev, mu_prev): 
        return mu_prev + beta_prev
    
    level_values, _ = scan(fn=level_step, sequences=[trend[:-1]], outputs_info=[level_init])
    components['level'] = pm.Deterministic(f'{vname}_level', pt.concatenate([pt.atleast_1d(level_init), level_values]))
    
    # Cycle component
    if metadata.series_type in ['flow', 'level'] and T > 20:
        cycle_period_bounds = (1.5*4, 12*4)
        cycle_freq_bounds = (2*np.pi / cycle_period_bounds[1], 2*np.pi / cycle_period_bounds[0])
        cycle_freq = pm.Uniform(f'{vname}_cycle_freq', lower=cycle_freq_bounds[0], upper=cycle_freq_bounds[1])
        cycle_damp = pm.Beta(f'{vname}_cycle_damp', alpha=8., beta=1.5)
        sigma_cycle = pm.HalfNormal(f'{vname}_sigma_cycle', sigma=float(np.sqrt(var_resid) * 0.5), dtype=dtype)
        cycle_innovations = pm.Normal(f'{vname}_cycle_innovations', mu=0, sigma=sigma_cycle, shape=(T, 2), dtype=dtype)
        
        def cycle_step(innov, prev_c, prev_cs, freq, damp):
            cos_f, sin_f = pt.cos(freq), pt.sin(freq)
            new_c = damp * (cos_f * prev_c + sin_f * prev_cs) + innov[0]
            new_cs = damp * (-sin_f * prev_c + cos_f * prev_cs) + innov[1]
            return new_c, new_cs
        
        initial_cycle_states = [pt.as_tensor_variable(0.0, dtype=dtype), pt.as_tensor_variable(0.0, dtype=dtype)]
        (cycle_states, _), _ = scan(
            fn=cycle_step, 
            sequences=[cycle_innovations], 
            outputs_info=initial_cycle_states, 
            non_sequences=[cycle_freq, cycle_damp]
        )
        components['cycle'] = pm.Deterministic(f'{vname}_cycle', cycle_states)
    else:
        components['cycle'] = pt.zeros(T, dtype=dtype)
    
    # Seasonal component
    if metadata.seasonal_pattern != 'none' and T > 8:
        sigma_seasonal = pm.HalfNormal(f'{vname}_sigma_seasonal', sigma=float(np.sqrt(var_resid)*0.5), dtype=dtype)
        seasonal_params = pm.Normal(f'{vname}_seasonal_params', mu=0, sigma=sigma_seasonal, shape=4, dtype=dtype)
        seasonal_sum_to_zero = seasonal_params - pt.mean(seasonal_params)
        quarter_indices = np.arange(T) % 4
        components['seasonal'] = pm.Deterministic(f'{vname}_seasonal', seasonal_sum_to_zero[quarter_indices])
    else:
        components['seasonal'] = pt.zeros(T, dtype=dtype)
    
    components['sigma_obs'] = pm.HalfNormal(f'{vname}_sigma_obs', sigma=float(np.sqrt(var_resid)), dtype=dtype)
    return components

def apply_constraint_to_model(constraint_data, constraint_name, components, T, constraint_idx):
    """Apply a constraint to the PyMC model"""
    dtype = 'float64'
    
    if constraint_data.get('identity_type') == 'flow_stock':
        # Flow-stock constraint
        level_series_code = next((s[1:] for s in constraint_data['left_side'] if s.startswith('Δ')), None)
        if level_series_code and level_series_code in components:
            level_comp = components[level_series_code]
            level_total = level_comp['level']
            level_change = pt.concatenate([
                pt.zeros(1, dtype=dtype), 
                level_total[1:] - level_total[:-1]
            ])
            
            flow_sum = pt.zeros(T, dtype=dtype)
            operators = constraint_data.get('operators', [])
            for i, series in enumerate(constraint_data['right_side']):
                if series in components:
                    comp = components[series]
                    total = comp['level'] + comp['cycle'] + comp['seasonal']
                    op = operators[i] if i < len(operators) else '+'
                    flow_sum = flow_sum + total if op == '+' else flow_sum - total
            
            identity_violation = level_change[1:] - flow_sum[1:]
            series_scale = pt.mean(pt.abs(flow_sum))
            scaled_tolerance = constraint_data.get('tolerance', 1e-6) * pt.maximum(1.0, series_scale * 0.001)
            
            penalty_dist = pm.StudentT.dist(nu=4, mu=0, sigma=scaled_tolerance)
            pm.Potential(f'flow_stock_penalty_{constraint_name}_{constraint_idx}', 
                        pm.logp(penalty_dist, identity_violation).sum())
    else:
        # Sum constraint
        left_sum = pt.zeros(T, dtype=dtype)
        for series in constraint_data['left_side']:
            if series in components:
                comp = components[series]
                left_sum = left_sum + comp['level'] + comp['cycle'] + comp['seasonal']
        
        right_sum = pt.zeros(T, dtype=dtype)
        operators = constraint_data.get('operators', [])
        for i, series in enumerate(constraint_data['right_side']):
            if series in components:
                comp = components[series]
                total = comp['level'] + comp['cycle'] + comp['seasonal']
                op = operators[i] if i < len(operators) else '+'
                right_sum = right_sum + total if op == '+' else right_sum - total
        
        identity_violation = left_sum - right_sum
        series_scale = pt.maximum(pt.mean(pt.abs(left_sum)), pt.mean(pt.abs(right_sum)))
        scaled_tolerance = constraint_data.get('tolerance', 1e-6) * pt.maximum(1.0, series_scale * 0.001)
        
        penalty_dist = pm.StudentT.dist(nu=4, mu=0, sigma=scaled_tolerance)
        pm.Potential(f'flow_stock_penalty_{constraint_name}_{constraint_idx}', 
                    pm.logp(penalty_dist, identity_violation).sum())

def extract_results_from_trace(trace, data):
    """Extract decomposition results from PyMC trace"""
    results = {}
    for series_name in data.columns:
        res = {}
        for component in ['level', 'trend', 'cycle', 'seasonal']:
            var_name = next(
                (v for v in trace.posterior.data_vars 
                 if v.startswith(f"{series_name}_") and v.endswith(f"_{component}")), 
                None
            )
            if var_name:
                mean_vals = trace.posterior[var_name].mean(dim=['chain', 'draw']).values
                res[component] = pd.Series(mean_vals, index=data.index)
                res[f'{component}_lower'] = pd.Series(
                    trace.posterior[var_name].quantile(0.025, dim=['chain', 'draw']).values, 
                    index=data.index
                )
                res[f'{component}_upper'] = pd.Series(
                    trace.posterior[var_name].quantile(0.975, dim=['chain', 'draw']).values, 
                    index=data.index
                )
            else:
                res[component] = pd.Series(0, index=data.index)
        results[series_name] = res
    return results

# --- Main Classes ---
@dataclass
class SeriesFormula:
    """Represents a series-specific formula from FOF data"""
    series_code: str
    series_name: str
    table_code: str
    line_number: int
    data_type: str
    formula: str
    components: List[str]
    derived_from: List[Dict[str, str]]
    used_in: List[Dict[str, str]]
    shown_on: List[str]

@dataclass
class AccountingIdentity:
    """Enhanced accounting identity with full formula support"""
    name: str
    identity_type: str
    formula: str
    left_side: List[str] = field(default_factory=list)
    right_side: List[str] = field(default_factory=list)
    operators: List[str] = field(default_factory=list)
    operator: str = '='
    tolerance: float = 1e-6
    parsed_formula: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    
    def check_validity(self, data: pd.DataFrame) -> pd.Series:
        """Check if identity holds in the data"""
        if self.identity_type == 'flow_stock' and any('Δ' in s for s in self.left_side):
            return self._check_flow_stock_validity(data)
        elif self.parsed_formula:
            left_value = self._evaluate_expression(self.parsed_formula['left'], data)
            right_value = self._evaluate_expression(self.parsed_formula['right'], data)
        else:
            left_value = self._evaluate_side_with_operators(data, self.left_side, ['+'] * len(self.left_side))
            right_value = self._evaluate_side_with_operators(data, self.right_side, self.operators)
        
        if self.operator == '=': 
            return np.abs(left_value - right_value) < self.tolerance
        elif self.operator == '<=': 
            return left_value <= right_value + self.tolerance
        else: 
            return left_value >= right_value - self.tolerance

    def _check_flow_stock_validity(self, data: pd.DataFrame) -> pd.Series:
        """Check flow-stock identity validity"""
        delta_series = [s for s in self.left_side if s.startswith('Δ')]
        if delta_series:
            level_code = delta_series[0][1:]
            if level_code in data.columns:
                level_change = data[level_code].diff()
                right_sum = self._evaluate_side_with_operators(data, self.right_side, self.operators)
                validity = pd.Series(True, index=data.index)
                validity.iloc[1:] = np.abs(level_change.iloc[1:] - right_sum.iloc[1:]) < self.tolerance
                return validity
        return pd.Series(True, index=data.index)

    def _evaluate_side_with_operators(self, data: pd.DataFrame, series_list: List[str], operators: List[str]) -> pd.Series:
        """Evaluate one side of the identity with operators"""
        result = pd.Series(0.0, index=data.index)
        for i, series in enumerate(series_list):
            op = operators[i] if i < len(operators) else '+'
            term = pd.Series(0.0, index=data.index)
            if series in data.columns:
                term = data[series].fillna(0)
            elif series.startswith('Δ'):
                level_code = series[1:]
                if level_code in data.columns:
                    term = data[level_code].diff().fillna(0)
            
            if op == '+': 
                result += term
            else: 
                result -= term
        return result

    def _evaluate_expression(self, expr: Dict, data: pd.DataFrame) -> pd.Series:
        """Evaluate a parsed expression tree"""
        if expr['type'] == 'series':
            series_name = expr['name']
            if series_name.startswith('Δ'):
                level_code = series_name[1:]
                return data[level_code].diff() if level_code in data.columns else pd.Series(0, index=data.index)
            elif series_name in data.columns:
                return data[series_name].shift(expr.get('lag', 0))
            else:
                return pd.Series(0, index=data.index)
        elif expr['type'] == 'constant':
            return pd.Series(expr['value'], index=data.index)
        elif expr['type'] == 'operation':
            left = self._evaluate_expression(expr['left'], data)
            right = self._evaluate_expression(expr['right'], data)
            op_map = {
                '+': left + right, 
                '-': left - right, 
                '*': left * right, 
                '/': left / right.replace(0, np.nan)
            }
            if expr['operator'] in op_map:
                return op_map[expr['operator']]
            raise ValueError(f"Unknown operator: {expr['operator']}")
        raise ValueError(f"Unknown expression type: {expr['type']}")

@dataclass
class SeriesMetadata:
    """Metadata for a series including all relationships"""
    name: str
    series_type: str
    crosses_zero: bool
    volatility_regime: str
    transformation: str
    seasonal_pattern: str
    outlier_dates: List[pd.Timestamp]
    structural_breaks: List[pd.Timestamp]
    identity_relationships: List[str] = field(default_factory=list)
    formula_relationships: List[str] = field(default_factory=list)
    is_derived: bool = False
    parent_series: List[str] = field(default_factory=list)
    data_type: str = "Source"
    table_codes: List[str] = field(default_factory=list)
    sector_code: str = ""
    instrument_code: str = ""


class ComprehensiveAlignedUCSystem:
    """Main system for comprehensive constrained decomposition"""
    
    def __init__(self, identities_file: str, n_cores: int = None, use_gpu: bool = True, cache_dir: str = './cache'):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.n_cores = n_cores or os.cpu_count()
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        
        # Load all data
        self.all_data = self._load_all_data(identities_file)
        self.identities = self._load_all_identities()
        self.series_formulas = self._load_series_formulas()
        self.series_metadata = self._load_series_metadata()
        self.sectors = self.all_data.get('sectors', {})
        self.identity_graph = self._build_comprehensive_graph()
        
        # Initialize Ray
        num_gpus = 1 if self.use_gpu else 0
        # Clear old Ray sessions
        import shutil
        ray_temp_dir = Path('/tmp/ray')
        if ray_temp_dir.exists():
            try:
                shutil.rmtree(ray_temp_dir)
                print("Cleared old Ray temp files")
            except:
                pass

        # Initialize Ray with custom temp dir
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.n_cores, 
                num_gpus=num_gpus, 
                ignore_reinit_error=True,
                _temp_dir='/tmp/ray',
                object_store_memory=4_000_000_000,  # Limit to 4GB
                _memory=8_000_000_000  # Limit to 8GB total
            )
        
        self.logger.info(
            f"Loaded {len(self.identities)} identities and "
            f"{len(self.series_formulas)} series formulas."
        )

    def _load_all_data(self, file_path):
        """Load all data from JSON file"""
        with open(file_path, 'r') as f: 
            return json.load(f)

    def _load_all_identities(self):
        """Load all identities including those from formulas"""
        identities = {}
        
        # Load explicit identities
        for identity_data in self.all_data.get('identities', []):
            identity = self._create_identity_from_data(identity_data)
            identities[identity.name] = identity
        
        # Convert formulas to identities
        for series_code, formula_data in self.all_data.get('formulas', {}).items():
            if formula_data.get('data_type') == 'Computed' and formula_data.get('formula'):
                identity = self._create_identity_from_formula(series_code, formula_data)
                if identity:
                    identities[identity.name] = identity
        
        return identities

    def _create_identity_from_data(self, identity_data: Dict) -> AccountingIdentity:
        """Create identity from data"""
        return AccountingIdentity(
            name=identity_data['identity_name'],
            identity_type=identity_data['identity_type'],
            formula=identity_data.get('description', ''),
            left_side=identity_data['left_side'],
            right_side=identity_data['right_side'],
            operators=['+'] * len(identity_data['right_side']),
            metadata={'table_codes': identity_data.get('table_codes', [])}
        )

    def _create_identity_from_formula(self, series_code: str, formula_data: Dict) -> Optional[AccountingIdentity]:
        """Create identity from formula data"""
        formula_str = formula_data.get('formula', '')
        if not formula_str: 
            return None

        components, operators = [], []
        
        # Use derived_from if available
        if formula_data.get('derived_from'):
            for item in formula_data['derived_from']:
                operators.append(item['operator'])
                components.append(item['code'])
        # Otherwise parse formula string
        elif formula_str.startswith('='):
            parts = formula_str[1:].strip().split()
            current_op = '+'
            for part in parts:
                if part in ['+', '-']:
                    current_op = part
                elif part and (part[0].isalpha() or part[0].isalnum()):
                    components.append(part)
                    operators.append(current_op)
        
        if components:
            return AccountingIdentity(
                name=f"Formula_{series_code}",
                identity_type='series_formula',
                formula=formula_str,
                left_side=[series_code],
                right_side=components,
                operators=operators,
                metadata={
                    'series_name': formula_data.get('series_name', ''), 
                    'table_code': formula_data.get('table_code', '')
                }
            )
        return None

    def _load_series_formulas(self):
        """Load all series formulas"""
        formulas = {}
        for series_code, data in self.all_data.get('formulas', {}).items():
            # Ensure all required fields exist
            formula_data = {
                'series_code': series_code,
                'series_name': data.get('series_name', ''),
                'table_code': data.get('table_code', ''),
                'line_number': data.get('line_number', 0),
                'data_type': data.get('data_type', 'Source'),
                'formula': data.get('formula', ''),
                'components': data.get('components', []),
                'derived_from': data.get('derived_from', []),
                'used_in': data.get('used_in', []),
                'shown_on': data.get('shown_on', [])
            }
            formulas[series_code] = SeriesFormula(**formula_data)
        return formulas

    def _load_series_metadata(self):
        """Load metadata for all series"""
        metadata = {}
        all_series_codes = set(self.all_data.get('series', {}).keys()) | set(self.all_data.get('formulas', {}).keys())

        for series_code in all_series_codes:
            series_data = self.all_data.get('series', {}).get(series_code, {})
            formula_data = self.all_data.get('formulas', {}).get(series_code, {})
            
            sector_code = series_code[2:4] if len(series_code) >= 4 else ''
            instrument_code = series_code[4:] if len(series_code) > 4 else ''
            
            # Find identity relationships
            identity_rels = []
            for name, identity in self.identities.items():
                all_identity_series = {s.lstrip('Δ') for s in identity.left_side + identity.right_side}
                if series_code in all_identity_series:
                    identity_rels.append(name)
            
            # Get parent and child series
            parent_series = [item['code'] for item in formula_data.get('derived_from', [])]
            child_series = [item['code'] for item in formula_data.get('used_in', [])]

            metadata[series_code] = SeriesMetadata(
                name=series_data.get('series_name', formula_data.get('series_name', series_code)),
                series_type=self._determine_series_type(series_code),
                crosses_zero=False, 
                volatility_regime='unknown', 
                transformation='unknown',
                seasonal_pattern='unknown', 
                outlier_dates=[], 
                structural_breaks=[],
                identity_relationships=identity_rels,
                formula_relationships=child_series,
                is_derived=(formula_data.get('data_type') == 'Computed'),
                parent_series=parent_series,
                data_type=formula_data.get('data_type', 'Source'),
                table_codes=list(set(series_data.get('shown_on', []) + formula_data.get('shown_on', []))),
                sector_code=sector_code,
                instrument_code=instrument_code
            )
        return metadata

    def _determine_series_type(self, series_code: str):
        """Determine series type from code prefix"""
        prefixes = {
            'FL': 'level', 
            'FU': 'flow', 
            'FA': 'flow_annual', 
            'FR': 'revaluation', 
            'FO': 'other_change'
        }
        return prefixes.get(series_code[:2], 'unknown')

    def _build_comprehensive_graph(self):
        """Build dependency graph"""
        G = nx.DiGraph()
        
        # Add nodes
        for series, metadata in self.series_metadata.items():
            G.add_node(
                series, 
                sector=metadata.sector_code, 
                instrument=metadata.instrument_code, 
                is_derived=metadata.is_derived
            )
        
        # Add edges from identities
        for identity in self.identities.values():
            if identity.identity_type == 'series_formula':
                target = identity.left_side[0]
                for component in identity.right_side:
                    if G.has_node(component) and G.has_node(target):
                        G.add_edge(component, target, identity=identity.name, type='formula')
        
        return G

    def analyze_series_with_context(self, series, name):
        """Analyze series characteristics"""
        metadata = self.series_metadata.get(name)
        if not metadata: 
            return None

        # Update metadata with data-driven analysis
        metadata.crosses_zero = (series.min() * series.max()) < 0 if pd.notna(series.min()) and pd.notna(series.max()) else False
        metadata.volatility_regime = 'high' if self._test_for_volatility_changes(series) else 'low'
        metadata.transformation = 'trend' if self._test_for_trend(series) else 'stationary'
        metadata.seasonal_pattern = 'quarterly' if self._test_for_seasonality(series) else 'none'
        metadata.outlier_dates = self._detect_outliers(series)
        metadata.structural_breaks = self._detect_trend_breaks(series)
        
        return metadata

    def get_decomposition_order(self, series_list):
        """Determine decomposition order based on dependencies"""
        subgraph = self.identity_graph.subgraph(series_list)
        
        if nx.is_directed_acyclic_graph(subgraph):
            # Return topological generations
            return list(nx.topological_generations(subgraph))
        else:
            # Handle cycles
            self.logger.warning("Cycles detected in dependency graph. Grouping cycles for joint estimation.")
            return [list(component) for component in nx.strongly_connected_components(subgraph)]

    def _group_by_all_constraints(self, series_list):
        """Group series by shared constraints"""
        series_graph = nx.Graph()
        series_graph.add_nodes_from(series_list)
        
        # Connect series that share constraints
        for identity_name, identity in self.identities.items():
            affected_series = {s.lstrip('Δ') for s in identity.left_side + identity.right_side}
            affected_series = list(affected_series.intersection(series_list))
            
            # Connect all pairs
            for i in range(len(affected_series)):
                for j in range(i + 1, len(affected_series)):
                    series_graph.add_edge(affected_series[i], affected_series[j], reason=identity_name)

        # Find connected components
        groups = {}
        processed_nodes = set()
        
        for i, group_nodes in enumerate(nx.connected_components(series_graph)):
            group_list = list(group_nodes)
            processed_nodes.update(group_list)
            
            # Collect all constraints for this group
            group_constraints = set()
            if len(group_list) > 1:
                sub_graph_edges = series_graph.subgraph(group_list).edges(data=True)
                for _, _, edge_data in sub_graph_edges:
                    if 'reason' in edge_data:
                        group_constraints.add(edge_data['reason'])

            # Name group after constraints
            group_name = '+'.join(sorted(list(group_constraints))[:3]) if group_constraints else f"Group_{i}"
            groups[group_name] = group_list

        # Handle independent series
        independent_series = [s for s in series_list if s not in processed_nodes]
        if independent_series:
            groups['independent'] = independent_series
            
        return groups

    def decompose_with_constraints(self, df: pd.DataFrame, series_list: Optional[List[str]] = None) -> Dict:
        """Main entry point for constrained decomposition"""
        series_list = series_list or [col for col in df.columns if col in self.series_metadata]
        
        self.logger.info("Phase 1: Validating all constraints...")
        validation_results = self._validate_all_constraints(df)
        
        self.logger.info("Phase 2: Determining decomposition order...")
        decomposition_levels = self.get_decomposition_order(series_list)
        
        self.logger.info("Phase 3: Analyzing series with full context...")
        metadata_dict = {}
        for s in series_list:
            if s in df.columns:
                metadata_dict[s] = self.analyze_series_with_context(df[s], s)
        
        self.logger.info("Phase 4: Performing constrained decomposition...")
        all_results = {}
        processed_series = set()

        # Process each level
        for level_idx, level_series in enumerate(decomposition_levels):
            level_series_to_process = [s for s in level_series if s not in processed_series]
            if not level_series_to_process: 
                continue

            self.logger.info(f"Processing level {level_idx + 1}/{len(decomposition_levels)}: {len(level_series_to_process)} series")
            constraint_groups = self._group_by_all_constraints(level_series_to_process)
            
            # Process groups in parallel
            tasks = []
            
            for group_name, group_series in constraint_groups.items():
                valid_series = [s for s in group_series if s in df.columns and s not in processed_series]
                if not valid_series: 
                    continue
                
                self.logger.info(f"Processing group '{group_name}' with {len(valid_series)} series")
                
                if group_name == 'independent':
                    # Process independent series in parallel
                    for series_name in valid_series:
                        task = decompose_single_series.remote(
                            df[[series_name]], 
                            metadata_dict[series_name],
                            series_name
                        )
                        tasks.append(task)
                else:
                    # Process constrained group
                    # Prepare system state for remote function
                    system_state = {
                        'identities': {
                            name: {
                                'left_side': identity.left_side,
                                'right_side': identity.right_side,
                                'operators': identity.operators,
                                'identity_type': identity.identity_type,
                                'tolerance': identity.tolerance
                            } for name, identity in self.identities.items()
                        },
                        'series_metadata': {
                            s: {
                                'series_type': m.series_type,
                                'seasonal_pattern': m.seasonal_pattern
                            } for s, m in self.series_metadata.items()
                        }
                    }
                    
                    task = decompose_constrained_group.remote(
                        system_state,
                        df[valid_series].values,
                        df[valid_series].index,
                        valid_series,
                        {s: metadata_dict[s] for s in valid_series},
                        group_name,
                        all_results
                    )
                    tasks.append(task)
            
            # Wait for results
            if tasks:
                level_results_list = ray.get(tasks)
                
                # Merge results
                for result_dict in level_results_list:
                    all_results.update(result_dict)
                
                processed_series.update(level_series_to_process)

        self.logger.info("Phase 5: Validating final decomposition...")
        final_validation = self._validate_decomposed_constraints(all_results)
        
        return self._package_comprehensive_results(all_results, validation_results, final_validation, metadata_dict)

    def _validate_all_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate constraints in raw data"""
        validation_data = []
        
        for identity_name, identity in self.identities.items():
            required_series = [s.lstrip('Δ') for s in identity.left_side + identity.right_side]
            if all(s in df.columns for s in required_series):
                validity = identity.check_validity(df)
                validation_data.append({
                    'constraint_name': identity_name, 
                    'constraint_type': identity.identity_type,
                    'valid_proportion': validity.mean(),
                    'total_periods': len(validity),
                    'invalid_periods': (~validity).sum()
                })
        
        return pd.DataFrame(validation_data)
        
    def _validate_decomposed_constraints(self, results: Dict) -> pd.DataFrame:
        """Validate constraints on decomposed results"""
        validation_data = []
        
        for identity_name, identity in self.identities.items():
            required_series = {s.lstrip('Δ') for s in identity.left_side + identity.right_side}
            if required_series.issubset(results.keys()):
                # Build temporary DataFrame
                temp_df = pd.DataFrame()
                for series in required_series:
                    comp = results[series]
                    temp_df[series] = comp['level'] + comp.get('trend', 0) + comp['cycle'] + comp['seasonal']
                
                # Calculate discrepancy
                left_sum = identity._evaluate_side_with_operators(
                    temp_df, identity.left_side, ['+'] * len(identity.left_side)
                )
                right_sum = identity._evaluate_side_with_operators(
                    temp_df, identity.right_side, identity.operators
                )
                discrepancy = left_sum - right_sum
                
                validation_data.append({
                    'constraint_name': identity_name,
                    'constraint_type': identity.identity_type,
                    'mean_discrepancy': discrepancy.mean(),
                    'std_discrepancy': discrepancy.std(),
                    'max_abs_discrepancy': discrepancy.abs().max()
                })
        
        return pd.DataFrame(validation_data)

    def _package_comprehensive_results(self, decomposition_results, initial_validation, final_validation, metadata_dict):
        """Package results into final format"""
        # Create component DataFrames
        components_dfs = {}
        for comp in ['level', 'trend', 'cycle', 'seasonal']:
            data = {}
            for s, r in decomposition_results.items():
                if comp in r:
                    data[s] = r[comp]
            if data:
                components_dfs[comp] = pd.DataFrame(data)
        
        # Create uncertainty DataFrames
        uncertainty_dfs = {}
        for comp in ['level', 'trend', 'cycle', 'seasonal']:
            lower_data = {}
            upper_data = {}
            for s, r in decomposition_results.items():
                if f'{comp}_lower' in r:
                    lower_data[s] = r[f'{comp}_lower']
                if f'{comp}_upper' in r:
                    upper_data[s] = r[f'{comp}_upper']
            if lower_data:
                uncertainty_dfs[f'{comp}_lower'] = pd.DataFrame(lower_data)
            if upper_data:
                uncertainty_dfs[f'{comp}_upper'] = pd.DataFrame(upper_data)

        return {
            'components': components_dfs,
            'uncertainty': uncertainty_dfs,
            'initial_validation': initial_validation,
            'final_validation': final_validation,
            'series_metadata': metadata_dict,
            'identity_graph': self.identity_graph,
            'identities': self.identities
        }

    # Statistical test methods
    def _test_for_trend(self, series: pd.Series) -> bool:
        """Test if series has significant trend"""
        series = series.dropna()
        if len(series) < 10: 
            return False
        try:
            result = adfuller(series, regression='ct')
            return result[1] > 0.05
        except:
            return False

    def _test_for_seasonality(self, series: pd.Series) -> bool:
        """Test for seasonal patterns"""
        series = series.dropna()
        if len(series) < 12: 
            return False
        try:
            df = pd.DataFrame({
                'y': series.values, 
                'quarter': np.tile(range(4), len(series)//4+1)[:len(series)], 
                'trend': range(len(series))
            })
            restricted = ols('y ~ trend', data=df).fit()
            unrestricted = ols('y ~ trend + C(quarter)', data=df).fit()
            f_stat = ((restricted.ssr - unrestricted.ssr) / 3) / (unrestricted.ssr / (len(series) - 5))
            return stats.f.sf(f_stat, 3, len(series) - 5) < 0.05
        except:
            return False

    def _test_for_volatility_changes(self, series: pd.Series) -> bool:
        """Test for time-varying volatility"""
        try:
            returns = series.pct_change().dropna()
            if len(returns) < 10: 
                return False
            acf_vals = acf(returns**2, nlags=min(4, len(returns)//4), fft=False)
            return np.any(np.abs(acf_vals[1:]) > 2 / np.sqrt(len(returns)))
        except:
            return False

    def _detect_trend_breaks(self, series: pd.Series) -> List[int]:
        """Detect structural breaks (simplified)"""
        return []

    def _detect_outliers(self, series: pd.Series) -> List[pd.Timestamp]:
        """Detect outliers in series"""
        series_no_na = series.dropna()
        if series_no_na.empty:
            return []
        try:
            z_scores = np.abs(stats.zscore(series_no_na))
            outlier_indices = np.where(z_scores > 3.5)[0]
            return [series_no_na.index[idx] for idx in outlier_indices]
        except:
            return []

    def generate_comprehensive_report(self, results: Dict) -> str:
        """Generate analysis report"""
        report_lines = [
            "Comprehensive Flow of Funds Decomposition Report",
            "=" * 60,
            f"\nAnalysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nTotal Series Processed: {len(results.get('components', {}).get('level', pd.DataFrame()).columns)}",
            "\n## Initial Constraint Validation:",
        ]
        
        if 'initial_validation' in results and not results['initial_validation'].empty:
            report_lines.append(results['initial_validation'].to_string())
        else:
            report_lines.append("No validation data available")
        
        report_lines.append("\n## Final Decomposed Constraint Validation:")
        
        if 'final_validation' in results and not results['final_validation'].empty:
            report_lines.append(results['final_validation'].to_string())
        else:
            report_lines.append("No validation data available")
        
        # Add summary statistics
        if 'final_validation' in results and not results['final_validation'].empty:
            val_df = results['final_validation']
            report_lines.extend([
                "\n## Constraint Satisfaction Summary:",
                f"Average absolute discrepancy: {val_df['mean_discrepancy'].abs().mean():.6f}",
                f"Maximum absolute discrepancy: {val_df['max_abs_discrepancy'].max():.6f}",
                f"Constraints checked: {len(val_df)}"
            ])
        
        return "\n".join(report_lines)

    def visualize_comprehensive_network(self, output_file: str = 'comprehensive_network.png', 
                                      highlight_series: Optional[List[str]] = None):
        """Visualize the dependency network"""
        plt.figure(figsize=(20, 20))
        
        # Use a better layout
        pos = nx.spring_layout(self.identity_graph, k=2, iterations=50, seed=42)
        
        # Color nodes by type
        node_colors = []
        for node in self.identity_graph.nodes():
            if node in self.series_metadata:
                if self.series_metadata[node].is_derived:
                    node_colors.append('lightcoral')
                else:
                    node_colors.append('lightblue')
            else:
                node_colors.append('lightgray')
        
        # Draw network
        nx.draw_networkx_nodes(
            self.identity_graph, pos, 
            node_color=node_colors,
            node_size=300,
            alpha=0.7
        )
        
        nx.draw_networkx_edges(
            self.identity_graph, pos,
            edge_color='gray',
            width=0.5,
            alpha=0.5,
            arrows=True
        )
        
        # Add labels for highlighted nodes or high-degree nodes
        labels = {}
        if highlight_series:
            labels = {n: n for n in highlight_series if n in self.identity_graph.nodes()}
        else:
            # Label nodes with degree > 5
            high_degree_nodes = [n for n, d in self.identity_graph.degree() if d > 5][:20]
            labels = {n: n for n in high_degree_nodes}

        nx.draw_networkx_labels(
            self.identity_graph, pos, 
            labels=labels, 
            font_size=8, 
            font_color='black'
        )
        
        plt.title("Flow of Funds Dependency Network", size=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Network visualization saved to {output_file}")

    def plot_constraint_satisfaction(self, results: Dict, constraint_name: str, 
                                   output_file: Optional[str] = None):
        """Plot constraint satisfaction over time"""
        self.logger.info(f"Plotting satisfaction for {constraint_name}")
        
        if constraint_name not in self.identities:
            self.logger.error(f"Constraint '{constraint_name}' not found.")
            return

        identity = self.identities[constraint_name]
        
        # Check if we have all required series
        required_series = {s.lstrip('Δ') for s in identity.left_side + identity.right_side}
        available_series = required_series.intersection(
            set(results.get('components', {}).get('level', pd.DataFrame()).columns)
        )
        
        if len(available_series) < len(required_series):
            self.logger.warning(
                f"Missing series for constraint '{constraint_name}'. "
                f"Required: {required_series}, Available: {available_series}"
            )
            return

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Build temporary DataFrame with decomposed values
        temp_df = pd.DataFrame()
        for series in required_series:
            if series in results['components']['level'].columns:
                comp_sum = (
                    results['components']['level'][series] +
                    results['components'].get('trend', pd.DataFrame()).get(series, 0) +
                    results['components']['cycle'][series] +
                    results['components']['seasonal'][series]
                )
                temp_df[series] = comp_sum

        # Calculate left and right sums
        left_sum = identity._evaluate_side_with_operators(
            temp_df, identity.left_side, ['+'] * len(identity.left_side)
        )
        right_sum = identity._evaluate_side_with_operators(
            temp_df, identity.right_side, identity.operators
        )
        discrepancy = left_sum - right_sum

        # Plot 1: Left vs Right
        ax = axes[0]
        left_sum.plot(ax=ax, label='Left Side', linewidth=2)
        right_sum.plot(ax=ax, label='Right Side', linewidth=2, linestyle='--')
        ax.set_title(f"Constraint '{constraint_name}': Components", fontsize=12)
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Discrepancy
        ax = axes[1]
        discrepancy.plot(ax=ax, label='Discrepancy', color='red', linewidth=2)
        ax.axhline(0, color='black', linestyle=':', linewidth=1)
        ax.fill_between(
            discrepancy.index,
            discrepancy,
            0,
            alpha=0.3,
            color='red'
        )
        ax.set_title("Constraint Violation", fontsize=12)
        ax.set_ylabel('Discrepancy')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Constraint plot saved to {output_file}")
        else:
            plt.show()
        
        plt.close()


# Main execution
if __name__ == "__main__":
    # File paths
    identities_path = Path('fof_formulas_extracted.json')
    data_path = Path('/home/tesla/Z1/temp/data/z1_quarterly/z1_quarterly_data_filtered.parquet')

    # Check if files exist
    if not identities_path.exists():
        print(f"Error: Identities file not found at {identities_path}")
        exit(1)
        
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        exit(1)

    try:
        # Initialize system
        print("Initializing system...")
        system = ComprehensiveAlignedUCSystem(
            identities_file=str(identities_path),
            n_cores=min(8, os.cpu_count()),
            use_gpu=False
        )
        
        # Load data
        print("Loading Z1 data...")
        z1_data = pd.read_parquet(data_path)
        print(f"Loaded data with {len(z1_data)} observations and {len(z1_data.columns)} series")
        
        # Select series to process - find series with the most constraints
        series_constraint_counts = {}
        for series in z1_data.columns:
            if series in system.series_metadata:
                constraint_count = len(system.series_metadata[series].identity_relationships)
                if constraint_count > 0:
                    series_constraint_counts[series] = constraint_count
        
        # Sort by number of constraints and select top series
        sorted_series = sorted(series_constraint_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Start with just 10 series for testing
        n_series_to_process = min(10, len(sorted_series))  # Start small!
        available_series = [s[0] for s in sorted_series[:n_series_to_process]]
        
        print(f"\nSelected {len(available_series)} most constrained series for processing")
        
        if available_series:
            print(f"\nProcessing {len(available_series)} series...")
            
            # Perform decomposition
            results = system.decompose_with_constraints(
                z1_data[available_series],
                series_list=available_series
            )
            
            print("\nDecomposition completed successfully!")
            
            # Generate report
            report = system.generate_comprehensive_report(results)
            print("\n" + report)
            
            # Save results
            for comp_name, comp_df in results['components'].items():
                output_file = f'decomposition_{comp_name}.csv'
                comp_df.to_csv(output_file)
                print(f"Saved {comp_name} components to {output_file}")
            
            # Generate visualizations
            system.visualize_comprehensive_network(
                output_file='network_visualization.png',
                highlight_series=available_series
            )
            
            # Plot constraint satisfaction if any constraints exist
            if system.identities:
                first_constraint = list(system.identities.keys())[0]
                system.plot_constraint_satisfaction(
                    results,
                    first_constraint,
                    output_file='constraint_satisfaction.png'
                )
        else:
            print("\nNo demo series found in the data file.")

    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()
            print("\nRay shutdown complete.")
