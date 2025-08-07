"""
Comprehensive Aligned Sum-Constrained Unobserved Components System
Utilizes ALL information from the extracted FOF formulas and identities
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
# from pytensor import scan
from pytensor.scan import scan
from statsmodels.tsa.statespace.structural import UnobservedComponents
import ray
from typing import Dict, List, Tuple, Optional, Set, Union
import logging
from dataclasses import dataclass, field
from scipy import stats
import arviz as az
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import json
import os
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SeriesFormula:
    """Represents a series-specific formula from FOF data"""
    series_code: str
    series_name: str
    table_code: str
    line_number: int
    data_type: str  # "Computed" or "Source"
    formula: str
    components: List[str]
    derived_from: List[Dict[str, str]]  # [{'operator': '+/-', 'code': '', 'name': ''}]
    used_in: List[Dict[str, str]]
    shown_on: List[str]
    
    def to_identity(self) -> 'AccountingIdentity':
        """Convert series formula to an accounting identity"""
        # Parse the formula to create identity
        if self.formula.startswith('='):
            # Formula like "= + FA116000105 + FA126000105"
            parts = self.formula[1:].strip().split()
            components = []
            operators = []
            
            i = 0
            while i < len(parts):
                if parts[i] in ['+', '-']:
                    operators.append(parts[i])
                    i += 1
                elif parts[i].startswith(('FA', 'FL', 'FU', 'FR', 'FO')):
                    components.append(parts[i])
                    i += 1
                else:
                    i += 1
            
            # Create identity: series_code = sum of components
            return AccountingIdentity(
                name=f"Formula_{self.series_code}",
                identity_type='series_formula',
                formula=f"{self.series_code} = {self.formula[1:].strip()}",
                left_side=[self.series_code],
                right_side=components,
                operators=operators,
                tolerance=1e-6
            )
        return None


@dataclass
class AccountingIdentity:
    """Enhanced accounting identity with full formula support"""
    name: str
    identity_type: str  # 'balance_sheet', 'flow_stock', 'sector_balance', 'market_clearing', 'series_formula'
    formula: str  # Original formula string
    left_side: List[str] = field(default_factory=list)
    right_side: List[str] = field(default_factory=list)
    operators: List[str] = field(default_factory=list)  # Operators for components
    operator: str = '='  # Main operator: '=', '<=', '>='
    tolerance: float = 1e-6
    parsed_formula: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)  # Additional metadata
    
    def check_validity(self, data: pd.DataFrame) -> pd.Series:
        """Check if identity holds in the data"""
        if self.identity_type == 'flow_stock' and any('Δ' in s for s in self.left_side):
            # Handle delta notation for flow-stock
            return self._check_flow_stock_validity(data)
        elif self.parsed_formula:
            # Use parsed formula for complex identities
            left_value = self._evaluate_expression(
                self.parsed_formula['left'], data
            )
            right_value = self._evaluate_expression(
                self.parsed_formula['right'], data
            )
        else:
            # Simple sum-based identity with operators
            left_value = self._evaluate_side_with_operators(data, self.left_side, ['+'] * len(self.left_side))
            right_value = self._evaluate_side_with_operators(data, self.right_side, self.operators)
        
        if self.operator == '=':
            return np.abs(left_value - right_value) < self.tolerance
        elif self.operator == '<=':
            return left_value <= right_value + self.tolerance
        else:  # '>='
            return left_value >= right_value - self.tolerance
    
    def _check_flow_stock_validity(self, data: pd.DataFrame) -> pd.Series:
        """Special handling for flow-stock identities with delta notation"""
        # Extract level series code from Δ notation
        delta_series = [s for s in self.left_side if s.startswith('Δ')]
        if delta_series:
            level_code = delta_series[0][1:]  # Remove Δ
            if level_code in data.columns:
                # Calculate change in level
                level_change = data[level_code].diff()
                
                # Sum right side (flows + revaluations + other)
                right_sum = pd.Series(0, index=data.index)
                for i, series in enumerate(self.right_side):
                    if series in data.columns:
                        op = self.operators[i] if i < len(self.operators) else '+'
                        if op == '+':
                            right_sum += data[series]
                        else:
                            right_sum -= data[series]
                
                # Check validity (ignoring first observation due to diff)
                validity = pd.Series(True, index=data.index)
                validity.iloc[1:] = np.abs(level_change.iloc[1:] - right_sum.iloc[1:]) < self.tolerance
                return validity
        
        return pd.Series(True, index=data.index)
    
    def _evaluate_side_with_operators(self, data: pd.DataFrame, 
                                    series_list: List[str], 
                                    operators: List[str]) -> pd.Series:
        """Evaluate a side of the identity with operators"""
        result = pd.Series(0, index=data.index)
        
        for i, series in enumerate(series_list):
            if series in data.columns:
                op = operators[i] if i < len(operators) else '+'
                if op == '+':
                    result += data[series]
                else:
                    result -= data[series]
            elif series.startswith('Δ'):
                # Handle delta notation
                level_code = series[1:]
                if level_code in data.columns:
                    result += data[level_code].diff()
        
        return result
    
    def _evaluate_expression(self, expr: Dict, data: pd.DataFrame) -> pd.Series:
        """Evaluate a parsed expression tree"""
        if expr['type'] == 'series':
            series_name = expr['name']
            lag = expr.get('lag', 0)
            
            # Handle delta notation
            if series_name.startswith('Δ'):
                level_code = series_name[1:]
                if level_code in data.columns:
                    return data[level_code].diff()
                else:
                    return pd.Series(0, index=data.index)
            elif series_name in data.columns:
                if lag > 0:
                    return data[series_name].shift(lag)
                else:
                    return data[series_name]
            else:
                return pd.Series(0, index=data.index)
                
        elif expr['type'] == 'constant':
            return pd.Series(expr['value'], index=data.index)
            
        elif expr['type'] == 'operation':
            op = expr['operator']
            left = self._evaluate_expression(expr['left'], data)
            right = self._evaluate_expression(expr['right'], data)
            
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                return left / right.replace(0, np.nan)
            else:
                raise ValueError(f"Unknown operator: {op}")
                
        else:
            raise ValueError(f"Unknown expression type: {expr['type']}")


@dataclass 
class SeriesMetadata:
    """Enhanced metadata including all extracted information"""
    name: str
    series_type: str
    crosses_zero: bool
    volatility_regime: str
    transformation: str
    seasonal_pattern: str
    outlier_dates: List[pd.Timestamp]
    structural_breaks: List[pd.Timestamp]
    identity_relationships: List[str] = field(default_factory=list)
    formula_relationships: List[str] = field(default_factory=list)  # Series used in formula
    is_derived: bool = False  # True if computed from formula
    parent_series: List[str] = field(default_factory=list)  # Components in formula
    data_type: str = "Source"  # "Computed" or "Source"
    table_codes: List[str] = field(default_factory=list)  # Tables where series appears
    sector_code: str = ""  # Extracted sector code
    instrument_code: str = ""  # Extracted instrument code


class ComprehensiveAlignedUCSystem:
    """
    Comprehensive system utilizing ALL extracted FOF information:
    - Individual series formulas
    - All types of accounting identities
    - Series metadata and relationships
    - Sector and instrument codes
    - Delta notation for flow-stock relationships
    """
    
    def __init__(self, 
                 identities_file: str,
                 n_cores: int = None,
                 use_gpu: bool = True,
                 cache_dir: str = './cache'):
                 
        self.logger = logging.getLogger(self.__class__.__name__)         
        
        self.n_cores = n_cores or os.cpu_count()
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        
        # Load all data from JSON
        self.all_data = self._load_all_data(identities_file)
        
        # Process all information
        self.identities = self._load_all_identities()
        self.series_formulas = self._load_series_formulas()
        self.series_metadata = self._load_series_metadata()
        self.sectors = self.all_data.get('sectors', {})
        
        # Build comprehensive dependency graph
        self.identity_graph = self._build_comprehensive_graph()
        
        # Initialize Ray
        # ray.init(num_cpus=self.n_cores, num_gpus=1 if use_gpu else 0)
        ray.init(num_cpus=self.n_cores, num_gpus=4 if self.use_gpu else 0)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Loaded {len(self.identities)} identities")
        self.logger.info(f"Loaded {len(self.series_formulas)} series formulas")
        self.logger.info(f"Total constraints: {len(self.identities) + len(self.series_formulas)}")
    
    def _load_all_data(self, identities_file: str) -> Dict:
        """Load all data from extracted JSON"""
        with open(identities_file, 'r') as f:
            return json.load(f)
    
    def _load_all_identities(self) -> Dict[str, AccountingIdentity]:
        """Load identities AND convert series formulas to identities"""
        identities = {}
        
        # 1. Load explicit identities
        for identity_data in self.all_data.get('identities', []):
            identity = self._create_identity_from_data(identity_data)
            identities[identity.name] = identity
        
        # 2. Convert series formulas to identities
        for series_code, formula_data in self.all_data.get('formulas', {}).items():
            if formula_data.get('data_type') == 'Computed' and formula_data.get('formula'):
                # Create identity from formula
                identity = self._create_identity_from_formula(series_code, formula_data)
                if identity:
                    identities[identity.name] = identity
        
        self.logger.info(f"Created {len(identities)} total identities (including formulas)")
        return identities
    
    def _create_identity_from_data(self, identity_data: Dict) -> AccountingIdentity:
        """Create identity from extracted data"""
        # Handle different identity types
        if identity_data['identity_type'] == 'flow_stock':
            # Special handling for flow-stock with delta notation
            return AccountingIdentity(
                name=identity_data['identity_name'],
                identity_type=identity_data['identity_type'],
                formula=identity_data.get('description', ''),
                left_side=identity_data['left_side'],
                right_side=identity_data['right_side'],
                operators=['+'] * len(identity_data['right_side']),
                metadata={'table_codes': identity_data.get('table_codes', [])}
            )
        else:
            # Standard identity
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
        """Create identity from series formula"""
        formula_str = formula_data.get('formula', '')
        
        if not formula_str:
            return None
        
        # Parse formula
        components = []
        operators = []
        
        if formula_str.startswith('='):
            # Parse components and operators from formula
            parts = formula_str[1:].strip().split()
            
            i = 0
            while i < len(parts):
                if parts[i] in ['+', '-']:
                    operators.append(parts[i])
                    i += 1
                elif parts[i] and parts[i][0].isalpha():  # Series code
                    components.append(parts[i])
                    if not operators:  # First component defaults to +
                        operators.append('+')
                    i += 1
                else:
                    i += 1
        
        # Use derived_from if available for more accurate parsing
        if formula_data.get('derived_from'):
            components = []
            operators = []
            for item in formula_data['derived_from']:
                operators.append(item['operator'])
                components.append(item['code'])
        
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
                    'table_code': formula_data.get('table_code', ''),
                    'data_type': formula_data.get('data_type', '')
                }
            )
        
        return None
    
    def _load_series_formulas(self) -> Dict[str, SeriesFormula]:
        """Load all series formulas"""
        formulas = {}
        
        for series_code, data in self.all_data.get('formulas', {}).items():
            formula = SeriesFormula(
                series_code=series_code,
                series_name=data.get('series_name', ''),
                table_code=data.get('table_code', ''),
                line_number=data.get('line_number', 0),
                data_type=data.get('data_type', ''),
                formula=data.get('formula', ''),
                components=data.get('components', []),
                derived_from=data.get('derived_from', []),
                used_in=data.get('used_in', []),
                shown_on=data.get('shown_on', [])
            )
            formulas[series_code] = formula
        
        return formulas
    
    def _load_series_metadata(self) -> Dict[str, SeriesMetadata]:
        """Build comprehensive metadata for all series"""
        metadata = {}
        
        # Process all series from the data
        for series_code, series_data in self.all_data.get('series', {}).items():
            # Extract sector and instrument codes
            sector_code = series_code[2:4] if len(series_code) >= 4 else ''
            instrument_code = series_code[4:] if len(series_code) > 4 else ''
            
            # Find all identity relationships
            identity_rels = []
            formula_rels = []
            
            for identity_name, identity in self.identities.items():
                if series_code in identity.left_side or series_code in identity.right_side:
                    identity_rels.append(identity_name)
            
            # Check if this series is computed
            is_computed = series_code in self.all_data.get('formulas', {})
            formula_data = self.all_data.get('formulas', {}).get(series_code, {})
            
            # Extract parent series from formula
            parent_series = []
            if formula_data:
                parent_series = formula_data.get('components', [])
                # Also check derived_from for more accurate component list
                if formula_data.get('derived_from'):
                    parent_series = [item['code'] for item in formula_data['derived_from']]
            
            # Find where this series is used
            for other_code, other_formula in self.all_data.get('formulas', {}).items():
                if series_code in other_formula.get('components', []):
                    formula_rels.append(other_code)
            
            metadata[series_code] = SeriesMetadata(
                name=series_data.get('series_name', ''),
                series_type=self._determine_series_type(series_code),
                crosses_zero=False,  # Will be determined from data
                volatility_regime='unknown',  # Will be determined from data
                transformation='unknown',  # Will be determined from data
                seasonal_pattern='unknown',  # Will be determined from data
                outlier_dates=[],
                structural_breaks=[],
                identity_relationships=identity_rels,
                formula_relationships=formula_rels,
                is_derived=is_computed,
                parent_series=parent_series,
                data_type=formula_data.get('data_type', 'Source'),
                table_codes=series_data.get('shown_on', []),
                sector_code=sector_code,
                instrument_code=instrument_code
            )
        
        return metadata
    
    def _determine_series_type(self, series_code: str) -> str:
        """Determine series type from code prefix"""
        if series_code.startswith('FL'):
            return 'level'
        elif series_code.startswith('FU'):
            return 'flow'
        elif series_code.startswith('FA'):
            return 'flow_annual'
        elif series_code.startswith('FR'):
            return 'revaluation'
        elif series_code.startswith('FO'):
            return 'other_change'
        else:
            return 'unknown'
    
    def _build_comprehensive_graph(self) -> nx.DiGraph:
        """Build comprehensive dependency graph including all relationships"""
        G = nx.DiGraph()
        
        # Add all series as nodes with metadata
        all_series = set()
        
        # From identities
        for identity in self.identities.values():
            all_series.update(identity.left_side)
            all_series.update(identity.right_side)
        
        # From formulas
        for formula in self.series_formulas.values():
            all_series.add(formula.series_code)
            all_series.update(formula.components)
        
        # Add nodes with attributes
        for series in all_series:
            metadata = self.series_metadata.get(series, SeriesMetadata(
                name=series,
                series_type='unknown',
                crosses_zero=False,
                volatility_regime='unknown',
                transformation='unknown',
                seasonal_pattern='unknown',
                outlier_dates=[],
                structural_breaks=[],
                sector_code=series[2:4] if len(series) >= 4 else '',
                instrument_code=series[4:] if len(series) > 4 else ''
            ))
            
            G.add_node(series, 
                      sector=metadata.sector_code,
                      instrument=metadata.instrument_code,
                      is_derived=metadata.is_derived,
                      data_type=metadata.data_type)
        
        # Add edges from identities
        for identity in self.identities.values():
            if identity.identity_type == 'balance_sheet':
                # Assets -> Net Worth, Liabilities -> Net Worth
                if len(identity.right_side) >= 2:
                    net_worth = identity.right_side[-1]
                    assets = identity.left_side[0] if identity.left_side else None
                    liabilities = identity.right_side[0] if identity.right_side else None
                    
                    if assets:
                        G.add_edge(assets, net_worth, 
                                 identity=identity.name,
                                 identity_type='balance_sheet',
                                 relationship='assets_to_networth')
                    if liabilities:
                        G.add_edge(liabilities, net_worth,
                                 identity=identity.name,
                                 identity_type='balance_sheet',
                                 relationship='liabilities_to_networth')
                        
            elif identity.identity_type == 'flow_stock':
                # Flows -> Level changes
                level_series = [s[1:] for s in identity.left_side if s.startswith('Δ')]
                if level_series:
                    level = level_series[0]
                    for flow in identity.right_side:
                        G.add_edge(flow, level,
                                 identity=identity.name,
                                 identity_type='flow_stock',
                                 relationship='flow_to_stock')
                        
            elif identity.identity_type == 'sector_balance':
                # Components -> Balance
                for left_item in identity.left_side:
                    for right_item in identity.right_side:
                        G.add_edge(left_item, right_item,
                                 identity=identity.name,
                                 identity_type='sector_balance',
                                 relationship='sector_balance')
                        
            elif identity.identity_type == 'series_formula':
                # Components -> Computed series
                target = identity.left_side[0] if identity.left_side else None
                if target:
                    for i, component in enumerate(identity.right_side):
                        operator = identity.operators[i] if i < len(identity.operators) else '+'
                        G.add_edge(component, target,
                                 identity=identity.name,
                                 identity_type='series_formula',
                                 operator=operator,
                                 relationship='formula_component')
        
        # Add edges from "used_in" relationships
        for series_code, formula in self.series_formulas.items():
            for used_item in formula.used_in:
                used_code = used_item['code']
                if used_code in all_series and series_code in all_series:
                    G.add_edge(series_code, used_code,
                             relationship='used_in',
                             operator=used_item['operator'])
        
        self.logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def analyze_series_with_context(self, series: pd.Series, name: str) -> SeriesMetadata:
        """Analyze series with all available context"""
        # Get existing metadata
        metadata = self.series_metadata.get(name, SeriesMetadata(
            name=name,
            series_type='unknown',
            crosses_zero=False,
            volatility_regime='unknown',
            transformation='unknown',
            seasonal_pattern='unknown',
            outlier_dates=[],
            structural_breaks=[],
            sector_code=name[2:4] if len(name) >= 4 else '',
            instrument_code=name[4:] if len(name) > 4 else ''
        ))
        
        # Update with data-driven analysis
        metadata.crosses_zero = series.min() < 0 < series.max()
        metadata.volatility_regime = 'high' if self._test_for_volatility_changes(series) else 'low'
        metadata.transformation = 'trend' if self._test_for_trend(series) else 'stationary'
        metadata.seasonal_pattern = 'quarterly' if self._test_for_seasonality(series) else 'none'
        metadata.outlier_dates = self._detect_outliers(series)
        metadata.structural_breaks = self._detect_trend_breaks(series)
        
        # Add sector name
        if metadata.sector_code in self.sectors:
            metadata.sector_name = self.sectors[metadata.sector_code]
        
        return metadata
    
    def get_decomposition_order(self, series_list: List[str]) -> List[List[str]]:
        """Determine optimal order considering all dependencies"""
        # Create subgraph with only requested series
        subgraph = self.identity_graph.subgraph(series_list)
        
        # Separate into source and computed series
        source_series = []
        computed_series = []
        
        for series in series_list:
            if series in self.series_metadata:
                if self.series_metadata[series].is_derived:
                    computed_series.append(series)
                else:
                    source_series.append(series)
            else:
                source_series.append(series)
        
        # Try topological sort
        try:
            topo_order = list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            # Handle cycles
            cycles = list(nx.simple_cycles(subgraph))
            self.logger.warning(f"Found {len(cycles)} cycles in dependency graph")
            
            # Break cycles by removing weakest links
            for cycle in cycles:
                # Find edge with fewest constraints
                min_edge = None
                min_constraints = float('inf')
                
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    if subgraph.has_edge(u, v):
                        edge_data = subgraph[u][v]
                        num_constraints = len([k for k in edge_data.keys() if 'identity' in k])
                        if num_constraints < min_constraints:
                            min_constraints = num_constraints
                            min_edge = (u, v)
                
                if min_edge:
                    subgraph.remove_edge(*min_edge)
            
            topo_order = list(nx.topological_sort(subgraph))
        
        # Group into levels ensuring source series come first
        levels = []
        
        # Level 0: Source series with no dependencies
        level0 = [s for s in source_series if s in topo_order and 
                  subgraph.in_degree(s) == 0]
        if level0:
            levels.append(level0)
        
        # Remaining levels
        remaining = set(topo_order) - set(level0)
        while remaining:
            level = []
            for node in remaining:
                predecessors = set(subgraph.predecessors(node))
                already_processed = set()
                for prev_level in levels:
                    already_processed.update(prev_level)
                
                if not predecessors.intersection(remaining):
                    level.append(node)
            
            if level:
                levels.append(level)
                remaining -= set(level)
            else:
                # Handle remaining nodes
                levels.append(list(remaining))
                break
        
        return levels
    
    def _group_by_all_constraints(self, series_list: List[str]) -> Dict[str, List[str]]:
        """Group series by ALL their constraints (identities + formulas)"""
        groups = {'independent': []}
        constraint_groups = {}
        
        # Build constraint mapping
        series_to_constraints = {}
        for series in series_list:
            constraints = set()
            
            # Add identity constraints
            for identity_name, identity in self.identities.items():
                if series in identity.left_side or series in identity.right_side:
                    constraints.add(identity_name)
            
            # Add formula constraints
            if series in self.series_metadata and self.series_metadata[series].is_derived:
                constraints.add(f"Formula_{series}")
            
            series_to_constraints[series] = constraints
        
        # Group series that share constraints
        processed = set()
        
        for series, constraints in series_to_constraints.items():
            if series in processed:
                continue
                
            if not constraints:
                groups['independent'].append(series)
                processed.add(series)
                continue
            
            # Find all series that share any constraint with this one
            group = {series}
            to_check = [series]
            
            while to_check:
                current = to_check.pop()
                current_constraints = series_to_constraints.get(current, set())
                
                for other_series, other_constraints in series_to_constraints.items():
                    if other_series not in processed and other_series != current:
                        if current_constraints.intersection(other_constraints):
                            if other_series not in group:
                                group.add(other_series)
                                to_check.append(other_series)
            
            # Add entire group to processed
            processed.update(group)
            
            # Name group by primary constraint
            primary_constraint = list(constraints)[0]
            if primary_constraint not in constraint_groups:
                constraint_groups[primary_constraint] = list(group)
        
        # Merge groups - NO DUPLICATE SERIES
        groups.update(constraint_groups)
        
        # Log the grouping
        self.logger.info(f"Grouping results:")
        for group_name, group_series in groups.items():
            self.logger.info(f"  Group '{group_name}': {len(group_series)} series")
            if len(group_series) <= 10:  # Only show series for small groups
                self.logger.info(f"    Series: {group_series}")
        
        # Check for duplicates across groups
        all_grouped = []
        for group_series in groups.values():
            all_grouped.extend(group_series)
        
        duplicates = [s for s in set(all_grouped) if all_grouped.count(s) > 1]
        if duplicates:
            self.logger.warning(f"WARNING: Series appear in multiple groups: {duplicates}")
        
        return groups
    
    def decompose_with_constraints(self,
                                  df: pd.DataFrame,
                                  series_list: Optional[List[str]] = None) -> Dict:
        """Main entry point with comprehensive constraint handling"""
        if series_list is None:
            series_list = df.columns.tolist()
        
        # Phase 1: Validate all constraints in raw data
        self.logger.info("Phase 1: Validating all constraints...")
        validation_results = self._validate_all_constraints(df)
        
        # Phase 2: Determine decomposition order
        self.logger.info("Phase 2: Determining decomposition order...")
        decomposition_levels = self.get_decomposition_order(series_list)
        self.logger.info(f"Will process in {len(decomposition_levels)} levels")
        
        # Phase 3: Analyze series with full context
        self.logger.info("Phase 3: Analyzing series with full context...")
        metadata_dict = {}
        for series in series_list:
            if series in df.columns:
                metadata = self.analyze_series_with_context(df[series], series)
                metadata_dict[series] = metadata
        
        # Phase 4: Decompose level by level with all constraints
        self.logger.info("Phase 4: Performing constrained decomposition...")
        all_results = {}
        processed_series = set()  # Track all processed series globally
        
        for level_idx, level_series in enumerate(decomposition_levels):
            # Filter out already processed series at the level
            level_series_to_process = [s for s in level_series if s not in processed_series]
            
            if not level_series_to_process:
                self.logger.info(f"Level {level_idx + 1}: All series already processed, skipping")
                continue
                
            self.logger.info(f"Processing level {level_idx + 1}/{len(decomposition_levels)}: {len(level_series_to_process)} series")
            
            # Group by all constraints
            constraint_groups = self._group_by_all_constraints(level_series_to_process)
            
            # Process each constraint group
            for group_name, group_series in constraint_groups.items():
                # Double-check: filter out already processed series
                valid_series = [s for s in group_series if s in df.columns and s not in processed_series]
                
                if not valid_series:
                    self.logger.info(f"Group {group_name}: All series already processed, skipping")
                    continue
                
                self.logger.info(f"Processing group {group_name} with {len(valid_series)} series: {valid_series}")
                
                if group_name == 'independent':
                    # Process independently
                    results = self._decompose_independent_batch(
                        df[valid_series],
                        {s: metadata_dict[s] for s in valid_series}
                    )
                else:
                    # Process with comprehensive constraints
                    results = self._decompose_with_all_constraints(
                        df[valid_series],
                        {s: metadata_dict[s] for s in valid_series},
                        group_name,
                        all_results
                    )
                
                all_results.update(results)
                processed_series.update(valid_series)
                
                # Log what was processed
                self.logger.info(f"Completed processing {len(results)} series in group {group_name}")
        
        # Phase 5: Validate final results
        self.logger.info("Phase 5: Validating final decomposition...")
        final_validation = self._validate_decomposed_constraints(all_results)
        
        return self._package_comprehensive_results(
            all_results, 
            validation_results, 
            final_validation,
            metadata_dict
        )
    
    def _decompose_with_all_constraints(self,
                                       df: pd.DataFrame,
                                       metadata_dict: Dict[str, SeriesMetadata],
                                       primary_constraint: str,
                                       previous_results: Dict) -> Dict:
        """Decompose with all applicable constraints"""
        self.logger.info(f"Comprehensive constrained decomposition for: {primary_constraint}")
        
        T = len(df)
        series_names = df.columns.tolist()
        
        # Collect all constraints affecting these series
        applicable_constraints = []
        
        for identity_name, identity in self.identities.items():
            affected_series = set(identity.left_side + identity.right_side)
            # Remove delta notation for checking
            affected_series = {s[1:] if s.startswith('Δ') else s for s in affected_series}
            if affected_series.intersection(series_names):
                applicable_constraints.append((identity_name, identity))
        
        self.logger.info(f"Applying {len(applicable_constraints)} constraints to {len(series_names)} series")
        
        # Create a completely fresh model - no context issues
        with pm.Model() as model:
            # Create flexible components for each series
            components = {}
            
            for series_name in series_names:
                series = df[series_name]
                self.logger.info(f"Creating components for {series_name}")
                
                # Add unique identifier to avoid conflicts
                import hashlib
                model_hash = hashlib.md5(f"{primary_constraint}_{series_name}".encode()).hexdigest()[:8]
                
                components[series_name] = self._create_contextual_components_unique(
                    series_name, series, metadata_dict[series_name], T, df, model_hash
                )
            
            # Apply ALL constraints
            for i, (constraint_name, constraint) in enumerate(applicable_constraints):
                self._apply_constraint(
                    constraint, constraint_name, components, T, previous_results, constraint_idx=i
                )
                
            # Observation equations
            for series_name in series_names:
                series = df[series_name]
                comp = components[series_name]
                
                # UCM-style: all components are additive
                mu = comp['level'] + comp['cycle'] + comp['seasonal']
                # Note: trend is already incorporated into level
                
                # Observation
                obs = pm.Normal(
                    f'obs_{series_name}',
                    mu=mu,
                    sigma=comp['sigma_obs'],
                    observed=series.values
                )                
            
            # Sample with appropriate settings
            # Replace pm.sample with:
            # Use ADVI for faster approximate inference
            approx = pm.fit(
                n=10000,  # iterations
                method='advi',
                obj_optimizer=pm.adam(learning_rate=0.01),
                progressbar=True
            )
            trace = approx.sample(500)  # Draw samples from approximation
        
        return self._extract_comprehensive_results(trace, components, df)
    
    def _create_contextual_components(self, 
                                    name: str, 
                                    series: pd.Series,
                                    metadata: SeriesMetadata,
                                    T: int,
                                    df: pd.DataFrame) -> Dict:
        """Create components using all available context"""
        components = {}
        init_mean = series.iloc[:5].mean()
        init_std = series.std()
        
        # Adjust priors based on series type and relationships
        if metadata.is_derived:
            # Tighter priors for computed series
            level_prior_std = init_std * 0.05
            trend_prior_std = init_std * 0.0005
        else:
            level_prior_std = init_std * 0.1
            trend_prior_std = init_std * 0.001
        
        # LEVEL
        level_init = pm.Normal(
            f'{name}_level_init',
            mu=init_mean,
            sigma=init_std
        )
        
        sigma_level = pm.HalfNormal(
            f'{name}_sigma_level',
            sigma=level_prior_std
        )
        
        level = pm.GaussianRandomWalk(
            f'{name}_level',
            mu=0,
            sigma=sigma_level,
            init_dist=pm.Normal.dist(level_init, sigma_level),
            shape=T
        )
        
        components['level'] = level
        
        # TREND - adjusted for series type
        if metadata.transformation == 'trend' or metadata.series_type in ['level', 'flow']:
            trend_init = pm.Normal(
                f'{name}_trend_init',
                mu=0,
                sigma=trend_prior_std
            )
            
            sigma_trend = pm.HalfNormal(
                f'{name}_sigma_trend',
                sigma=trend_prior_std
            )
            
            # Include structural breaks from metadata
            trend_innovations = pm.Normal(
                f'{name}_trend_innov',
                mu=0,
                sigma=1,
                shape=T
            )
            
            # Add known structural breaks
            for idx, break_point in enumerate(metadata.structural_breaks):
                if isinstance(break_point, pd.Timestamp):
                    break_idx = df.index.get_loc(break_point)
                else:
                    break_idx = break_point
                    
                if 0 < break_idx < T:
                    break_indicator = pm.Bernoulli(
                        f'{name}_break_{idx}',
                        p=0.9  # High probability since we detected it
                    )
                    break_size = pm.Normal(
                        f'{name}_break_size_{idx}',
                        mu=0,
                        sigma=init_std * 0.1
                    )
                    trend_innovations = pt.set_subtensor(
                        trend_innovations[break_idx],
                        trend_innovations[break_idx] + break_indicator * break_size
                    )
            
            trend = pm.Deterministic(
                f'{name}_trend',
                (trend_init + pt.cumsum(trend_innovations * sigma_trend)) * pt.arange(T, dtype='float64')
            )
        else:
            trend = pt.zeros(T)

        components['trend'] = trend
                
        # CYCLE - adjusted based on relationships
        if metadata.series_type in ['flow', 'level']:
            # Check if series shares cycles with related series
            related_sectors = []
            for other_series, other_meta in self.series_metadata.items():
                if (other_meta.sector_code == metadata.sector_code and 
                    other_series != name):
                    related_sectors.append(other_series)
            
            # Sector-specific cycle period prior
            if metadata.sector_code in ['10', '11', '12']:  # Real sectors
                cycle_period_mu = 6 * 4  # 6 years
                cycle_period_sd = 2
            elif metadata.sector_code in ['40', '50', '60']:  # Financial sectors
                cycle_period_mu = 4 * 4  # 4 years  
                cycle_period_sd = 1
            else:
                cycle_period_mu = 5 * 4  # 5 years default
                cycle_period_sd = 2
            
            cycle_period = pm.Normal(
                f'{name}_cycle_period',
                mu=cycle_period_mu,
                sigma=cycle_period_sd
            )
            
            rho = pm.Beta(
                f'{name}_rho',
                alpha=8,
                beta=2
            )
            
            sigma_cycle = pm.HalfNormal(
                f'{name}_sigma_cycle',
                sigma=init_std * 0.1
            )
            
            # Phase shift - can be constrained for related series
            phase_shift = pm.Uniform(
                f'{name}_phase_shift',
                lower=0,
                upper=2 * np.pi
            )
            
            # Build cycle
            cycle_freq = 2 * np.pi / cycle_period
            cos_omega = pt.cos(cycle_freq)
            sin_omega = pt.sin(cycle_freq)

            cycle_innovations = pm.Normal(
                f'{name}_cycle_innov',
                mu=0,
                sigma=sigma_cycle,
                shape=(T, 2)
            )

            # Initialize with phase
            cycle_init = cycle_innovations[0, 0] * pt.cos(phase_shift)
            cycle_star_init = cycle_innovations[0, 0] * pt.sin(phase_shift)

            # Define the step function for scan
            def cycle_step(innov, prev_cycle, prev_cycle_star, rho, cos_omega, sin_omega):
                new_cycle = (rho * cos_omega * prev_cycle +
                             rho * sin_omega * prev_cycle_star + innov[0])
                new_cycle_star = (-rho * sin_omega * prev_cycle +
                                  rho * cos_omega * prev_cycle_star + innov[1])
                return new_cycle, new_cycle_star

            # Run the scan
            (cycle_states, cycle_star_states), _ = scan(
                fn=cycle_step,
                sequences=[cycle_innovations[1:]],
                outputs_info=[cycle_init, cycle_star_init],
                non_sequences=[rho, cos_omega, sin_omega]
            )

            # 2. ASSEMBLE THE FINAL CYCLE AFTER THE SCAN
            #    Concatenate the initial state with the generated states
            full_cycle = pt.concatenate([pt.atleast_1d(cycle_init), cycle_states])

            cycle = pm.Deterministic(
                f'{name}_cycle',
                full_cycle
            )

        else:
            cycle = pt.zeros(T)
        
        components['cycle'] = cycle
        
        # SEASONAL - with metadata-informed patterns
        if metadata.seasonal_pattern != 'none':
            # Instrument-specific seasonal patterns
            if metadata.instrument_code in ['103000', '203000']:  # Loans
                seasonal_prior_std = init_std * 0.3  # Stronger seasonality
            else:
                seasonal_prior_std = init_std * 0.2
            
            seasonal_pattern = pm.Normal(
                f'{name}_seasonal_pattern',
                mu=0,
                sigma=seasonal_prior_std,
                shape=4
            )
            
            # Build seasonal using indexing
            quarter_indices = pt.arange(T) % 4
            seasonal = pm.Deterministic(
                f'{name}_seasonal',
                seasonal_pattern[quarter_indices]
            )
        else:
            seasonal = pt.zeros(T)
        
        components['seasonal'] = seasonal
        
        # STOCHASTIC VOLATILITY - based on metadata
        components['sigma_obs'] = pm.HalfNormal(
            f'{name}_sigma_obs', sigma=init_std * 0.1
        )
        components['stochastic_vol'] = False

        
        return components
    
    def _apply_constraint(self,
                          constraint: AccountingIdentity,
                          constraint_name: str,
                          components: Dict,
                          T: int,
                          previous_results: Dict,
                          constraint_idx: int):
        """Apply a specific constraint to the model"""
        
        if constraint.identity_type == 'flow_stock':
            # Pass constraint_name here
            self._apply_flow_stock_constraint(constraint, constraint_name, components, T, previous_results, constraint_idx)
        elif constraint.parsed_formula:
            # Pass constraint_name here
            self._apply_formula_constraint(constraint, constraint_name, components, T, constraint_idx)
        else:
            # This call is already correct
            self._apply_sum_constraint(constraint, constraint_name, components, T, constraint_idx)
    
    def _apply_flow_stock_constraint(self,
                                     constraint: AccountingIdentity,
                                     constraint_name: str,
                                     components: Dict,
                                     T: int,
                                     previous_results: Dict,
                                     constraint_idx: int):
        """Apply flow-stock constraint with delta notation"""
        # Find level series
        level_series = None
        for s in constraint.left_side:
            if s.startswith('Δ'):
                level_series = s[1:]
                break
        
        if not level_series or level_series not in components:
            self._apply_sum_constraint(constraint, constraint_name, components, T, constraint_idx)
            return
        
        # Calculate level change
        level_comp = components[level_series]
        # UCM: level already includes trend
        level_total = level_comp['level'] + level_comp['cycle'] + level_comp['seasonal']
        
        # Calculate differences directly
        level_change = pt.concatenate([
            pt.zeros(1),
            level_total[1:] - level_total[:-1]
        ])
        
        # Calculate flow sum
        flow_sum = pt.zeros(T)
        for i, series in enumerate(constraint.right_side):
            if series in components:
                comp = components[series]
                # UCM: level includes trend
                total = comp['level'] + comp['cycle'] + comp['seasonal']
                
                op = constraint.operators[i] if i < len(constraint.operators) else '+'
                if op == '+':
                    flow_sum = flow_sum + total
                else:
                    flow_sum = flow_sum - total

        # Apply constraint (ignoring first observation)
        identity_violation = level_change[1:] - flow_sum[1:]
        series_scale       = pt.mean(pt.abs(flow_sum))
        scaled_tolerance   = constraint.tolerance * pt.maximum(100.0,
                                                               series_scale * 0.001)

        # Student‑t penalty
        penalty_logp = pm.logp(
            pm.StudentT.dist(nu=5, mu=0, sigma=scaled_tolerance),
            identity_violation
        )

        pm.Potential(
            f'flow_stock_penalty_{constraint.name}_{constraint_idx}',
            penalty_logp.sum()
        )
    
    def _apply_formula_constraint(
            self,
            constraint: AccountingIdentity,
            constraint_name: str,
            components: Dict,
            T: int,
            constraint_idx: int
        ):
        """Apply a formula constraint using Student‑T penalty (no fallback)."""

        # 1) Build series–sum lookup for every component that is present
        comp_sums = {}
        for series in constraint.left_side + constraint.right_side:
            if series in components:
                c = components[series]
                comp_sums[series] = (
                    c['level'] +               # level already contains trend
                    c['cycle'] +
                    c['seasonal']
                )

        # ----- LEFT‑HAND SIDE -----------------------------------------------
        left_sum = pt.zeros(T)
        for series in constraint.left_side:
            if series in comp_sums:
                left_sum += comp_sums[series]

        # ----- RIGHT‑HAND SIDE ----------------------------------------------
        right_sum = pt.zeros(T)
        for idx, series in enumerate(constraint.right_side):
            if series in comp_sums:
                op = constraint.operators[idx] if idx < len(constraint.operators) else '+'
                if op == '+':
                    right_sum += comp_sums[series]
                else:  # '-'
                    right_sum -= comp_sums[series]

        # 2) Vector of violations
        identity_violation = left_sum - right_sum

        # 3) Scale‑dependent tolerance (same recipe as other helpers)
        series_scale = pt.maximum(
            pt.mean(pt.abs(left_sum)),
            pt.mean(pt.abs(right_sum))
        )
        scaled_tolerance = constraint.tolerance * pt.maximum(
            100.0,
            series_scale * 0.001
        )

        # 4) Student‑t penalty
        penalty_logp = pm.logp(
            pm.StudentT.dist(nu=5, mu=0, sigma=scaled_tolerance),
            identity_violation
        )

        pm.Potential(
            f'formula_penalty_{constraint_name}_{constraint_idx}',
            penalty_logp.sum()
        )

    
    def _apply_sum_constraint(self,
                              constraint: AccountingIdentity,
                              constraint_name: str,
                              components: Dict,
                              T: int,
                              constraint_idx: int):
        """Apply standard sum constraint with smooth trend UCM components"""
        # Calculate left side sum
        left_sum = pt.zeros(T)
        for series in constraint.left_side:
            if series in components:
                comp = components[series]
                # Note: level already includes integrated trend
                left_sum = left_sum + (comp['level'] + comp['cycle'] + comp['seasonal'])
        
        # Calculate right side sum with operators
        right_sum = pt.zeros(T)
        for i, series in enumerate(constraint.right_side):
            if series in components:
                comp = components[series]
                # Note: level already includes integrated trend
                total = comp['level'] + comp['cycle'] + comp['seasonal']
                
                op = constraint.operators[i] if i < len(constraint.operators) else '+'
                if op == '+':
                    right_sum = right_sum + total
                else:
                    right_sum = right_sum - total
        
        # Apply soft constraint (unchanged)
        identity_violation = left_sum - right_sum

        series_scale     = pt.maximum(pt.mean(pt.abs(left_sum)),
                                      pt.mean(pt.abs(right_sum)))
        scaled_tolerance = constraint.tolerance * pt.maximum(100.0,
                                                             series_scale * 0.001)

        penalty_logp = pm.logp(
            pm.StudentT.dist(nu=5, mu=0, sigma=scaled_tolerance),
            identity_violation
        )

        pm.Potential(
            f'sum_penalty_{constraint_name}_{constraint_idx}',
            penalty_logp.sum()
        )
    
    def _validate_all_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate all constraints in raw data"""
        validation_data = []
        
        # Validate identities
        for identity_name, identity in self.identities.items():
            # Check if we have required series
            required_series = identity.left_side + identity.right_side
            # Remove delta notation for checking
            required_series = [s[1:] if s.startswith('Δ') else s for s in required_series]
            
            available_series = [s for s in required_series if s in df.columns]
            
            if len(available_series) >= 2:  # Need at least 2 series
                validity = identity.check_validity(df)
                
                validation_data.append({
                    'constraint_name': identity_name,
                    'constraint_type': identity.identity_type,
                    'valid_proportion': validity.mean(),
                    'max_violation': (~validity).sum(),
                    'series_count': len(available_series),
                    'missing_series': len(required_series) - len(available_series)
                })
        
        return pd.DataFrame(validation_data)
    
    def _validate_decomposed_constraints(self, results: Dict) -> pd.DataFrame:
        """Validate constraints on decomposed components"""
        validation_data = []
        
        for identity_name, identity in self.identities.items():
            # Check if we have all required series
            required_series = identity.left_side + identity.right_side
            required_series = [s[1:] if s.startswith('Δ') else s for s in required_series]
            
            available_series = [s for s in required_series if s in results]
            
            if len(available_series) >= 2:
                # Build DataFrame with component sums
                temp_df = pd.DataFrame()
                for series in available_series:
                    if series in results:
                        comp = results[series]
                        temp_df[series] = (comp['level'] + comp['trend'] + 
                                         comp['cycle'] + comp['seasonal'])
                
                # Check validity
                if identity.identity_type == 'flow_stock':
                    # Special handling for flow-stock
                    discrepancy = self._validate_flow_stock_decomposed(
                        identity, results, temp_df
                    )
                else:
                    # Standard validation
                    left_sum = pd.Series(0, index=temp_df.index)
                    right_sum = pd.Series(0, index=temp_df.index)
                    
                    for series in identity.left_side:
                        if series in temp_df.columns:
                            left_sum += temp_df[series]
                    
                    for i, series in enumerate(identity.right_side):
                        if series in temp_df.columns:
                            op = identity.operators[i] if i < len(identity.operators) else '+'
                            if op == '+':
                                right_sum += temp_df[series]
                            else:
                                right_sum -= temp_df[series]
                    
                    discrepancy = left_sum - right_sum
                
                validation_data.append({
                    'constraint_name': identity_name,
                    'constraint_type': identity.identity_type,
                    'mean_discrepancy': discrepancy.mean(),
                    'std_discrepancy': discrepancy.std(),
                    'max_abs_discrepancy': discrepancy.abs().max(),
                    'series_count': len(available_series)
                })
        
        return pd.DataFrame(validation_data)
    
    def _validate_flow_stock_decomposed(self,
                                      identity: AccountingIdentity,
                                      results: Dict,
                                      temp_df: pd.DataFrame) -> pd.Series:
        """Validate flow-stock identity on decomposed results"""
        # Find level series
        level_series = None
        for s in identity.left_side:
            if s.startswith('Δ'):
                level_series = s[1:]
                break
        
        if level_series and level_series in temp_df.columns:
            level_change = temp_df[level_series].diff()
            
            # Sum flows
            flow_sum = pd.Series(0, index=temp_df.index)
            for i, series in enumerate(identity.right_side):
                if series in temp_df.columns:
                    op = identity.operators[i] if i < len(identity.operators) else '+'
                    if op == '+':
                        flow_sum += temp_df[series]
                    else:
                        flow_sum -= temp_df[series]
            
            # Ignore first observation
            discrepancy = level_change - flow_sum
            discrepancy.iloc[0] = 0
            
            return discrepancy
        
        return pd.Series(0, index=temp_df.index)
    
    def _extract_comprehensive_results(self, 
                                     trace,
                                     components: Dict,
                                     data: pd.DataFrame) -> Dict:
        """Extract results with smooth trend UCM component separation"""
        results = {}
        
        trace_vars = list(trace.posterior.data_vars)
        var_mapping = {}
        
        for series_name in data.columns:
            var_mapping[series_name] = {}
            series_vars = [v for v in trace_vars if v.startswith(f'{series_name}_')]
            
            for var in series_vars:
                if var.endswith('_level') and not var.endswith('_level_init'):
                    var_mapping[series_name]['level'] = var
                elif var.endswith('_trend') and not var.endswith('_trend_init'):
                    var_mapping[series_name]['slope'] = var  # Changed from 'trend'
                elif var.endswith('_cycle') and not var.endswith('_cycle_period'):
                    var_mapping[series_name]['cycle'] = var
                elif var.endswith('_seasonal') and not var.endswith('_seasonal_init'):
                    var_mapping[series_name]['seasonal'] = var
        
        for series_name in data.columns:
            vars_dict = var_mapping.get(series_name, {})
            
            # Extract smooth trend components
            if 'level' in vars_dict:
                level_values = trace.posterior[vars_dict['level']].mean(dim=['chain', 'draw']).values
            else:
                level_values = np.zeros(len(data))
            
            if 'slope' in vars_dict:
                slope_values = trace.posterior[vars_dict['slope']].mean(dim=['chain', 'draw']).values
            else:
                slope_values = np.zeros(len(data))
            
            # For display and analysis:
            # - level: the smooth trend level (includes integrated slope)
            # - trend: the slope/growth rate (β_t)
            results[series_name] = {
                'level': pd.Series(
                    level_values,  # Smooth trend level
                    index=data.index
                ),
                'trend': pd.Series(
                    slope_values,  # Slope (growth rate indicator)
                    index=data.index
                ),
                'cycle': pd.Series(
                    trace.posterior[vars_dict['cycle']].mean(dim=['chain', 'draw']).values 
                    if 'cycle' in vars_dict else np.zeros(len(data)),
                    index=data.index
                ),
                'seasonal': pd.Series(
                    trace.posterior[vars_dict['seasonal']].mean(dim=['chain', 'draw']).values 
                    if 'seasonal' in vars_dict else np.zeros(len(data)),
                    index=data.index
                ),
                'identity_constrained': True,
                'model_type': 'pymc_smooth_trend_ucm'
            }
            
            # Add growth rate calculation for exponential series
            results[series_name]['growth_rate'] = pd.Series(
                np.where(level_values == 0, np.nan, slope_values / level_values * 100),
                index=data.index
            )
            
            # Add uncertainty intervals (same as before)
            for component in ['level', 'slope', 'cycle', 'seasonal']:
                if component in vars_dict:
                    var_name = vars_dict[component]
                    results[series_name][f'{component}_lower'] = pd.Series(
                        trace.posterior[var_name].quantile(0.025, dim=['chain', 'draw']).values,
                        index=data.index
                    )
                    results[series_name][f'{component}_upper'] = pd.Series(
                        trace.posterior[var_name].quantile(0.975, dim=['chain', 'draw']).values,
                        index=data.index
                    )
            
            # Add diagnostics (same as before)
            if 'level' in vars_dict:
                rhat_da = az.rhat(trace, var_names=[vars_dict['level']])
                rhat_values = rhat_da[vars_dict['level']].values
                rhat_max = float(np.max(rhat_values))
                
                ess_bulk_da = az.ess(trace, var_names=[vars_dict['level']], method='bulk')
                ess_bulk_values = ess_bulk_da[vars_dict['level']].values
                ess_bulk_min = float(np.min(ess_bulk_values))
                
                ess_tail_da = az.ess(trace, var_names=[vars_dict['level']], method='tail')
                ess_tail_values = ess_tail_da[vars_dict['level']].values
                ess_tail_min = float(np.min(ess_tail_values))
                
                results[series_name]['diagnostics'] = {
                    'r_hat': rhat_max,
                    'ess_bulk': ess_bulk_min,
                    'ess_tail': ess_tail_min
                }
            else:
                results[series_name]['diagnostics'] = {
                    'r_hat': np.nan,
                    'ess_bulk': np.nan,
                    'ess_tail': np.nan
                }
        
        return results
    
    def _package_comprehensive_results(self,
                                     decomposition_results: Dict,
                                     initial_validation: pd.DataFrame,
                                     final_validation: pd.DataFrame,
                                     metadata_dict: Dict) -> Dict:
        """Package all results with comprehensive information"""
        # Create component DataFrames
        components = {}
        for component in ['level', 'trend', 'cycle', 'seasonal']:
            data = {}
            for series, result in decomposition_results.items():
                if component in result:
                    data[series] = result[component]
            components[component] = pd.DataFrame(data)
        
        # Create uncertainty DataFrames
        uncertainty = {}
        for component in ['level', 'trend', 'cycle', 'seasonal']:
            lower_data = {}
            upper_data = {}
            for series, result in decomposition_results.items():
                if f'{component}_lower' in result:
                    lower_data[series] = result[f'{component}_lower']
                    upper_data[series] = result[f'{component}_upper']
            if lower_data:
                uncertainty[f'{component}_lower'] = pd.DataFrame(lower_data)
                uncertainty[f'{component}_upper'] = pd.DataFrame(upper_data)
        
        # Compile metadata
        metadata_df = pd.DataFrame({
            series: {
                'identity_constrained': result.get('identity_constrained', False),
                'model_type': result.get('model_type', 'unknown'),
                'sector_code': metadata_dict.get(series, SeriesMetadata(
                    name=series, series_type='unknown', crosses_zero=False,
                    volatility_regime='unknown', transformation='unknown',
                    seasonal_pattern='unknown', outlier_dates=[], structural_breaks=[]
                )).sector_code,
                'instrument_code': metadata_dict.get(series, SeriesMetadata(
                    name=series, series_type='unknown', crosses_zero=False,
                    volatility_regime='unknown', transformation='unknown',
                    seasonal_pattern='unknown', outlier_dates=[], structural_breaks=[]
                )).instrument_code,
                'is_derived': metadata_dict.get(series, SeriesMetadata(
                    name=series, series_type='unknown', crosses_zero=False,
                    volatility_regime='unknown', transformation='unknown',
                    seasonal_pattern='unknown', outlier_dates=[], structural_breaks=[]
                )).is_derived,
                'num_constraints': len(metadata_dict.get(series, SeriesMetadata(
                    name=series, series_type='unknown', crosses_zero=False,
                    volatility_regime='unknown', transformation='unknown',
                    seasonal_pattern='unknown', outlier_dates=[], structural_breaks=[]
                )).identity_relationships) + len(metadata_dict.get(series, SeriesMetadata(
                    name=series, series_type='unknown', crosses_zero=False,
                    volatility_regime='unknown', transformation='unknown',
                    seasonal_pattern='unknown', outlier_dates=[], structural_breaks=[]
                )).formula_relationships)
            }
            for series, result in decomposition_results.items()
        }).T
        
        # Compile diagnostics
        diagnostics_data = []
        for series, result in decomposition_results.items():
            if 'diagnostics' in result:
                diag = result['diagnostics']
                diag['series'] = series
                diagnostics_data.append(diag)
        
        diagnostics_df = pd.DataFrame(diagnostics_data) if diagnostics_data else pd.DataFrame()
        
        return {
            'components': components,
            'uncertainty': uncertainty,
            'metadata': metadata_df,
            'initial_validation': initial_validation,
            'final_validation': final_validation,
            'identity_graph': self.identity_graph,
            'identities': self.identities,
            'series_formulas': self.series_formulas,
            'sectors': self.sectors,
            'diagnostics': diagnostics_df,
            'series_metadata': metadata_dict
        }
    
    def visualize_comprehensive_network(self, 
                                      output_file: str = 'comprehensive_network.png',
                                      highlight_series: Optional[List[str]] = None):
        """Visualize comprehensive network with all relationships"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
        
        # Left plot: Identity relationships
        ax1.set_title('Identity and Formula Relationships', fontsize=16)
        
        # Create subgraph with highlighted series if specified
        if highlight_series:
            # Find all connected series
            connected = set(highlight_series)
            for series in highlight_series:
                if series in self.identity_graph:
                    connected.update(nx.ancestors(self.identity_graph, series))
                    connected.update(nx.descendants(self.identity_graph, series))
            
            subgraph = self.identity_graph.subgraph(connected)
        else:
            subgraph = self.identity_graph
        
        pos = nx.spring_layout(subgraph, k=3, iterations=50)
        
        # Node colors by type and highlighting
        node_colors = []
        node_sizes = []
        for node in subgraph.nodes():
            if highlight_series and node in highlight_series:
                node_sizes.append(1000)
            else:
                node_sizes.append(500)
            
            if node in self.series_metadata:
                meta = self.series_metadata[node]
                if meta.is_derived:
                    node_colors.append('lightcoral')  # Computed series
                elif meta.series_type == 'level':
                    node_colors.append('lightblue')
                elif meta.series_type == 'flow':
                    node_colors.append('lightgreen')
                elif meta.series_type == 'revaluation':
                    node_colors.append('lightyellow')
                else:
                    node_colors.append('lightgray')
            else:
                node_colors.append('lightgray')
        
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            ax=ax1
        )
        
        # Edge colors by relationship type
        edge_colors = []
        edge_widths = []
        for u, v, data in subgraph.edges(data=True):
            if data.get('identity_type') == 'balance_sheet':
                edge_colors.append('red')
                edge_widths.append(2)
            elif data.get('identity_type') == 'flow_stock':
                edge_colors.append('blue')
                edge_widths.append(2)
            elif data.get('identity_type') == 'sector_balance':
                edge_colors.append('green')
                edge_widths.append(2)
            elif data.get('identity_type') == 'series_formula':
                edge_colors.append('purple')
                edge_widths.append(1.5)
            elif data.get('relationship') == 'used_in':
                edge_colors.append('orange')
                edge_widths.append(1)
            else:
                edge_colors.append('gray')
                edge_widths.append(0.5)
        
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6,
            arrows=True,
            ax=ax1
        )
        
        # Add labels for highlighted series
        if highlight_series:
            labels = {n: n for n in highlight_series}
        else:
            # Show labels for high-degree nodes
            degrees = dict(subgraph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
            labels = {n: n for n, _ in top_nodes}
        
        nx.draw_networkx_labels(
            subgraph,
            pos,
            labels,
            font_size=10,
            ax=ax1
        )
        
        # Right plot: Sector relationships
        ax2.set_title('Sector and Instrument Relationships', fontsize=16)
        
        # Create sector graph
        sector_graph = nx.Graph()
        
        # Add sectors as nodes
        for sector_code, sector_name in self.sectors.items():
            sector_graph.add_node(sector_code, name=sector_name)
        
        # Add edges based on shared identities
        sector_connections = {}
        for identity in self.identities.values():
            sectors_in_identity = set()
            for series in identity.left_side + identity.right_side:
                if len(series) >= 4:
                    sector = series[2:4]
                    if sector in self.sectors:
                        sectors_in_identity.add(sector)
            
            # Create edges between all sectors in this identity
            for s1 in sectors_in_identity:
                for s2 in sectors_in_identity:
                    if s1 < s2:
                        key = (s1, s2)
                        if key not in sector_connections:
                            sector_connections[key] = 0
                        sector_connections[key] += 1
        
        # Add weighted edges
        for (s1, s2), weight in sector_connections.items():
            sector_graph.add_edge(s1, s2, weight=weight)
        
        # Layout and draw
        pos2 = nx.spring_layout(sector_graph, k=5, iterations=50)
        
        # Node sizes based on number of series
        sector_sizes = {}
        for series in self.series_metadata.keys():
            if len(series) >= 4:
                sector = series[2:4]
                if sector not in sector_sizes:
                    sector_sizes[sector] = 0
                sector_sizes[sector] += 1
        
        node_sizes2 = [min(sector_sizes.get(node, 50) * 20, 3000) 
                      for node in sector_graph.nodes()]
        
        nx.draw_networkx_nodes(
            sector_graph,
            pos2,
            node_size=node_sizes2,
            node_color='lightsteelblue',
            alpha=0.7,
            ax=ax2
        )
        
        # Edge widths based on connection strength
        edge_widths2 = [sector_graph[u][v]['weight'] * 0.5 
                       for u, v in sector_graph.edges()]
        
        nx.draw_networkx_edges(
            sector_graph,
            pos2,
            width=edge_widths2,
            alpha=0.5,
            ax=ax2
        )
        
        # Labels with sector names
        sector_labels = {code: f"{code}\n{name[:20]}..." 
                        if len(name) > 20 else f"{code}\n{name}"
                        for code, name in self.sectors.items()
                        if code in sector_graph.nodes()}
        
        nx.draw_networkx_labels(
            sector_graph,
            pos2,
            sector_labels,
            font_size=8,
            ax=ax2
        )
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comprehensive network visualization saved to {output_file}")
    
    def plot_constraint_satisfaction(self, 
                                   results: Dict,
                                   constraint_name: str,
                                   output_file: Optional[str] = None):
        """Visualize how well a constraint is satisfied over time"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        
        # Get constraint
        if constraint_name not in self.identities:
            self.logger.error(f"Constraint {constraint_name} not found")
            return
        
        constraint = self.identities[constraint_name]
        
        # Handle different constraint types
        if constraint.identity_type == 'flow_stock':
            self._plot_flow_stock_constraint(results, constraint, axes)
        else:
            self._plot_standard_constraint(results, constraint, axes)
        
        plt.suptitle(f'Constraint Analysis: {constraint_name}', fontsize=16)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def _plot_flow_stock_constraint(self, results, constraint, axes):
        """Plot flow-stock constraint satisfaction"""
        # Find level series
        level_series = None
        for s in constraint.left_side:
            if s.startswith('Δ'):
                level_series = s[1:]
                break
        
        if not level_series or level_series not in results['components']['level'].columns:
            self.logger.error(f"Level series {level_series} not found")
            return
        
        # Calculate level change
        level_total = (results['components']['level'][level_series] +
                      results['components']['trend'][level_series] +
                      results['components']['cycle'][level_series] +
                      results['components']['seasonal'][level_series])
        
        level_change = level_total.diff()
        
        # Calculate flow sum
        flow_sum = pd.Series(0, index=level_total.index)
        flow_components = {}
        
        for i, series in enumerate(constraint.right_side):
            if series in results['components']['level'].columns:
                comp_sum = (results['components']['level'][series] +
                           results['components']['trend'][series] +
                           results['components']['cycle'][series] +
                           results['components']['seasonal'][series])
                
                op = constraint.operators[i] if i < len(constraint.operators) else '+'
                if op == '+':
                    flow_sum += comp_sum
                    flow_components[f"+{series}"] = comp_sum
                else:
                    flow_sum -= comp_sum
                    flow_components[f"-{series}"] = -comp_sum
        
        # Plot 1: Level and cumulative flows
        axes[0].plot(level_total, label=f'Level: {level_series}', linewidth=2)
        axes[0].plot(flow_sum.cumsum() + level_total.iloc[0], 
                    label='Cumulative Flows', linestyle='--', linewidth=2)
        axes[0].set_ylabel('Value')
        axes[0].set_title('Level vs Cumulative Flows')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Change in level vs flows
        axes[1].plot(level_change, label='Δ Level', linewidth=2)
        axes[1].plot(flow_sum, label='Flow Sum', linestyle='--', linewidth=2)
        axes[1].fill_between(level_change.index, 
                           level_change - flow_sum,
                           alpha=0.3, color='red', 
                           label='Discrepancy')
        axes[1].set_ylabel('Change')
        axes[1].set_title('Period Changes: Level vs Flows')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Individual flow components
        for name, flow in list(flow_components.items())[:5]:  # Limit to 5
            axes[2].plot(flow, label=name, alpha=0.7)
        axes[2].set_ylabel('Flow Value')
        axes[2].set_title('Individual Flow Components')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Violation over time
        violation = level_change - flow_sum
        violation_pct = (violation / level_total.abs().rolling(4).mean() * 100).fillna(0)
        
        axes[3].plot(violation_pct, label='Violation %', color='red')
        axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[3].fill_between(violation_pct.index,
                           violation_pct.rolling(4).quantile(0.25),
                           violation_pct.rolling(4).quantile(0.75),
                           alpha=0.2, color='red',
                           label='25-75% range')
        axes[3].set_ylabel('Violation (%)')
        axes[3].set_xlabel('Time')
        axes[3].set_title('Constraint Violation as % of Level')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    def _plot_standard_constraint(self, results, constraint, axes):
        """Plot standard constraint satisfaction"""
        # Calculate left and right sums
        left_sum = pd.Series(0, index=results['components']['level'].index)
        right_sum = pd.Series(0, index=results['components']['level'].index)
        
        left_components = {}
        right_components = {}
        
        # Left side
        for series in constraint.left_side:
            if series in results['components']['level'].columns:
                comp_sum = (results['components']['level'][series] +
                           results['components']['trend'][series] +
                           results['components']['cycle'][series] +
                           results['components']['seasonal'][series])
                left_sum += comp_sum
                left_components[series] = comp_sum
        
        # Right side with operators
        for i, series in enumerate(constraint.right_side):
            if series in results['components']['level'].columns:
                comp_sum = (results['components']['level'][series] +
                           results['components']['trend'][series] +
                           results['components']['cycle'][series] +
                           results['components']['seasonal'][series])
                
                op = constraint.operators[i] if i < len(constraint.operators) else '+'
                if op == '+':
                    right_sum += comp_sum
                    right_components[f"+{series}"] = comp_sum
                else:
                    right_sum -= comp_sum
                    right_components[f"-{series}"] = -comp_sum
        
        violation = left_sum - right_sum
        
        # Plot 1: Left vs Right totals
        axes[0].plot(left_sum, label='Left Side Total', linewidth=2)
        axes[0].plot(right_sum, label='Right Side Total', linestyle='--', linewidth=2)
        axes[0].set_ylabel('Value')
        axes[0].set_title('Identity Components: Left vs Right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Violation over time
        axes[1].plot(violation, label='Violation', color='red')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1].fill_between(violation.index,
                           -2*violation.std(), 
                           2*violation.std(),
                           alpha=0.2, color='gray',
                           label='±2 std bands')
        axes[1].set_ylabel('Violation')
        axes[1].set_title('Constraint Violation Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Left side components
        for i, (name, comp) in enumerate(list(left_components.items())[:5]):
            axes[2].plot(comp, label=name, alpha=0.7)
        axes[2].set_ylabel('Value')
        axes[2].set_title('Left Side Components')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Right side components
        for i, (name, comp) in enumerate(list(right_components.items())[:5]):
            axes[3].plot(comp, label=name, alpha=0.7)
        axes[3].set_ylabel('Value')
        axes[3].set_xlabel('Time')
        axes[3].set_title('Right Side Components')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    def generate_comprehensive_report(self, results: Dict) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("Comprehensive Flow of Funds Decomposition Report")
        report.append("=" * 60)
        report.append(f"\nAnalysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary statistics
        report.append("\n## Summary Statistics")
        report.append(f"Total Series Processed: {len(results['components']['level'].columns)}")
        report.append(f"Total Constraints Applied: {len(self.identities)}")
        report.append(f"Series with Formulas: {len(self.series_formulas)}")
        report.append(f"Sectors Covered: {len(self.sectors)}")
        
        # Constraint satisfaction
        report.append("\n## Constraint Satisfaction")
        if not results['final_validation'].empty:
            by_type = results['final_validation'].groupby('constraint_type').agg({
                'mean_discrepancy': ['mean', 'std'],
                'max_abs_discrepancy': 'max',
                'constraint_name': 'count'
            })
            report.append("\nBy Constraint Type:")
            report.append(by_type.to_string())
        
        # Series metadata summary
        report.append("\n## Series Characteristics")
        if 'metadata' in results:
            meta_summary = results['metadata'].groupby(['sector_code', 'is_derived']).size().unstack(fill_value=0)
            report.append("\nSeries by Sector and Type:")
            report.append(meta_summary.to_string())
        
        # Model diagnostics
        report.append("\n## Model Diagnostics")
        if 'diagnostics' in results and not results['diagnostics'].empty:
            diag_summary = results['diagnostics'][['r_hat', 'ess_bulk', 'ess_tail']].describe()
            report.append("\nConvergence Statistics:")
            report.append(diag_summary.to_string())
            
            # Flag problematic series
            problematic = results['diagnostics'][results['diagnostics']['r_hat'] > 1.01]
            if not problematic.empty:
                report.append(f"\nSeries with convergence issues: {len(problematic)}")
                report.append(problematic[['series', 'r_hat']].to_string())
        
        # Network statistics
        report.append("\n## Network Analysis")
        report.append(f"Total Nodes: {self.identity_graph.number_of_nodes()}")
        report.append(f"Total Edges: {self.identity_graph.number_of_edges()}")
        report.append(f"Average Degree: {sum(dict(self.identity_graph.degree()).values()) / self.identity_graph.number_of_nodes():.2f}")
        
        # Most connected series
        degrees = dict(self.identity_graph.degree())
        top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        report.append("\nMost Connected Series:")
        for series, degree in top_connected:
            name = self.series_metadata.get(series, SeriesMetadata(name=series, series_type='unknown', 
                                                                  crosses_zero=False, volatility_regime='unknown',
                                                                  transformation='unknown', seasonal_pattern='unknown',
                                                                  outlier_dates=[], structural_breaks=[])).name
            report.append(f"  {series}: {degree} connections - {name[:50]}...")
        
        return "\n".join(report)
    
    # Keep all the statistical test methods from the original
    def _test_for_trend(self, series: pd.Series) -> bool:
        """Test if series has significant trend"""
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna(), regression='ct')
        return result[1] > 0.05
    
    def _test_for_seasonality(self, series: pd.Series) -> bool:
        """Test for seasonal patterns"""
        if len(series) < 3 * 4:
            return False
            
        from statsmodels.formula.api import ols
        
        df = pd.DataFrame({
            'y': series.values,
            'quarter': np.tile([1, 2, 3, 4], len(series) // 4 + 1)[:len(series)],
            'trend': np.arange(len(series))
        })
        
        restricted = ols('y ~ trend', data=df).fit()
        unrestricted = ols('y ~ trend + C(quarter)', data=df).fit()
        
        f_stat = ((restricted.ssr - unrestricted.ssr) / 3) / (unrestricted.ssr / (len(series) - 5))
        p_value = 1 - stats.f.cdf(f_stat, 3, len(series) - 5)
        
        return p_value < 0.05
    
    def _test_for_volatility_changes(self, series: pd.Series) -> bool:
        """Test for time-varying volatility"""
        returns = series.pct_change().dropna()
        squared_returns = returns ** 2
        
        from statsmodels.tsa.stattools import acf
        acf_sq = acf(squared_returns, nlags=4)
        
        return np.any(np.abs(acf_sq[1:]) > 2 / np.sqrt(len(returns)))
    
    def _detect_trend_breaks(self, series: pd.Series) -> List[int]:
        """Detect potential structural breaks in trend"""
        window = min(20, len(series) // 5)
        rolling_var = pd.Series(series).rolling(window).var()
        
        var_changes = rolling_var.diff().abs()
        threshold = var_changes.quantile(0.95)
        
        potential_breaks = np.where(var_changes > threshold)[0]
        
        break_points = []
        if len(potential_breaks) > 0:
            break_points = [potential_breaks[0]]
            for bp in potential_breaks[1:]:
                if bp - break_points[-1] > window:
                    break_points.append(bp)
                    
        return break_points[:3]
    
    def _detect_outliers(self, series: pd.Series) -> List[pd.Timestamp]:
        """Detect outliers in series"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        outlier_indices = np.where(z_scores > 3)[0]
        return [series.index[idx] for idx in outlier_indices]
    
    def _decompose_independent_batch(self, df: pd.DataFrame, metadata_dict: Dict) -> Dict:
        """Decompose independent series using statsmodels"""
        results = {}
        
        for series_name in df.columns:
            metadata = metadata_dict[series_name]
            
            # Configure model based on metadata
            model_config = {
                'level': True,
                'trend': metadata.transformation == 'trend',
                'seasonal': metadata.seasonal_pattern != 'none',
                'freq_seasonal': [{'period': 4, 'harmonics': 2}] if metadata.seasonal_pattern != 'none' else None,
                'cycle': True,
                'irregular': True,
                'stochastic_level': False,
                'stochastic_trend': True,
                'stochastic_seasonal': True,
                'stochastic_cycle': True,
                'damped_cycle': True,
            }
            
            try:
                model = UnobservedComponents(df[series_name], **model_config)
                fit = model.fit(method='powell', disp=False)
                
                results[series_name] = {
                    'level': pd.Series(fit.level.smoothed, index=df.index),
                    'trend': pd.Series(fit.trend.smoothed if model_config['trend'] else 0, index=df.index),
                    'cycle': pd.Series(fit.cycle.smoothed, index=df.index),
                    'seasonal': pd.Series(fit.seasonal.smoothed if model_config['seasonal'] else 0, index=df.index),
                    'identity_constrained': False,
                    'model_type': 'statsmodels_uc'
                }
            except Exception as e:
                self.logger.warning(f"Failed to decompose {series_name}: {e}")
                # Fallback to simple decomposition
                results[series_name] = self._simple_decomposition(df[series_name])
                
        return results
    
    def _simple_decomposition(self, series: pd.Series) -> Dict:
        """Simple fallback decomposition"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomp = seasonal_decompose(series, model='additive', period=4)
        
        return {
            'level': pd.Series(decomp.trend.bfill().ffill(), index=series.index),
            'trend': pd.Series(0, index=series.index),
            'cycle': pd.Series(decomp.resid.fillna(0), index=series.index),
            'seasonal': pd.Series(decomp.seasonal.fillna(0), index=series.index),
            'identity_constrained': False,
            'model_type': 'seasonal_decompose'
        }
        
    def _create_contextual_components_with_suffix(self, 
                                                name: str, 
                                                series: pd.Series,
                                                metadata: SeriesMetadata,
                                                T: int,
                                                df: pd.DataFrame,
                                                suffix: str) -> Dict:
        """Create components with unique suffix to avoid naming conflicts"""
        components = {}
        init_mean = series.iloc[:5].mean()
        init_std = series.std()
        
        # Use suffix in all variable names
        var_name = f"{name}_{suffix}"
        
        # LEVEL
        level_init = pm.Normal(
            f'{var_name}_level_init',
            mu=init_mean,
            sigma=init_std
        )
        
        sigma_level = pm.HalfNormal(
            f'{var_name}_sigma_level',
            sigma=init_std * 0.1
        )
        
        level = pm.GaussianRandomWalk(
            f'{var_name}_level',
            mu=0,
            sigma=sigma_level,
            init_dist=pm.Normal.dist(level_init, sigma_level),
            shape=T
        )
        
        components['level'] = level
        
        # Continue with other components using var_name instead of name...
        # (implement the rest similarly)
        
        return components        

    def _create_contextual_components_unique(self,
                                             name: str,
                                             series: pd.Series,
                                             metadata: SeriesMetadata,
                                             T: int,
                                             df: pd.DataFrame,
                                             unique_id: str) -> Dict:
        """Create UCM-style components with smooth trend model exactly like statsmodels"""
        components = {}
        vname = f"{name}_{unique_id}"
        series_values = series.values
        
        # Get variance estimates using HP filter (statsmodels approach)
        from statsmodels.tsa.filters.hp_filter import hpfilter
        
        # First HP filter
        cycle1, trend1 = hpfilter(series_values, lamb=1600)
        
        # Second HP filter for smooth trend variance
        cycle2, trend2 = hpfilter(trend1, lamb=1600)
        
        # Variance estimates
        trend_var_estimate = np.var(trend2)
        irregular_var_estimate = np.var(series_values - trend1)
        var_resid = irregular_var_estimate  # For other components
        
        # Initial values
        init_level = float(series_values[0])
        init_trend = float(np.mean(np.diff(series_values[:10])))
        
        # 1. SMOOTH TREND MODEL (level + stochastic trend)
        # Initial level
        level_init = pm.Normal(
            f'{vname}_level_init',
            mu=init_level,
            sigma=np.sqrt(var_resid)
        )
        
        # Initial trend
        trend_init = pm.Normal(
            f'{vname}_trend_init',
            mu=init_trend,
            sigma=np.sqrt(trend_var_estimate)
        )
        
        # Trend variance parameter (to be estimated)
        sigma_trend = pm.HalfNormal(
            f'{vname}_sigma_trend',
            sigma=np.sqrt(trend_var_estimate)  # Use sqrt of variance
        )
        
        # Trend innovations
        trend_innovations = pm.Normal(
            f'{vname}_trend_innovations',
            mu=0,
            sigma=sigma_trend,
            shape=T-1
        )
        
        # Trend evolution: β_t = β_{t-1} + ζ_t
        trend = pm.Deterministic(
            f'{vname}_trend',
            pt.concatenate([
                pt.atleast_1d(trend_init),
                trend_init + pt.cumsum(trend_innovations)
            ])
        )
        
        # Level evolution: μ_t = μ_{t-1} + β_{t-1}
        def level_step(beta_prev, mu_prev):
            return mu_prev + beta_prev
        
        level_values, _ = scan(
            fn=level_step,
            sequences=[trend[:-1]],  # Use β_{t-1} for updating μ_t
            outputs_info=[level_init]
        )
        
        level = pm.Deterministic(
            f'{vname}_level',
            pt.concatenate([pt.atleast_1d(level_init), level_values])
        )
        
        components['level'] = level
        components['trend'] = trend  # This is the slope β_t
        
        # 2. CYCLE (if applicable)
        if metadata.series_type in ['flow', 'level'] and T > 20:
            # Cycle period bounds 
            cycle_period_bounds = (1.5*4, 12*4)  # 1.5 to 12 years for quarterly
            cycle_freq_bounds = (
                2*np.pi / cycle_period_bounds[1], 
                2*np.pi / cycle_period_bounds[0]
            )
            
            # Cycle frequency parameter
            cycle_freq_unconstrained = pm.Normal(f'{vname}_cycle_freq_raw', mu=0, sigma=1)
            cycle_freq = pm.Deterministic(
                f'{vname}_cycle_freq',
                cycle_freq_bounds[0] + 
                (cycle_freq_bounds[1] - cycle_freq_bounds[0]) * 
                pm.math.sigmoid(cycle_freq_unconstrained)
            )
            
            # Damping parameter
            cycle_damp_unconstrained = pm.Normal(f'{vname}_cycle_damp_raw', mu=0, sigma=1)
            cycle_damp = pm.Deterministic(
                f'{vname}_cycle_damp',
                pm.math.sigmoid(cycle_damp_unconstrained)
            )
            
            # Cycle variance
            sigma_cycle = pm.HalfNormal(
                f'{vname}_sigma_cycle',
                sigma=np.sqrt(var_resid)
            )
            
            # Initial cycle states
            cycle_init = pm.Normal(
                f'{vname}_cycle_init',
                mu=0,
                sigma=np.sqrt(var_resid),
                shape=2
            )
            
            # Cycle innovations
            cycle_innovations = pm.Normal(
                f'{vname}_cycle_innovations',
                mu=0,
                sigma=sigma_cycle,
                shape=(T-1, 2)
            )
            
            # Cycle evolution
            def cycle_step(innov, prev_state, freq, damp):
                c = prev_state[0]
                c_star = prev_state[1]
                
                cos_freq = pm.math.cos(freq)
                sin_freq = pm.math.sin(freq)
                
                new_c = damp * (cos_freq * c + sin_freq * c_star) + innov[0]
                new_c_star = damp * (-sin_freq * c + cos_freq * c_star) + innov[1]
                
                return pt.stack([new_c, new_c_star])
            
            cycle_states, _ = scan(
                fn=cycle_step,
                sequences=[cycle_innovations],
                outputs_info=[cycle_init],
                non_sequences=[cycle_freq, cycle_damp]
            )
            
            cycle = pm.Deterministic(
                f'{vname}_cycle',
                pt.concatenate([
                    cycle_init[0:1],
                    cycle_states[:, 0]
                ])
            )
        else:
            cycle = pt.zeros(T)
        
        components['cycle'] = cycle
        
        # 3. SEASONAL (frequency domain with period=4, harmonics=2)
        if metadata.seasonal_pattern != 'none':
            period = 4
            harmonics = 2
            lambda_p = 2 * np.pi / float(period)
            
            # Initialize seasonal states
            seasonal_init = pm.Normal(
                f'{vname}_seasonal_init',
                mu=0,
                sigma=np.sqrt(var_resid),
                shape=2 * harmonics
            )
            
            # Seasonal variance
            sigma_seasonal = pm.HalfNormal(
                f'{vname}_sigma_seasonal',
                sigma=np.sqrt(var_resid)
            )
            
            # Seasonal innovations
            seasonal_innovations = pm.Normal(
                f'{vname}_seasonal_innovations',
                mu=0,
                sigma=sigma_seasonal,
                shape=(T-1, 2 * harmonics)
            )
            
            # Build seasonal components
            seasonal_states = []
            state_idx = 0
            
            for j in range(1, harmonics + 1):
                cos_lambda_j = np.cos(lambda_p * j)
                sin_lambda_j = np.sin(lambda_p * j)
                
                def seasonal_step(innov, prev_state):
                    gamma = prev_state[0]
                    gamma_star = prev_state[1]
                    
                    new_gamma = cos_lambda_j * gamma + sin_lambda_j * gamma_star + innov[0]
                    new_gamma_star = -sin_lambda_j * gamma + cos_lambda_j * gamma_star + innov[1]
                    
                    return pt.stack([new_gamma, new_gamma_star])
                
                harmonic_innovations = seasonal_innovations[:, state_idx:state_idx+2]
                harmonic_init = seasonal_init[state_idx:state_idx+2]
                
                harmonic_states, _ = scan(
                    fn=seasonal_step,
                    sequences=[harmonic_innovations],
                    outputs_info=[harmonic_init]
                )
                
                gamma_j = pt.concatenate([
                    harmonic_init[0:1],
                    harmonic_states[:, 0]
                ])
                
                seasonal_states.append(gamma_j)
                state_idx += 2
            
            seasonal = pm.Deterministic(
                f'{vname}_seasonal',
                sum(seasonal_states)
            )
        else:
            seasonal = pt.zeros(T)
        
        components['seasonal'] = seasonal
        
        # 4. IRREGULAR (observation error)
        sigma_irregular = pm.HalfNormal(
            f'{vname}_sigma_irregular',
            sigma=np.sqrt(irregular_var_estimate)
        )
        
        components['sigma_obs'] = sigma_irregular
        components['stochastic_vol'] = False
        
        return components
        
    def calculate_growth_rates(self, results: Dict, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate growth rates from UCM decomposition"""
        growth_rates = {}
        
        for series_name in results.keys():
            if series_name in data.columns:
                # Get components
                level = results[series_name]['level']
                trend = results[series_name]['trend']
                cycle = results[series_name]['cycle']
                seasonal = results[series_name]['seasonal']
                
                # Fitted values (level already includes trend in UCM)
                fitted_values = level + cycle + seasonal
                
                # Growth rate = trend (slope) / fitted values
                growth_rate = trend / fitted_values.replace(0, np.nan)
                
                # Convert to percentage
                growth_rates[series_name] = growth_rate * 100
        
        return pd.DataFrame(growth_rates)
        
    def analyze_growth_patterns(self, results: Dict, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze growth patterns from smooth trend decomposition"""
        growth_analysis = []
        
        for series_name, result in results.items():
            if series_name in data.columns:
                # Get components
                level = result['level']
                slope = result['trend']  # This is β_t
                
                # Calculate growth metrics
                growth_rate = slope / level * 100
                avg_growth = growth_rate.mean()
                growth_volatility = growth_rate.std()
                
                # Check if growth is accelerating
                slope_change = slope.diff()
                accelerating = slope_change.mean() > 0
                
                # Estimate if series is approximately exponential
                # For exponential: slope/level should be roughly constant
                growth_cv = growth_volatility / abs(avg_growth) if avg_growth != 0 else np.inf
                is_exponential = growth_cv < 0.5  # Coefficient of variation < 50%
                
                growth_analysis.append({
                    'series': series_name,
                    'avg_growth_rate': avg_growth,
                    'growth_volatility': growth_volatility,
                    'is_accelerating': accelerating,
                    'likely_exponential': is_exponential,
                    'growth_cv': growth_cv,
                    'final_level': level.iloc[-1],
                    'final_slope': slope.iloc[-1]
                })
        
        return pd.DataFrame(growth_analysis)        
 
    # Add these diagnostic functions to your class

    def diagnose_model_issues(self, df: pd.DataFrame, series_list: List[str]):
        """Diagnose why ADVI loss might be constant"""
        
        print("=== MODEL DIAGNOSTICS ===\n")
        
        # 1. Check data characteristics
        print("1. DATA CHARACTERISTICS:")
        for series in series_list[:3]:  # Check first 3 series
            if series in df.columns:
                s = df[series]
                print(f"\n{series}:")
                print(f"  Mean: {s.mean():.2e}")
                print(f"  Std: {s.std():.2e}")
                print(f"  Min: {s.min():.2e}")
                print(f"  Max: {s.max():.2e}")
                print(f"  Scale (max-min): {s.max() - s.min():.2e}")
                print(f"  Has NaNs: {s.isna().any()}")
                print(f"  Zeros: {(s == 0).sum()}")
        
        # 2. Check constraints
        print("\n2. CONSTRAINT ANALYSIS:")
        constraint_overlap = {}
        for identity_name, identity in self.identities.items():
            series_in_constraint = set(identity.left_side + identity.right_side)
            series_in_constraint = {s[1:] if s.startswith('Δ') else s for s in series_in_constraint}
            overlap = series_in_constraint.intersection(series_list)
            if overlap:
                constraint_overlap[identity_name] = len(overlap)
        
        print(f"Total constraints affecting these series: {len(constraint_overlap)}")
        print(f"Average series per constraint: {np.mean(list(constraint_overlap.values())):.1f}")
        print(f"Max series in a constraint: {max(constraint_overlap.values()) if constraint_overlap else 0}")
        
        # 3. Test simple model without constraints
        print("\n3. TESTING SIMPLE MODEL (no constraints):")
        test_series = series_list[0] if series_list else None
        if test_series and test_series in df.columns:
            self._test_simple_model(df[test_series], test_series)
        
        # 4. Check for multicollinearity
        print("\n4. CORRELATION ANALYSIS:")
        if len(series_list) > 1:
            corr_matrix = df[series_list[:5]].corr()  # First 5 series
            high_corr = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        high_corr.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr:
                print(f"Found {len(high_corr)} highly correlated pairs (>0.95):")
                for s1, s2, corr in high_corr[:3]:
                    print(f"  {s1} - {s2}: {corr:.3f}")
            else:
                print("No extremely high correlations found")
        
        return constraint_overlap

    def _test_simple_model(self, series: pd.Series, name: str):
        """Test a simple model to see if ADVI works at all"""
        import pymc as pm
        import pytensor.tensor as pt
        
        print(f"\nTesting simple random walk model for {name}...")
        
        # Standardize data
        y_std = (series - series.mean()) / series.std()
        
        with pm.Model() as simple_model:
            # Very simple model
            sigma = pm.HalfNormal('sigma', sigma=1.0)
            mu = pm.GaussianRandomWalk('mu', sigma=sigma, shape=len(series))
            
            # Likelihood
            obs = pm.Normal('obs', mu=mu, sigma=0.1, observed=y_std.values)
            
            # Try ADVI
            try:
                print("  Fitting with ADVI...")
                approx = pm.fit(
                    n=5000,
                    method='advi',
                    obj_optimizer=pm.adam(learning_rate=0.01),
                    progressbar=False
                )
                
                # Check ELBO history
                elbo_hist = approx.hist
                print(f"  Initial ELBO: {elbo_hist[0]:.2f}")
                print(f"  Final ELBO: {elbo_hist[-1]:.2f}")
                print(f"  ELBO change: {elbo_hist[-1] - elbo_hist[0]:.2f}")
                print(f"  Last 100 iterations std: {np.std(elbo_hist[-100:]):.4f}")
                
                # Plot ELBO if needed
                if len(elbo_hist) > 100:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 4))
                    plt.plot(elbo_hist)
                    plt.title(f'ELBO History - {name}')
                    plt.xlabel('Iteration')
                    plt.ylabel('ELBO')
                    plt.grid(True)
                    plt.show()
                    
            except Exception as e:
                print(f"  Simple model failed: {e}")

    def check_data_scales(self, df: pd.DataFrame, series_list: List[str]):
        """Check if data scales might be causing numerical issues"""
        
        scales = []
        for series in series_list:
            if series in df.columns:
                s = df[series]
                scales.append({
                    'series': series,
                    'mean': s.mean(),
                    'std': s.std(),
                    'max': s.max(),
                    'min': s.min(),
                    'range': s.max() - s.min(),
                    'cv': s.std() / abs(s.mean()) if s.mean() != 0 else np.inf
                })
        
        scales_df = pd.DataFrame(scales)
        
        print("\n=== DATA SCALE ANALYSIS ===")
        print("\nSeries with extreme scales:")
        
        # Check for very large values
        large_scale = scales_df[scales_df['max'].abs() > 1e6]
        if not large_scale.empty:
            print(f"\nVery large values (>1e6): {len(large_scale)} series")
            print(large_scale[['series', 'max', 'mean']].head())
        
        # Check for very small values
        small_scale = scales_df[scales_df['std'] < 1e-6]
        if not small_scale.empty:
            print(f"\nVery small std (<1e-6): {len(small_scale)} series")
            print(small_scale[['series', 'std', 'mean']].head())
        
        # Check for high coefficient of variation
        high_cv = scales_df[scales_df['cv'] > 10]
        if not high_cv.empty:
            print(f"\nHigh coefficient of variation (>10): {len(high_cv)} series")
            print(high_cv[['series', 'cv', 'mean', 'std']].head())
        
        return scales_df


# Usage example
if __name__ == "__main__":
    
    # Initialize comprehensive system
    system = ComprehensiveAlignedUCSystem(
        identities_file='fof_formulas_extracted.json',
        n_cores=32,
        use_gpu=True
    )
    
    # Load Z1 data
    z1_data = pd.read_parquet('/home/tesla/Z1/temp/data/z1_quarterly/z1_quarterly_data_filtered.parquet')
    
    # Select a subset of series for demonstration
    # In practice, you might process all series or specific groups
    demo_series = [
        'FL104090005',  # Households net worth
        'FL104000005',  # Households total assets
        'FL104190005',  # Households total liabilities
        'FU104090005',  # Change in households net worth
        'FL103064105',  # Households corporate equities
        'FL103065005',  # Households mutual fund shares
    ]
    
    available_series = [s for s in demo_series if s in z1_data.columns]
    
    # Perform comprehensive decomposition
    results = system.decompose_with_constraints(
        z1_data[available_series],
        series_list=available_series
    )
    
    # Visualize comprehensive network
    system.visualize_comprehensive_network(
        output_file='comprehensive_fof_network.png',
        highlight_series=available_series
    )
    
    # Generate report
    report = system.generate_comprehensive_report(results)
    print(report)
    
    # Save report
    with open('comprehensive_decomposition_report.txt', 'w') as f:
        f.write(report)
    
    # Analyze specific constraints
    print("\n\nConstraint Analysis:")
    print(results['final_validation'].head(10))
    
    # Plot constraint satisfaction for a specific identity
    if len(system.identities) > 0:
        # Find a balance sheet identity to visualize
        balance_sheet_identities = [
            name for name, identity in system.identities.items()
            if identity.identity_type == 'balance_sheet'
        ]
        
        if balance_sheet_identities:
            system.plot_constraint_satisfaction(
                results,
                balance_sheet_identities[0],
                output_file='balance_sheet_constraint.png'
            )
    
    # Export results
    for component in ['level', 'trend', 'cycle', 'seasonal']:
        results['components'][component].to_csv(f'comprehensive_{component}_components.csv')
    
    # Clean up
    ray.shutdown()
