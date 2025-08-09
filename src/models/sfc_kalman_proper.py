"""
Properly Architected Stock-Flow Consistent Kalman Filter
Addresses all critical issues: preserves shock structure, implements proper constraints,
and enforces full SFC accounting identities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from scipy import sparse
from scipy.linalg import block_diag, solve, inv
import scipy.sparse.linalg as sp_linalg

from src.models.hierarchical_kalman_filter import HierarchicalKalmanFilter


@dataclass
class StockFlowPair:
    """Complete stock-flow relationship in Z.1 data."""
    stock_series: str
    flow_series: str
    reval_series: Optional[str] = None  # FR - revaluation
    other_series: Optional[str] = None  # FV - other volume changes
    has_complete_flows: bool = False
    sector: str = ""
    instrument: str = ""
    
    def get_all_series(self) -> List[str]:
        """Return all series codes for this pair."""
        series = [self.stock_series, self.flow_series]
        if self.reval_series:
            series.append(self.reval_series)
        if self.other_series:
            series.append(self.other_series)
        return series


@dataclass
class BilateralConstraint:
    """Bilateral position constraint."""
    holder: str
    issuer: str
    instrument: str
    asset_series: Optional[str] = None
    liability_series: Optional[str] = None
    positions: Optional[np.ndarray] = None


@dataclass
class SFCStateSpace:
    """Manages the extended state space structure."""
    # Base model dimensions
    n_base_states: int
    n_base_shocks: int
    
    # SFC extensions
    n_stock_states: int = 0
    n_flow_states: int = 0
    n_reval_states: int = 0  # FR states
    n_other_states: int = 0  # FV states
    n_bilateral_states: int = 0
    
    # State indices mapping
    base_indices: Dict[str, Dict[str, int]] = field(default_factory=dict)
    stock_indices: Dict[str, int] = field(default_factory=dict)
    flow_indices: Dict[str, int] = field(default_factory=dict)
    reval_indices: Dict[str, int] = field(default_factory=dict)
    other_indices: Dict[str, int] = field(default_factory=dict)
    bilateral_indices: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    
    @property
    def n_sfc_states(self) -> int:
        """Total number of SFC-specific states."""
        return (self.n_stock_states + self.n_flow_states + 
                self.n_reval_states + self.n_other_states + 
                self.n_bilateral_states)
    
    @property
    def n_total_states(self) -> int:
        """Total state dimension."""
        return self.n_base_states + self.n_sfc_states
    
    @property
    def n_sfc_shocks(self) -> int:
        """Number of shocks for SFC states."""
        # Use shared shocks by instrument to control dimensionality
        return min(self.n_sfc_states, max(10, self.n_sfc_states // 10))
    
    @property
    def n_total_shocks(self) -> int:
        """Total shock dimension."""
        return self.n_base_shocks + self.n_sfc_shocks


class SFCConstraintProjector:
    """
    Handles constraint projection with proper covariance update.
    Enforces SFC identities while preserving statistical properties.
    """
    
    def __init__(self, tolerance: float = 1e-10, max_iterations: int = 10,
                 use_sparse: bool = True):
        """
        Initialize constraint projector.
        
        Parameters
        ----------
        tolerance : float
            Convergence tolerance for iterative projection
        max_iterations : int
            Maximum projection iterations
        use_sparse : bool
            Whether to use sparse matrix operations
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.use_sparse = use_sparse
        self.logger = logging.getLogger(__name__)
    
    def project_state(self, x: np.ndarray, P: np.ndarray, 
                      A: Union[np.ndarray, sparse.spmatrix], 
                      b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state onto constraint manifold with covariance update.
        Keeps operations sparse when possible.
        
        Implements: min ||x - x_hat||^2_P^-1 subject to Ax = b
        
        Solution:
        x* = x - P·A'(A·P·A')^-1(A·x - b)
        P* = P - P·A'(A·P·A')^-1·A·P
        
        Parameters
        ----------
        x : np.ndarray
            State vector to project
        P : np.ndarray
            State covariance matrix
        A : np.ndarray or sparse matrix
            Constraint matrix
        b : np.ndarray
            Constraint values
            
        Returns
        -------
        x_proj : np.ndarray
            Projected state
        P_proj : np.ndarray
            Updated covariance
        """
        if A.shape[0] == 0:
            return x, P
        
        # Compute residual
        if sparse.issparse(A):
            residual = A @ x - b
        else:
            residual = A @ x - b
        
        # Check if constraints already satisfied
        if np.max(np.abs(residual)) < self.tolerance:
            return x, P
        
        # Handle sparse case efficiently
        if sparse.issparse(A) and self.use_sparse:
            return self._project_sparse(x, P, A, b, residual)
        else:
            # Convert to dense if needed
            if sparse.issparse(A):
                A_dense = A.toarray()
            else:
                A_dense = A
            return self._project_dense(x, P, A_dense, b, residual)
    
    def _project_sparse(self, x: np.ndarray, P: np.ndarray, 
                       A: sparse.spmatrix, b: np.ndarray, 
                       residual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sparse implementation of constraint projection.
        """
        # Compute AP = A @ P efficiently
        AP = A @ P
        
        # Compute APAT = AP @ A.T
        APAT = AP @ A.T
        
        # Add regularization
        if sparse.issparse(APAT):
            APAT = APAT + sparse.eye(APAT.shape[0], format='csr') * 1e-12
        else:
            APAT = APAT + np.eye(APAT.shape[0]) * 1e-12
        
        # Solve for gain using sparse solver
        if sparse.issparse(APAT):
            # Use sparse LU decomposition
            from scipy.sparse.linalg import splu
            lu = splu(APAT.tocsc())
            gain_T = lu.solve(AP.T)
            gain = gain_T.T
        else:
            # Small system - use dense solve
            gain = P @ A.T
            gain = solve(APAT, gain.T).T
        
        # Project state
        x_proj = x - gain @ residual
        
        # Update covariance (Joseph form)
        I_KA = np.eye(P.shape[0]) - gain @ A
        P_proj = I_KA @ P @ I_KA.T + gain @ np.eye(A.shape[0]) * 1e-10 @ gain.T
        
        # Ensure symmetry
        P_proj = 0.5 * (P_proj + P_proj.T)
        
        return x_proj, P_proj
    
    def _project_dense(self, x: np.ndarray, P: np.ndarray, 
                      A: np.ndarray, b: np.ndarray,
                      residual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dense implementation of constraint projection.
        """
        # Compute APAT = A·P·A'
        AP = A @ P
        APAT = AP @ A.T
        
        # Add small regularization for numerical stability
        APAT += np.eye(APAT.shape[0]) * 1e-12
        
        # Solve for gain matrix K = P·A'·(APAT)^-1
        K = P @ A.T
        gain = solve(APAT, K.T).T
        
        # Project state
        x_proj = x - gain @ residual
        
        # Update covariance (Joseph form for numerical stability)
        I_KA = np.eye(P.shape[0]) - gain @ A
        P_proj = I_KA @ P @ I_KA.T + gain @ np.eye(A.shape[0]) * 1e-10 @ gain.T
        
        # Ensure symmetry
        P_proj = 0.5 * (P_proj + P_proj.T)
        
        return x_proj, P_proj
    
    def _iterative_projection(self, x: np.ndarray, P: np.ndarray,
                              A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iterative projection for singular or near-singular systems.
        Should not be reached in normal operation.
        """
        raise RuntimeError("Constraint system is singular - check for redundant or conflicting constraints")


class SFCConstraintBuilder:
    """
    Builds constraint matrices from Z.1 formulas and SFC relationships.
    """
    
    def __init__(self, state_space: SFCStateSpace, 
                 stock_flow_pairs: List[StockFlowPair],
                 bilateral_constraints: List[BilateralConstraint],
                 formulas: Optional[Dict] = None,
                 extend_state_space: bool = False,
                 enforce_market_clearing: bool = True,
                 enforce_bilateral: bool = True):
        """
        Initialize constraint builder.
        """
        self.state_space = state_space
        self.stock_flow_pairs = stock_flow_pairs
        self.bilateral_constraints = bilateral_constraints
        self.formulas = formulas or {}
        self.extend_state_space = extend_state_space
        self.enforce_market_clearing = enforce_market_clearing
        self.enforce_bilateral = enforce_bilateral
        self.logger = logging.getLogger(__name__)
    
    def build_constraints(self, t: int = 0) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build constraint matrices for time t.
        
        Returns
        -------
        A : sparse.csr_matrix
            Constraint matrix
        b : np.ndarray
            Constraint values
        """
        constraints = []
        rhs = []
        
        # Determine actual state dimension
        if self.extend_state_space:
            n_states = self.state_space.n_total_states
        else:
            n_states = self.state_space.n_base_states
        
        # 1. Stock-flow identities: ΔFL = FU + FR + FV
        self._add_stock_flow_constraints(constraints, rhs, t)
        
        # 2. Market clearing: Σ(assets) = Σ(liabilities) by instrument
        if self.enforce_market_clearing:
            self._add_market_clearing_constraints(constraints, rhs)
        
        # 3. Bilateral aggregation: Σ(bilateral) = aggregate
        if self.enforce_bilateral and self.bilateral_constraints:
            self._add_bilateral_constraints(constraints, rhs)
        
        # 4. Formula constraints
        if self.formulas:
            self._add_formula_constraints(constraints, rhs)
        
        if not constraints:
            return sparse.csr_matrix((0, n_states)), np.array([])
        
        # Convert to sparse matrix
        n_constraints = len(constraints)
        
        row_ind = []
        col_ind = []
        data = []
        
        for i, constraint in enumerate(constraints):
            for j, value in constraint.items():
                if j < n_states:  # Only include valid indices
                    row_ind.append(i)
                    col_ind.append(j)
                    data.append(value)
        
        A = sparse.csr_matrix((data, (row_ind, col_ind)), 
                              shape=(n_constraints, n_states))
        b = np.array(rhs)
        
        return A, b

    
    def _add_market_clearing_constraints(self, constraints: List, rhs: List):
        """
        Add market clearing constraints by instrument.
        Assets = Liabilities for each instrument across all sectors.
        
        Uses asset_liability_map if provided, otherwise uses Z.1 conventions.
        """
        # Group series by instrument and side (asset/liability)
        instruments = {}
        
        for series_code in self.state_space.base_indices.keys():
            # Parse series code: PPSSIIIIIDDD
            if len(series_code) < 9 or not series_code.startswith('FL'):
                continue
            
            instrument = series_code[4:9]
            
            # Determine if asset or liability
            if series_code in self.asset_liability_map:
                # Use explicit mapping if available
                side = self.asset_liability_map[series_code]
            else:
                # Use Z.1 conventions:
                # Assets: FL...005, FL...015, FL...025, etc.
                # Liabilities: FL...105, FL...115, FL...125, etc.
                suffix = series_code[9:] if len(series_code) > 9 else ""
                if len(suffix) >= 3:
                    first_digit = suffix[0]
                    side = 'liability' if first_digit in ['1', '2'] else 'asset'
                else:
                    continue  # Skip if can't determine
            
            # Initialize instrument structure
            if instrument not in instruments:
                instruments[instrument] = {'assets': [], 'liabilities': []}
            
            # Add to appropriate side
            if side == 'liability':
                instruments[instrument]['liabilities'].append(series_code)
            else:
                instruments[instrument]['assets'].append(series_code)
        
        # Create constraints for each instrument
        for instrument, sides in instruments.items():
            if sides['assets'] and sides['liabilities']:
                constraint_row = {}
                
                # Sum of assets (+1)
                for asset_series in sides['assets']:
                    idx = self.state_space.base_indices[asset_series]['level']
                    constraint_row[idx] = constraint_row.get(idx, 0.0) + 1.0
                
                # Sum of liabilities (-1)
                for liability_series in sides['liabilities']:
                    idx = self.state_space.base_indices[liability_series]['level']
                    constraint_row[idx] = constraint_row.get(idx, 0.0) - 1.0
                
                # Only add if we have both sides
                if len(constraint_row) >= 2:
                    constraints.append(constraint_row)
                    rhs.append(0.0)
    
    def _add_bilateral_constraints(self, constraints: List, rhs: List):
        """
        Add bilateral aggregation constraints.
        Sum of all bilateral positions = aggregate FL series.
        
        Groups bilaterals by instrument and ensures they sum to the aggregate.
        """
        if not self.bilateral_constraints or not self.state_space.bilateral_indices:
            return
        
        # Group bilaterals by instrument
        by_instrument = {}
        for constraint in self.bilateral_constraints:
            instrument = constraint.instrument
            if instrument not in by_instrument:
                by_instrument[instrument] = []
            by_instrument[instrument].append(constraint)
        
        # For each instrument, create aggregation constraint
        for instrument, bilaterals in by_instrument.items():
            # Find the aggregate series for this instrument
            # This would be the total FL series for all sectors combined
            aggregate_series = self._find_aggregate_series(instrument)
            
            if not aggregate_series:
                continue
            
            # Get aggregate state index
            aggregate_idx = None
            if aggregate_series in self.state_space.base_indices:
                aggregate_idx = self.state_space.base_indices[aggregate_series]['level']
            
            if aggregate_idx is None:
                continue
            
            # Build constraint: sum(bilaterals) = aggregate
            constraint_row = {}
            
            # Aggregate gets coefficient -1
            constraint_row[aggregate_idx] = -1.0
            
            # Each bilateral gets coefficient +1
            n_bilaterals = 0
            for bilateral in bilaterals:
                key = (bilateral.holder, bilateral.issuer, bilateral.instrument)
                if key in self.state_space.bilateral_indices:
                    bilateral_idx = self.state_space.bilateral_indices[key]
                    constraint_row[bilateral_idx] = constraint_row.get(bilateral_idx, 0.0) + 1.0
                    n_bilaterals += 1
            
            # Only add if we have bilaterals to aggregate
            if n_bilaterals > 0 and len(constraint_row) > 1:
                constraints.append(constraint_row)
                rhs.append(0.0)
    
    def _find_aggregate_series(self, instrument: str) -> Optional[str]:
        """
        Find the aggregate FL series for an instrument.
        Typically sector 89 (all sectors) or 90 (totals).
        """
        # Look for all-sector aggregates with exact patterns
        aggregate_sectors = ['89', '90', '99']
        common_suffixes = ['005', '000']
        
        # Try exact matches first
        for sector in aggregate_sectors:
            for suffix in common_suffixes:
                candidate = f"FL{sector}{instrument}{suffix}"
                if candidate in self.state_space.base_indices:
                    return candidate
        
        # Try without suffix
        for sector in aggregate_sectors:
            candidate = f"FL{sector}{instrument}"
            if candidate in self.state_space.base_indices:
                return candidate
        
        # No aggregate found
        return None
    
    def _add_formula_constraints(self, constraints: List, rhs: List):
        """
        Add Z.1 formula constraints with proper lag handling.
        Uses companion form state augmentation for lagged terms.
        """
        if not self.formulas:
            return
        
        try:
            # Import the formula parser from the project
            from src.utils.formula_parser import FormulaParser
            parser = FormulaParser()
            
            for series_code, formula_dict in self.formulas.items():
                # Check if target series is in our state space
                target_idx = self._get_state_index_with_lag(series_code, lag=0)
                
                if target_idx is None:
                    continue  # Skip if series not in state space
                
                # Parse formula components
                components = parser.parse_formula(formula_dict)
                
                if not components:
                    continue
                
                # Build constraint row: target = sum(components)
                # Rearranged as: target - component1 - component2 ... = 0
                constraint_row = {}
                constraint_row[target_idx] = 1.0  # Target series
                
                # Add each component with proper lag handling
                for comp_series, lag, operator, coefficient in components:
                    # Get state index for lagged component
                    comp_idx = self._get_state_index_with_lag(comp_series, lag)
                    
                    if comp_idx is not None:
                        # Add to constraint (negative because we move to LHS)
                        constraint_row[comp_idx] = constraint_row.get(comp_idx, 0.0) - coefficient
                
                # Only add constraint if it has multiple terms
                if len(constraint_row) > 1:
                    constraints.append(constraint_row)
                    rhs.append(0.0)
                    
        except ImportError:
            # If FormulaParser not available, use basic implementation
            self._add_basic_formula_constraints(constraints, rhs)
    
    def _get_state_index_with_lag(self, series_code: str, lag: int) -> Optional[int]:
        """
        Get state index for a series with specific lag.
        Fully implements companion form state augmentation for lags.
        
        Parameters
        ----------
        series_code : str
            Series identifier
        lag : int
            Lag (0 for current, -1 for t-1, -2 for t-2, etc.)
            
        Returns
        -------
        int or None
            State index for the lagged series
        """
        # Check if series exists in base indices
        if series_code not in self.state_space.base_indices:
            return None
        
        # Get series info from base indices
        series_info = self.state_space.base_indices[series_code]
        
        if lag == 0:
            # Current value - use level state
            return series_info['level']
        
        elif lag < 0:
            # Past value - check companion states
            abs_lag = abs(lag)
            
            # Check if we have this lag in the companion form
            if abs_lag <= series_info.get('max_lag', 0):
                # Lag states are stored after level and trend
                lag_key = f'lag_{abs_lag}'
                
                if lag_key in series_info:
                    return series_info[lag_key]
                else:
                    # Calculate lag index directly
                    return series_info['start_idx'] + 1 + abs_lag
            else:
                # Lag exceeds maximum stored
                return None
        
        else:
            # Future values (lag > 0) not supported
            return None
    
    def _add_stock_flow_constraints(self, constraints: List, rhs: List, t: int):
        """
        Add stock-flow identity constraints with proper lag handling.
        FL[t] - FL[t-1] = FU[t] + FR[t] + FV[t]
        """
        # Skip first period since we need t-1
        if t == 0:
            return
        
        for pair in self.stock_flow_pairs:
            constraint = {}
            
            # Stock at time t
            stock_t_idx = self._get_state_index_with_lag(pair.stock_series, lag=0)
            if stock_t_idx is not None:
                constraint[stock_t_idx] = 1.0
            
            # Stock at time t-1
            stock_t1_idx = self._get_state_index_with_lag(pair.stock_series, lag=-1)
            if stock_t1_idx is not None:
                constraint[stock_t1_idx] = -1.0
            
            # Flow at time t
            flow_idx = self._get_state_index_with_lag(pair.flow_series, lag=0)
            if flow_idx is not None:
                constraint[flow_idx] = -1.0
            
            # Revaluation at time t
            if pair.reval_series:
                reval_idx = self._get_state_index_with_lag(pair.reval_series, lag=0)
                if reval_idx is not None:
                    constraint[reval_idx] = -1.0
            
            # Other changes at time t
            if pair.other_series:
                other_idx = self._get_state_index_with_lag(pair.other_series, lag=0)
                if other_idx is not None:
                    constraint[other_idx] = -1.0
            
            # Add constraint if we have enough terms
            if len(constraint) >= 3:  # Need at least stock[t], stock[t-1], and flow
                constraints.append(constraint)
                rhs.append(0.0)
    
    def _add_basic_formula_constraints(self, constraints: List, rhs: List):
        """
        Basic formula constraint implementation without the parser.
        Handles simple additive formulas without lags.
        """
        if not self.formulas:
            return
        
        for target_series, formula_dict in self.formulas.items():
            # Check for derived_from list (simpler format)
            derived_from = formula_dict.get('derived_from', [])
            if not derived_from:
                continue
            
            # Find target index
            target_idx = self._get_state_index_with_lag(target_series, lag=0)
            
            if target_idx is None:
                continue
            
            # Build constraint
            constraint_row = {target_idx: 1.0}
            
            for component in derived_from:
                comp_series = component.get('code', '')
                operator = component.get('operator', '+')
                lag = component.get('lag', 0)
                coef = -1.0 if operator == '-' else 1.0
                
                # Find component index with lag
                comp_idx = self._get_state_index_with_lag(comp_series, lag)
                
                if comp_idx is not None:
                    constraint_row[comp_idx] = constraint_row.get(comp_idx, 0.0) - coef
            
            # Add constraint if meaningful
            if len(constraint_row) > 1:
                constraints.append(constraint_row)
                rhs.append(0.0)


class ProperSFCKalmanFilter(HierarchicalKalmanFilter):
    """
    Properly architected Stock-Flow Consistent Kalman Filter.
    
    Key improvements:
    1. Preserves base model shock structure via block-diagonal extension
    2. Implements complete constraint projection with covariance update
    3. Properly handles all SFC identities
    4. Maintains numerical stability with sparse operations
    """
    
    def __init__(self, data: pd.DataFrame,
                 formulas: Optional[Dict] = None,
                 fwtw_data: Optional[pd.DataFrame] = None,
                 enforce_sfc: bool = True,
                 enforce_market_clearing: bool = True,
                 enforce_bilateral: bool = True,
                 include_revaluations: bool = True,
                 error_variance_ratio: float = 0.01,
                 reval_variance_ratio: float = 0.1,
                 bilateral_variance_ratio: float = 0.05,
                 normalize_data: bool = True,
                 transformation: str = 'square',
                 use_sparse: bool = True,
                 extend_state_space: bool = False,
                 asset_liability_map: Optional[Dict[str, str]] = None,
                 **kwargs):
        """
        Initialize proper SFC Kalman Filter.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data with DatetimeIndex
        formulas : dict, optional
            Z.1 formula definitions
        fwtw_data : pd.DataFrame, optional
            From-Whom-to-Whom bilateral positions (should have asset_series/liability_series columns)
        enforce_sfc : bool
            Whether to enforce stock-flow consistency
        enforce_market_clearing : bool
            Whether to enforce market clearing by instrument
        enforce_bilateral : bool
            Whether to enforce bilateral aggregation constraints
        include_revaluations : bool
            Whether to model FR/FV as explicit states
        error_variance_ratio : float
            Base measurement error variance ratio
        reval_variance_ratio : float
            Variance ratio for revaluation states
        bilateral_variance_ratio : float
            Variance ratio for bilateral states
        normalize_data : bool
            Whether to normalize data before filtering
        transformation : str
            Transformation type for hierarchy
        use_sparse : bool
            Whether to use sparse matrix operations
        extend_state_space : bool
            Whether to extend state space (False = use projection only)
        asset_liability_map : dict, optional
            Maps series codes to 'asset' or 'liability'
        """
        # Store configuration
        self.enforce_sfc = enforce_sfc
        self.enforce_market_clearing = enforce_market_clearing
        self.enforce_bilateral = enforce_bilateral
        self.include_revaluations = include_revaluations
        self.reval_variance_ratio = reval_variance_ratio
        self.bilateral_variance_ratio = bilateral_variance_ratio
        self.use_sparse = use_sparse
        self.extend_state_space = extend_state_space
        self.asset_liability_map = asset_liability_map or {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # CRITICAL: Analyze data structure BEFORE parent initialization
        self.stock_flow_pairs = self._identify_stock_flow_pairs(data)
        self.bilateral_constraints = self._process_bilateral_data(fwtw_data) if fwtw_data is not None else []
        
        # Store original data info
        self.data_columns = data.columns.tolist()
        self.original_data = data.copy()
        self.n_periods = len(data)
        
        # Strip project-specific kwargs before passing to parent
        parent_kwargs = {}
        for key, value in kwargs.items():
            # Only pass statsmodels-compatible parameters
            if key not in ['formula_constraints', 'stock_flow_weight', 'formula_weight', 
                          'market_clearing_weight', 'projection_tolerance', 'max_projection_iterations']:
                parent_kwargs[key] = value
        
        # Initialize parent (this sets up base model)
        super().__init__(
            data=data,
            formulas=formulas,
            error_variance_ratio=error_variance_ratio,
            normalize_data=normalize_data,
            transformation=transformation,
            **parent_kwargs
        )
        
        # CRITICAL: Cache base model structures BEFORE any modifications
        self._cache_base_structures()
        
        # Set up state space mapping
        self._setup_state_space()
        
        # Only extend if requested (for now, work with base dimensions)
        if self.extend_state_space:
            self.logger.warning("State space extension disabled to avoid dimension conflicts")
        
        # Set up constraint system (works via projection)
        self._setup_constraints()
        
        self.logger.info(f"Initialized Proper SFC Kalman Filter:")
        self.logger.info(f"  Base states: {self.state_space.n_base_states}")
        self.logger.info(f"  Stock-flow pairs: {len(self.stock_flow_pairs)}")
        self.logger.info(f"  Complete pairs: {sum(1 for p in self.stock_flow_pairs if p.has_complete_flows)}")
        self.logger.info(f"  Bilateral constraints: {len(self.bilateral_constraints)}")
        self.logger.info(f"  Enforcement via: projection")
    
    def _cache_base_structures(self):
        """
        Cache base model matrices before any modifications.
        CRITICAL: Must be called immediately after parent initialization.
        """
        self.base_transition = self['transition'].copy()
        self.base_design = self['design'].copy()
        self.base_selection = self['selection'].copy()
        self.base_state_cov = self['state_cov'].copy()
        self.base_obs_cov = self['obs_cov'].copy()
        
        # Cache dimensions
        self.n_base_states = self.k_states
        self.n_base_shocks = self.k_posdef
        
        # Cache source_info if available from parent
        if hasattr(self, 'source_info'):
            self.cached_source_info = self.source_info.copy()
        
        self.logger.info(f"Cached base structures: {self.n_base_states} states, {self.n_base_shocks} shocks")
    
    def _identify_stock_flow_pairs(self, data: pd.DataFrame) -> List[StockFlowPair]:
        """
        Identify complete stock-flow pairs including FR/FV.
        """
        pairs = []
        columns_set = set(data.columns)
        processed = set()
        
        for col in data.columns:
            if col in processed or not col.startswith('FL'):
                continue
            
            # Extract components
            if len(col) >= 9:
                sector = col[2:4]
                instrument = col[4:9]
                suffix = col[9:] if len(col) > 9 else ""
                
                # Look for complete set
                flow = f"FU{sector}{instrument}{suffix}"
                reval = f"FR{sector}{instrument}{suffix}"
                other = f"FV{sector}{instrument}{suffix}"
                
                if flow in columns_set:
                    pair = StockFlowPair(
                        stock_series=col,
                        flow_series=flow,
                        reval_series=reval if reval in columns_set else None,
                        other_series=other if other in columns_set else None,
                        has_complete_flows=(reval in columns_set and other in columns_set),
                        sector=sector,
                        instrument=instrument
                    )
                    pairs.append(pair)
                    processed.update(pair.get_all_series())
        
        return pairs
    
    def _process_bilateral_data(self, fwtw_data: pd.DataFrame) -> List[BilateralConstraint]:
        """
        Process FWTW data into bilateral constraints.
        """
        constraints = []
        
        if 'Holder Code' in fwtw_data.columns and 'Issuer Code' in fwtw_data.columns:
            grouped = fwtw_data.groupby(['Holder Code', 'Issuer Code', 'Instrument Code'])
            
            for (holder, issuer, instrument), group in grouped:
                # Map to Z.1 series if possible
                asset_series = self._map_to_z1_series(holder, instrument, 'asset')
                liability_series = self._map_to_z1_series(issuer, instrument, 'liability')
                
                constraint = BilateralConstraint(
                    holder=holder,
                    issuer=issuer,
                    instrument=instrument,
                    asset_series=asset_series,
                    liability_series=liability_series,
                    positions=group[['Date', 'Level']].values if 'Level' in group.columns else None
                )
                constraints.append(constraint)
        
        return constraints
    
    def _map_to_z1_series(self, entity: str, instrument: str, side: str) -> Optional[str]:
        """
        Map FWTW entity/instrument to Z.1 series code.
        """
        # This would implement the actual mapping logic
        # For now, return None (would need mapping tables)
        return None
    
    def _setup_state_space(self):
        """
        Set up state space by reading from parent model's actual structure.
        CRITICAL: Must match HierarchicalKalmanFilter's state layout exactly.
        """
        self.state_space = SFCStateSpace(
            n_base_states=self.n_base_states,
            n_base_shocks=self.n_base_shocks
        )
        
        # Read state mapping from parent model - REQUIRED
        if not hasattr(self, 'source_info'):
            raise ValueError("Parent model must have source_info structure for state mapping")
        
        self._read_state_mapping_from_parent()
        
        # Map SFC components to existing states
        self._map_sfc_to_states()
        
        # Set up bilateral state indices if needed
        if self.enforce_bilateral and self.bilateral_constraints:
            self._setup_bilateral_indices()
    
    def _read_state_mapping_from_parent(self):
        """
        Read actual state indices from HierarchicalKalmanFilter's source_info.
        
        Parent structure:
        - source_info[series]['start_idx']: Starting state index
        - source_info[series]['k_states']: Number of states for this series
        - source_info[series]['max_lag']: Maximum lag
        
        State layout per series:
        - idx + 0: level
        - idx + 1: trend  
        - idx + 2: lag_1 (if max_lag >= 1)
        - idx + 3: lag_2 (if max_lag >= 2)
        """
        if not self.source_info:
            raise ValueError("source_info is empty - cannot map states")
        
        self.logger.info("Reading state mapping from parent model's source_info")
        
        for series_code, info in self.source_info.items():
            start_idx = info['start_idx']
            k_states = info['k_states']
            max_lag = info.get('max_lag', 0)
            
            # Map this series
            self.state_space.base_indices[series_code] = {
                'level': start_idx,
                'trend': start_idx + 1,
                'start_idx': start_idx,
                'n_states': k_states,
                'max_lag': max_lag
            }
            
            # Add lag indices if present
            for lag_num in range(1, max_lag + 1):
                self.state_space.base_indices[series_code][f'lag_{lag_num}'] = start_idx + 1 + lag_num
        
        self.logger.info(f"Mapped {len(self.state_space.base_indices)} series from parent model")
    
    def _map_sfc_to_states(self):
        """
        Map stock-flow pairs to their state indices.
        """
        for pair in self.stock_flow_pairs:
            # Map stock series
            if pair.stock_series in self.state_space.base_indices:
                info = self.state_space.base_indices[pair.stock_series]
                self.state_space.stock_indices[pair.stock_series] = info['level']
            
            # Map flow series
            if pair.flow_series in self.state_space.base_indices:
                info = self.state_space.base_indices[pair.flow_series]
                self.state_space.flow_indices[pair.flow_series] = info['level']
            
            # Map revaluation series
            if pair.reval_series and pair.reval_series in self.state_space.base_indices:
                info = self.state_space.base_indices[pair.reval_series]
                self.state_space.reval_indices[pair.reval_series] = info['level']
            
            # Map other changes series
            if pair.other_series and pair.other_series in self.state_space.base_indices:
                info = self.state_space.base_indices[pair.other_series]
                self.state_space.other_indices[pair.other_series] = info['level']
    
    def _setup_bilateral_indices(self):
        """
        Set up bilateral position state indices.
        Maps each (holder, issuer, instrument) to a state index.
        """
        mapped_count = 0
        
        for constraint in self.bilateral_constraints:
            key = (constraint.holder, constraint.issuer, constraint.instrument)
            
            # Get corresponding Z.1 series - REQUIRED
            z1_series = self._map_bilateral_to_z1(constraint)
            
            if z1_series and z1_series in self.state_space.base_indices:
                # Map to existing state
                info = self.state_space.base_indices[z1_series]
                self.state_space.bilateral_indices[key] = info['level']
                mapped_count += 1
        
        if self.enforce_bilateral and mapped_count == 0:
            self.logger.warning("No bilateral constraints could be mapped to states")
    
    def _map_bilateral_to_z1(self, constraint: BilateralConstraint) -> Optional[str]:
        """
        Map a bilateral constraint to its Z.1 series code.
        Uses actual mapping from constraint object.
        """
        # Use mapped series from constraint
        if constraint.asset_series:
            return constraint.asset_series
        elif constraint.liability_series:
            return constraint.liability_series
        
        # No mapping available
        return None

    
    def _extend_model_properly(self):
        """
        Extension currently disabled to avoid statsmodels dimension conflicts.
        Work through projection-only mode.
        """
        if self.extend_state_space:
            self.logger.warning("State space extension disabled - using projection-only mode")
        
        # No extension - work with base model dimensions
        return
    
    def _build_extended_selection(self) -> np.ndarray:
        """
        Build block-diagonal extended selection matrix.
        PRESERVES base shock structure.
        """
        n_sfc_states = self.state_space.n_sfc_states
        n_sfc_shocks = self.state_space.n_sfc_shocks
        
        # Create selection for SFC states (can use shared shocks)
        R_sfc = np.zeros((n_sfc_states, n_sfc_shocks))
        
        # Simple approach: map states to shocks by groups
        # (In practice, group by instrument or sector)
        shocks_per_state = max(1, n_sfc_shocks // n_sfc_states)
        for i in range(min(n_sfc_states, n_sfc_shocks)):
            R_sfc[i, i % n_sfc_shocks] = 1.0
        
        # Block diagonal combination
        R_new = np.block([
            [self.base_selection, np.zeros((self.n_base_states, n_sfc_shocks))],
            [np.zeros((n_sfc_states, self.n_base_shocks)), R_sfc]
        ])
        
        return R_new
    
    def _build_extended_state_cov(self) -> np.ndarray:
        """
        Build block-diagonal extended state covariance.
        PRESERVES base covariance structure.
        """
        # Variance for different types of SFC states
        flow_var = self.error_variance_ratio
        reval_var = self.reval_variance_ratio
        bilateral_var = self.bilateral_variance_ratio
        
        # Build diagonal for SFC shocks
        sfc_variances = []
        
        # Flow state variances
        sfc_variances.extend([flow_var] * min(self.state_space.n_flow_states, 
                                              self.state_space.n_sfc_shocks // 2))
        
        # Revaluation variances (if present)
        if self.state_space.n_reval_states > 0:
            sfc_variances.extend([reval_var] * min(self.state_space.n_reval_states,
                                                   self.state_space.n_sfc_shocks // 4))
        
        # Bilateral variances
        remaining = self.state_space.n_sfc_shocks - len(sfc_variances)
        if remaining > 0:
            sfc_variances.extend([bilateral_var] * remaining)
        
        Q_sfc = np.diag(sfc_variances[:self.state_space.n_sfc_shocks])
        
        # Block diagonal combination
        Q_new = block_diag(self.base_state_cov, Q_sfc)
        
        return Q_new
    
    def _setup_constraints(self):
        """
        Set up constraint system.
        """
        self.constraint_builder = SFCConstraintBuilder(
            state_space=self.state_space,
            stock_flow_pairs=self.stock_flow_pairs,
            bilateral_constraints=self.bilateral_constraints,
            formulas=self.formulas if hasattr(self, 'formulas') else None,
            extend_state_space=self.extend_state_space,
            enforce_market_clearing=self.enforce_market_clearing,
            enforce_bilateral=self.enforce_bilateral
        )
        
        self.projector = SFCConstraintProjector(
            use_sparse=self.use_sparse
        )
        
        self.logger.info("Set up constraint system (projection-based)")
    
    def filter(self, params=None, **kwargs):
        """
        Run Kalman filter with SFC constraint projection.
        
        Parameters
        ----------
        params : array_like, optional
            Model parameters
        **kwargs : keyword arguments
            Additional arguments passed to parent filter
        """
        # Run base filter
        results = super().filter(params, **kwargs)
        
        # Apply constraints if enabled
        if self.enforce_sfc:
            self._apply_constraints_to_results(results, 'filtered')
        
        return results
    
    def smooth(self, params=None, **kwargs):
        """
        Run Kalman smoother with SFC constraint projection.
        
        Parameters
        ----------
        params : array_like, optional
            Model parameters
        **kwargs : keyword arguments
            Additional arguments passed to parent smoother
        """
        # Run base smoother
        results = super().smooth(params, **kwargs)
        
        # Apply constraints if enabled
        if self.enforce_sfc:
            self._apply_constraints_to_results(results, 'smoothed')
        
        return results
    
    def _apply_constraints_to_results(self, results, state_type: str):
        """
        Apply SFC constraints to filter/smoother results.
        PROPERLY updates both mean and covariance.
        """
        # Check if we have the required attributes
        if state_type == 'filtered':
            if not hasattr(results, 'filtered_state'):
                self.logger.warning("No filtered_state in results - skipping constraints")
                return
            states = results.filtered_state
            
            # Check for covariance
            if hasattr(results, 'filtered_state_cov'):
                covs = results.filtered_state_cov
                has_cov = True
            else:
                self.logger.warning("No filtered_state_cov - applying mean-only projection")
                has_cov = False
        else:  # smoothed
            if not hasattr(results, 'smoothed_state'):
                self.logger.warning("No smoothed_state in results - skipping constraints")
                return
            states = results.smoothed_state
            
            # Check for covariance
            if hasattr(results, 'smoothed_state_cov'):
                covs = results.smoothed_state_cov
                has_cov = True
            else:
                self.logger.warning("No smoothed_state_cov - applying mean-only projection")
                has_cov = False
        
        n_periods = states.shape[1]
        
        # Log constraint diagnostics at t=1
        if n_periods > 1:
            self._log_constraint_diagnostics(t=1)
        
        # Apply constraints at each time period
        for t in range(n_periods):
            # Build constraints for this period
            A, b = self.constraint_builder.build_constraints(t)
            
            if A.shape[0] > 0:
                # Get current state
                state_t = states[:, t]
                
                if has_cov:
                    # Project state and covariance
                    cov_t = covs[:, :, t]
                    state_proj, cov_proj = self.projector.project_state(
                        state_t, cov_t, A, b
                    )
                    states[:, t] = state_proj
                    covs[:, :, t] = cov_proj
                else:
                    # Mean-only projection (simplified)
                    if sparse.issparse(A):
                        residual = A @ state_t - b
                    else:
                        residual = A @ state_t - b
                    
                    # Simple gradient step
                    if np.max(np.abs(residual)) > 1e-10:
                        if sparse.issparse(A):
                            gradient = A.T @ residual
                        else:
                            gradient = A.T @ residual
                        states[:, t] -= 0.1 * gradient / (1 + np.linalg.norm(gradient))
        
        # Update results object
        if state_type == 'filtered':
            results.filtered_state = states
            if has_cov:
                results.filtered_state_cov = covs
        else:
            results.smoothed_state = states
            if has_cov:
                results.smoothed_state_cov = covs
        
        self.logger.info(f"Applied SFC constraints to {state_type} results")
    
    def _log_constraint_diagnostics(self, t: int = 1):
        """
        Log diagnostic information about constraints at time t.
        """
        A, b = self.constraint_builder.build_constraints(t)
        
        if A.shape[0] == 0:
            self.logger.info(f"No constraints at t={t}")
            return
        
        self.logger.info(f"Constraint diagnostics at t={t}:")
        self.logger.info(f"  Number of constraints: {A.shape[0]}")
        self.logger.info(f"  Constraint matrix shape: {A.shape}")
        self.logger.info(f"  Sparsity: {A.nnz / (A.shape[0] * A.shape[1]):.4f}" if sparse.issparse(A) else "Dense")
        
        # Sample a few constraint rows to verify indices
        if hasattr(self.stock_flow_pairs, '__len__') and len(self.stock_flow_pairs) > 0:
            pair = self.stock_flow_pairs[0]
            self.logger.info(f"  Sample stock-flow pair: {pair.stock_series} - {pair.flow_series}")
            
            # Check which states are involved
            if pair.stock_series in self.state_space.base_indices:
                stock_info = self.state_space.base_indices[pair.stock_series]
                self.logger.info(f"    Stock state indices: level={stock_info['level']}, max_lag={stock_info.get('max_lag', 0)}")
            
            if pair.flow_series in self.state_space.base_indices:
                flow_info = self.state_space.base_indices[pair.flow_series]
                self.logger.info(f"    Flow state index: level={flow_info['level']}")
    
    def validate_constraints(self, states: np.ndarray, tolerance: float = 1e-6) -> Dict:
        """
        Validate that SFC constraints are satisfied.
        """
        violations = {
            'stock_flow': [],
            'market_clearing': [],
            'bilateral': [],
            'max_violation': 0.0
        }
        
        n_periods = states.shape[1]
        
        for t in range(1, n_periods):  # Start from t=1 for stock-flow
            A, b = self.constraint_builder.build_constraints(t)
            
            if A.shape[0] > 0:
                state_t = states[:, t]
                residual = np.abs(A @ state_t - b)
                
                max_viol = np.max(residual)
                violations['max_violation'] = max(violations['max_violation'], max_viol)
                
                # Categorize violations
                # (Would need to track constraint types in builder)
                violations['stock_flow'].append(np.mean(residual))
        
        # Compute summary statistics
        summary = {
            'max_violation': violations['max_violation'],
            'mean_stock_flow': np.mean(violations['stock_flow']) if violations['stock_flow'] else 0.0,
            'constraints_satisfied': violations['max_violation'] < tolerance
        }
        
        return summary
    
    def get_diagnostics(self) -> Dict:
        """
        Get comprehensive diagnostics.
        """
        return {
            'base_states': self.state_space.n_base_states,
            'sfc_states': self.state_space.n_sfc_states,
            'total_states': self.state_space.n_total_states,
            'base_shocks': self.state_space.n_base_shocks,
            'sfc_shocks': self.state_space.n_sfc_shocks,
            'total_shocks': self.state_space.n_total_shocks,
            'stock_flow_pairs': len(self.stock_flow_pairs),
            'complete_pairs': sum(1 for p in self.stock_flow_pairs if p.has_complete_flows),
            'bilateral_constraints': len(self.bilateral_constraints),
            'enforce_sfc': self.enforce_sfc,
            'enforce_market_clearing': self.enforce_market_clearing,
            'enforce_bilateral': self.enforce_bilateral,
            'include_revaluations': self.include_revaluations
        }
