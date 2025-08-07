# ==============================================================================
# FILE: src/utils/formula_parser.py
# ==============================================================================
"""
Complete formula parser with lag support and proper evaluation.
"""

import re
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FormulaParser:
    """Parse and evaluate Z.1 formulas with complete lag support."""
    
    @staticmethod
    def parse_formula(formula_dict: Dict) -> List[Tuple[str, int, str, float]]:
        """
        Parse a formula into components with lags and operators.
        
        Returns:
            List of (series_code, lag, operator, coefficient) tuples
        """
        if not formula_dict:
            return []
            
        formula_str = formula_dict.get('formula', '')
        if not formula_str:
            return []
        
        components = []
        
        # Handle different formula formats
        # Format 1: Explicit derived_from list
        if 'derived_from' in formula_dict and formula_dict['derived_from']:
            for item in formula_dict['derived_from']:
                series = item.get('code', '')
                if not series:
                    continue
                operator = item.get('operator', '+')
                lag = item.get('lag', 0)
                coef = -1.0 if operator == '-' else 1.0
                components.append((series, lag, operator, coef))
        
        # Format 2: Parse formula string
        else:
            # Pattern for Z.1 series with optional lag
            # Matches: FA103064105[t-1] or FA103064105(-1) or FA103064105
            pattern = re.compile(
                r'([+-]?)\s*'                    # Optional sign
                r'(F[ALRUV]\d{2}\d{5}\d{3})'    # Series code (correct pattern)
                r'(?:'                           # Optional lag group
                r'\[t\s*([+-]?\d+)\]'           # [t-1] format
                r'|'                            # OR
                r'\(([+-]?\d+)\)'               # (-1) format
                r')?'
            )
            
            for match in pattern.finditer(formula_str):
                sign = match.group(1) or '+'
                series = match.group(2)
                lag_bracket = match.group(3)
                lag_paren = match.group(4)
                
                lag = 0
                if lag_bracket:
                    lag = int(lag_bracket)
                elif lag_paren:
                    lag = int(lag_paren)
                
                coef = -1.0 if sign == '-' else 1.0
                components.append((series, lag, sign, coef))
        
        return components
    
    @staticmethod
    def evaluate_formula(formula_components: List[Tuple[str, int, str, float]],
                        data: pd.DataFrame,
                        time_index: int,
                        allow_partial: bool = True) -> Tuple[float, List[str]]:
        """
        Evaluate a parsed formula at a specific time.
        
        Parameters:
        -----------
        formula_components : List of (series, lag, operator, coefficient)
        data : DataFrame with series as columns
        time_index : Current time index
        allow_partial : If True, compute with available components
        
        Returns:
        --------
        (result, missing_components) tuple
        """
        result = 0.0
        missing_components = []
        available_components = 0
        
        for series, lag, operator, coef in formula_components:
            # Calculate lagged time index
            lagged_index = time_index + lag  # lag is negative for past
            
            # Check bounds
            if lagged_index < 0 or lagged_index >= len(data):
                missing_components.append(f"{series}[{lag}]")
                if not allow_partial:
                    return np.nan, missing_components
                continue
            
            # Get value
            if series in data.columns:
                value = data.iloc[lagged_index][series]
                if not pd.isna(value):
                    result += coef * value
                    available_components += 1
                else:
                    missing_components.append(f"{series}(NaN)")
                    if not allow_partial:
                        return np.nan, missing_components
            else:
                missing_components.append(f"{series}(missing)")
                if not allow_partial:
                    return np.nan, missing_components
        
        # Return NaN if no components available
        if available_components == 0:
            return np.nan, missing_components
            
        return result, missing_components
