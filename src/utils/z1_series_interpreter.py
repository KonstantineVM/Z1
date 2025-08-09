"""
Z1 Series Interpreter - Correct implementation based on Federal Reserve documentation.
Properly identifies series types and relationships.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Z1Series:
    """Represents a properly parsed Z1 series."""
    full_code: str
    prefix: str  # FA, FL, FU, FR, FV, LA, etc.
    sector: str  # 2-digit sector code
    instrument: str  # 5-digit instrument code
    calculation: str  # 1-digit (0, 1, 3, 5, 6)
    frequency: str  # Q or A
    
    @property
    def series_type(self) -> str:
        """Get the type of series based on prefix."""
        type_map = {
            'FA': 'flow_saar',  # Flow, seasonally adjusted annual rate
            'FL': 'level',      # Level/stock, not seasonally adjusted
            'FU': 'flow',        # Flow/transaction, not seasonally adjusted
            'FR': 'revaluation', # Revaluation
            'FV': 'other_change', # Other volume changes
            'LA': 'level_sa',    # Level, seasonally adjusted
            'FC': 'change',      # Change in level
            'FG': 'growth_rate'  # Growth rate
        }
        return type_map.get(self.prefix, 'unknown')
    
    @property
    def is_stock(self) -> bool:
        """Check if this is a stock/level series."""
        return self.prefix in ('FL', 'LA', 'LM')
    
    @property
    def is_flow(self) -> bool:
        """Check if this is a flow/transaction series."""
        return self.prefix in ('FA', 'FU')
    
    @property
    def is_seasonally_adjusted(self) -> bool:
        """Check if series is seasonally adjusted."""
        return self.prefix in ('FA', 'LA', 'FG')
    
    @classmethod
    def parse(cls, series_code: str) -> Optional['Z1Series']:
        """Parse a Z1 series code into components."""
        # Remove .Q or .A suffix for parsing
        if series_code.endswith(('.Q', '.A')):
            frequency = series_code[-1]
            base_code = series_code[:-2]
        else:
            frequency = ''
            base_code = series_code
        
        # Pattern: 2 letters + 2 digits + 5 digits + 1 digit
        pattern = r'^([A-Z]{2})(\d{2})(\d{5})(\d)$'
        match = re.match(pattern, base_code)
        
        if not match:
            return None
        
        return cls(
            full_code=series_code,
            prefix=match.group(1),
            sector=match.group(2),
            instrument=match.group(3),
            calculation=match.group(4),
            frequency=frequency
        )


class Z1SeriesInterpreter:
    """Correctly interprets Z1 series codes and relationships."""
    
    # Sector codes from Federal Reserve documentation
    SECTOR_CODES = {
        '10': 'Nonfinancial corporate business',
        '15': 'Households and nonprofit organizations',
        '26': 'Rest of the world',
        '31': 'Federal government',
        '70': 'Private depository institutions',
        '89': 'All sectors',
        # Add more as needed
    }
    
    # Common instrument patterns
    ASSET_INSTRUMENTS = {
        '20000': 'Total assets',
        '30641': 'Corporate equities (as asset)',
        '30611': 'Treasury securities (as asset)',
        # Add more
    }
    
    LIABILITY_INSTRUMENTS = {
        '21000': 'Total liabilities and net worth',
        '31641': 'Corporate equities (as liability)',
        '31611': 'Treasury securities (as liability)',
        # Add more
    }
    
    def __init__(self):
        self.stock_flow_pairs = {}
        self.market_clearing_groups = {}
    
    def identify_stock_flow_pairs(self, series_list: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Identify stock-flow-revaluation sets in the data.
        Returns dict keyed by stock series with corresponding flows.
        """
        pairs = {}
        
        # Parse all series
        parsed = {}
        for series in series_list:
            p = Z1Series.parse(series)
            if p:
                parsed[series] = p
        
        # Find FL (stock) series and their corresponding flows
        for series, parsed_series in parsed.items():
            if parsed_series.prefix == 'FL':
                # This is a stock series
                key = f"{parsed_series.sector}{parsed_series.instrument}{parsed_series.calculation}"
                
                # Look for corresponding flow series
                flow_code = f"FU{key}"
                reval_code = f"FR{key}"
                other_code = f"FV{key}"
                
                # Add frequency suffix
                if parsed_series.frequency:
                    flow_code += f".{parsed_series.frequency}"
                    reval_code += f".{parsed_series.frequency}"
                    other_code += f".{parsed_series.frequency}"
                
                pairs[series] = {
                    'stock': series,
                    'flow': flow_code if flow_code in series_list else None,
                    'revaluation': reval_code if reval_code in series_list else None,
                    'other_change': other_code if other_code in series_list else None
                }
        
        return pairs
    
    def build_stock_flow_constraint(self, stock: str, flow: str, reval: str = None, other: str = None) -> str:
        """
        Build the stock-flow consistency constraint equation.
        FL[t] = FL[t-1] + FU + FR + FV
        """
        constraint = f"{stock}[t] = {stock}[t-1] + {flow}"
        if reval:
            constraint += f" + {reval}"
        if other:
            constraint += f" + {other}"
        return constraint
    
    def identify_market_clearing_groups(self, series_list: List[str]) -> Dict[str, List[str]]:
        """
        Group series by instrument for market clearing constraints.
        For each instrument, assets should equal liabilities.
        """
        groups = {}
        
        for series in series_list:
            parsed = Z1Series.parse(series)
            if not parsed or not parsed.is_stock:
                continue
            
            instrument = parsed.instrument
            if instrument not in groups:
                groups[instrument] = {'holders': [], 'issuers': []}
            
            # Determine if this sector is typically a holder or issuer for this instrument
            # This requires domain knowledge or a mapping table
            # Simplified: some sectors are typically holders, others issuers
            holder_sectors = ['15', '59', '65']  # Households, pension funds, mutual funds
            issuer_sectors = ['10', '31', '40']  # Corporations, government, GSEs
            
            if parsed.sector in holder_sectors:
                groups[instrument]['holders'].append(series)
            elif parsed.sector in issuer_sectors:
                groups[instrument]['issuers'].append(series)
            # Some sectors can be both
            
        return groups
    
    def validate_series_relationships(self, data: dict) -> Dict[str, bool]:
        """
        Validate that series follow expected relationships.
        Returns dict of constraint_name: is_satisfied
        """
        results = {}
        
        # Check stock-flow consistency
        for stock_series, components in self.stock_flow_pairs.items():
            if components['flow']:
                # Check FL[t] - FL[t-1] ≈ FU + FR + FV
                # Implementation depends on data structure
                pass
        
        return results
    
    @staticmethod
    def correct_series_name_interpretation(series: str) -> Dict[str, str]:
        """
        Provide correct interpretation of a series name.
        """
        parsed = Z1Series.parse(series)
        if not parsed:
            return {'error': 'Invalid series format'}
        
        interpretation = {
            'series': series,
            'prefix': parsed.prefix,
            'prefix_meaning': {
                'FA': 'Flow at seasonally adjusted annual rate',
                'FL': 'Level (stock), not seasonally adjusted',
                'FU': 'Flow (transaction), not seasonally adjusted',
                'FR': 'Revaluation',
                'FV': 'Other volume changes',
                'LA': 'Level, seasonally adjusted'
            }.get(parsed.prefix, 'Unknown prefix'),
            'sector': Z1SeriesInterpreter.SECTOR_CODES.get(parsed.sector, f'Sector {parsed.sector}'),
            'instrument': parsed.instrument,
            'is_stock': parsed.is_stock,
            'is_flow': parsed.is_flow,
            'is_seasonally_adjusted': parsed.is_seasonally_adjusted,
            'calculation_type': {
                '0': 'Input with seasonal factor',
                '1': 'Input from NIPA',
                '3': 'Input with zero seasonal factor',
                '5': 'Computed series',
                '6': 'Percent series'
            }.get(parsed.calculation, 'Unknown')
        }
        
        # Note about FA/FL confusion
        if parsed.prefix == 'FA':
            interpretation['warning'] = "FA is a FLOW (not assets!), seasonally adjusted at annual rate"
        elif parsed.prefix == 'FL':
            interpretation['warning'] = "FL is a LEVEL/STOCK (not liabilities!), can be either asset or liability"
        
        return interpretation


def demonstrate_correct_usage():
    """Demonstrate the correct interpretation of Z1 series."""
    
    interpreter = Z1SeriesInterpreter()
    
    # Example series that are commonly misunderstood
    examples = [
        'FA156090005.Q',  # Households financial assets - WRONG NAME! This is actually a FLOW
        'FL156090005.Q',  # Households financial assets level - This is the actual STOCK
        'FL103064105.Q',  # Nonfinancial corporate equity - This is a LIABILITY (equity issued)
        'FL153064105.Q',  # Households corporate equity - This is an ASSET (equity owned)
    ]
    
    print("CORRECT Z1 SERIES INTERPRETATION")
    print("=" * 80)
    
    for series in examples:
        result = interpreter.correct_series_name_interpretation(series)
        print(f"\nSeries: {series}")
        print(f"  Type: {result['prefix_meaning']}")
        print(f"  Sector: {result['sector']}")
        print(f"  Is Stock?: {result['is_stock']}")
        print(f"  Is Flow?: {result['is_flow']}")
        if 'warning' in result:
            print(f"  ⚠️ WARNING: {result['warning']}")
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS:")
    print("1. FA series are FLOWS (seasonally adjusted), not assets!")
    print("2. FL series are LEVELS/STOCKS, not liabilities!")
    print("3. Whether something is an asset or liability depends on CONTEXT, not prefix")
    print("4. Stock-flow identity: FL[t] = FL[t-1] + FU + FR + FV")


if __name__ == "__main__":
    demonstrate_correct_usage()