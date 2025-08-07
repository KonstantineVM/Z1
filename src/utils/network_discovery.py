# ==============================================================================
# FILE: src/utils/network_discovery.py
# ==============================================================================
"""
Dynamic network discovery for complete SFC analysis.
"""

from typing import Set, List, Dict, Optional
import pandas as pd
import logging
from src.network.fwtw_z1_mapper import FWTWtoZ1Mapper, Z1SeriesCode
from src.utils.formula_parser import FormulaParser

logger = logging.getLogger(__name__)


class NetworkDiscovery:
    """Discover complete network of series from data and formulas."""
    
    @staticmethod
    def discover_complete_network(initial_series: Optional[List[str]] = None,
                                 formulas: Optional[Dict] = None,
                                 fwtw_data: Optional[pd.DataFrame] = None,
                                 z1_data: Optional[pd.DataFrame] = None,
                                 include_all: bool = True) -> Dict[str, Set[str]]:
        """
        Discover ALL series needed for complete SFC consistency.
        
        Parameters:
        -----------
        initial_series : Starting series to trace from
        formulas : Z.1 formula definitions
        fwtw_data : FWTW bilateral position data
        z1_data : Z.1 time series data
        include_all : If True, include everything found
        
        Returns:
        --------
        Dictionary with discovered series by category
        """
        discovery = {
            'initial': set(initial_series) if initial_series else set(),
            'from_formulas': set(),
            'from_fwtw': set(),
            'from_z1_columns': set(),
            'market_totals': set(),
            'all_series': set()
        }
        
        # 1. Discover from Z.1 column names if provided
        if z1_data is not None:
            for col in z1_data.columns:
                parsed = Z1SeriesCode.parse(col)
                if parsed:
                    discovery['from_z1_columns'].add(col)
        
        # 2. Trace formula dependencies recursively
        if formulas and initial_series:
            parser = FormulaParser()
            
            required = set(initial_series)
            processed = set()
            to_process = list(initial_series)
            
            while to_process:
                current = to_process.pop(0)
                if current in processed:
                    continue
                processed.add(current)
                
                # Check if this series has a formula
                if current in formulas:
                    components = parser.parse_formula(formulas[current])
                    for series_code, _, _, _ in components:
                        if series_code not in required:
                            required.add(series_code)
                            to_process.append(series_code)
                
                # Find series that depend on current
                for other_series, formula_dict in formulas.items():
                    if other_series not in required:
                        components = parser.parse_formula(formula_dict)
                        for dep_series, _, _, _ in components:
                            if dep_series == current:
                                required.add(other_series)
                                to_process.append(other_series)
            
            discovery['from_formulas'] = required
        
        # 3. Discover from FWTW data
        if fwtw_data is not None:
            mapper = FWTWtoZ1Mapper()
            
            try:
                fwtw_clean = mapper.validate_and_standardize_fwtw(fwtw_data)
                
                # Get all unique combinations
                instruments = set(fwtw_clean['Instrument Code'].unique())
                holder_sectors = set(fwtw_clean['Holder Code'].unique())
                issuer_sectors = set(fwtw_clean['Issuer Code'].unique())
                all_sectors = holder_sectors | issuer_sectors
                
                # Build all implied series
                for sector in all_sectors:
                    for instrument in instruments:
                        # Include all prefixes for complete accounting
                        for prefix in ['FA', 'FL', 'FR', 'FU', 'FV']:
                            series = f"{prefix}{sector.zfill(2)}{instrument.zfill(5)}05"
                            discovery['from_fwtw'].add(series)
                
                # Add market totals (sector 89)
                for instrument in instruments:
                    for prefix in ['FA', 'FL']:
                        total_series = f"{prefix}89{instrument.zfill(5)}05"
                        discovery['market_totals'].add(total_series)
                        
            except Exception as e:
                logger.error(f"Error processing FWTW data: {e}")
        
        # 4. Add market clearing series for all discovered instruments
        all_discovered = discovery['from_formulas'] | discovery['from_fwtw'] | discovery['from_z1_columns']
        
        instruments_found = set()
        for series in all_discovered:
            parsed = Z1SeriesCode.parse(series)
            if parsed:
                instruments_found.add(parsed.instrument)
        
        # Add totals for all instruments
        for instrument in instruments_found:
            for prefix in ['FA', 'FL']:
                discovery['market_totals'].add(f"{prefix}89{instrument}05")
        
        # Combine all
        discovery['all_series'] = (
            discovery['initial'] |
            discovery['from_formulas'] |
            discovery['from_fwtw'] |
            discovery['from_z1_columns'] |
            discovery['market_totals']
        )
        
        # Log discovery statistics
        logger.info("Network discovery complete:")
        logger.info(f"  Initial series: {len(discovery['initial'])}")
        logger.info(f"  From formulas: {len(discovery['from_formulas'])}")
        logger.info(f"  From FWTW: {len(discovery['from_fwtw'])}")
        logger.info(f"  From Z.1 columns: {len(discovery['from_z1_columns'])}")
        logger.info(f"  Market totals: {len(discovery['market_totals'])}")
        logger.info(f"  Total discovered: {len(discovery['all_series'])}")
        
        return discovery
