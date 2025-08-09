"""
FWTWtoZ1Mapper - Clean implementation with no fuzzy logic.
Maps FWTW bilateral positions to Z.1 FL (level/stock) series codes.
FL series represent LEVELS (stocks), not liabilities!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Z1SeriesCode:
    """Represents a complete Z.1 series code."""
    prefix: str  # FL for levels, FU for flows, FR for revaluations, etc.
    sector: str  # 2-digit sector code (Fed digits 1-2)
    instrument: str  # 5-digit instrument code (Fed digits 3-7)
    suffix: str  # 2-digit suffix (digit 8 always 0, digit 9 is calculation type)
    
    @property
    def full_code(self) -> str:
        return f"{self.prefix}{self.sector}{self.instrument}{self.suffix}"
    
    @property
    def series_type(self) -> str:
        """Return the type based on prefix."""
        type_map = {
            'FA': 'flow_saar',      # Flow, seasonally adjusted annual rate (NOT assets!)
            'FL': 'level',          # Level/stock, not seasonally adjusted (NOT liabilities!)
            'FU': 'flow',           # Flow/transaction, not seasonally adjusted
            'FR': 'revaluation',    # Revaluation
            'FV': 'other_change',   # Other volume changes
            'LA': 'level_sa'        # Level, seasonally adjusted
        }
        return type_map.get(self.prefix, 'unknown')


class FWTWtoZ1Mapper:
    """
    Maps FWTW bilateral positions to Z.1 series codes.
    Both holder and issuer positions map to FL (level) series.
    No fuzzy logic - requires exact column names and formats.
    """
    
    # Official Federal Reserve Sector Codes - Hardcoded
    SECTOR_CODES = {
        '10': 'Nonfinancial Corporate Business',
        '11': 'Nonfinancial Noncorporate Business',
        '15': 'Households and Nonprofit Organizations',
        '21': 'State and Local Governments',
        '26': 'Rest of World',
        '31': 'Federal Government',
        '42': 'Government-Sponsored Enterprises',
        '47': 'Credit Unions',
        '50': 'Other Financial Business',
        '51': 'Property-Casualty Insurance Companies',
        '54': 'Life Insurance Companies',
        '55': 'Closed-End Funds',
        '56': 'Exchange-Traded Funds',
        '59': 'Private and Public Pension Funds',
        '61': 'Finance Companies',
        '63': 'Money Market Funds',
        '64': 'Mortgage Real Estate Investment Trusts',
        '65': 'Mutual Funds',
        '66': 'Security Brokers and Dealers',
        '67': 'Issuers of Asset-Backed Securities',
        '71': 'Monetary Authority',
        '73': 'Holding Companies',
        '74': 'Banks in U.S.-Affiliated Areas',
        '75': 'Foreign Banking Offices in U.S.',
        '76': 'U.S.-Chartered Depository Institutions',
        '89': 'All Sectors',
        '90': 'Instrument Discrepancies Sector'
    }
    
    # Required FWTW columns - exact names required
    REQUIRED_COLUMNS = {
        'Date',
        'Holder Code', 
        'Issuer Code',
        'Instrument Code',
        'Level'
    }
    
    # Optional columns
    OPTIONAL_COLUMNS = {
        'Holder Name',
        'Issuer Name', 
        'Instrument Name'
    }
    
    def __init__(self):
        self.valid_sectors = set(self.SECTOR_CODES.keys())
        
    def validate_fwtw_data(self, fwtw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate FWTW data structure. No fuzzy matching - exact requirements.
        
        Parameters:
        -----------
        fwtw_data : pd.DataFrame
            FWTW data with required columns
            
        Returns:
        --------
        pd.DataFrame
            Validated and formatted data
            
        Raises:
        -------
        ValueError
            If required columns are missing or data is invalid
        """
        # Check required columns exist
        missing_columns = self.REQUIRED_COLUMNS - set(fwtw_data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {list(fwtw_data.columns)}"
            )
        
        # Create clean copy
        clean_data = fwtw_data.copy()
        
        # Validate and format Date column
        if not pd.api.types.is_datetime64_any_dtype(clean_data['Date']):
            # Require datetime type
            raise ValueError("'Date' column must be datetime64 type")
        
        # Format sector codes - must be 2 digits
        clean_data['Holder Code'] = clean_data['Holder Code'].astype(str).str.zfill(2)
        clean_data['Issuer Code'] = clean_data['Issuer Code'].astype(str).str.zfill(2)
        
        # Format instrument codes - must be 5 digits
        clean_data['Instrument Code'] = clean_data['Instrument Code'].astype(str).str.zfill(5)
        
        # Validate Level is numeric
        if not pd.api.types.is_numeric_dtype(clean_data['Level']):
            clean_data['Level'] = pd.to_numeric(clean_data['Level'], errors='raise')
        
        # Remove rows with null values in required columns
        initial_len = len(clean_data)
        clean_data = clean_data.dropna(subset=list(self.REQUIRED_COLUMNS))
        
        if len(clean_data) < initial_len:
            logger.info(f"Removed {initial_len - len(clean_data)} rows with null values")
        
        if clean_data.empty:
            raise ValueError("No valid data remains after cleaning")
        
        return clean_data
    
    def map_to_z1_series(self, 
                         fwtw_data: pd.DataFrame,
                         available_z1_series: Optional[Set[str]] = None) -> pd.DataFrame:
        """
        Map FWTW bilateral positions to Z.1 FL (level) series codes.
        
        FL represents levels/stocks that are not seasonally adjusted.
        Both holder and issuer positions map to FL series.
        
        Parameters:
        -----------
        fwtw_data : pd.DataFrame
            FWTW data with required columns
        available_z1_series : Set[str], optional
            Set of available Z.1 series for validation (base codes without .Q)
            
        Returns:
        --------
        pd.DataFrame with columns:
            - date: Position date
            - holder_code: 2-digit sector code
            - issuer_code: 2-digit sector code
            - instrument_code: 5-digit instrument code
            - holder_series: FL series for holder (e.g., FL1530641005)
            - issuer_series: FL series for issuer (e.g., FL1030641005)
            - holder_flow_series: FU series for holder
            - issuer_flow_series: FU series for issuer
            - level: Position value (stock amount)
            - holder_name: Sector name or from data
            - issuer_name: Sector name or from data
            - instrument_name: From data if available
        """
        # Validate input data
        clean_data = self.validate_fwtw_data(fwtw_data)
        
        # Map each row
        mapped_rows = []
        
        for _, row in clean_data.iterrows():
            holder_code = row['Holder Code']
            issuer_code = row['Issuer Code']
            instrument_code = row['Instrument Code']
            
            # Build Z.1 series codes
            # FL = Level (stock), not seasonally adjusted
            # FU = Flow (transaction), not seasonally adjusted
            # Suffix: digit 8 is always 0, digit 9 is 5 (calculated series)
            holder_series = f"FL{holder_code}{instrument_code}05"
            issuer_series = f"FL{issuer_code}{instrument_code}05"
            holder_flow_series = f"FU{holder_code}{instrument_code}05"
            issuer_flow_series = f"FU{issuer_code}{instrument_code}05"
            
            # Check if series exist in Z.1 if validation set provided
            if available_z1_series is not None:
                if holder_series not in available_z1_series and issuer_series not in available_z1_series:
                    continue  # Skip if neither series exists
            
            mapped_rows.append({
                'date': row['Date'],
                'holder_code': holder_code,
                'issuer_code': issuer_code,
                'instrument_code': instrument_code,
                'holder_series': holder_series,
                'issuer_series': issuer_series,
                'holder_flow_series': holder_flow_series,
                'issuer_flow_series': issuer_flow_series,
                'level': row['Level'],
                'holder_name': row.get('Holder Name', self.SECTOR_CODES.get(holder_code, f'Sector {holder_code}')),
                'issuer_name': row.get('Issuer Name', self.SECTOR_CODES.get(issuer_code, f'Sector {issuer_code}')),
                'instrument_name': row.get('Instrument Name', f'Instrument {instrument_code}')
            })
        
        result = pd.DataFrame(mapped_rows)
        
        if result.empty:
            logger.warning("No FWTW positions mapped to Z.1 series")
        else:
            logger.info(f"Mapped {len(result)} FWTW positions to Z.1 series")
            logger.info(f"Unique bilateral relationships: {len(result.groupby(['holder_code', 'issuer_code', 'instrument_code']))}")
        
        return result
    
    def calculate_bilateral_flows(self, mapped_fwtw: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate bilateral flows from changes in FWTW stock positions.
        Flow[t] = Level[t] - Level[t-1]
        
        Parameters:
        -----------
        mapped_fwtw : pd.DataFrame
            Output from map_to_z1_series()
            
        Returns:
        --------
        pd.DataFrame with bilateral flows
        """
        if mapped_fwtw.empty:
            return pd.DataFrame()
        
        # Sort by date
        mapped_fwtw = mapped_fwtw.sort_values('date')
        
        # Group by bilateral relationship
        grouped = mapped_fwtw.groupby(['holder_code', 'issuer_code', 'instrument_code'])
        
        flow_records = []
        
        for (holder, issuer, instrument), group in grouped:
            group = group.sort_values('date')
            
            if len(group) < 2:
                continue  # Need at least 2 periods for flow
            
            # Calculate flows as first differences
            for i in range(1, len(group)):
                current = group.iloc[i]
                previous = group.iloc[i-1]
                
                flow_records.append({
                    'date': current['date'],
                    'holder_code': holder,
                    'issuer_code': issuer,
                    'instrument_code': instrument,
                    'holder_flow_series': current['holder_flow_series'],
                    'issuer_flow_series': current['issuer_flow_series'],
                    'flow': current['level'] - previous['level'],
                    'level_t': current['level'],
                    'level_t_minus_1': previous['level']
                })
        
        return pd.DataFrame(flow_records)