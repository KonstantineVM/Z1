# ==============================================================================
# FILE: src/network/fwtw_z1_mapper.py
# ==============================================================================
"""
Maps FWTW bilateral positions to Z.1 series codes.
Uses official Federal Reserve sector and instrument codes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
import re
from dataclasses import dataclass
from pathlib import Path
logger = logging.getLogger(__name__)


@dataclass
class Z1SeriesCode:
    """Represents a complete Z.1 series code."""
    prefix: str  # FA, FL, FR, FU, FV (2 chars)
    sector: str  # 2-digit sector code  
    instrument: str  # 5-digit instrument code
    suffix: str  # 3-digit suffix (usually "005")
    
    @property
    def full_code(self) -> str:
        return f"{self.prefix}{self.sector}{self.instrument}{self.suffix}"
    
    @classmethod
    def parse(cls, code: str) -> Optional['Z1SeriesCode']:
        """Parse a Z.1 series code string."""
        # Correct pattern: 2-char prefix + 2-digit sector + 5-digit instrument + 3-digit suffix
        pattern = re.compile(r'^(F[ALRUV])(\d{2})(\d{5})(\d{3})$')
        match = pattern.match(code)
        if match:
            return cls(
                prefix=match.group(1),
                sector=match.group(2),
                instrument=match.group(3),
                suffix=match.group(4)
            )
        return None


class FWTWtoZ1Mapper:
    """Complete mapper with proper validation and flexible column handling."""
    
    # Official Federal Reserve Sector Codes
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
    
    def __init__(self):
        self.valid_sectors = set(self.SECTOR_CODES.keys())
        self.discovered_instruments = set()
        self.unmapped_items = []
        
    def validate_and_standardize_fwtw(self, fwtw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate FWTW data and standardize column names.
        Handles multiple possible column name formats.
        """
        # Map possible column names to standard names
        column_mappings = {
            'Date': ['Date', 'date', 'DATE', 'Period', 'period'],
            'Holder Code': ['Holder Code', 'holder_code', 'HolderCode', 'holder'],
            'Issuer Code': ['Issuer Code', 'issuer_code', 'IssuerCode', 'issuer'],
            'Instrument Code': ['Instrument Code', 'instrument_code', 'InstrumentCode'],
            'Level': ['Level', 'level', 'Value', 'value', 'Amount'],
            'Holder Name': ['Holder Name', 'holder_name', 'HolderName'],
            'Issuer Name': ['Issuer Name', 'issuer_name', 'IssuerName'],
            'Instrument Name': ['Instrument Name', 'instrument_name', 'InstrumentName']
        }
        
        standardized = fwtw_data.copy()
        
        for standard_name, alternatives in column_mappings.items():
            found = False
            for alt in alternatives:
                if alt in standardized.columns:
                    if alt != standard_name:
                        standardized = standardized.rename(columns={alt: standard_name})
                    found = True
                    break
            
            if not found and standard_name in ['Holder Code', 'Issuer Code', 'Instrument Code', 'Date', 'Level']:
                raise ValueError(f"Required column '{standard_name}' not found. "
                               f"Available columns: {list(fwtw_data.columns)}")
        
        # Validate and clean data types
        standardized['Holder Code'] = standardized['Holder Code'].astype(str).str.zfill(2)
        standardized['Issuer Code'] = standardized['Issuer Code'].astype(str).str.zfill(2)
        standardized['Instrument Code'] = standardized['Instrument Code'].astype(str).str.zfill(5)
        standardized['Level'] = pd.to_numeric(standardized['Level'], errors='coerce')
        
        # Handle date format
        standardized['Date'] = self._parse_fwtw_dates(standardized['Date'])
        
        # Discover instruments
        self.discovered_instruments.update(standardized['Instrument Code'].unique())
        
        # Remove invalid rows
        initial_len = len(standardized)
        standardized = standardized.dropna(subset=['Level', 'Date'])
        if len(standardized) < initial_len:
            logger.warning(f"Dropped {initial_len - len(standardized)} rows with invalid data")
        
        return standardized
    
    def _parse_fwtw_dates(self, date_series: pd.Series) -> pd.Series:
        """Parse FWTW dates handling multiple formats."""
        if pd.api.types.is_datetime64_any_dtype(date_series):
            return date_series
        
        # Try different date formats
        formats_to_try = [
            '%Y%m%d',     # YYYYMMDD
            '%Y-%m-%d',   # YYYY-MM-DD
            '%m/%d/%Y',   # MM/DD/YYYY
            '%Y%m',       # YYYYMM (monthly)
            None          # Let pandas infer
        ]
        
        for fmt in formats_to_try:
            try:
                return pd.to_datetime(date_series, format=fmt)
            except:
                continue
        
        # Handle YYYYQQ format (e.g., "2024Q1")
        if date_series.astype(str).str.match(r'^\d{4}Q\d$').all():
            year = date_series.str[:4].astype(int)
            quarter = date_series.str[5:].astype(int)
            month = (quarter - 1) * 3 + 1
            return pd.to_datetime(year.astype(str) + '-' + month.astype(str).str.zfill(2) + '-01')
        
        # Handle YYYY:Q format (e.g., "2024:1")
        if date_series.astype(str).str.match(r'^\d{4}:\d$').all():
            year = date_series.str[:4].astype(int)
            quarter = date_series.str[5:].astype(int)
            month = (quarter - 1) * 3 + 1
            return pd.to_datetime(year.astype(str) + '-' + month.astype(str).str.zfill(2) + '-01')
        
        raise ValueError(f"Could not parse date format. Sample values: {date_series.head()}")
    
    def map_to_z1_series(self, fwtw_data: pd.DataFrame,
                         available_z1_series: Optional[Set[str]] = None,
                         include_all: bool = True) -> pd.DataFrame:
        """
        Map FWTW to Z.1 series with complete validation.
        
        Parameters:
        -----------
        include_all : bool
            If True, include ALL positions (no filtering)
        """
        # Standardize FWTW data first
        fwtw_clean = self.validate_and_standardize_fwtw(fwtw_data)
        
        mapped_positions = []
        self.unmapped_items = []
        
        for _, row in fwtw_clean.iterrows():
            holder_code = row['Holder Code']
            issuer_code = row['Issuer Code']
            instrument_code = row['Instrument Code']
            
            # Track unmapped items but don't skip them if include_all=True
            if holder_code not in self.valid_sectors:
                self.unmapped_items.append(('holder', holder_code))
                if not include_all:
                    continue
                    
            if issuer_code not in self.valid_sectors:
                self.unmapped_items.append(('issuer', issuer_code))
                if not include_all:
                    continue
            
            # Build Z.1 series codes
            asset_series = f"FA{holder_code}{instrument_code}005"
            liability_series = f"FL{issuer_code}{instrument_code}005"
            
            # Check availability if series list provided
            include_asset = True
            include_liability = True
            
            if available_z1_series and not include_all:
                include_asset = asset_series in available_z1_series
                include_liability = liability_series in available_z1_series
                
                if not include_asset and not include_liability:
                    continue
            
            mapped_positions.append({
                'date': row['Date'],
                'holder_code': holder_code,
                'issuer_code': issuer_code,
                'instrument_code': instrument_code,
                'asset_series': asset_series if include_asset else None,
                'liability_series': liability_series if include_liability else None,
                'level': float(row['Level']),
                'holder_name': row.get('Holder Name', f'Sector_{holder_code}'),
                'issuer_name': row.get('Issuer Name', f'Sector_{issuer_code}'),
                'instrument_name': row.get('Instrument Name', f'Inst_{instrument_code}')
            })
        
        if self.unmapped_items:
            unique_unmapped = set(self.unmapped_items)
            logger.warning(f"Found {len(unique_unmapped)} unmapped codes (included anyway): {list(unique_unmapped)[:5]}")
        
        return pd.DataFrame(mapped_positions)
