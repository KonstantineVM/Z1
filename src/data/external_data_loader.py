"""
External Data Loader
Handles loading of non-Fed data sources including market data, FRED series, etc.
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from io import BytesIO, StringIO
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


class ExternalDataLoader:
    """
    Loader for external (non-Fed) data sources with caching support
    """
    
    def __init__(self, cache_directory: str = "./data/cache/external", 
                 cache_expiry_days: int = 7):
        """
        Initialize external data loader
        
        Parameters:
        -----------
        cache_directory : str
            Directory to cache downloaded data
        cache_expiry_days : int
            Number of days before cache expires
        """
        self.cache_directory = Path(cache_directory)
        self.cache_expiry_days = cache_expiry_days
        
        # Create cache subdirectories
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        (self.cache_directory / 'metadata').mkdir(exist_ok=True)
        
    def _get_cache_path(self, source: str) -> Path:
        """Get path for cached file"""
        return self.cache_directory / f"{source}.parquet"
    
    def _get_metadata_path(self, source: str) -> Path:
        """Get path for metadata file"""
        return self.cache_directory / 'metadata' / f"{source}.json"
    
    def _is_cache_valid(self, source: str) -> bool:
        """Check if cache is still valid"""
        cache_path = self._get_cache_path(source)
        metadata_path = self._get_metadata_path(source)
        
        if not cache_path.exists() or not metadata_path.exists():
            return False
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            cached_at = datetime.fromisoformat(metadata['cached_at'])
            age = datetime.now() - cached_at
            
            return age.days < self.cache_expiry_days
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    def _save_to_cache(self, data: pd.DataFrame, source: str):
        """Save data to cache"""
        cache_path = self._get_cache_path(source)
        metadata_path = self._get_metadata_path(source)
        
        try:
            # Save data
            data.to_parquet(cache_path, compression='snappy')
            
            # Save metadata
            metadata = {
                'source': source,
                'cached_at': datetime.now().isoformat(),
                'shape': data.shape,
                'columns': list(data.columns)[:10],  # First 10 columns as sample
                'date_range': [str(data.index.min()), str(data.index.max())] if not data.empty else []
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Cached {source} data: {data.shape}")
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _load_from_cache(self, source: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        if not self._is_cache_valid(source):
            return None
            
        cache_path = self._get_cache_path(source)
        
        try:
            data = pd.read_parquet(cache_path)
            logger.info(f"Loaded {source} from cache: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None
    
    def get_cache_info(self) -> Dict[str, Dict]:
        """Get information about cached data"""
        cache_info = {}
        
        # Check all parquet files in cache directory
        for cache_file in self.cache_directory.glob("*.parquet"):
            source = cache_file.stem
            metadata_path = self._get_metadata_path(source)
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    cached_at = datetime.fromisoformat(metadata['cached_at'])
                    age = datetime.now() - cached_at
                    
                    cache_info[source] = {
                        'cached_at': metadata['cached_at'],
                        'valid': age.days < self.cache_expiry_days,
                        'shape': tuple(metadata['shape']),
                        'size_bytes': cache_file.stat().st_size
                    }
                except Exception as e:
                    logger.error(f"Error reading metadata for {source}: {e}")
                    
        return cache_info
    
    def clear_cache(self, source: Optional[str] = None):
        """Clear cache for specific source or all sources"""
        if source:
            # Clear specific source
            cache_path = self._get_cache_path(source)
            metadata_path = self._get_metadata_path(source)
            
            if cache_path.exists():
                cache_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
                
            logger.info(f"Cleared cache for {source}")
        else:
            # Clear all cache
            for cache_file in self.cache_directory.glob("*.parquet"):
                cache_file.unlink()
            for metadata_file in (self.cache_directory / 'metadata').glob("*.json"):
                metadata_file.unlink()
                
            logger.info("Cleared all external data cache")
    
    def load_shiller_data(self, url: Optional[str] = None, 
                         force_download: bool = False) -> pd.DataFrame:
        """
        Load Robert Shiller's S&P 500 data
        
        Parameters:
        -----------
        url : str, optional
            URL to Shiller data
        force_download : bool
            Force fresh download even if cache exists
            
        Returns:
        --------
        pd.DataFrame
            S&P 500 data
        """
        source = 'shiller_sp500'
        
        # Try loading from cache first
        if not force_download:
            cached_data = self._load_from_cache(source)
            if cached_data is not None:
                return cached_data
        
        # Download fresh data
        if url is None:
            url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
            
        logger.info(f"Downloading Shiller data from {url}")
        
        try:
            # Read Excel file
            df = pd.read_excel(url, sheet_name='Data', skiprows=7)
            
            # Process columns
            df = df.rename(columns={
                'Date': 'date',
                'P': 'price',
                'D': 'dividend',
                'E': 'earnings',
                'CPI': 'cpi',
                'Long Rate': 'long_rate',
                'Real Price': 'real_price',
                'Real Dividend': 'real_dividend',
                'Real Earnings': 'real_earnings',
                'CAPE': 'cape'
            })
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m', errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.set_index('date')
            
            # Save to cache
            self._save_to_cache(df, source)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Shiller data: {e}")
            return pd.DataFrame()
    
    def load_dallas_fed_debt(self, url: Optional[str] = None,
                            force_download: bool = False) -> pd.DataFrame:
        """
        Load Dallas Fed International House Price Database
        """
        source = 'dallas_fed_debt'
        
        # Try loading from cache first
        if not force_download:
            cached_data = self._load_from_cache(source)
            if cached_data is not None:
                return cached_data
        
        if url is None:
            url = "https://www.dallasfed.org/-/media/documents/institute/houseprice/hp2401.xlsx"
            
        logger.info(f"Downloading Dallas Fed data from {url}")
        
        try:
            # Read Excel file
            df = pd.read_excel(url, sheet_name='hp2401', skiprows=5)
            
            # Process the data (this is a placeholder - adjust based on actual data structure)
            df['date'] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index('date')
            df = df.iloc[:, 1:]  # Remove first column after setting as index
            
            # Save to cache
            self._save_to_cache(df, source)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Dallas Fed data: {e}")
            return pd.DataFrame()
    
    def load_gold_prices(self, url: Optional[str] = None,
                        force_download: bool = False) -> pd.DataFrame:
        """
        Load historical gold prices
        """
        source = 'gold_prices'
        
        # Try loading from cache first
        if not force_download:
            cached_data = self._load_from_cache(source)
            if cached_data is not None:
                return cached_data
        
        # This is a placeholder - you'll need to implement actual gold price loading
        logger.warning("Gold price loading not fully implemented")
        return pd.DataFrame()
    
    def load_fred_series(self, series_id: str, 
                        force_download: bool = False) -> pd.DataFrame:
        """
        Load FRED series data
        
        Parameters:
        -----------
        series_id : str
            FRED series ID
        force_download : bool
            Force fresh download
            
        Returns:
        --------
        pd.DataFrame
            FRED series data
        """
        source = f'fred_{series_id}'
        
        # Try loading from cache first
        if not force_download:
            cached_data = self._load_from_cache(source)
            if cached_data is not None:
                return cached_data
        
        # Placeholder for FRED API implementation
        logger.warning(f"FRED API not implemented. Could not load {series_id}")
        return pd.DataFrame()
    
    def combine_external_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple external data sources into single DataFrame
        
        Parameters:
        -----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary of DataFrames to combine
            
        Returns:
        --------
        pd.DataFrame
            Combined DataFrame
        """
        if not data_dict:
            return pd.DataFrame()
        
        # Start with first non-empty DataFrame
        combined = None
        for name, df in data_dict.items():
            if not df.empty:
                if combined is None:
                    combined = df.copy()
                else:
                    # Join on index
                    combined = combined.join(df, how='outer', rsuffix=f'_{name}')
        
        if combined is not None:
            # Sort by date
            combined = combined.sort_index()
            
            logger.info(f"Combined external data: {combined.shape}")
            
        return combined if combined is not None else pd.DataFrame()
