# src/network/fwtw_loader.py
"""
FWTW Data Loader
Specialized loader for Flow of Funds Through Wall Street data
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class FWTWDataLoader:
    """
    Flow of Funds Through Wall Street (FWTW) Data Loader
    
    This class provides tools for downloading, caching, and processing FWTW data
    from the Federal Reserve's Flow of Funds dataset.
    """
    
    FWTW_URL = "https://www.federalreserve.gov/releases/efa/fwtw_data.csv"
    
    def __init__(self, cache_dir: str = "./data/cache/fwtw", cache_expiry_days: int = 7):
        """
        Initialize FWTW Data Loader
        
        Parameters:
        -----------
        cache_dir : str
            Directory for caching downloaded data
        cache_expiry_days : int
            Number of days before cache expires
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_days = cache_expiry_days
        self.metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid"""
        if cache_key not in self.metadata:
            return False
        
        cached_time = datetime.fromisoformat(self.metadata[cache_key]['timestamp'])
        expiry_time = cached_time + timedelta(days=self.cache_expiry_days)
        
        return datetime.now() < expiry_time
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.parquet"
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """Save data to cache"""
        cache_path = self._get_cache_path(cache_key)
        data.to_parquet(cache_path)
        
        self.metadata[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'shape': list(data.shape),
            'columns': list(data.columns)
        }
        self._save_metadata()
        
        logger.info(f"Saved {cache_key} to cache: {data.shape}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        if not self._is_cache_valid(cache_key):
            return None
        
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None
        
        try:
            data = pd.read_parquet(cache_path)
            logger.info(f"Loaded {cache_key} from cache: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading cache {cache_key}: {e}")
            return None
    
    def download_fwtw_data(self) -> pd.DataFrame:
        """
        Download FWTW data from Federal Reserve website
        
        Returns:
        --------
        pd.DataFrame
            Raw FWTW data
        """
        logger.info(f"Downloading FWTW data from {self.FWTW_URL}")
        
        try:
            response = requests.get(self.FWTW_URL, timeout=30)
            response.raise_for_status()
            
            # Parse CSV data
            data = pd.read_csv(StringIO(response.text))
            logger.info(f"Downloaded FWTW data: {data.shape}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading FWTW data: {e}")
            raise
    
    def process_fwtw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw FWTW data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw FWTW data
            
        Returns:
        --------
        pd.DataFrame
            Processed FWTW data
        """
        # Create copy to avoid modifying original
        processed = df.copy()
        
        # Convert date column
        processed['Date'] = pd.to_datetime(processed['Date'])
        
        # Convert level to numeric
        processed['Level'] = pd.to_numeric(processed['Level'], errors='coerce')
        
        # Add additional metadata columns
        processed['Year'] = processed['Date'].dt.year
        processed['Quarter'] = processed['Date'].dt.quarter
        
        # Calculate net positions
        processed['Net_Position'] = processed.groupby(['Date', 'Holder Name'])['Level'].transform('sum') - \
                                   processed.groupby(['Date', 'Issuer Name'])['Level'].transform('sum')
        
        # Remove any rows with invalid data
        processed = processed.dropna(subset=['Level'])
        
        logger.info(f"Processed FWTW data: {processed.shape}")
        
        return processed
    
    def load_fwtw_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Load FWTW data with caching support
        
        Parameters:
        -----------
        force_download : bool
            Force fresh download even if cache exists
            
        Returns:
        --------
        pd.DataFrame
            Processed FWTW data
        """
        cache_key = "fwtw_processed"
        
        # Try to load from cache first
        if not force_download:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Download and process fresh data
        raw_data = self.download_fwtw_data()
        processed_data = self.process_fwtw_data(raw_data)
        
        # Save to cache
        self._save_to_cache(processed_data, cache_key)
        
        return processed_data
    
    def get_unique_entities(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get unique entities from FWTW data
        
        Parameters:
        -----------
        data : pd.DataFrame
            FWTW data
            
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary with holders, issuers, and instruments
        """
        return {
            'holders': sorted(data['Holder Name'].unique()),
            'issuers': sorted(data['Issuer Name'].unique()),
            'instruments': sorted(data['Instrument Name'].unique()),
            'dates': sorted(data['Date'].unique())
        }
    
    def get_date_range(self, data: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get date range of data"""
        return data['Date'].min(), data['Date'].max()
    
    def filter_by_date(self, data: pd.DataFrame, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Filter data by date range
        
        Parameters:
        -----------
        data : pd.DataFrame
            FWTW data
        start_date : str, optional
            Start date (YYYY-MM-DD format)
        end_date : str, optional
            End date (YYYY-MM-DD format)
            
        Returns:
        --------
        pd.DataFrame
            Filtered data
        """
        filtered = data.copy()
        
        if start_date:
            filtered = filtered[filtered['Date'] >= pd.to_datetime(start_date)]
        
        if end_date:
            filtered = filtered[filtered['Date'] <= pd.to_datetime(end_date)]
        
        return filtered
    
    def filter_by_entities(self, data: pd.DataFrame,
                          holders: Optional[List[str]] = None,
                          issuers: Optional[List[str]] = None,
                          instruments: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter data by specific entities
        
        Parameters:
        -----------
        data : pd.DataFrame
            FWTW data
        holders : List[str], optional
            List of holder names to include
        issuers : List[str], optional
            List of issuer names to include
        instruments : List[str], optional
            List of instrument names to include
            
        Returns:
        --------
        pd.DataFrame
            Filtered data
        """
        filtered = data.copy()
        
        if holders:
            filtered = filtered[filtered['Holder Name'].isin(holders)]
        
        if issuers:
            filtered = filtered[filtered['Issuer Name'].isin(issuers)]
        
        if instruments:
            filtered = filtered[filtered['Instrument Name'].isin(instruments)]
        
        return filtered
    
    def aggregate_by_sector(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate flows by sector
        
        Parameters:
        -----------
        data : pd.DataFrame
            FWTW data
            
        Returns:
        --------
        pd.DataFrame
            Aggregated data by sector
        """
        # Group by date and holder/issuer
        holder_flows = data.groupby(['Date', 'Holder Name'])['Level'].sum().reset_index()
        holder_flows.columns = ['Date', 'Sector', 'Outflow']
        
        issuer_flows = data.groupby(['Date', 'Issuer Name'])['Level'].sum().reset_index()
        issuer_flows.columns = ['Date', 'Sector', 'Inflow']
        
        # Merge inflows and outflows
        sector_flows = pd.merge(holder_flows, issuer_flows, on=['Date', 'Sector'], how='outer')
        sector_flows = sector_flows.fillna(0)
        
        # Calculate net flow
        sector_flows['Net_Flow'] = sector_flows['Inflow'] - sector_flows['Outflow']
        
        return sector_flows
    
    def calculate_network_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic network metrics for each date
        
        Parameters:
        -----------
        data : pd.DataFrame
            FWTW data
            
        Returns:
        --------
        pd.DataFrame
            Network metrics by date
        """
        metrics = []
        
        for date in data['Date'].unique():
            date_data = data[data['Date'] == date]
            
            metric = {
                'Date': date,
                'num_holders': date_data['Holder Name'].nunique(),
                'num_issuers': date_data['Issuer Name'].nunique(),
                'num_instruments': date_data['Instrument Name'].nunique(),
                'num_transactions': len(date_data),
                'total_volume': date_data['Level'].sum(),
                'avg_transaction_size': date_data['Level'].mean(),
                'max_transaction': date_data['Level'].max(),
                'min_transaction': date_data['Level'].min()
            }
            
            metrics.append(metric)
        
        return pd.DataFrame(metrics).sort_values('Date')
    
    def get_top_flows(self, data: pd.DataFrame, 
                     date: Optional[str] = None,
                     top_n: int = 10) -> pd.DataFrame:
        """
        Get top flows by size
        
        Parameters:
        -----------
        data : pd.DataFrame
            FWTW data
        date : str, optional
            Specific date to filter (if None, uses latest date)
        top_n : int
            Number of top flows to return
            
        Returns:
        --------
        pd.DataFrame
            Top flows
        """
        if date:
            filtered = data[data['Date'] == pd.to_datetime(date)]
        else:
            # Use latest date
            latest_date = data['Date'].max()
            filtered = data[data['Date'] == latest_date]
        
        return filtered.nlargest(top_n, 'Level')[
            ['Holder Name', 'Issuer Name', 'Instrument Name', 'Level']
        ]