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
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ExternalDataLoader:
    """
    Loader for external (non-Fed) data sources
    """
    
    def __init__(self, cache_dir: str = "./data/external_cache"):
        """
        Initialize external data loader
        
        Parameters:
        -----------
        cache_dir : str
            Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_shiller_data(self, url: Optional[str] = None) -> pd.DataFrame:
        """
        Load Robert Shiller's S&P 500 data
        
        Parameters:
        -----------
        url : str, optional
            URL to Shiller data. If None, uses default URL
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with Date index and columns: P, D, E, CPI, Rate GS10
        """
        if url is None:
            url = "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls"
        
        cache_file = self.cache_dir / "shiller_sp500.pkl"
        
        # Check cache first
        if cache_file.exists() and self._is_cache_valid(cache_file):
            logger.info("Loading Shiller data from cache")
            return pd.read_pickle(cache_file)
        
        logger.info("Downloading Shiller S&P 500 data...")
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Read Excel file
                df = pd.read_excel(
                    BytesIO(response.content),
                    sheet_name='Data',
                    usecols="A:E,G",
                    header=7
                )
                
                # Clean and process data
                df = df.dropna(subset=['Date'])
                
                # Convert decimal year to proper dates
                df['Date'] = df['Date'].apply(self._decimal_year_to_month_end)
                df.set_index('Date', inplace=True)
                
                # Resample to quarterly
                df_quarterly = df.resample('Q').last()
                
                # Convert index to string format for consistency
                df_quarterly.index = df_quarterly.index.strftime('%Y-%m-%d')
                
                # Apply log transformation to price data (not rates)
                price_cols = ['P', 'D', 'E', 'CPI']
                for col in price_cols:
                    if col in df_quarterly.columns:
                        df_quarterly[col] = np.log(df_quarterly[col])
                
                # Save to cache
                df_quarterly.to_pickle(cache_file)
                
                logger.info(f"Loaded Shiller data: {df_quarterly.shape}")
                return df_quarterly
                
            else:
                logger.error(f"Failed to download Shiller data. Status code: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading Shiller data: {str(e)}")
            return pd.DataFrame()
    
    def load_dallas_fed_debt(self, url: Optional[str] = None) -> pd.DataFrame:
        """
        Load Dallas Fed government debt data
        
        Parameters:
        -----------
        url : str, optional
            URL to Dallas Fed data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with debt measures
        """
        if url is None:
            url = 'https://www.dallasfed.org/~/media/documents/research/govdebt.xlsx'
        
        cache_file = self.cache_dir / "dallas_fed_debt.pkl"
        
        # Check cache
        if cache_file.exists() and self._is_cache_valid(cache_file):
            logger.info("Loading Dallas Fed data from cache")
            return pd.read_pickle(cache_file)
        
        logger.info("Downloading Dallas Fed debt data...")
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Read Excel with multi-level headers
                df = pd.read_excel(BytesIO(response.content), header=[1, 2], index_col=0)
                
                # Process dates
                df.index = pd.to_datetime(df.index)
                df.index = df.index - pd.offsets.MonthEnd(1)
                df.index = df.index.strftime('%Y-%m-%d')
                
                # Flatten column names
                new_columns = [
                    'Par value Gross federal debt',
                    'Par value Privately held gross federal debt',
                    'Par value Marketable Treasury debt',
                    'Market value Gross federal debt',
                    'Market value Privately held gross federal debt',
                    'Market value Marketable Treasury debt'
                ]
                
                if len(df.columns) >= len(new_columns):
                    df.columns = new_columns[:len(df.columns)]
                
                # Apply log transformation
                df = np.log(df)
                
                # Save to cache
                df.to_pickle(cache_file)
                
                logger.info(f"Loaded Dallas Fed data: {df.shape}")
                return df
                
            else:
                logger.error(f"Failed to download Dallas Fed data. Status code: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading Dallas Fed data: {str(e)}")
            return pd.DataFrame()
    
    def load_fred_series(self, series_id: str) -> pd.DataFrame:
        """
        Load a single FRED series
        
        Parameters:
        -----------
        series_id : str
            FRED series ID (e.g., 'UNRATENSA', 'POPTHM')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with series data
        """
        cache_file = self.cache_dir / f"fred_{series_id}.pkl"
        
        # Check cache
        if cache_file.exists() and self._is_cache_valid(cache_file):
            logger.info(f"Loading FRED series {series_id} from cache")
            return pd.read_pickle(cache_file)
        
        logger.info(f"Downloading FRED series {series_id}...")
        
        # Try to scrape from FRED website
        url = f'https://fred.stlouisfed.org/data/{series_id}'
        
        try:
            # Use pandas to read HTML tables
            tables = pd.read_html(url)
            
            if tables:
                # The data table is usually the last one
                df = tables[-1]
                
                # Rename columns
                if len(df.columns) >= 2:
                    df.columns = ['DATE', series_id]
                    
                    # Convert date column
                    df['DATE'] = pd.to_datetime(df['DATE'])
                    df.set_index('DATE', inplace=True)
                    
                    # Resample to quarterly
                    df_quarterly = df.resample('Q').last()
                    
                    # Convert index to string
                    df_quarterly.index = df_quarterly.index.strftime('%Y-%m-%d')
                    
                    # Apply log transformation (except for rate series)
                    if not any(term in series_id.upper() for term in ['RATE', 'RIF', 'DFF', 'TB']):
                        df_quarterly = np.log(df_quarterly)
                    
                    # Save to cache
                    df_quarterly.to_pickle(cache_file)
                    
                    logger.info(f"Loaded FRED series {series_id}: {df_quarterly.shape}")
                    return df_quarterly
                    
            logger.warning(f"No data found for FRED series {series_id}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading FRED series {series_id}: {str(e)}")
            
            # Alternative: Use FRED API if available
            return self._load_fred_api(series_id)
    
    def load_gold_prices(self) -> pd.DataFrame:
        """
        Load LBMA gold prices
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with gold prices
        """
        url = "https://prices.lbma.org.uk/json/gold_am.json"
        cache_file = self.cache_dir / "gold_prices.pkl"
        
        # Check cache
        if cache_file.exists() and self._is_cache_valid(cache_file):
            logger.info("Loading gold prices from cache")
            return pd.read_pickle(cache_file)
        
        logger.info("Downloading gold prices...")
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Process dates
                df['d'] = pd.to_datetime(df['d'])
                df.set_index('d', inplace=True)
                df.index.name = 'DATE'
                
                # Extract USD price (first element)
                df['GOLD'] = df['v'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
                
                # Drop unnecessary columns
                df = df[['GOLD']]
                
                # Resample to quarterly
                df_quarterly = df.resample('Q').last()
                
                # Convert index to string
                df_quarterly.index = df_quarterly.index.strftime('%Y-%m-%d')
                
                # Apply log transformation
                df_quarterly = np.log(df_quarterly)
                
                # Save to cache
                df_quarterly.to_pickle(cache_file)
                
                logger.info(f"Loaded gold prices: {df_quarterly.shape}")
                return df_quarterly
                
            else:
                logger.error(f"Failed to download gold prices. Status code: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading gold prices: {str(e)}")
            return pd.DataFrame()
    
    def load_oil_prices(self) -> pd.DataFrame:
        """
        Load WTI oil prices from FRED
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with oil prices
        """
        return self.load_fred_series('WTISPLC')
    
    def load_multiple_fred_series(self, series_list: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load multiple FRED series
        
        Parameters:
        -----------
        series_list : List[str]
            List of FRED series IDs
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping series ID to DataFrame
        """
        results = {}
        
        for series_id in series_list:
            df = self.load_fred_series(series_id)
            if not df.empty:
                results[series_id] = df
                
        return results
    
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
                    combined = combined.join(df, how='outer')
        
        if combined is not None:
            # Sort by date
            combined = combined.sort_index()
            
            logger.info(f"Combined external data: {combined.shape}")
            
        return combined if combined is not None else pd.DataFrame()
    
    def _decimal_year_to_month_end(self, decimal_year: float) -> pd.Timestamp:
        """Convert decimal year to month end date"""
        year = int(decimal_year)
        month = round((decimal_year - year) * 100)
        
        # Handle edge cases
        if month == 0:
            month = 1
        elif month > 12:
            month = 12
            
        # Create date for first of month
        date = pd.Timestamp(year=year, month=month, day=1)
        
        # Adjust to end of month
        return date + pd.offsets.MonthEnd(0)
    
    def _is_cache_valid(self, cache_file: Path, days: int = 7) -> bool:
        """Check if cache file is still valid"""
        if not cache_file.exists():
            return False
            
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        
        return file_age.days < days
    
    def _load_fred_api(self, series_id: str) -> pd.DataFrame:
        """
        Alternative method to load FRED data via API
        Requires FRED API key
        """
        # This would require fred or fredapi package
        # Placeholder for API implementation
        logger.warning(f"FRED API not implemented. Could not load {series_id}")
        return pd.DataFrame()
    
    def clear_cache(self):
        """Clear all cached data"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")