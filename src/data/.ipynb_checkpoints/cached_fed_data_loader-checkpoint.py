"""
Federal Reserve Data Loader with Caching Support
"""

import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from zipfile import ZipFile
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class CachedFedDataLoader:
    """Unified loader for Federal Reserve economic data with caching support"""
    
    BASE_URL = "https://www.federalreserve.gov/DataDownload/Output.aspx"
    
    # Namespace mappings (same as before)
    NAMESPACES = {
        'Z1': {
            'message': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message',
            'common': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common',
            'compact': 'http://www.federalreserve.gov/structure/compact/common',
            'z1': 'http://www.federalreserve.gov/structure/compact/Z1_Z1'
        },
        'H6': {
            'message': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message',
            'common': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common',
            'compact': 'http://www.federalreserve.gov/structure/compact/common',
            'h6_m1': 'http://www.federalreserve.gov/structure/compact/H6_H6_M1',
            'h6_m2': 'http://www.federalreserve.gov/structure/compact/H6_H6_M2',
            'h6_mbase': 'http://www.federalreserve.gov/structure/compact/H6_H6_MBASE',
            'h6_memo': 'http://www.federalreserve.gov/structure/compact/H6_H6_MEMO'
        },
        # ... other namespaces remain the same
    }
    
    def __init__(self, base_directory: str = "./data/fed_data", 
                 cache_directory: str = "./data/cache",
                 start_year: int = 1959, 
                 end_year: int = 2024,
                 cache_expiry_days: int = 7):
        """
        Initialize the Fed Data Loader with caching
        
        Parameters:
        -----------
        base_directory : str
            Directory to store downloaded data
        cache_directory : str
            Directory to store cached processed data
        start_year : int
            Start year for data extraction
        end_year : int
            End year for data extraction
        cache_expiry_days : int
            Number of days before cache expires
        """
        self.base_directory = base_directory
        self.cache_directory = cache_directory
        self.start_year = start_year
        self.end_year = end_year
        self.cache_expiry_days = cache_expiry_days
        
        self._ensure_directories_exist()
        self._generate_quarterly_dates()
        
    def _ensure_directories_exist(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.base_directory, exist_ok=True)
        os.makedirs(self.cache_directory, exist_ok=True)
        os.makedirs(os.path.join(self.cache_directory, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.cache_directory, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(self.cache_directory, 'metadata'), exist_ok=True)
        
    def _get_cache_key(self, source: str, data_type: str = 'processed') -> str:
        """Generate cache key for a data source"""
        key_parts = [source, str(self.start_year), str(self.end_year), data_type]
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, source: str, data_type: str = 'processed') -> Path:
        """Get path for cached file"""
        cache_key = self._get_cache_key(source, data_type)
        return Path(self.cache_directory) / data_type / f"{source}_{cache_key}.parquet"
    
    def _get_metadata_path(self, source: str) -> Path:
        """Get path for metadata file"""
        return Path(self.cache_directory) / 'metadata' / f"{source}_metadata.json"
    
    def _is_cache_valid(self, source: str, data_type: str = 'processed') -> bool:
        """Check if cached data is still valid"""
        cache_path = self._get_cache_path(source, data_type)
        metadata_path = self._get_metadata_path(source)
        
        if not cache_path.exists() or not metadata_path.exists():
            return False
            
        # Check metadata
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Check if cache has expired
            cached_time = datetime.fromisoformat(metadata['cached_at'])
            expiry_time = cached_time + timedelta(days=self.cache_expiry_days)
            
            if datetime.now() > expiry_time:
                logger.info(f"Cache expired for {source}")
                return False
                
            # Check if parameters match
            if (metadata.get('start_year') != self.start_year or 
                metadata.get('end_year') != self.end_year):
                logger.info(f"Cache parameters mismatch for {source}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    def _save_to_cache(self, data: pd.DataFrame, source: str, 
                      data_type: str = 'processed'):
        """Save data to cache"""
        cache_path = self._get_cache_path(source, data_type)
        metadata_path = self._get_metadata_path(source)
        
        try:
            # Save data
            data.to_parquet(cache_path, compression='snappy')
            
            # Save metadata
            metadata = {
                'source': source,
                'data_type': data_type,
                'cached_at': datetime.now().isoformat(),
                'start_year': self.start_year,
                'end_year': self.end_year,
                'shape': data.shape,
                'columns': list(data.columns)[:10]  # First 10 columns as sample
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Cached {source} data: {data.shape}")
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _load_from_cache(self, source: str, data_type: str = 'processed') -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        if not self._is_cache_valid(source, data_type):
            return None
            
        cache_path = self._get_cache_path(source, data_type)
        
        try:
            data = pd.read_parquet(cache_path)
            logger.info(f"Loaded {source} from cache: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None
    
    def _download_if_needed(self, source: str) -> Optional[str]:
        """Download data if not already present or if forced"""
        # Check if raw file exists and is recent
        raw_file_pattern = os.path.join(self.base_directory, f"FRB_{source}", "*.xml")
        existing_files = list(Path(self.base_directory).glob(f"FRB_{source}/*.xml"))
        
        if existing_files:
            # Check file age
            file_stat = os.stat(existing_files[0])
            file_age_days = (datetime.now() - datetime.fromtimestamp(file_stat.st_mtime)).days
            
            if file_age_days < self.cache_expiry_days:
                logger.info(f"Using existing file for {source} (age: {file_age_days} days)")
                return str(existing_files[0])
        
        # Download new data
        logger.info(f"Downloading fresh data for {source}")
        url = f"{self.BASE_URL}?rel={source}&filetype=zip"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Extract filename from headers or use default
                content_disposition = response.headers.get('Content-Disposition')
                if content_disposition and 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[-1].strip('"')
                else:
                    filename = f"{source}.zip"
                
                # Create subdirectory for extracted files
                archive_name = os.path.splitext(filename)[0]
                extract_directory = os.path.join(self.base_directory, archive_name)
                
                if not os.path.exists(extract_directory):
                    os.makedirs(extract_directory)
                
                # Extract ZIP file
                zip_file = ZipFile(BytesIO(response.content))
                zip_file.extractall(extract_directory)
                
                # Find XML file
                xml_files = list(Path(extract_directory).glob("*.xml"))
                if xml_files:
                    return str(xml_files[0])
                    
            else:
                logger.error(f"Failed to download {source}. Status code: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error downloading {source}: {str(e)}")
            
        return None
    
    def load_single_source(self, source: str, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Load a single data source with caching
        
        Parameters:
        -----------
        source : str
            Source identifier (e.g., 'Z1', 'H6')
        force_download : bool
            Force fresh download even if cache exists
            
        Returns:
        --------
        pd.DataFrame or None
            Parsed data
        """
        # Try to load from cache first
        if not force_download:
            cached_data = self._load_from_cache(source)
            if cached_data is not None:
                return cached_data
        
        # Download/locate raw file
        xml_path = self._download_if_needed(source)
        if not xml_path:
            return None
            
        # Parse data
        data = self.parse_xml_data(xml_path, source)
        
        # Save to cache
        if data is not None and not data.empty:
            self._save_to_cache(data, source)
            
        return data
    
    def load_multiple_sources(self, sources: List[str], 
                            force_download: bool = False,
                            parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load data from multiple Fed sources with caching
        
        Parameters:
        -----------
        sources : List[str]
            List of source identifiers
        force_download : bool
            Force fresh download even if cache exists
        parallel : bool
            Whether to load sources in parallel
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping source to its DataFrame
        """
        data_dict = {}
        
        if parallel and len(sources) > 1:
            # Load in parallel
            with ThreadPoolExecutor(max_workers=min(4, len(sources))) as executor:
                futures = {
                    executor.submit(self.load_single_source, source, force_download): source
                    for source in sources
                }
                
                for future in futures:
                    source = futures[future]
                    try:
                        data = future.result()
                        if data is not None:
                            data_dict[source] = data
                    except Exception as e:
                        logger.error(f"Error loading {source}: {e}")
        else:
            # Load sequentially
            for source in sources:
                data = self.load_single_source(source, force_download)
                if data is not None:
                    data_dict[source] = data
                    
        return data_dict
    
    def clear_cache(self, source: Optional[str] = None):
        """
        Clear cached data
        
        Parameters:
        -----------
        source : str, optional
            Specific source to clear, or None to clear all
        """
        if source:
            # Clear specific source
            for data_type in ['raw', 'processed']:
                cache_path = self._get_cache_path(source, data_type)
                if cache_path.exists():
                    cache_path.unlink()
                    
            metadata_path = self._get_metadata_path(source)
            if metadata_path.exists():
                metadata_path.unlink()
                
            logger.info(f"Cleared cache for {source}")
        else:
            # Clear all cache
            import shutil
            shutil.rmtree(self.cache_directory)
            self._ensure_directories_exist()
            logger.info("Cleared all cache")
    
    def get_cache_info(self) -> Dict[str, Dict]:
        """Get information about cached data"""
        cache_info = {}
        
        metadata_dir = Path(self.cache_directory) / 'metadata'
        for metadata_file in metadata_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    source = metadata['source']
                    cache_info[source] = {
                        'cached_at': metadata['cached_at'],
                        'shape': metadata['shape'],
                        'valid': self._is_cache_valid(source)
                    }
            except Exception as e:
                logger.error(f"Error reading metadata: {e}")
                
        return cache_info
    
    # Include all the parsing methods from the original loader
    def _generate_quarterly_dates(self):
        """Generate quarterly date strings"""
        self.quarterly_dates = [
            f"{year}-{month:02d}-{'30' if month in [6, 9] else '31'}"
            for year in range(self.start_year, self.end_year + 1) 
            for month in (3, 6, 9, 12)
        ]
        # Remove last 3 quarters if needed
        self.quarterly_dates = self.quarterly_dates[:-3]
    
    def parse_xml_data(self, file_path: str, source_type: str) -> pd.DataFrame:
        """Parse XML data file based on source type (same as original)"""
        # Implementation remains the same as in the original fed_data_loader.py
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        namespaces = self.NAMESPACES.get(source_type, {})
        all_series_data = []
        
        # Handle different namespace structures (same implementation as before)
        if source_type == 'Z1':
            series_elements = root.findall('.//z1:Series', namespaces)
            self._parse_series_elements(series_elements, namespaces, all_series_data, 'z1')
        # ... (rest of the implementation remains the same)
        
        return pd.DataFrame(all_series_data)
    
    def _parse_series_elements(self, series_elements, namespaces, all_series_data, dataset_key):
        """Helper method to parse series elements (same as original)"""
        for series in series_elements:
            series_attributes = series.attrib
            observations = {date: None for date in self.quarterly_dates}
            
            for obs in series.findall('.//compact:Obs', namespaces):
                time_period = obs.get('TIME_PERIOD')
                obs_value = obs.get('OBS_VALUE')
                if time_period in observations:
                    observations[time_period] = obs_value
            
            series_data = {**series_attributes, **observations}
            all_series_data.append(series_data)