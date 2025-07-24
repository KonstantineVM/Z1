"""
Federal Reserve Data Loader
Handles loading and parsing of various Fed data sources
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

logger = logging.getLogger(__name__)


class FedDataLoader:
    """Unified loader for Federal Reserve economic data"""
    
    BASE_URL = "https://www.federalreserve.gov/DataDownload/Output.aspx"
    
    # Namespace mappings for different data sources
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
        'H8': {
            'message': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message',
            'common': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common',
            'compact': 'http://www.federalreserve.gov/structure/compact/common',
            'h8': 'http://www.federalreserve.gov/structure/compact/H8_H8'
        },
        'G17': {
            'message': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message',
            'common': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common',
            'compact': 'http://www.federalreserve.gov/structure/compact/common',
            'g17_cap': 'http://www.federalreserve.gov/structure/compact/G17_CAP',
            'g17_caputil': 'http://www.federalreserve.gov/structure/compact/G17_CAPUTL',
            'g17_diff': 'http://www.federalreserve.gov/structure/compact/G17_DIFF',
            'g17_gvip': 'http://www.federalreserve.gov/structure/compact/G17_GVIP',
            'g17_ip_durable': 'http://www.federalreserve.gov/structure/compact/G17_IP_DURABLE_GOODS_DETAIL',
            'g17_ip_gross': 'http://www.federalreserve.gov/structure/compact/G17_IP_GROSS_VALUE_STAGE_OF_PROCESS_GROUPS',
            'g17_ip_major': 'http://www.federalreserve.gov/structure/compact/G17_IP_MAJOR_INDUSTRY_GROUPS',
            'g17_market': 'http://www.federalreserve.gov/structure/compact/G17_IP_MARKET_GROUPS',
            'g17_mining': 'http://www.federalreserve.gov/structure/compact/G17_IP_MINING_AND_UTILITY_DETAIL',
            'g17_nondurable': 'http://www.federalreserve.gov/structure/compact/G17_IP_NONDURABLE_GOODS_DETAIL',
            'g17_special': 'http://www.federalreserve.gov/structure/compact/G17_IP_SPECIAL_AGGREGATES',
            'g17_kw': 'http://www.federalreserve.gov/structure/compact/G17_KW',
            'g17_mva': 'http://www.federalreserve.gov/structure/compact/G17_MVA',
            'g17_riw': 'http://www.federalreserve.gov/structure/compact/G17_RIW'
        },
        'G19': {
            'message': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message',
            'common': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common',
            'compact': 'http://www.federalreserve.gov/structure/compact/common',
            'g19': 'http://www.federalreserve.gov/structure/compact/G19_CCOUT'
        },
        'G20': {
            'message': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message',
            'common': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common',
            'compact': 'http://www.federalreserve.gov/structure/compact/common',
            'g20_owned': 'http://www.federalreserve.gov/structure/compact/G20_OWNED',
            'g20_hist': 'http://www.federalreserve.gov/structure/compact/G20_HIST',
            'g20_terms': 'http://www.federalreserve.gov/structure/compact/G20_TERMS'
        },
        'H15': {
            'message': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message',
            'common': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common',
            'compact': 'http://www.federalreserve.gov/structure/compact/common',
            'h15': 'http://www.federalreserve.gov/structure/compact/H15_H15'
        }
    }
    
    def __init__(self, base_directory: str = "./data/fed_data", 
                 start_year: int = 1959, 
                 end_year: int = 2024):
        """
        Initialize the Fed Data Loader
        
        Parameters:
        -----------
        base_directory : str
            Directory to store downloaded data
        start_year : int
            Start year for data extraction
        end_year : int
            End year for data extraction
        """
        self.base_directory = base_directory
        self.start_year = start_year
        self.end_year = end_year
        self._ensure_directory_exists()
        self._generate_quarterly_dates()
        
    def _ensure_directory_exists(self):
        """Create base directory if it doesn't exist"""
        os.makedirs(self.base_directory, exist_ok=True)
        
    def _generate_quarterly_dates(self):
        """Generate quarterly date strings"""
        self.quarterly_dates = [
            f"{year}-{month:02d}-{'30' if month in [6, 9] else '31'}"
            for year in range(self.start_year, self.end_year + 1) 
            for month in (3, 6, 9, 12)
        ]
        # Remove last 3 quarters if needed (based on original notebook)
        self.quarterly_dates = self.quarterly_dates[:-3]
        
    def download_and_extract(self, rel_list: List[str]) -> Dict[str, str]:
        """
        Download and extract Fed data files
        
        Parameters:
        -----------
        rel_list : List[str]
            List of data source identifiers (e.g., ['Z1', 'H6'])
            
        Returns:
        --------
        Dict[str, str]
            Mapping of source identifier to extracted directory path
        """
        extracted_paths = {}
        
        for rel in rel_list:
            url = f"{self.BASE_URL}?rel={rel}&filetype=zip"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    # Extract filename from headers or use default
                    content_disposition = response.headers.get('Content-Disposition')
                    if content_disposition and 'filename=' in content_disposition:
                        filename = content_disposition.split('filename=')[-1].strip('"')
                    else:
                        filename = f"{rel}.zip"
                    
                    # Create subdirectory for extracted files
                    archive_name = os.path.splitext(filename)[0]
                    extract_directory = os.path.join(self.base_directory, archive_name)
                    
                    if not os.path.exists(extract_directory):
                        os.makedirs(extract_directory)
                    
                    # Extract ZIP file
                    zip_file = ZipFile(BytesIO(response.content))
                    zip_file.extractall(extract_directory)
                    
                    extracted_paths[rel] = extract_directory
                    logger.info(f"Successfully extracted {rel} to {extract_directory}")
                else:
                    logger.error(f"Failed to download {rel}. Status code: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error processing {rel}: {str(e)}")
                
        return extracted_paths
    
    def parse_xml_data(self, file_path: str, source_type: str) -> pd.DataFrame:
        """
        Parse XML data file based on source type
        
        Parameters:
        -----------
        file_path : str
            Path to XML file
        source_type : str
            Type of data source (e.g., 'Z1', 'H6')
            
        Returns:
        --------
        pd.DataFrame
            Parsed data with series as columns and dates as index
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        namespaces = self.NAMESPACES.get(source_type, {})
        all_series_data = []
        
        # Handle different namespace structures
        if source_type == 'Z1':
            series_elements = root.findall('.//z1:Series', namespaces)
            self._parse_series_elements(series_elements, namespaces, all_series_data, 'z1')
            
        elif source_type == 'H6':
            for dataset in ['h6_m1', 'h6_m2', 'h6_mbase', 'h6_memo']:
                series_elements = root.findall(f'.//{dataset}:Series', namespaces)
                self._parse_series_elements(series_elements, namespaces, all_series_data, dataset)
                
        elif source_type == 'H8':
            series_elements = root.findall('.//h8:Series', namespaces)
            self._parse_series_elements(series_elements, namespaces, all_series_data, 'h8')
            
        elif source_type == 'G17':
            for dataset in ['g17_cap', 'g17_caputil', 'g17_diff', 'g17_gvip', 'g17_ip_durable',
                           'g17_ip_gross', 'g17_ip_major', 'g17_market', 'g17_mining',
                           'g17_nondurable', 'g17_special', 'g17_kw', 'g17_mva', 'g17_riw']:
                series_elements = root.findall(f'.//{dataset}:Series', namespaces)
                self._parse_series_elements(series_elements, namespaces, all_series_data, dataset)
                
        elif source_type == 'G19':
            series_elements = root.findall('.//g19:Series', namespaces)
            self._parse_series_elements(series_elements, namespaces, all_series_data, 'g19')
            
        elif source_type == 'G20':
            for dataset in ['g20_owned', 'g20_hist', 'g20_terms']:
                series_elements = root.findall(f'.//{dataset}:Series', namespaces)
                self._parse_series_elements(series_elements, namespaces, all_series_data, dataset)
                
        elif source_type == 'H15':
            series_elements = root.findall('.//h15:Series', namespaces)
            self._parse_series_elements(series_elements, namespaces, all_series_data, 'h15')
        
        return pd.DataFrame(all_series_data)
    
    def _parse_series_elements(self, series_elements, namespaces, all_series_data, dataset_key):
        """Helper method to parse series elements"""
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
    
    def extract_series_descriptions(self, file_path: str, source_type: str) -> pd.DataFrame:
        """
        Extract series names and descriptions from XML
        
        Parameters:
        -----------
        file_path : str
            Path to XML file
        source_type : str
            Type of data source
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with SERIES_NAME and Long Description columns
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        data = []
        namespaces = self.NAMESPACES.get(source_type, {})
        
        # Find all series based on source type
        if source_type == 'Z1':
            series_xpath = './/{http://www.federalreserve.gov/structure/compact/Z1_Z1}Series'
        else:
            # Generic approach for other sources
            series_xpath = './/Series'
            
        for series in root.findall(series_xpath):
            series_name = series.get('SERIES_NAME')
            
            # Find Long Description annotation
            long_description = None
            for annotation in series.findall('.//{http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common}Annotation'):
                annotation_type = annotation.find('{http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common}AnnotationType')
                if annotation_type is not None and annotation_type.text == 'Long Description':
                    annotation_text = annotation.find('{http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common}AnnotationText')
                    if annotation_text is not None:
                        long_description = annotation_text.text
                        break
            
            data.append((series_name, long_description))
            
        return pd.DataFrame(data, columns=['SERIES_NAME', 'Long Description'])
    
    def load_multiple_sources(self, sources: List[str], 
                             download: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load data from multiple Fed sources
        
        Parameters:
        -----------
        sources : List[str]
            List of source identifiers
        download : bool
            Whether to download data first
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping source to its DataFrame
        """
        if download:
            paths = self.download_and_extract(sources)
        
        data_dict = {}
        
        for source in sources:
            # Find XML file in source directory
            source_dir = os.path.join(self.base_directory, f"FRB_{source}")
            if os.path.exists(source_dir):
                xml_files = [f for f in os.listdir(source_dir) if f.endswith('.xml')]
                if xml_files:
                    xml_path = os.path.join(source_dir, xml_files[0])
                    data_dict[source] = self.parse_xml_data(xml_path, source)
                    logger.info(f"Loaded {source} data: {data_dict[source].shape}")
                    
        return data_dict