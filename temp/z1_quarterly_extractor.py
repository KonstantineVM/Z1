"""
Extract and prepare all quarterly Z1 series for the Comprehensive Aligned 
Sum-Constrained Unobserved Components System
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import requests
from zipfile import ZipFile
from io import BytesIO
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Z1QuarterlyExtractor:
    """
    Extract all quarterly Z1 series and prepare them for UC analysis
    """
    
    def __init__(self, 
                 output_dir: str = "./data",
                 start_year: int = 1952,
                 end_year: int = 2024):
        """
        Initialize extractor
        
        Parameters:
        -----------
        output_dir : str
            Directory to save parquet files
        start_year : int
            Start year for data extraction
        end_year : int
            End year for data extraction
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_year = start_year
        self.end_year = end_year
        
        # Z1 specific namespaces
        self.namespaces = {
            'message': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message',
            'common': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common',
            'compact': 'http://www.federalreserve.gov/structure/compact/common',
            'z1': 'http://www.federalreserve.gov/structure/compact/Z1_Z1'
        }
        
        # Generate quarterly dates
        self.quarterly_dates = self._generate_quarterly_dates()
        
    def _generate_quarterly_dates(self) -> List[str]:
        """Generate list of quarterly dates"""
        dates = []
        for year in range(self.start_year, self.end_year + 1):
            for month in [3, 6, 9, 12]:
                day = '31' if month in [3, 12] else '30'
                dates.append(f"{year}-{month:02d}-{day}")
        return dates
    
    def download_z1_data(self, force_download: bool = False) -> str:
        """
        Download latest Z1 data from Federal Reserve
        
        Parameters:
        -----------
        force_download : bool
            Force download even if file exists
            
        Returns:
        --------
        str
            Path to extracted XML file
        """
        xml_path = self.output_dir / "Z1_data.xml"
        
        if xml_path.exists() and not force_download:
            logger.info(f"Using existing Z1 data: {xml_path}")
            return str(xml_path)
        
        logger.info("Downloading Z1 data from Federal Reserve...")
        
        # Fed data download URL
        url = "https://www.federalreserve.gov/DataDownload/Output.aspx"
        params = {
            'rel': 'Z1',
            'filetype': 'zip'
        }
        
        # Download ZIP file
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Extract XML from ZIP
        with ZipFile(BytesIO(response.content)) as zip_file:
            # Find XML file (could be Z1_data.xml or FRB_Z1.xml)
            xml_files = [f for f in zip_file.namelist() if f.endswith('.xml')]
            if not xml_files:
                raise ValueError("No XML file found in Z1 ZIP")
            
            # Extract the first XML file
            zip_file.extract(xml_files[0], self.output_dir)
            extracted_path = self.output_dir / xml_files[0]
            
            # Rename to standard name if needed
            if extracted_path != xml_path:
                extracted_path.rename(xml_path)
        
        logger.info(f"Downloaded and extracted Z1 data to: {xml_path}")
        return str(xml_path)
    
    def extract_quarterly_series(self, xml_path: str) -> pd.DataFrame:
        """
        Extract all quarterly series from Z1 XML
        
        Parameters:
        -----------
        xml_path : str
            Path to Z1 XML file
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with metadata and all quarterly series
        """
        logger.info("Parsing Z1 XML file...")
        
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        all_series_data = []
        quarterly_count = 0
        total_count = 0
        
        # Find all series
        for series in root.findall('.//z1:Series', self.namespaces):
            total_count += 1
            series_attributes = series.attrib
            
            # Check if it's quarterly (FREQ='162' or Q in series name)
            freq = series_attributes.get('FREQ', '')
            series_name = series_attributes.get('SERIES_NAME', '')
            
            is_quarterly = (
                freq == '162' or 
                series_name.endswith('.Q') or
                'quarterly' in series_attributes.get('SERIES_DESC', '').lower()
            )
            
            if is_quarterly:
                quarterly_count += 1
                
                # Initialize observations with all dates
                observations = {date: None for date in self.quarterly_dates}
                
                # Extract observations
                for obs in series.findall('.//compact:Obs', self.namespaces):
                    time_period = obs.get('TIME_PERIOD')
                    obs_value = obs.get('OBS_VALUE')
                    
                    if time_period in observations:
                        try:
                            observations[time_period] = float(obs_value)
                        except (ValueError, TypeError):
                            observations[time_period] = None
                
                # Combine attributes and observations
                series_data = {**series_attributes, **observations}
                all_series_data.append(series_data)
        
        logger.info(f"Found {quarterly_count} quarterly series out of {total_count} total series")
        
        # Create DataFrame
        df = pd.DataFrame(all_series_data)
        
        # Identify metadata columns (non-date columns)
        metadata_cols = [col for col in df.columns if not col.startswith(('19', '20'))]
        date_cols = [col for col in df.columns if col.startswith(('19', '20'))]
        
        # Reorder columns: metadata first, then dates
        df = df[metadata_cols + sorted(date_cols)]
        
        logger.info(f"Created DataFrame with shape: {df.shape}")
        
        return df
        
    def create_time_series_format(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert to time series format (dates as index, series as columns)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with series as rows and dates as columns
                
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Time series formatted DataFrame and metadata DataFrame
        """
        logger.info("Converting to time series format...")
        
        # Get all potential metadata and date columns first
        all_metadata_cols = [col for col in df.columns if not col.startswith(('19', '20'))]
        date_cols = [col for col in df.columns if col.startswith(('19', '20'))]
        
        # Set series name as index on the main dataframe
        if 'SERIES_NAME' not in df.columns:
            raise KeyError("SERIES_NAME column is missing, cannot create time series format.")
        df = df.set_index('SERIES_NAME')

        # Now, define the final metadata columns for selection, EXCLUDING SERIES_NAME
        metadata_cols_for_selection = [col for col in all_metadata_cols if col != 'SERIES_NAME']
        
        # Create metadata from the remaining columns
        metadata = df[metadata_cols_for_selection].copy()
        
        # Extract just the time series data
        ts_data = df[date_cols].T
        
        # Convert index to datetime
        ts_data.index = pd.to_datetime(ts_data.index)
        ts_data.index.name = 'Date'
        
        # Convert all columns to numeric
        ts_data = ts_data.apply(pd.to_numeric, errors='coerce')
        
        # Remove .Q suffix from column names to match formula file
        ts_data.columns = ts_data.columns.str.replace('.Q', '', regex=False)
        
        # Also remove .Q from the metadata index for consistency
        metadata.index = metadata.index.str.replace('.Q', '', regex=False)
        
        logger.info(f"Time series shape: {ts_data.shape}")
        logger.info(f"Date range: {ts_data.index.min()} to {ts_data.index.max()}")
        
        return ts_data, metadata
    
    def filter_for_uc_analysis(self, 
                              ts_data: pd.DataFrame, 
                              metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Filter series appropriate for UC analysis
        
        Parameters:
        -----------
        ts_data : pd.DataFrame
            Time series data
        metadata : pd.DataFrame
            Series metadata
            
        Returns:
        --------
        pd.DataFrame
            Filtered time series data
        """
        logger.info("Filtering series for UC analysis...")
        
        # Start with all series
        ts_filtered = ts_data.copy()
        initial_count = len(ts_filtered.columns)
        
        # 1. Remove series with too many missing values (relaxed to 50%)
        missing_pct = ts_filtered.isna().sum() / len(ts_filtered)
        valid_series = missing_pct[missing_pct < 0.5].index
        ts_filtered = ts_filtered[valid_series]
        logger.info(f"After removing >50% missing: {len(ts_filtered.columns)} series remain")
        
        # 2. Remove series that are all zeros or constant
        # Calculate std only on non-null values
        std_dev = ts_filtered.std(skipna=True)
        non_constant = std_dev[std_dev > 1e-10].index  # Very small threshold
        ts_filtered = ts_filtered[non_constant]
        logger.info(f"After removing constant series: {len(ts_filtered.columns)} series remain")
        
        # 3. Filter by series prefix if metadata available
        if 'SERIES_PREFIX' in metadata.columns and len(metadata) > 0:
            # Keep main data series types
            valid_prefixes = ['FL', 'FU', 'FA', 'FR', 'FV', 'FD', 'LM', 'LA', 'FC', 'FG']
            
            # Need to match the series names (which may have .Q removed)
            # Get the metadata index without .Q if it exists
            metadata_index_clean = metadata.index.str.replace('.Q', '', regex=False)
            
            # Find series that have valid prefixes
            valid_mask = metadata_index_clean.isin(ts_filtered.columns)
            metadata_matched = metadata[valid_mask].copy()
            metadata_matched.index = metadata_matched.index.str.replace('.Q', '', regex=False)
            
            if len(metadata_matched) > 0:
                valid_series_by_prefix = metadata_matched[
                    metadata_matched['SERIES_PREFIX'].isin(valid_prefixes)
                ].index
                
                # Only filter if we have valid series
                if len(valid_series_by_prefix) > 0:
                    ts_filtered = ts_filtered[valid_series_by_prefix]
                    logger.info(f"After prefix filter: {len(ts_filtered.columns)} series remain")
                else:
                    logger.warning("No series found with valid prefixes, keeping all")
        
        # 4. Final check - ensure we have some data
        if len(ts_filtered.columns) == 0:
            logger.warning("All series filtered out! Returning series with minimal filtering")
            # Return just non-constant series with <50% missing
            return ts_data[non_constant]
        
        logger.info(f"Final filtered shape: {ts_filtered.shape}")
        logger.info(f"Filtered {initial_count - len(ts_filtered.columns)} series out of {initial_count}")
        
        return ts_filtered
    
    def save_to_parquet(self, 
                       ts_data: pd.DataFrame,
                       metadata: pd.DataFrame,
                       suffix: str = "") -> Dict[str, str]:
        """
        Save data to parquet files
        
        Parameters:
        -----------
        ts_data : pd.DataFrame
            Time series data
        metadata : pd.DataFrame
            Series metadata
        suffix : str
            Optional suffix for filename
            
        Returns:
        --------
        dict
            Paths to saved files
        """
        # Save time series data
        ts_filename = f"z1_quarterly_data{suffix}.parquet"
        ts_path = self.output_dir / ts_filename
        ts_data.to_parquet(ts_path, compression='snappy')
        logger.info(f"Saved time series data to: {ts_path}")
        
        # Save metadata
        meta_filename = f"z1_quarterly_metadata{suffix}.parquet"
        meta_path = self.output_dir / meta_filename
        metadata.to_parquet(meta_path, compression='snappy')
        logger.info(f"Saved metadata to: {meta_path}")
        
        # Also save a sample CSV for inspection
        sample_filename = f"z1_quarterly_sample{suffix}.csv"
        sample_path = self.output_dir / sample_filename
        ts_data.iloc[:20, :10].to_csv(sample_path)
        logger.info(f"Saved sample CSV to: {sample_path}")
        
        return {
            'timeseries': str(ts_path),
            'metadata': str(meta_path),
            'sample': str(sample_path)
        }
    
    def create_series_groups(self, 
                           ts_data: pd.DataFrame,
                           metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create logical groups of series for easier analysis
        
        Parameters:
        -----------
        ts_data : pd.DataFrame
            Time series data
        metadata : pd.DataFrame
            Series metadata
            
        Returns:
        --------
        dict
            Dictionary of DataFrames by group
        """
        groups = {}
        
        # Group by instrument type (last 6 digits of series code)
        # Common instruments in Z1
        instrument_groups = {
            '005005': 'net_lending',
            '000005': 'total_assets',
            '190005': 'total_liabilities',
            '090005': 'net_worth',
            '064105': 'corporate_equities',
            '063063': 'mutual_funds',
            '033029': 'mortgages',
            '004005': 'credit_market_instruments'
        }
        
        for instrument_code, instrument_name in instrument_groups.items():
            series_list = [col for col in ts_data.columns if col.endswith(instrument_code)]
            if series_list:
                groups[instrument_name] = ts_data[series_list]
                logger.info(f"Group '{instrument_name}': {len(series_list)} series")
        
        # Group by sector (digits 3-4 of series code)
        # Common sectors
        sector_groups = {
            '10': 'households',
            '11': 'nonfinancial_corporate',
            '12': 'nonfinancial_noncorporate',
            '31': 'federal_government',
            '21': 'state_local_government',
            '40': 'banking',
            '50': 'insurance',
            '60': 'pension_funds',
            '89': 'all_sectors'
        }
        
        for sector_code, sector_name in sector_groups.items():
            series_list = [col for col in ts_data.columns 
                          if len(col) > 4 and col[2:4] == sector_code]
            if series_list:
                groups[f"sector_{sector_name}"] = ts_data[series_list]
                logger.info(f"Sector '{sector_name}': {len(series_list)} series")
        
        return groups
    
    def run_full_extraction(self, 
                          force_download: bool = False,
                          save_groups: bool = True) -> Dict[str, str]:
        """
        Run the complete extraction pipeline
        
        Parameters:
        -----------
        force_download : bool
            Force fresh download of Z1 data
        save_groups : bool
            Save grouped series separately
            
        Returns:
        --------
        dict
            Paths to all saved files
        """
        logger.info("Starting Z1 quarterly series extraction...")
        
        # Step 1: Download data
        xml_path = self.download_z1_data(force_download)
        
        # Step 2: Extract quarterly series
        df_raw = self.extract_quarterly_series(xml_path)
        
        # Step 3: Convert to time series format
        ts_data, metadata = self.create_time_series_format(df_raw)
        
        # Step 4: Save complete dataset
        paths = self.save_to_parquet(ts_data, metadata, suffix="_complete")
        
        # Step 5: Filter for UC analysis
        ts_filtered = self.filter_for_uc_analysis(ts_data, metadata)
        
        # Step 6: Save filtered dataset
        filtered_paths = self.save_to_parquet(
            ts_filtered, 
            metadata.loc[ts_filtered.columns],
            suffix="_filtered"
        )
        paths.update({f"filtered_{k}": v for k, v in filtered_paths.items()})
        
        # Step 7: Create and save groups
        if save_groups:
            groups = self.create_series_groups(ts_filtered, metadata)
            
            groups_dir = self.output_dir / "groups"
            groups_dir.mkdir(exist_ok=True)
            
            for group_name, group_data in groups.items():
                group_path = groups_dir / f"{group_name}.parquet"
                group_data.to_parquet(group_path, compression='snappy')
                paths[f"group_{group_name}"] = str(group_path)
        
        # Create summary report
        self._create_summary_report(ts_data, ts_filtered, metadata, paths)
        
        logger.info("Extraction complete!")
        return paths
    
    def _create_summary_report(self, 
                             ts_complete: pd.DataFrame,
                             ts_filtered: pd.DataFrame,
                             metadata: pd.DataFrame,
                             paths: Dict[str, str]):
        """Create summary report of extraction"""
        report_path = self.output_dir / "z1_extraction_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Z1 QUARTERLY SERIES EXTRACTION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Extraction Date: {datetime.now()}\n\n")
            
            f.write("DATA SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total quarterly series: {len(ts_complete.columns)}\n")
            f.write(f"Filtered series for UC: {len(ts_filtered.columns)}\n")
            f.write(f"Date range: {ts_complete.index.min()} to {ts_complete.index.max()}\n")
            f.write(f"Number of time periods: {len(ts_complete)}\n\n")
            
            f.write("SERIES BREAKDOWN\n")
            f.write("-" * 30 + "\n")
            
            # Count by prefix
            if 'SERIES_PREFIX' in metadata.columns:
                prefix_counts = metadata['SERIES_PREFIX'].value_counts()
                f.write("By Prefix:\n")
                for prefix, count in prefix_counts.head(10).items():
                    f.write(f"  {prefix}: {count}\n")
                f.write("\n")
            
            # Count by unit
            if 'UNIT' in metadata.columns:
                unit_counts = metadata['UNIT'].value_counts()
                f.write("By Unit:\n")
                for unit, count in unit_counts.head(5).items():
                    f.write(f"  {unit}: {count}\n")
                f.write("\n")
            
            f.write("OUTPUT FILES\n")
            f.write("-" * 30 + "\n")
            for file_type, path in paths.items():
                f.write(f"{file_type}: {Path(path).name}\n")
        
        logger.info(f"Summary report saved to: {report_path}")


# Main execution
if __name__ == "__main__":
    # Initialize extractor
    extractor = Z1QuarterlyExtractor(
        output_dir="./data/z1_quarterly",
        start_year=1952,  # Adjust as needed
        end_year=2024
    )
    
    # Run extraction
    paths = extractor.run_full_extraction(
        force_download=False,  # Set to True to force fresh download
        save_groups=False       # Save series groups separately
    )
    
    print("\nExtraction complete! Files saved:")
    for file_type, path in paths.items():
        print(f"  {file_type}: {path}")
    
    # Example: Load the filtered data for UC analysis
    print("\nLoading filtered data for UC analysis...")
    z1_data = pd.read_parquet(paths['filtered_timeseries'])
    print(f"Loaded data shape: {z1_data.shape}")
    print(f"Sample series: {list(z1_data.columns[:5])}")
