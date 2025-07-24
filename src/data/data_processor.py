"""
Data Processor
Handles processing, filtering, and combining of Fed and external data sources
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Process and combine economic time series data
    """
    
    def __init__(self):
        """Initialize data processor"""
        self.processed_data = {}
        
    def process_fed_data(self, df: pd.DataFrame, 
                        source_type: str,
                        series_start_col: Optional[int] = None) -> pd.DataFrame:
        """
        Process Fed data based on source type
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw Fed data
        source_type : str
            Type of Fed data (Z1, H6, H8, etc.)
        series_start_col : int, optional
            Column index where time series data starts
            
        Returns:
        --------
        pd.DataFrame
            Processed data with dates as index and series as columns
        """
        logger.info(f"Processing {source_type} data...")
        
        # Determine series start column based on source type
        if series_start_col is None:
            series_start_col = self._get_series_start_col(source_type)
        
        # Get time series columns
        time_columns = self._identify_time_columns(df, series_start_col)
        
        if not time_columns:
            logger.warning(f"No time series columns found in {source_type} data")
            return pd.DataFrame()
        
        # Filter based on source type
        df_filtered = self._apply_source_filters(df, source_type)
        
        if df_filtered.empty:
            logger.warning(f"No data remaining after filtering {source_type}")
            return pd.DataFrame()
        
        # Process time series
        df_processed = self._process_time_series(df_filtered, time_columns, source_type)
        
        logger.info(f"Processed {source_type}: {df_processed.shape}")
        
        return df_processed
    
    def _get_series_start_col(self, source_type: str) -> int:
        """Get the starting column index for time series data"""
        start_cols = {
            'Z1': 9,
            'H6': 6,
            'H8': 10,
            'G17': 7,
            'G19': 9,
            'G20': 10,
            'H15': 7
        }
        return start_cols.get(source_type, 9)
    
    def _identify_time_columns(self, df: pd.DataFrame, start_col: int) -> List[str]:
        """Identify columns containing time series data"""
        # Time columns typically start with years (19xx or 20xx)
        time_columns = []
        
        for col in df.columns[start_col:]:
            if (col.startswith('19') or col.startswith('20')) and '-' in col:
                time_columns.append(col)
                
        return time_columns
    
    def _apply_source_filters(self, df: pd.DataFrame, source_type: str) -> pd.DataFrame:
        """Apply source-specific filters"""
        df_filtered = df.copy()
        
        if source_type == 'Z1':
            # Filter for quarterly frequency and exclude flow/adjustment series
            if 'FREQ' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['FREQ'] == '162']
            if 'SERIES_PREFIX' in df_filtered.columns:
                exclude_prefixes = ['FA', 'PC', 'LA', 'FC', 'FG']
                df_filtered = df_filtered[~df_filtered['SERIES_PREFIX'].isin(exclude_prefixes)]
                
        elif source_type == 'H6':
            # Filter for non-seasonally adjusted
            if 'ADJUSTED' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['ADJUSTED'] == 'NSA']
                
        elif source_type == 'H8':
            # Filter for non-seasonally adjusted
            if 'SA' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['SA'] == 'NSA']
                
        elif source_type in ['G17', 'G19', 'G20']:
            # Filter for non-seasonally adjusted
            if 'SA' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['SA'] != 'SA']
                
        elif source_type == 'H15':
            # Filter for quarterly frequency
            if 'FREQ' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['FREQ'] == '129']
                
        return df_filtered
    
    def _process_time_series(self, df: pd.DataFrame, 
                           time_columns: List[str],
                           source_type: str) -> pd.DataFrame:
        """Process time series data exactly as in Jupyter notebooks"""
        
        # 1. Check for leading NAs or zeros (on string data, before conversion)
        def has_leading_nan_or_zero(series):
            return pd.isna(series.iloc[0]) or series.iloc[0] == '0'
        
        # Apply the function to each row and create a boolean filter
        filter_no_leading_nas = df[time_columns].apply(
            lambda row: not has_leading_nan_or_zero(row), axis=1
        )
        
        # Filter the DataFrame to remove rows with leading NAs/zeros
        df_filtered = df[filter_no_leading_nas].reset_index(drop=True)
        
        # 2. Transpose the DataFrame BEFORE numeric conversion
        series_name_col = 'SERIES_NAME' if 'SERIES_NAME' in df_filtered.columns else df_filtered.columns[0]
        df_transposed = df_filtered.set_index(series_name_col)[time_columns].T
        
        # 3. Drop non-numeric rows (metadata rows) after transposing
        # The number of metadata rows equals the number of non-time-series columns
        # This is the index where time_columns start in the original DataFrame
        metadata_rows = self._get_series_start_col(source_type)
        df_transposed = df_transposed.iloc[metadata_rows:, :]
        
        # 4. Convert to float
        df_transposed = df_transposed.astype(float)
        
        # 5. Set proper datetime index
        df_transposed.index = pd.to_datetime(df_transposed.index)
        
        # 6. Remove duplicate columns
        df_transposed = df_transposed.loc[:, ~df_transposed.columns.duplicated()]
        
        return df_transposed
    
    def _apply_log_transformation(self, df: pd.DataFrame, 
                                time_columns: List[str]) -> pd.DataFrame:
        """Apply log transformation to appropriate series"""
        df_processed = df.copy()
        
        # Create mask for percent units
        percent_mask = pd.Series(False, index=df.index)
        if 'UNIT' in df.columns:
            percent_mask = df['UNIT'].str.contains(r'\bpercent\b', case=False, na=False)
        
        # Identify series that cross zero (excluding percent series)
        rows_crossing_zero = []
        
        for idx, row in df[~percent_mask].iterrows():
            if (row[time_columns] < 0).any():
                rows_crossing_zero.append(idx)
        
        # Create mask for series to transform
        transform_mask = ~percent_mask & ~df.index.isin(rows_crossing_zero)
        
        # Apply log transformation
        if transform_mask.any():
            df_processed.loc[transform_mask, time_columns] = df_processed.loc[
                transform_mask, time_columns
            ].applymap(lambda x: np.log(x) if x > 0 else np.nan)
        
        logger.info(f"Applied log transformation to {transform_mask.sum()} series")
        logger.info(f"Skipped {len(rows_crossing_zero)} zero-crossing series")
        logger.info(f"Skipped {percent_mask.sum()} percent-unit series")
        
        return df_processed
        
    def _apply_log_transformation_transposed(self, df: pd.DataFrame, 
                                            source_type: str) -> pd.DataFrame:
        """Apply log transformation to already transposed data"""
        df_processed = df.copy()
        
        # For Z1 data, we need to check the original metadata
        # Since we don't have UNIT column after transposition, 
        # we'll apply log to all positive series that don't cross zero
        
        for col in df.columns:
            series = df[col]
            # Check if series crosses zero
            if (series < 0).any():
                continue  # Skip series that cross zero
            
            # Apply log transformation to positive values
            df_processed[col] = series.apply(lambda x: np.log(x) if x > 0 else np.nan)
        
        logger.info(f"Applied log transformation to {source_type} data")
        
        return df_processed        
    
    def combine_data_sources(self, fed_data: Dict[str, pd.DataFrame],
                           external_data: Dict[str, pd.DataFrame],
                           join_method: str = 'outer') -> pd.DataFrame:
        """
        Combine Fed and external data sources
        
        Parameters:
        -----------
        fed_data : Dict[str, pd.DataFrame]
            Dictionary of processed Fed data
        external_data : Dict[str, pd.DataFrame]
            Dictionary of external data
        join_method : str
            How to join data ('outer', 'inner', 'left')
            
        Returns:
        --------
        pd.DataFrame
            Combined dataset
        """
        logger.info("Combining data sources...")
        
        # Start with Fed data
        combined = None
        
        # Combine Fed sources
        for source_name, df in fed_data.items():
            if df.empty:
                continue
                
            if combined is None:
                combined = df.copy()
            else:
                # Join on index (dates)
                combined = combined.join(df, how=join_method, rsuffix=f'_{source_name}')
        
        # Add external data
        for source_name, df in external_data.items():
            if df.empty:
                continue
                
            # Ensure string index for consistency
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.strftime('%Y-%m-%d')
            
            if combined is None:
                combined = df.copy()
            else:
                # Align indices
                combined_dates = pd.to_datetime(combined.index)
                df_dates = pd.to_datetime(df.index)
                
                # Create temporary DataFrames with datetime index
                temp_combined = combined.copy()
                temp_combined.index = combined_dates
                
                temp_df = df.copy()
                temp_df.index = df_dates
                
                # Join
                temp_combined = temp_combined.join(temp_df, how=join_method)
                
                # Convert back to string index
                temp_combined.index = temp_combined.index.strftime('%Y-%m-%d')
                combined = temp_combined
        
        if combined is not None:
            # Remove any remaining duplicates
            combined = combined.loc[:, ~combined.columns.duplicated()]
            
            # Drop columns with all NaN
            combined = combined.dropna(axis=1, how='all')
            
            # Sort by date
            combined = combined.sort_index()
            
            logger.info(f"Combined dataset shape: {combined.shape}")
            logger.info(f"Date range: {combined.index[0]} to {combined.index[-1]}")
            logger.info(f"Number of series: {len(combined.columns)}")
            
        return combined if combined is not None else pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate processed data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to validate
            
        Returns:
        --------
        Dict[str, any]
            Validation results
        """
        validation_results = {
            'shape': df.shape,
            'date_range': (df.index[0], df.index[-1]) if not df.empty else (None, None),
            'missing_values': df.isna().sum().sum(),
            'missing_by_column': df.isna().sum().to_dict(),
            'infinite_values': np.isinf(df.values).sum(),
            'duplicate_columns': df.columns[df.columns.duplicated()].tolist(),
            'constant_columns': [],
            'high_correlation_pairs': []
        }
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                validation_results['constant_columns'].append(col)
        
        # Check for highly correlated columns (sample if too many)
        if len(df.columns) < 100:
            corr_matrix = df.corr().abs()
            # Get upper triangle
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find pairs with correlation > 0.99
            high_corr = np.where(upper_tri > 0.99)
            validation_results['high_correlation_pairs'] = [
                (df.columns[i], df.columns[j], upper_tri.iloc[i, j])
                for i, j in zip(*high_corr)
            ]
        
        return validation_results
    
    def create_analysis_dataset(self, df: pd.DataFrame,
                              min_observations: int = 100,
                              max_missing_pct: float = 0.2) -> pd.DataFrame:
        """
        Create a clean dataset for analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Combined data
        min_observations : int
            Minimum number of non-missing observations per series
        max_missing_pct : float
            Maximum percentage of missing values allowed
            
        Returns:
        --------
        pd.DataFrame
            Cleaned dataset
        """
        logger.info("Creating analysis dataset...")
        
        # Remove series with too many missing values
        missing_pct = df.isna().sum() / len(df)
        valid_columns = missing_pct[missing_pct <= max_missing_pct].index
        
        # Remove series with too few observations
        non_missing_counts = df[valid_columns].notna().sum()
        valid_columns = non_missing_counts[non_missing_counts >= min_observations].index
        
        # Create filtered dataset
        df_analysis = df[valid_columns].copy()
        
        # Forward fill missing values (for small gaps)
        df_analysis = df_analysis.fillna(method='ffill', limit=4)
        
        # Remove any remaining rows with all NaN
        df_analysis = df_analysis.dropna(how='all')
        
        logger.info(f"Analysis dataset: {df_analysis.shape}")
        logger.info(f"Removed {len(df.columns) - len(df_analysis.columns)} series")
        
        return df_analysis
