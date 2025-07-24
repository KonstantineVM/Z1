"""
Unobserved Components Model implementation with parallel processing
"""

import statsmodels.api as sm
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings
import statsmodels.tools.sm_exceptions
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class UnobservedComponentsModel:
    """
    Wrapper for statsmodels UnobservedComponents with parallel processing
    """
    
    DEFAULT_MODEL_PARAMS = {
        'level': True,
        'trend': True,
        'seasonal': False,
        'freq_seasonal': [{'period': 4, 'harmonics': 2}],
        'cycle': True,
        'autoregressive': None,
        'exog': None,
        'irregular': True,
        'stochastic_level': False,
        'stochastic_trend': True,
        'stochastic_seasonal': True,
        'stochastic_freq_seasonal': [True],
        'stochastic_cycle': True,
        'damped_cycle': True,
        'cycle_period_bounds': None,
        'mle_regression': True,
        'use_exact_diffuse': False
    }
    
    def __init__(self, model_params: Optional[Dict] = None):
        """
        Initialize the UC model
        
        Parameters:
        -----------
        model_params : Dict, optional
            Model parameters to override defaults
        """
        self.model_params = self.DEFAULT_MODEL_PARAMS.copy()
        if model_params:
            self.model_params.update(model_params)
            
    def _suppress_warnings(self):
        """Suppress specific warnings during model fitting"""
        warnings.filterwarnings('ignore', message="`product` is deprecated as of NumPy 1.25.0")
        warnings.filterwarnings('ignore', category=statsmodels.tools.sm_exceptions.ValueWarning,
                               message="No frequency information was provided")
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                               message="invalid value encountered in subtract")
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                               message="divide by zero encountered in log")
    
    def decompose_single_series(self, series: pd.Series, 
                               series_name: str) -> Dict[str, pd.Series]:
        """
        Decompose a single time series
        
        Parameters:
        -----------
        series : pd.Series
            Time series to decompose
        series_name : str
            Name of the series
            
        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary with component names as keys
        """
        with warnings.catch_warnings():
            self._suppress_warnings()
            
            try:
                # Ensure numeric and forward-fill NaNs
                time_series_data = series.astype(float).ffill()
                
                # Create and fit model
                model = sm.tsa.UnobservedComponents(time_series_data, **self.model_params)
                
                # Set filter method for numerical stability
                model.ssm.set_filter_method(filter_conventional=True, filter_concentrated=True)
                
                # Fit model
                result = model.fit(method='powell', maxiter=10000, disp=False)
                
                return {
                    'level': result.level.smoothed,
                    'trend': result.trend.smoothed,
                    'cycle': result.cycle.smoothed,
                    'seasonal': result.freq_seasonal[0]['smoothed'] if result.freq_seasonal else None
                }
                
            except Exception as e:
                logger.error(f"Error processing series {series_name}: {e}")
                return None
    
    def decompose_parallel(self, df: pd.DataFrame, 
                          n_jobs: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Decompose multiple series in parallel
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with series as columns
        n_jobs : int, optional
            Number of parallel jobs
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with component DataFrames
        """
        columns = df.columns
        
        # Pre-allocate dictionaries
        level_dict = {col: None for col in columns}
        trend_dict = {col: None for col in columns}
        cycle_dict = {col: None for col in columns}
        seasonal_dict = {col: None for col in columns}
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self._process_column_wrapper, col, df[col])
                for col in columns
            ]
            
            for future in tqdm(futures, desc="Decomposing series"):
                col, result = future.result()
                if result:
                    level_dict[col] = result['level']
                    trend_dict[col] = result['trend']
                    cycle_dict[col] = result['cycle']
                    seasonal_dict[col] = result['seasonal']
        
        # Convert to DataFrames
        return {
            'level': pd.DataFrame(level_dict),
            'trend': pd.DataFrame(trend_dict),
            'cycle': pd.DataFrame(cycle_dict),
            'seasonal': pd.DataFrame(seasonal_dict)
        }
    
    def _process_column_wrapper(self, col: str, series: pd.Series) -> Tuple[str, Dict]:
        """Wrapper for parallel processing"""
        return col, self.decompose_single_series(series, col)
    
    def identify_zero_crossing_series(self, df: pd.DataFrame, 
                                     components: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Identify series where both the original and level component cross zero
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original data
        components : Dict[str, pd.DataFrame]
            Decomposed components
            
        Returns:
        --------
        List[str]
            List of column names that cross zero
        """
        level_df = components['level']
        columns_crossing_zero = []
        
        for col in tqdm(level_df.columns, desc='Checking zero crossings'):
            level_series = level_df[col]
            original_series = df[col]
            
            # Check for sign changes in both series
            level_crosses = ((level_series.shift(1) * level_series) < 0).any()
            original_crosses = ((original_series.shift(1) * original_series) < 0).any()
            
            if level_crosses and original_crosses:
                columns_crossing_zero.append(col)
                
        return columns_crossing_zero
    
    def calculate_seasonal_amplitude(self, seasonal_component: pd.Series) -> np.ndarray:
        """
        Calculate time-varying amplitude of seasonal component
        
        Parameters:
        -----------
        seasonal_component : pd.Series
            Seasonal component series
            
        Returns:
        --------
        np.ndarray
            Interpolated amplitude values
        """
        from scipy.signal import find_peaks
        from scipy.interpolate import interp1d
        
        # Find peaks and troughs
        peaks, _ = find_peaks(seasonal_component, distance=3)
        troughs, _ = find_peaks(-seasonal_component, distance=3)
        
        # Ensure equal number
        min_length = min(len(peaks), len(troughs))
        peaks = peaks[:min_length]
        troughs = troughs[:min_length]
        
        if len(peaks) == 0 or len(troughs) == 0:
            # Return constant amplitude if no peaks/troughs found
            return np.ones(len(seasonal_component))
        
        # Calculate amplitude
        amplitudes = seasonal_component.iloc[peaks].values - seasonal_component.iloc[troughs].values
        amplitude_times = peaks
        
        # Interpolate to original length
        interp_func = interp1d(amplitude_times, amplitudes, 
                              kind='linear', bounds_error=False, 
                              fill_value="extrapolate")
        interpolated_amplitudes = interp_func(np.arange(len(seasonal_component)))
        
        return interpolated_amplitudes
    
    def normalize_components_by_amplitude(self, components: Dict[str, pd.DataFrame], 
                                         zero_crossing_cols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Normalize components by seasonal amplitude for zero-crossing series
        
        Parameters:
        -----------
        components : Dict[str, pd.DataFrame]
            Original components
        zero_crossing_cols : List[str]
            Columns that cross zero
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Normalized components
        """
        normalized_components = {}
        
        for component_name in ['level', 'trend', 'cycle', 'seasonal']:
            if component_name not in components:
                continue
                
            df = components[component_name].copy()
            
            for col in zero_crossing_cols:
                if col in df.columns and components['seasonal'] is not None and col in components['seasonal'].columns:
                    # Calculate amplitude
                    amplitude = self.calculate_seasonal_amplitude(components['seasonal'][col])
                    
                    # Normalize by squared amplitude
                    df[col] = df[col] / (amplitude ** 2)
                    
            normalized_components[component_name] = df
            
        return normalized_components