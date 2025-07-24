"""
Feature engineering for time series analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for economic time series
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        pass
    
    def create_component_features(self, components: Dict[str, pd.DataFrame], 
                                 zero_crossing_cols: List[str]) -> pd.DataFrame:
        """
        Create features from decomposed components
        
        Parameters:
        -----------
        components : Dict[str, pd.DataFrame]
            Dictionary of component DataFrames
        zero_crossing_cols : List[str]
            Columns that cross zero
            
        Returns:
        --------
        pd.DataFrame
            Feature DataFrame with multi-level columns
        """
        frames = []
        
        # Process each component type
        for component_name in ['Level', 'Trend', 'Cycle', 'Seasonal']:
            component_lower = component_name.lower()
            
            if component_lower not in components or components[component_lower] is None:
                continue
                
            df = components[component_lower]
            
            # Get columns that don't cross zero
            valid_cols = df.columns.difference(zero_crossing_cols)
            
            if component_name == 'Level':
                # Calculate percent change for level
                frame = df[valid_cols].pct_change().add_prefix(f'{component_name}_')
            else:
                # Divide by level for other components
                if 'level' in components and components['level'] is not None:
                    level_df = components['level'][valid_cols]
                    frame = df[valid_cols].div(level_df.values).add_prefix(f'{component_name}_')
                else:
                    frame = df[valid_cols].add_prefix(f'{component_name}_')
                    
            frames.append(frame)
        
        # Handle zero-crossing columns separately
        if zero_crossing_cols:
            zero_crossing_frames = self._process_zero_crossing_features(
                components, zero_crossing_cols
            )
            frames.extend(zero_crossing_frames)
        
        # Concatenate all frames
        if frames:
            feature_df = pd.concat(frames, axis=1)
            
            # Create multi-level columns
            feature_df.columns = pd.MultiIndex.from_tuples(
                [tuple(col.split('_', 1)) for col in feature_df.columns],
                names=['Component', 'Series']
            )
            
            return feature_df
        else:
            return pd.DataFrame()
    
    def _process_zero_crossing_features(self, components: Dict[str, pd.DataFrame], 
                                       zero_crossing_cols: List[str]) -> List[pd.DataFrame]:
        """Process features for zero-crossing series"""
        frames = []
        
        # These would be the amplitude-normalized components
        # In the full implementation, we'd use the normalized components
        
        for component_name in ['Level', 'Trend', 'Cycle', 'Seasonal']:
            component_lower = component_name.lower()
            
            if component_lower not in components or components[component_lower] is None:
                continue
                
            df = components[component_lower]
            cols_to_process = [col for col in zero_crossing_cols if col in df.columns]
            
            if cols_to_process:
                frame = df[cols_to_process].add_prefix(f'{component_name}_ZC_')
                frames.append(frame)
                
        return frames
    
    def create_lagged_features(self, df: pd.DataFrame, 
                              max_lags: int = 16,
                              min_lag: int = 3) -> pd.DataFrame:
        """
        Create lagged features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input features
        max_lags : int
            Maximum number of lags
        min_lag : int
            Minimum lag to start from
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with original and lagged features
        """
        lagged_dfs = [df]
        
        for lag in range(min_lag, max_lags + 1):
            lagged_df = df.shift(lag).rename(columns=lambda x: f'{x}_lag{lag}')
            lagged_dfs.append(lagged_df)
            
        # Concatenate all
        result = pd.concat(lagged_dfs, axis=1)
        
        # Drop initial rows with NaN
        result = result.iloc[max_lags:]
        
        return result
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   interaction_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified pairs
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input features
        interaction_pairs : List[Tuple[str, str]]
            List of column pairs to create interactions
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional interaction features
        """
        result = df.copy()
        
        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Multiplicative interaction
                result[f'{col1}_X_{col2}'] = df[col1] * df[col2]
                
                # Ratio (with small epsilon to avoid division by zero)
                result[f'{col1}_DIV_{col2}'] = df[col1] / (df[col2] + 1e-10)
                
        return result
    
    def create_economic_indicators(self, components: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create specific economic indicators from components
        
        Parameters:
        -----------
        components : Dict[str, pd.DataFrame]
            Component DataFrames
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with economic indicators
        """
        indicators = pd.DataFrame(index=components['trend'].index)
        
        # Velocity of money (if available)
        if 'Trend' in components:
            trend_df = components['trend']
            if 'FU086902005.Q' in trend_df.columns and 'M2_N.M' in trend_df.columns:
                indicators['velocity_of_money'] = (
                    trend_df['FU086902005.Q'] - trend_df['M2_N.M']
                )
        
        # Average interest rate (if multiple rate series available)
        rate_cols = [col for col in components.get('trend', pd.DataFrame()).columns 
                     if 'Rate' in col or 'RIF' in col]
        if len(rate_cols) > 1 and 'level' in components:
            level_df = components['level']
            trend_df = components['trend']
            
            # Weighted average by level
            weights = level_df[rate_cols]
            rates = trend_df[rate_cols] / level_df[rate_cols]
            indicators['avg_interest_rate'] = (rates * weights).sum(axis=1) / weights.sum(axis=1)
        
        return indicators
    
    def prepare_train_test_split(self, features: pd.DataFrame, 
                               target: pd.Series,
                               split_ratio: float = 0.8,
                               reverse_time: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                                   pd.Series, pd.Series]:
        """
        Prepare train/test split for time series
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature DataFrame
        target : pd.Series
            Target variable
        split_ratio : float
            Ratio for training data
        reverse_time : bool
            Whether to reverse time order (for some models)
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        # Ensure alignment
        features = features.loc[target.index]
        
        # Drop any remaining NaNs
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_idx]
        target = target[valid_idx]
        
        # Reverse time if requested
        if reverse_time:
            features = features.iloc[::-1].reset_index(drop=True)
            target = target.iloc[::-1].reset_index(drop=True)
        
        # Split point
        split_point = int(len(features) * split_ratio)
        
        # Split without shuffling
        X_train = features.iloc[:split_point]
        X_test = features.iloc[split_point:]
        y_train = target.iloc[:split_point]
        y_test = target.iloc[split_point:]
        
        return X_train, X_test, y_train, y_test