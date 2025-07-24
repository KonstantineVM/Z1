"""
Economic analysis and relationship identification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from celer import Lasso
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)


class EconomicAnalysis:
    """
    Analyze economic relationships using decomposed components
    """
    
    def __init__(self, components: Dict[str, pd.DataFrame], 
                 original_data: Optional[pd.DataFrame] = None):
        """
        Initialize economic analysis
        
        Parameters:
        -----------
        components : Dict[str, pd.DataFrame]
            Decomposed components
        original_data : pd.DataFrame, optional
            Original time series data
        """
        self.components = components
        self.original_data = original_data
        self.scaler = StandardScaler()
        
    def analyze_velocity_of_money(self) -> pd.DataFrame:
        """
        Analyze velocity of money relationships
        
        Returns:
        --------
        pd.DataFrame
            Analysis results
        """
        results = pd.DataFrame()
        
        if 'trend' not in self.components:
            logger.warning("No trend component available for velocity analysis")
            return results
            
        trend_df = self.components['trend']
        
        # Check for required series
        gdp_series = self._find_gdp_series(trend_df.columns)
        m2_series = self._find_m2_series(trend_df.columns)
        
        if gdp_series and m2_series:
            results['gdp_growth'] = trend_df[gdp_series]
            results['m2_growth'] = trend_df[m2_series]
            results['velocity_change'] = results['gdp_growth'] - results['m2_growth']
            
            # Add consumption if available
            pce_series = self._find_pce_series(trend_df.columns)
            if pce_series:
                results['pce_growth'] = trend_df[pce_series]
                
        return results
    
    def analyze_interest_rate_relationships(self, target_series: str,
                                          lasso_alpha: float = 0.001,
                                          exclude_patterns: Optional[List[str]] = None) -> Dict:
        """
        Analyze relationships between interest rates and target variable
        
        Parameters:
        -----------
        target_series : str
            Target series name
        lasso_alpha : float
            LASSO regularization parameter
        exclude_patterns : List[str], optional
            Patterns to exclude from features
            
        Returns:
        --------
        Dict
            Analysis results including coefficients and predictions
        """
        if exclude_patterns is None:
            exclude_patterns = []
            
        # Prepare features and target
        features, target = self._prepare_features_target(target_series, exclude_patterns)
        
        if features.empty or target.empty:
            logger.warning(f"No valid features or target for {target_series}")
            return {}
            
        # Split data
        split_point = int(len(features) * 0.8)
        X_train, X_test = features.iloc[:split_point], features.iloc[split_point:]
        y_train, y_test = target.iloc[:split_point], target.iloc[split_point:]
        
        # Standardize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit LASSO
        lasso = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=100000, tol=1e-5)
        lasso.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = lasso.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        
        # Extract important features
        coef_df = pd.DataFrame({
            'Feature': features.columns,
            'Coefficient': lasso.coef_
        })
        coef_df = coef_df[coef_df['Coefficient'] != 0].sort_values(
            'Coefficient', key=abs, ascending=False
        )
        
        return {
            'coefficients': coef_df,
            'predictions': y_pred,
            'actual': y_test,
            'mse': mse,
            'feature_importance': coef_df.head(20)
        }
    
    def analyze_savings_dynamics(self) -> pd.DataFrame:
        """
        Analyze personal savings dynamics
        
        Returns:
        --------
        pd.DataFrame
            Savings analysis results
        """
        results = pd.DataFrame()
        
        if 'trend' not in self.components or 'level' not in self.components:
            return results
            
        trend_df = self.components['trend']
        level_df = self.components['level']
        
        # Find savings-related series
        savings_cols = [col for col in trend_df.columns 
                       if any(pattern in col for pattern in ['156007', '156012', 'saving', 'SAVING'])]
        
        for col in savings_cols:
            if col in level_df.columns:
                # Normalized savings rate change
                results[f'{col}_normalized'] = trend_df[col] / level_df[col]
            else:
                results[f'{col}_trend'] = trend_df[col]
                
        return results
    
    def identify_leading_indicators(self, target_series: str,
                                   max_lag: int = 8) -> pd.DataFrame:
        """
        Identify leading indicators for a target series
        
        Parameters:
        -----------
        target_series : str
            Target series to predict
        max_lag : int
            Maximum lag to consider
            
        Returns:
        --------
        pd.DataFrame
            Leading indicators ranked by importance
        """
        if 'cycle' not in self.components:
            logger.warning("No cycle component available for leading indicator analysis")
            return pd.DataFrame()
            
        cycle_df = self.components['cycle']
        
        if target_series not in cycle_df.columns:
            logger.warning(f"Target series {target_series} not found")
            return pd.DataFrame()
            
        correlations = []
        
        for col in cycle_df.columns:
            if col == target_series:
                continue
                
            for lag in range(1, max_lag + 1):
                # Calculate correlation at different lags
                lagged_series = cycle_df[col].shift(lag)
                corr = cycle_df[target_series].corr(lagged_series)
                
                if not np.isnan(corr):
                    correlations.append({
                        'indicator': col,
                        'lag': lag,
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
        
        # Convert to DataFrame and sort
        corr_df = pd.DataFrame(correlations)
        if not corr_df.empty:
            corr_df = corr_df.sort_values('abs_correlation', ascending=False)
            
        return corr_df
    
    def _find_gdp_series(self, columns: List[str]) -> Optional[str]:
        """Find GDP series in columns"""
        gdp_patterns = ['FU086902005', 'GDP', 'gdp']
        for col in columns:
            if any(pattern in col for pattern in gdp_patterns):
                return col
        return None
    
    def _find_m2_series(self, columns: List[str]) -> Optional[str]:
        """Find M2 money supply series in columns"""
        m2_patterns = ['M2_N.M', 'M2SL', 'M2']
        for col in columns:
            if any(pattern in col for pattern in m2_patterns):
                return col
        return None
    
    def _find_pce_series(self, columns: List[str]) -> Optional[str]:
        """Find PCE series in columns"""
        pce_patterns = ['FU156901001', 'PCE', 'pce']
        for col in columns:
            if any(pattern in col for pattern in pce_patterns):
                return col
        return None
    
    def _prepare_features_target(self, target_series: str,
                               exclude_patterns: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for analysis"""
        # Combine all components into features
        frames = []
        
        for comp_name, comp_df in self.components.items():
            if comp_df is not None and not comp_df.empty:
                # Prefix columns with component name
                prefixed = comp_df.add_prefix(f'{comp_name}_')
                frames.append(prefixed)
                
        if not frames:
            return pd.DataFrame(), pd.Series()
            
        all_features = pd.concat(frames, axis=1)
        
        # Extract target
        target_col = None
        for col in all_features.columns:
            if target_series in col:
                target_col = col
                break
                
        if target_col is None:
            return pd.DataFrame(), pd.Series()
            
        target = all_features[target_col]
        
        # Remove target and excluded patterns from features
        feature_cols = [col for col in all_features.columns 
                       if col != target_col and 
                       not any(pattern in col for pattern in exclude_patterns)]
        
        features = all_features[feature_cols]
        
        # Remove NaNs
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        
        return features[valid_idx], target[valid_idx]