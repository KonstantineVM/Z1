# src/analysis/economic_analysis.py
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

    def __init__(self, components: dict, original_data: pd.DataFrame):
        """
        Initializes the EconomicAnalysis class.

        Args:
            components (dict): A dictionary where keys are component names (e.g., 'trend', 'cycle')
                               and values are the corresponding NumPy arrays.
            original_data (pd.DataFrame): The original DataFrame from which the components were derived.
        """
        self.original_data = original_data
        self.scaler = StandardScaler()

        # Convert NumPy array components into Pandas Series, using the original data's index.
        # This makes them compatible with analysis methods that expect Pandas objects.
        self.components = {
            key: pd.Series(value, index=self.original_data.index, name=key)
            for key, value in components.items()
        }

    def analyze_velocity_of_money(self) -> pd.DataFrame:
        """
        Analyze velocity of money relationships.
        Velocity of Money = GDP / M2. The change in velocity is approximated by
        the growth rate of GDP minus the growth rate of M2.

        Returns:
        --------
        pd.DataFrame
            Analysis results including GDP growth, M2 growth, and the change in velocity.
        """
        results = pd.DataFrame()

        if 'trend' not in self.components:
            logger.warning("No trend component available for velocity analysis")
            return results

        # The 'trend' component is the trend of the primary series (e.g., GDP)
        gdp_trend_growth = self.components['trend']

        # Find the M2 series in the *original data*
        m2_series_name = self._find_m2_series(self.original_data.columns)
        if not m2_series_name:
            logger.warning("M2 series not found in original data for velocity analysis")
            return results

        # For this analysis, we assume the trend of M2 is what matters
        m2_series = self.original_data[m2_series_name]

        results['gdp_growth'] = gdp_trend_growth
        results['m2_growth'] = m2_series
        results['velocity_change'] = results['gdp_growth'] - results['m2_growth']

        # Add consumption (PCE) if available from the original data
        pce_series_name = self._find_pce_series(self.original_data.columns)
        if pce_series_name:
            results['pce_growth'] = self.original_data[pce_series_name]

        return results.dropna()

    def analyze_interest_rate_relationships(self, target_series: str,
                                          lasso_alpha: float = 0.001,
                                          exclude_patterns: Optional[List[str]] = None) -> Dict:
        """
        Analyze relationships between interest rates and target variable using LASSO regression.

        Parameters:
        -----------
        target_series : str
            Target series name (e.g., 'GDP', 'PCE').
        lasso_alpha : float
            LASSO regularization parameter.
        exclude_patterns : List[str], optional
            Patterns to exclude from features.

        Returns:
        --------
        Dict
            Analysis results including coefficients and predictions.
        """
        if exclude_patterns is None:
            exclude_patterns = []

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
        lasso = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=10000, tol=1e-5)
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
            'actual': y_test.values,
            'mse': mse,
            'feature_importance': coef_df.head(20)
        }

    def identify_leading_indicators(self, target_series: str,
                                      max_lag: int = 8) -> pd.DataFrame:
        """
        Identify leading indicators for a target series using cross-correlation on the cycle component.

        Parameters:
        -----------
        target_series : str
            Target series to predict (e.g., 'GDP').
        max_lag : int
            Maximum lag in quarters to consider.

        Returns:
        --------
        pd.DataFrame
            Leading indicators ranked by the absolute correlation value.
        """
        if 'cycle' not in self.components:
            logger.warning("No cycle component available for leading indicator analysis")
            return pd.DataFrame()

        # The cycle component is now a Series
        cycle_series = self.components['cycle']
        
        # We need the other series from the original data to correlate against
        all_series = self.original_data.copy()
        
        # Ensure target is in the data
        target_col_name = self._find_series_by_keyword(target_series, all_series.columns)
        if not target_col_name:
            logger.warning(f"Target series '{target_series}' not found in data.")
            return pd.DataFrame()

        correlations = []

        for col in all_series.columns:
            if col == target_col_name:
                continue

            for lag in range(1, max_lag + 1):
                # Calculate correlation between the target's cycle and the lagged original series
                lagged_series = all_series[col].shift(lag)
                corr = cycle_series.corr(lagged_series)

                if not np.isnan(corr):
                    correlations.append({
                        'indicator': col,
                        'lag': lag,
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })

        if not correlations:
            return pd.DataFrame()

        # Convert to DataFrame and sort
        corr_df = pd.DataFrame(correlations)
        return corr_df.sort_values('abs_correlation', ascending=False)

    ### Helper Methods ###

    def _find_series_by_keyword(self, keyword: str, columns: List[str]) -> Optional[str]:
        """Generic helper to find a series name by keyword."""
        for col in columns:
            if keyword.lower() in col.lower():
                return col
        return None

    def _find_gdp_series(self, columns: List[str]) -> Optional[str]:
        """Find GDP series in a list of column names."""
        return self._find_series_by_keyword('gdp', columns)

    def _find_m2_series(self, columns: List[str]) -> Optional[str]:
        """Find M2 money supply series in a list of column names."""
        return self._find_series_by_keyword('m2', columns)

    def _find_pce_series(self, columns: List[str]) -> Optional[str]:
        """Find PCE series in a list of column names."""
        return self._find_series_by_keyword('pce', columns)

    def _prepare_features_target(self, target_series: str,
                                  exclude_patterns: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for analysis from the original data."""
        
        all_features = self.original_data.copy()
        
        # Find the full name of the target column
        target_col = self._find_series_by_keyword(target_series, all_features.columns)
        if not target_col:
            logger.warning(f"Target series '{target_series}' could not be found.")
            return pd.DataFrame(), pd.Series(dtype='float64')

        target = all_features[target_col]
        
        # Exclude the target column itself
        features_to_drop = [target_col]
        # Exclude other patterns
        for pattern in exclude_patterns:
            for col in all_features.columns:
                if pattern in col:
                    features_to_drop.append(col)
                    
        features = all_features.drop(columns=list(set(features_to_drop)))
        
        # Align data and remove NaNs that may have resulted from lagging
        features, target = features.align(target, join='inner', axis=0)
        combined = pd.concat([features, target], axis=1).dropna()
        
        if combined.empty:
            return pd.DataFrame(), pd.Series(dtype='float64')
            
        target = combined[target_col]
        features = combined.drop(columns=[target_col])
        
        return features, target
