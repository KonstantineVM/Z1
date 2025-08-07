"""
Risk indicator analysis for forecast improvement.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .config_manager import RiskIndicatorConfig


@dataclass
class CorrelationResult:
    """Results from correlation analysis."""
    series: str
    lag: int
    correlation: float
    abs_correlation: float
    n_obs: int
    series_mean: float
    series_std: float


@dataclass
class QualityMetrics:
    """Quality metrics for a series."""
    series: str
    years_of_data: float
    first_valid: pd.Timestamp
    last_valid: pd.Timestamp
    variation_pct: float
    unique_pct: float
    unique_values: int
    crisis_ratio: float
    mean: float
    std: float
    cv: float


class RiskIndicatorAnalyzer:
    """Analyzes and builds risk indicators from forecast errors."""
    
    def __init__(self, config: RiskIndicatorConfig):
        """
        Initialize risk indicator analyzer.
        
        Parameters
        ----------
        config : RiskIndicatorConfig
            Configuration for risk indicators
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage for results
        self.correlation_results = None
        self.quality_metrics = None
        self.selected_indicators = None
        self.risk_scores = None
        
    def find_error_correlates(self, data: pd.DataFrame, 
                            forecast_errors: pd.Series,
                            target_series: str) -> pd.DataFrame:
        """
        Find series that correlate with forecast errors.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full dataset
        forecast_errors : pd.Series
            Forecast errors indexed by date
        target_series : str
            Name of target series (to exclude)
            
        Returns
        -------
        pd.DataFrame
            Correlation results sorted by absolute correlation
        """
        self.logger.info(f"Finding series that correlate with {target_series} forecast errors...")
        
        # Standardize errors
        errors_std = (forecast_errors - forecast_errors.mean()) / forecast_errors.std()
        
        # Get all series except target
        all_series = [col for col in data.columns if col != target_series]
        self.logger.info(f"Checking correlations with {len(all_series)} series...")
        
        # Store results
        correlation_results = []
        
        for series_name in tqdm(all_series, desc="Computing correlations"):
            if series_name not in data.columns:
                continue
                
            series = data[series_name].dropna()
            
            # Skip if too few observations
            if len(series) < self.config.min_obs:
                continue
            
            # Standardize series
            if series.std() > 0:
                series_std = (series - series.mean()) / series.std()
            else:
                continue
            
            # Check correlations at different lags
            for lag in range(self.config.min_lag, self.config.max_lag + 1):
                # Series leads errors by 'lag' periods
                series_shifted = series_std.shift(lag)
                
                # Find common dates
                common_idx = errors_std.index.intersection(series_shifted.index)
                
                if len(common_idx) < self.config.min_obs:
                    continue
                
                # Calculate correlation
                error_aligned = errors_std.loc[common_idx]
                series_aligned = series_shifted.loc[common_idx]
                
                if len(error_aligned) > 0 and series_aligned.std() > 0:
                    corr = error_aligned.corr(series_aligned)
                    
                    if not np.isnan(corr):
                        result = CorrelationResult(
                            series=series_name,
                            lag=lag,
                            correlation=corr,
                            abs_correlation=abs(corr),
                            n_obs=len(common_idx),
                            series_mean=data[series_name].mean(),
                            series_std=data[series_name].std()
                        )
                        correlation_results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame([vars(r) for r in correlation_results])
        
        if results_df.empty:
            self.logger.warning("No valid correlations found!")
            return pd.DataFrame()
        
        # Sort by absolute correlation
        results_df = results_df.sort_values('abs_correlation', ascending=False)
        
        # Keep best lag per series
        best_results = []
        seen_series = set()
        
        for _, row in results_df.iterrows():
            if row['series'] not in seen_series:
                seen_series.add(row['series'])
                best_results.append(row)
                if len(best_results) >= self.config.top_n:
                    break
        
        self.correlation_results = pd.DataFrame(best_results)
        
        # Log top results
        self.logger.info(f"\nTop {len(self.correlation_results)} series correlating with forecast errors:")
        for _, row in self.correlation_results.head(10).iterrows():
            lag_desc = f"leads by {row['lag']}Q" if row['lag'] > 0 else "contemporaneous"
            self.logger.info(f"  {row['series']}: r={row['correlation']:.3f} ({lag_desc}, n={row['n_obs']})")
        
        return self.correlation_results
    
    def analyze_series_quality(self, data: pd.DataFrame, 
                              series_list: List[str]) -> pd.DataFrame:
        """
        Analyze quality metrics for candidate series.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full dataset
        series_list : List[str]
            List of series to analyze
            
        Returns
        -------
        pd.DataFrame
            Quality metrics for each series
        """
        quality_metrics = []
        
        # Define crisis periods for ratio calculation
        crisis_periods = [
            ('2000-01-01', '2002-12-31'),  # Dot-com
            ('2007-01-01', '2009-12-31'),  # Financial crisis
            ('2020-01-01', '2021-12-31'),  # COVID
        ]
        
        for series in series_list:
            if series not in data.columns:
                continue
                
            series_data = data[series]
            
            # Basic statistics
            first_valid = series_data.first_valid_index()
            last_valid = series_data.last_valid_index()
            
            if first_valid is None:
                continue
                
            # Years of data
            years_of_data = (last_valid - first_valid).days / 365.25
            
            # Variation metrics
            pct_change = series_data.pct_change().abs()
            meaningful_changes = (pct_change > 0.001).sum()
            variation_pct = (meaningful_changes / series_data.notna().sum()) * 100
            
            # Unique values
            unique_values = series_data.nunique()
            unique_pct = (unique_values / series_data.notna().sum()) * 100
            
            # Crisis ratio
            crisis_mask = pd.Series(False, index=data.index)
            for start, end in crisis_periods:
                crisis_mask |= (data.index >= start) & (data.index <= end)
            
            crisis_variation = pct_change[crisis_mask].mean()
            normal_variation = pct_change[~crisis_mask].mean()
            crisis_ratio = crisis_variation / (normal_variation + 1e-10)
            
            metrics = QualityMetrics(
                series=series,
                years_of_data=years_of_data,
                first_valid=first_valid,
                last_valid=last_valid,
                variation_pct=variation_pct,
                unique_pct=unique_pct,
                unique_values=unique_values,
                crisis_ratio=crisis_ratio,
                mean=series_data.mean(),
                std=series_data.std(),
                cv=series_data.std() / (abs(series_data.mean()) + 1e-10)
            )
            
            quality_metrics.append(metrics)
        
        self.quality_metrics = pd.DataFrame([vars(m) for m in quality_metrics])
        return self.quality_metrics
    
    def select_quality_indicators(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Select high-quality indicators based on correlation and quality metrics.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full dataset
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with 'risk_on' and 'risk_off' indicators
        """
        if self.correlation_results is None:
            raise ValueError("Must run find_error_correlates first")
        
        # Get all series from correlation results
        all_series = self.correlation_results['series'].unique()
        
        # Analyze quality
        quality_df = self.analyze_series_quality(data, all_series)
        
        # Apply quality filters
        quality_series = quality_df[
            (quality_df['years_of_data'] >= self.config.min_years) &
            (quality_df['variation_pct'] >= self.config.min_variation_pct) &
            (quality_df['crisis_ratio'] <= self.config.max_crisis_ratio) &
            (quality_df['unique_pct'] > 1)
        ]
        
        self.logger.info(f"Quality filters applied:")
        self.logger.info(f"  - Minimum {self.config.min_years} years of data")
        self.logger.info(f"  - Minimum {self.config.min_variation_pct}% variation")
        self.logger.info(f"  - Crisis ratio < {self.config.max_crisis_ratio}")
        self.logger.info(f"  - Started with {len(all_series)} series")
        self.logger.info(f"  - {len(quality_series)} series pass quality filters")
        
        # Merge with correlation results
        quality_correlations = self.correlation_results.merge(
            quality_series[['series', 'years_of_data', 'variation_pct', 'crisis_ratio']], 
            on='series'
        )
        
        # Separate risk-on and risk-off
        risk_on_candidates = quality_correlations[
            quality_correlations['correlation'] > self.config.min_abs_correlation
        ].sort_values('abs_correlation', ascending=False)
        
        risk_off_candidates = quality_correlations[
            quality_correlations['correlation'] < -self.config.min_abs_correlation
        ].sort_values('abs_correlation', ascending=False)
        
        # Select diverse indicators
        risk_on_selected = self._select_diverse_indicators(risk_on_candidates)
        risk_off_selected = self._select_diverse_indicators(risk_off_candidates)
        
        self.selected_indicators = {
            'risk_on': risk_on_selected,
            'risk_off': risk_off_selected,
            'quality_df': quality_df
        }
        
        return self.selected_indicators
    
    def _select_diverse_indicators(self, candidates: pd.DataFrame, 
                                  n_select: int = 10) -> pd.DataFrame:
        """Select diverse indicators with different lags and types."""
        selected = []
        used_prefixes = set()
        used_lags = {}
        
        for _, row in candidates.iterrows():
            # Extract prefix (first 3 characters)
            prefix = row['series'][:3]
            lag = row['lag']
            
            # Skip if we already have this type with similar lag
            if prefix in used_lags and abs(used_lags[prefix] - lag) < 2:
                continue
                
            selected.append(row)
            used_prefixes.add(prefix)
            used_lags[prefix] = lag
            
            if len(selected) >= n_select:
                break
                
        return pd.DataFrame(selected)
    
    def compute_rolling_ridge_weights(self, zscore_df: pd.DataFrame,
                                    forecast_errors: pd.Series) -> pd.DataFrame:
        """
        Compute time-varying weights using rolling ridge regression.
        
        Parameters
        ----------
        zscore_df : pd.DataFrame
            Panel of z-scores for indicators
        forecast_errors : pd.Series
            Forecast errors (4Q ahead)
            
        Returns
        -------
        pd.DataFrame
            Time-varying coefficients
        """
        coef_hist = []
        dates = []
        
        n_features = zscore_df.shape[1]
        window = self.config.ridge_window
        
        for t in range(window, len(zscore_df)):
            X_win = zscore_df.iloc[t-window:t]
            y_win = forecast_errors.shift(-4).iloc[t-window:t]
            
            # Create mask for non-NaN rows
            mask = X_win.notna().all(axis=1) & y_win.notna()
            
            X_sub = X_win.loc[mask]
            y_sub = y_win.loc[mask]
            
            if len(X_sub) < window * self.config.min_data_pct:
                coef_hist.append(np.full(n_features, np.nan))
                dates.append(zscore_df.index[t])
                continue
            
            try:
                model = RidgeCV(alphas=self.config.ridge_alphas, 
                              fit_intercept=False, cv=5)
                model.fit(X_sub, y_sub)
                coef_hist.append(model.coef_)
                dates.append(zscore_df.index[t])
            except Exception as e:
                self.logger.warning(f"Ridge regression failed at time {t}: {e}")
                coef_hist.append(np.full(n_features, np.nan))
                dates.append(zscore_df.index[t])
        
        coef_df = pd.DataFrame(coef_hist, index=dates, columns=zscore_df.columns)
        return coef_df
    
    def build_z_scores(self, data: pd.DataFrame, 
                      indicator_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build panel of z-scores for indicators.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full dataset
        indicator_df : pd.DataFrame
            DataFrame with indicator information
            
        Returns
        -------
        pd.DataFrame
            Panel of z-scores
        """
        z_scores = {}
        
        for _, row in indicator_df.iterrows():
            series_name = row['series']
            if series_name not in data.columns:
                continue
                
            lag = int(row['lag'])
            series = data[series_name]
            
            # Use expanding window for early periods, then rolling
            expanding_mean = series.expanding(min_periods=self.config.zscore_min_periods).mean()
            expanding_std = series.expanding(min_periods=self.config.zscore_min_periods).std()
            
            rolling_mean = series.rolling(self.config.zscore_window, 
                                        min_periods=self.config.zscore_min_periods).mean()
            rolling_std = series.rolling(self.config.zscore_window,
                                       min_periods=self.config.zscore_min_periods).std()
            
            # Combine
            threshold = self.config.zscore_expanding_threshold
            mean = expanding_mean.copy()
            std = expanding_std.copy()
            if len(series) > threshold:
                mean.iloc[threshold:] = rolling_mean.iloc[threshold:]
                std.iloc[threshold:] = rolling_std.iloc[threshold:]
            
            # Calculate z-score
            z = pd.Series(0.0, index=series.index)
            valid_mask = std > 1e-8
            z[valid_mask] = (series[valid_mask] - mean[valid_mask]) / std[valid_mask]
            
            # Clip and shift
            z = z.clip(*self.config.zscore_clip_range).shift(lag)
            z_scores[f"{series_name}_{lag}"] = z
        
        return pd.DataFrame(z_scores)
    
    def build_dynamic_risk_scores(self, data: pd.DataFrame,
                                forecast_errors: pd.Series) -> Dict[str, pd.Series]:
        """
        Build dynamic risk scores using rolling ridge regression.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full dataset
        forecast_errors : pd.Series
            Forecast errors
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary containing various risk scores
        """
        if self.selected_indicators is None:
            raise ValueError("Must run select_quality_indicators first")
        
        # Combine all selected indicators
        all_indicators = pd.concat([
            self.selected_indicators['risk_on'],
            self.selected_indicators['risk_off']
        ])
        
        # Build z-score panel
        self.logger.info("Building z-score panel...")
        z_panel = self.build_z_scores(data, all_indicators)
        
        # Compute rolling weights
        self.logger.info("Computing rolling ridge weights...")
        coef_df = self.compute_rolling_ridge_weights(z_panel, forecast_errors)
        
        # Build dynamic composite
        common_idx = z_panel.index.intersection(coef_df.index)
        beta = coef_df.loc[common_idx]
        z = z_panel.loc[common_idx]
        
        dynamic_risk = (beta * z).sum(axis=1)
        
        # Calculate thresholds
        lookback = self.config.ridge_window
        upper_threshold = dynamic_risk.rolling(lookback, min_periods=40).quantile(0.85)
        lower_threshold = dynamic_risk.rolling(lookback, min_periods=40).quantile(0.15)
        
        # Build traditional composites for comparison
        risk_on_composite, risk_off_composite = self._build_traditional_composites(
            data, self.selected_indicators['risk_on'], self.selected_indicators['risk_off']
        )
        
        self.risk_scores = {
            'dynamic_risk': dynamic_risk,
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'risk_on_composite': risk_on_composite,
            'risk_off_composite': risk_off_composite,
            'net_risk_traditional': risk_on_composite - risk_off_composite,
            'z_panel': z_panel,
            'coef_df': coef_df
        }
        
        return self.risk_scores
    
    def _build_traditional_composites(self, data: pd.DataFrame,
                                    risk_on_df: pd.DataFrame,
                                    risk_off_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Build traditional weighted composites."""
        risk_on_composite = pd.Series(0.0, index=data.index)
        risk_off_composite = pd.Series(0.0, index=data.index)
        
        # Risk-on composite
        risk_on_weights = []
        for _, row in risk_on_df.iterrows():
            z_scores = self._calculate_single_zscore(data, row['series'])
            if z_scores is not None:
                weight = abs(row['correlation'])
                risk_on_composite += z_scores * weight
                risk_on_weights.append(weight)
        
        if risk_on_weights:
            risk_on_composite = risk_on_composite / sum(risk_on_weights)
        
        # Risk-off composite
        risk_off_weights = []
        for _, row in risk_off_df.iterrows():
            z_scores = self._calculate_single_zscore(data, row['series'])
            if z_scores is not None:
                weight = abs(row['correlation'])
                risk_off_composite += z_scores * weight
                risk_off_weights.append(weight)
        
        if risk_off_weights:
            risk_off_composite = risk_off_composite / sum(risk_off_weights)
        
        return risk_on_composite, risk_off_composite
    
    def _calculate_single_zscore(self, data: pd.DataFrame, 
                               series_name: str) -> Optional[pd.Series]:
        """Calculate z-score for a single series."""
        if series_name not in data.columns:
            return None
            
        series = data[series_name]
        
        # Use same logic as build_z_scores
        expanding_mean = series.expanding(min_periods=self.config.zscore_min_periods).mean()
        expanding_std = series.expanding(min_periods=self.config.zscore_min_periods).std()
        
        rolling_mean = series.rolling(self.config.zscore_window, 
                                    min_periods=self.config.zscore_min_periods).mean()
        rolling_std = series.rolling(self.config.zscore_window,
                                   min_periods=self.config.zscore_min_periods).std()
        
        threshold = self.config.zscore_expanding_threshold
        mean = expanding_mean.copy()
        std = expanding_std.copy()
        if len(series) > threshold:
            mean.iloc[threshold:] = rolling_mean.iloc[threshold:]
            std.iloc[threshold:] = rolling_std.iloc[threshold:]
        
        z = pd.Series(0.0, index=series.index)
        valid_mask = std > 1e-8
        z[valid_mask] = (series[valid_mask] - mean[valid_mask]) / std[valid_mask]
        
        return z.clip(*self.config.zscore_clip_range)
