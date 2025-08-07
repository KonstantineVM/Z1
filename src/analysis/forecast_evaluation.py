"""
Forecast evaluation module for Kalman filter analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from .config_manager import ForecastConfig


@dataclass
class ForecastMetrics:
    """Container for forecast evaluation metrics."""
    rmse: float
    mae: float
    mape: float
    me: float  # Mean error (bias)
    medae: float  # Median absolute error
    n_forecasts: int
    coverage_68: float  # % of actuals within 68% CI
    coverage_95: float  # % of actuals within 95% CI
    directional_accuracy: Optional[float] = None


class ForecastEvaluator:
    """Evaluates forecast accuracy for Kalman filter models."""
    
    def __init__(self, config: ForecastConfig):
        """
        Initialize forecast evaluator.
        
        Parameters
        ----------
        config : ForecastConfig
            Configuration for forecast evaluation
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage for evaluation results
        self.forecasts = None
        self.actuals = None
        self.errors = None
        self.metrics = None
        
    def evaluate_rolling_forecasts(self, model: Any, fitted_results: Any,
                                 data: pd.DataFrame, target_series: str,
                                 start_date: Optional[pd.Timestamp] = None,
                                 end_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Evaluate rolling h-step ahead forecasts.
        
        Parameters
        ----------
        model : KalmanFilter
            Fitted Kalman filter model
        fitted_results : FilterResults
            Results from model fitting
        data : pd.DataFrame
            Full dataset
        target_series : str
            Name of target series
        start_date : pd.Timestamp, optional
            Start date for evaluation
        end_date : pd.Timestamp, optional
            End date for evaluation
            
        Returns
        -------
        Dict[str, Any]
            Evaluation statistics and results
        """
        self.logger.info(f"Evaluating {self.config.horizon}-step ahead rolling forecasts...")
        
        # Determine evaluation period
        if start_date is None:
            start_idx = int(len(data) * self.config.evaluation_start_pct)
            start_date = data.index[start_idx]
        else:
            start_idx = data.index.get_loc(start_date)
        
        if end_date is None:
            # End early enough to have actuals to compare
            end_idx = len(data) - self.config.horizon
            end_date = data.index[end_idx]
        else:
            end_idx = data.index.get_loc(end_date)
        
        self.logger.info(f"Evaluation period: {start_date} to {end_date}")
        self.logger.info(f"Number of forecasts: {end_idx - start_idx}")
        
        # Storage for results
        forecast_dates = []
        forecast_values = []
        forecast_lower_68 = []
        forecast_upper_68 = []
        forecast_lower_95 = []
        forecast_upper_95 = []
        actual_values = []
        forecast_se = []
        
        # Generate forecasts at each time point
        self.logger.info("Generating rolling forecasts...")
        
        for t in tqdm(range(start_idx, end_idx), desc="Computing forecasts"):
            base_date = data.index[t]
            
            # Get forecast using model's predict method
            forecast_result = self._generate_forecast(
                model, fitted_results, base_date, self.config.horizon, target_series
            )
            
            # Extract forecast for target series at horizon
            fc_value = forecast_result['point_forecast'][target_series].iloc[self.config.horizon - 1]
            fc_se = forecast_result['forecast_se'][target_series].iloc[self.config.horizon - 1]
            
            # Get actual value
            actual_date = data.index[t + self.config.horizon]
            actual_value = data[target_series].iloc[t + self.config.horizon]
            
            # Store results
            forecast_dates.append(base_date)
            forecast_values.append(fc_value)
            forecast_se.append(fc_se)
            actual_values.append(actual_value)
            
            # Calculate confidence intervals
            forecast_lower_68.append(fc_value - fc_se)
            forecast_upper_68.append(fc_value + fc_se)
            forecast_lower_95.append(fc_value - 2 * fc_se)
            forecast_upper_95.append(fc_value + 2 * fc_se)
        
        # Convert to arrays
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        self.forecasts = np.array(forecast_values)
        self.actuals = np.array(actual_values)
        self.errors = self.actuals - self.forecasts
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(
            self.forecasts, self.actuals, self.errors,
            np.array(forecast_lower_68), np.array(forecast_upper_68),
            np.array(forecast_lower_95), np.array(forecast_upper_95)
        )
        
        # Log summary statistics
        self._log_evaluation_summary()
        
        # Return comprehensive results
        return {
            'rmse': self.metrics.rmse,
            'mae': self.metrics.mae,
            'mape': self.metrics.mape,
            'me': self.metrics.me,
            'medae': self.metrics.medae,
            'forecast_dates': forecast_dates,
            'forecasts': self.forecasts,
            'actuals': self.actuals,
            'errors': self.errors,
            'forecast_se': np.array(forecast_se),
            'lower_68': np.array(forecast_lower_68),
            'upper_68': np.array(forecast_upper_68),
            'lower_95': np.array(forecast_lower_95),
            'upper_95': np.array(forecast_upper_95),
            'coverage_68': self.metrics.coverage_68,
            'coverage_95': self.metrics.coverage_95,
            'directional_accuracy': self.metrics.directional_accuracy,
            'metrics': self.metrics
        }
    
    def _generate_forecast(self, model, fitted_results, base_date, horizon, target_series):
        """Generate forecast from a specific base date."""
        
        # Use model's get_most_probable_path method
        path_result = model.get_most_probable_path(fitted_results, base_date=base_date)
        
        # Get the path values (these are the forecasts)
        path_values = path_result['path'][target_series]
        uncertainty_values = path_result['uncertainty'][target_series]
        
        # Extract only the values we need for the horizon
        point_forecasts = path_values.iloc[:horizon].values
        forecast_ses = uncertainty_values.iloc[:horizon].values
        
        # Create forecast DataFrame with proper index
        forecast_index = path_values.index[:horizon]
        
        return {
            'point_forecast': pd.DataFrame(
                {target_series: point_forecasts},
                index=forecast_index
            ),
            'forecast_se': pd.DataFrame(
                {target_series: forecast_ses},
                index=forecast_index
            )
        }
    
    def _calculate_metrics(self, forecasts, actuals, errors,
                          lower_68, upper_68, lower_95, upper_95):
        """Calculate comprehensive forecast metrics."""
        # Basic error metrics
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        
        # MAPE (handle zeros)
        nonzero_mask = actuals != 0
        if nonzero_mask.any():
            mape = np.mean(np.abs(errors[nonzero_mask] / actuals[nonzero_mask])) * 100
        else:
            mape = np.nan
        
        # Bias and median error
        me = np.mean(errors)
        medae = np.median(np.abs(errors))
        
        # Coverage rates
        coverage_68 = np.mean((actuals >= lower_68) & (actuals <= upper_68)) * 100
        coverage_95 = np.mean((actuals >= lower_95) & (actuals <= upper_95)) * 100
        
        # Directional accuracy
        if len(forecasts) > 1:
            forecast_changes = np.diff(forecasts)
            actual_changes = np.diff(actuals)
            same_direction = np.sign(forecast_changes) == np.sign(actual_changes)
            directional_accuracy = np.mean(same_direction) * 100
        else:
            directional_accuracy = None
        
        return ForecastMetrics(
            rmse=rmse,
            mae=mae,
            mape=mape,
            me=me,
            medae=medae,
            n_forecasts=len(forecasts),
            coverage_68=coverage_68,
            coverage_95=coverage_95,
            directional_accuracy=directional_accuracy
        )
    
    def _log_evaluation_summary(self):
        """Log evaluation summary statistics."""
        self.logger.info(f"\nForecast Evaluation Summary ({self.config.horizon}-step ahead):")
        self.logger.info(f"  Number of forecasts: {self.metrics.n_forecasts}")
        self.logger.info(f"  RMSE: {self.metrics.rmse:,.0f}")
        self.logger.info(f"  MAE: {self.metrics.mae:,.0f}")
        self.logger.info(f"  MAPE: {self.metrics.mape:.1f}%")
        self.logger.info(f"  Mean Error (Bias): {self.metrics.me:,.0f}")
        self.logger.info(f"  Coverage (68% CI): {self.metrics.coverage_68:.1f}%")
        self.logger.info(f"  Coverage (95% CI): {self.metrics.coverage_95:.1f}%")
        
        if self.metrics.directional_accuracy is not None:
            self.logger.info(f"  Directional Accuracy: {self.metrics.directional_accuracy:.1f}%")
    
    def evaluate_forecast_by_horizon(self, model, fitted_results, data, target_series,
                                   horizons: Optional[List[int]] = None) -> Dict[int, ForecastMetrics]:
        """
        Evaluate forecasts at multiple horizons.
        
        Parameters
        ----------
        model : KalmanFilter
            Fitted model
        fitted_results : FilterResults
            Fitting results
        data : pd.DataFrame
            Full dataset
        target_series : str
            Target series name
        horizons : List[int], optional
            List of horizons to evaluate
            
        Returns
        -------
        Dict[int, ForecastMetrics]
            Metrics for each horizon
        """
        if horizons is None:
            horizons = [1, 2, 4, 8]  # Default horizons
        
        self.logger.info(f"Evaluating forecasts at horizons: {horizons}")
        
        results = {}
        
        for h in horizons:
            # Temporarily update config
            original_horizon = self.config.horizon
            self.config.horizon = h
            
            # Evaluate at this horizon
            eval_results = self.evaluate_rolling_forecasts(
                model, fitted_results, data, target_series
            )
            
            results[h] = eval_results['metrics']
            
            # Restore config
            self.config.horizon = original_horizon
        
        return results
    
    def evaluate_conditional_performance(self, eval_results: Dict[str, Any],
                                       condition_series: pd.Series,
                                       n_quantiles: int = 5) -> Dict[str, ForecastMetrics]:
        """
        Evaluate forecast performance conditional on another variable.
        
        Parameters
        ----------
        eval_results : Dict[str, Any]
            Results from evaluate_rolling_forecasts
        condition_series : pd.Series
            Series to condition on (e.g., volatility, level)
        n_quantiles : int
            Number of quantiles to split conditions
            
        Returns
        -------
        Dict[str, ForecastMetrics]
            Metrics for each condition quantile
        """
        self.logger.info("Evaluating conditional forecast performance...")
        
        # Align condition series with forecast dates
        forecast_dates = eval_results['forecast_dates']
        conditions = condition_series.loc[forecast_dates]
        
        # Create quantile buckets
        quantiles = pd.qcut(conditions, n_quantiles, labels=False)
        
        conditional_results = {}
        
        for q in range(n_quantiles):
            mask = quantiles == q
            
            if mask.sum() > 10:  # Need minimum observations
                # Calculate metrics for this subset
                subset_metrics = self._calculate_metrics(
                    eval_results['forecasts'][mask],
                    eval_results['actuals'][mask],
                    eval_results['errors'][mask],
                    eval_results['lower_68'][mask],
                    eval_results['upper_68'][mask],
                    eval_results['lower_95'][mask],
                    eval_results['upper_95'][mask]
                )
                
                conditional_results[f'Q{q+1}'] = subset_metrics
                
                # Log results
                cond_range = f"{conditions[mask].min():.2f} to {conditions[mask].max():.2f}"
                self.logger.info(f"  Quantile {q+1} ({cond_range}): RMSE={subset_metrics.rmse:.0f}")
        
        return conditional_results
    
    def decompose_forecast_errors(self, model, fitted_results, eval_results: Dict[str, Any],
                                 data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Decompose forecast errors into components.
        
        Parameters
        ----------
        model : KalmanFilter
            Fitted model
        fitted_results : FilterResults
            Fitting results
        eval_results : Dict[str, Any]
            Evaluation results
        data : pd.DataFrame
            Full dataset
            
        Returns
        -------
        Dict[str, np.ndarray]
            Error components
        """
        self.logger.info("Decomposing forecast errors...")
        
        forecast_dates = eval_results['forecast_dates']
        errors = eval_results['errors']
        
        # Initialize components
        n_errors = len(errors)
        bias_component = np.full(n_errors, eval_results['me'])
        
        # Variance component (squared deviation from bias)
        variance_component = (errors - eval_results['me'])**2
        
        # Covariance component (serial correlation)
        if n_errors > 1:
            error_centered = errors - eval_results['me']
            covariance_component = np.zeros(n_errors)
            
            for i in range(1, n_errors):
                covariance_component[i] = error_centered[i] * error_centered[i-1]
        else:
            covariance_component = np.zeros(n_errors)
        
        # Model uncertainty vs observation noise
        model_uncertainty = eval_results['forecast_se']**2
        
        # Total MSE decomposition
        mse_total = errors**2
        
        components = {
            'bias_squared': bias_component**2,
            'variance': variance_component,
            'covariance': covariance_component,
            'model_uncertainty': model_uncertainty,
            'mse_total': mse_total,
            'dates': forecast_dates
        }
        
        # Log decomposition summary
        bias_pct = np.mean(bias_component**2) / np.mean(mse_total) * 100
        var_pct = np.mean(variance_component) / np.mean(mse_total) * 100
        
        self.logger.info(f"  Bias contribution to MSE: {bias_pct:.1f}%")
        self.logger.info(f"  Variance contribution to MSE: {var_pct:.1f}%")
        
        return components
    
    def test_forecast_efficiency(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test forecast efficiency (unbiasedness and no serial correlation).
        
        Parameters
        ----------
        eval_results : Dict[str, Any]
            Evaluation results
            
        Returns
        -------
        Dict[str, Any]
            Test results
        """
        from scipy import stats
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        self.logger.info("Testing forecast efficiency...")
        
        errors = eval_results['errors']
        forecasts = eval_results['forecasts']
        
        # Test 1: Unbiasedness (t-test that mean error = 0)
        t_stat, p_value_bias = stats.ttest_1samp(errors, 0)
        
        # Test 2: No serial correlation (Ljung-Box test)
        lb_result = acorr_ljungbox(errors, lags=10, return_df=True)
        p_value_serial = lb_result['lb_pvalue'].iloc[-1]  # p-value at lag 10
        
        # Test 3: Errors uncorrelated with forecasts (efficiency)
        corr_ef, p_value_efficiency = stats.pearsonr(errors, forecasts)
        
        # Test 4: Errors have no trend (regression test)
        time_trend = np.arange(len(errors))
        slope, intercept, r_value, p_value_trend, std_err = stats.linregress(time_trend, errors)
        
        efficiency_results = {
            'unbiased': {
                't_statistic': t_stat,
                'p_value': p_value_bias,
                'reject_h0': p_value_bias < 0.05,
                'interpretation': 'Biased' if p_value_bias < 0.05 else 'Unbiased'
            },
            'no_serial_correlation': {
                'ljung_box_stat': lb_result['lb_stat'].iloc[-1],
                'p_value': p_value_serial,
                'reject_h0': p_value_serial < 0.05,
                'interpretation': 'Serial correlation present' if p_value_serial < 0.05 else 'No serial correlation'
            },
            'uncorrelated_with_forecast': {
                'correlation': corr_ef,
                'p_value': p_value_efficiency,
                'reject_h0': p_value_efficiency < 0.05,
                'interpretation': 'Inefficient' if p_value_efficiency < 0.05 else 'Efficient'
            },
            'no_trend': {
                'slope': slope,
                'p_value': p_value_trend,
                'reject_h0': p_value_trend < 0.05,
                'interpretation': 'Trend present' if p_value_trend < 0.05 else 'No trend'
            }
        }
        
        # Overall assessment
        all_efficient = all(not test['reject_h0'] for test in efficiency_results.values())
        efficiency_results['overall_efficient'] = all_efficient
        
        # Log results
        self.logger.info("  Forecast Efficiency Tests:")
        for test_name, result in efficiency_results.items():
            if test_name != 'overall_efficient':
                self.logger.info(f"    {test_name}: {result['interpretation']} (p={result['p_value']:.3f})")
        self.logger.info(f"  Overall: {'Efficient' if all_efficient else 'Not efficient'}")
        
        return efficiency_results
    
    def compare_forecast_methods(self, data: pd.DataFrame, target_series: str,
                               methods: Dict[str, Any]) -> pd.DataFrame:
        """
        Compare multiple forecasting methods.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full dataset
        target_series : str
            Target series
        methods : Dict[str, Any]
            Dictionary of method_name: (model, fitted_results)
            
        Returns
        -------
        pd.DataFrame
            Comparison results
        """
        self.logger.info("Comparing forecast methods...")
        
        comparison_results = []
        
        for method_name, (model, fitted_results) in methods.items():
            self.logger.info(f"  Evaluating {method_name}...")
            
            # Evaluate this method
            eval_results = self.evaluate_rolling_forecasts(
                model, fitted_results, data, target_series
            )
            
            # Store results
            comparison_results.append({
                'Method': method_name,
                'RMSE': eval_results['rmse'],
                'MAE': eval_results['mae'],
                'MAPE': eval_results['mape'],
                'Bias': eval_results['me'],
                'Coverage_68': eval_results['coverage_68'],
                'Coverage_95': eval_results['coverage_95'],
                'Dir_Accuracy': eval_results['directional_accuracy']
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.set_index('Method')
        
        # Add relative performance
        best_rmse = comparison_df['RMSE'].min()
        comparison_df['Rel_RMSE'] = comparison_df['RMSE'] / best_rmse
        
        # Sort by RMSE
        comparison_df = comparison_df.sort_values('RMSE')
        
        self.logger.info("\nMethod Comparison Summary:")
        self.logger.info(comparison_df.to_string())
        
        return comparison_df
    
    def create_forecast_report(self, eval_results: Dict[str, Any],
                             target_series: str,
                             output_path: Optional[str] = None) -> str:
        """
        Create comprehensive forecast evaluation report.
        
        Parameters
        ----------
        eval_results : Dict[str, Any]
            Evaluation results
        target_series : str
            Target series name
        output_path : str, optional
            Path to save report
            
        Returns
        -------
        str
            Report content
        """
        report = []
        report.append("# Forecast Evaluation Report")
        report.append(f"\nTarget Series: {target_series}")
        report.append(f"Forecast Horizon: {self.config.horizon} quarters")
        report.append(f"Number of Forecasts: {eval_results['metrics'].n_forecasts}")
        report.append(f"Evaluation Period: {eval_results['forecast_dates'][0]} to {eval_results['forecast_dates'][-1]}")
        
        # Performance metrics
        report.append("\n## Performance Metrics")
        report.append(f"- RMSE: {eval_results['rmse']:,.0f}")
        report.append(f"- MAE: {eval_results['mae']:,.0f}")
        report.append(f"- MAPE: {eval_results['mape']:.1f}%")
        report.append(f"- Mean Error (Bias): {eval_results['me']:,.0f}")
        report.append(f"- Median Absolute Error: {eval_results['medae']:,.0f}")
        
        # Coverage rates
        report.append("\n## Prediction Interval Coverage")
        report.append(f"- 68% Interval: {eval_results['coverage_68']:.1f}% (Expected: 68%)")
        report.append(f"- 95% Interval: {eval_results['coverage_95']:.1f}% (Expected: 95%)")
        
        # Directional accuracy
        if eval_results['directional_accuracy'] is not None:
            report.append(f"\n## Directional Accuracy")
            report.append(f"- Correctly predicted direction: {eval_results['directional_accuracy']:.1f}%")
        
        # Error analysis
        report.append("\n## Error Analysis")
        errors = eval_results['errors']
        report.append(f"- Error Standard Deviation: {np.std(errors):,.0f}")
        report.append(f"- Error Skewness: {stats.skew(errors):.2f}")
        report.append(f"- Error Kurtosis: {stats.kurtosis(errors):.2f}")
        
        # Efficiency tests
        efficiency_results = self.test_forecast_efficiency(eval_results)
        report.append("\n## Forecast Efficiency Tests")
        for test_name, result in efficiency_results.items():
            if test_name != 'overall_efficient':
                report.append(f"- {test_name.replace('_', ' ').title()}: {result['interpretation']}")
        report.append(f"\nOverall Assessment: {'Efficient forecasts' if efficiency_results['overall_efficient'] else 'Forecasts show inefficiency'}")
        
        # Recommendations
        report.append("\n## Recommendations")
        if abs(eval_results['me']) > eval_results['mae'] * 0.2:
            report.append("- Consider bias correction methods")
        if eval_results['coverage_68'] < 60 or eval_results['coverage_68'] > 76:
            report.append("- Prediction intervals need calibration")
        if not efficiency_results['overall_efficient']:
            report.append("- Investigate sources of forecast inefficiency")
        
        report_content = '\n'.join(report)
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            self.logger.info(f"Forecast report saved to: {output_path}")
        
        return report_content
