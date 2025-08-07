"""
Refactored main driver for Kalman filter analysis.
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from .config_manager import ConfigManager
from .results_manager import ResultsManager
from .data_pipeline import DataPipeline
from .risk_indicators import RiskIndicatorAnalyzer
from models.hierarchical_kalman_filter import fit_hierarchical_kalman_filter


class KalmanFilterAnalysis:
    """Main analysis orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Kalman filter analysis.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        """
        # Initialize configuration
        self.config = ConfigManager(config_path)
        
        # Initialize results manager
        self.results = ResultsManager(
            output_dir=self.config.data.output_dir,
            timestamp_outputs=self.config.timestamp_outputs
        )
        
        # Initialize data pipeline
        self.pipeline = DataPipeline(self.config, self.results)
        
        # Initialize risk analyzer
        self.risk_analyzer = RiskIndicatorAnalyzer(self.config.risk_indicators)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Storage for main components
        self.data = None
        self.model = None
        self.fitted_results = None
        self.eval_stats = None
        
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        self.logger.info("="*60)
        self.logger.info("Starting Kalman Filter Analysis")
        self.logger.info("="*60)
        
        try:
            # Step 1: Load and prepare data
            self._load_data()
            
            # Step 2: Fit Kalman filter model
            self._fit_model()
            
            # Step 3: Evaluate forecasts
            self._evaluate_forecasts()
            
            # Step 4: Analyze risk indicators
            self._analyze_risk_indicators()
            
            # Step 5: Generate visualizations
            self._create_visualizations()
            
            # Step 6: Export results
            self._export_results()
            
            # Create summary report
            report_path = self.results.create_summary_report()
            self.logger.info(f"\nAnalysis complete! Summary report: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise
    
    def _load_data(self):
        """Load and prepare data."""
        self.logger.info("\nStep 1: Loading data...")
        
        # Run data pipeline
        pipeline_results = self.pipeline.run_full_pipeline()
        self.data = pipeline_results['data']
        
        # Set metadata
        self.results.set_metadata(
            config_path=str(self.config.config_path),
            target_series=self.config.model.target_series,
            data=self.data
        )
        
        # Save raw data info
        self.results.add_dataframe(
            'data_summary',
            self.data.describe(),
            subdir='diagnostics'
        )
    
    def _fit_model(self):
        """Fit Kalman filter model."""
        self.logger.info("\nStep 2: Fitting Kalman filter model...")
        
        # Fit model
        self.model, results = fit_hierarchical_kalman_filter(
            data=self.data,
            formulas_file=self.config.model.formulas_file,
            series_list=[self.config.model.target_series],
            normalize=self.config.model.normalize,
            error_variance_ratio=self.config.model.error_variance_ratio,
            loglikelihood_burn=self.config.model.loglikelihood_burn,
            use_exact_diffuse=self.config.model.use_exact_diffuse,
            transformation=self.config.model.transformation,
            max_attempts=self.config.model.max_attempts
        )
        
        self.fitted_results = results['fitted_results']
        
        # Debug: Print model structure
        print("\n=== MODEL STRUCTURE DEBUG ===")
        print(f"Model type: {type(self.model)}")
        print(f"Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")

        # Check for data-related attributes
        if hasattr(self.model, 'data'):
            print(f"model.data type: {type(self.model.data)}")
            if hasattr(self.model.data, 'index'):
                print(f"model.data.index exists: {self.model.data.index[:5]}")
                
        if hasattr(self.model, 'data_normalized'):
            print(f"model.data_normalized type: {type(self.model.data_normalized)}")
            print(f"model.data_normalized.index[:5]: {self.model.data_normalized.index[:5]}")
            
        if hasattr(self.model, 'endog'):
            print(f"model.endog shape: {self.model.endog.shape}")
            
        print("=== END DEBUG ===\n")        
        
        # Store model results
        model_info = {
            'log_likelihood': self.fitted_results.llf,
            'aic': self.fitted_results.aic,
            'n_parameters': len(self.fitted_results.params),
            'parameters': self.fitted_results.params.tolist(),
            'param_names': self.model.param_names
        }
        
        self.results.save_model_results('kalman_model', model_info)
        self.results.add_result('log_likelihood', self.fitted_results.llf, 'model')
        self.results.add_result('aic', self.fitted_results.aic, 'model')
        
        # Save filtered series
        filtered_series = results['filtered_series']
        self.results.add_dataframe(
            'filtered_series',
            filtered_series['filtered'],
            subdir='model_output'
        )
        self.results.add_dataframe(
            'smoothed_series',
            filtered_series['smoothed'],
            subdir='model_output'
        )
    
    def _evaluate_forecasts(self):
        """Evaluate forecast accuracy."""
        self.logger.info("\nStep 3: Evaluating forecasts...")
        
        # Generate and evaluate 4-quarter ahead forecasts
        from .forecast_evaluation import ForecastEvaluator
        
        evaluator = ForecastEvaluator(self.config.forecast)
        self.eval_stats = evaluator.evaluate_rolling_forecasts(
            model=self.model,
            fitted_results=self.fitted_results,
            data=self.data,
            target_series=self.config.model.target_series
        )
        
        # Store evaluation results
        self.results.add_result('forecast_rmse', self.eval_stats['rmse'], 'evaluation')
        self.results.add_result('forecast_mae', self.eval_stats['mae'], 'evaluation')
        self.results.add_result('forecast_mape', self.eval_stats['mape'], 'evaluation')
        
        # Save detailed evaluation
        eval_df = pd.DataFrame({
            'forecast_date': self.eval_stats['forecast_dates'],
            'forecast': self.eval_stats['forecasts'],
            'actual': self.eval_stats['actuals'],
            'error': self.eval_stats['errors']
        })
        self.results.add_dataframe('forecast_evaluation', eval_df, subdir='evaluation')
    
    def _analyze_risk_indicators(self):
        """Analyze risk indicators."""
        self.logger.info("\nStep 4: Analyzing risk indicators...")
        
        # Find error correlates
        forecast_errors = pd.Series(
            self.eval_stats['errors'],
            index=self.eval_stats['forecast_dates']
        )
        
        correlations = self.risk_analyzer.find_error_correlates(
            data=self.data,
            forecast_errors=forecast_errors,
            target_series=self.config.model.target_series
        )
        
        self.results.add_dataframe(
            'error_correlations',
            correlations,
            subdir='risk_analysis'
        )
        
        # Select quality indicators
        selected = self.risk_analyzer.select_quality_indicators(self.data)
        
        self.results.add_dataframe(
            'selected_risk_indicators',
            pd.concat([selected['risk_on'], selected['risk_off']]),
            subdir='risk_analysis'
        )
        
        # Build dynamic risk scores
        risk_scores = self.risk_analyzer.build_dynamic_risk_scores(
            self.data,
            forecast_errors
        )
        
        # Save risk scores
        risk_df = pd.DataFrame({
            'dynamic_risk': risk_scores['dynamic_risk'],
            'upper_threshold': risk_scores['upper_threshold'],
            'lower_threshold': risk_scores['lower_threshold'],
            'risk_on_composite': risk_scores['risk_on_composite'],
            'risk_off_composite': risk_scores['risk_off_composite']
        })
        self.results.add_dataframe('risk_scores', risk_df, subdir='risk_analysis')
        
        # Store key risk metrics
        current_risk = risk_scores['dynamic_risk'].iloc[-1]
        self.results.add_result('current_risk_score', current_risk, 'risk')
        self.results.add_result('risk_regime', self._determine_regime(current_risk), 'risk')
        
    def _train_error_model(self):
        """Train XGBoost model for error prediction."""
        self.logger.info("\nStep 4b: Training XGBoost error model...")
        
        from .xgboost_error_model import KalmanErrorPredictor, ErrorModelConfig
        
        # Get config
        xgb_config = ErrorModelConfig(
            use_differences=self.config.xgboost_error_model.get('use_differences', True),
            use_growth_rates=self.config.xgboost_error_model.get('use_growth_rates', True),
            n_features_select=self.config.xgboost_error_model.get('n_features_select', 100)
        )
        
        # Initialize and train
        self.error_model = KalmanErrorPredictor(xgb_config)
        self.error_model.train(self.data, self.eval_stats['errors'])
        
        # Save feature importance
        feature_importance = self.error_model.get_feature_importance(30)
        self.results.add_dataframe(
            'xgboost_feature_importance',
            feature_importance,
            subdir='model_output'
        )        
    
    def _determine_regime(self, risk_score: float) -> str:
        """Determine risk regime from score."""
        if risk_score > 1:
            return "STRONG RISK-ON"
        elif risk_score > 0:
            return "MILD RISK-ON"
        elif risk_score > -1:
            return "MILD RISK-OFF"
        else:
            return "STRONG RISK-OFF"
    
    def _create_visualizations(self):
        """Create all visualizations."""
        self.logger.info("\nStep 5: Creating visualizations...")
        
        from .visualization import VisualizationManager
        
        viz_manager = VisualizationManager(self.config.plots, self.results)
        
        # Create standard plots
        viz_manager.create_model_diagnostics(
            self.data,
            self.model,
            self.fitted_results,
            self.config.model.target_series
        )
        
        viz_manager.create_forecast_evaluation_plots(
            self.eval_stats,
            self.config.model.target_series
        )
        
        viz_manager.create_risk_indicator_dashboard(
            self.risk_analyzer.risk_scores,
            self.data
        )
        
        # Create recession analysis if enabled
        if hasattr(self.config, 'recession'):
            from .recession_analysis import RecessionAnalyzer
            
            recession_analyzer = RecessionAnalyzer(self.config.recession)
            recession_results = recession_analyzer.analyze_recession_risk(
                risk_scores=self.risk_analyzer.risk_scores,
                data=self.data
            )
            
            viz_manager.create_recession_analysis_plots(
                recession_results,
                self.risk_analyzer.risk_scores
            )
    
    def _export_results(self):
        """Export all results."""
        self.logger.info("\nStep 6: Exporting results...")
        
        # Key results summary
        key_results = {
            'target_series': self.config.model.target_series,
            'data_range': {
                'start': str(self.data.index[0]),
                'end': str(self.data.index[-1]),
                'n_observations': len(self.data)
            },
            'model_fit': {
                'log_likelihood': float(self.fitted_results.llf),
                'aic': float(self.fitted_results.aic),
                'n_parameters': len(self.fitted_results.params)
            },
            'forecast_performance': {
                'rmse': float(self.eval_stats['rmse']),
                'mae': float(self.eval_stats['mae']),
                'mape': float(self.eval_stats['mape'])
            },
            'current_risk': {
                'score': float(self.risk_analyzer.risk_scores['dynamic_risk'].iloc[-1]),
                'regime': self._determine_regime(
                    self.risk_analyzer.risk_scores['dynamic_risk'].iloc[-1]
                )
            }
        }
        
        self.results.export_key_results(key_results)
        
        # Save diagnostics
        diagnostics = {
            'config': vars(self.config),
            'model_diagnostics': {
                'transformation': self.model.transformation,
                'normalization_scale': float(self.model.scale_factor),
                'n_source_series': len(self.model.source_series),
                'n_computed_series': len(self.model.computed_series)
            },
            'risk_indicators': {
                'n_correlations_tested': len(self.risk_analyzer.correlation_results),
                'n_quality_indicators': len(self.risk_analyzer.selected_indicators['risk_on']) + 
                                       len(self.risk_analyzer.selected_indicators['risk_off']),
                'ridge_window': self.config.risk_indicators.ridge_window
            }
        }
        
        self.results.save_diagnostics(diagnostics)
        
        # List all outputs
        outputs = self.results.list_outputs()
        self.logger.info("\nGenerated outputs:")
        for category, files in outputs.items():
            if files:
                self.logger.info(f"\n{category.capitalize()}:")
                for file in files[:5]:  # Show first 5
                    self.logger.info(f"  - {file}")
                if len(files) > 5:
                    self.logger.info(f"  ... and {len(files)-5} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Kalman Filter Analysis')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--target',
        type=str,
        help='Override target series from config'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Disable timestamp in output filenames'
    )
    
    args = parser.parse_args()
    
    # Create analysis instance
    analysis = KalmanFilterAnalysis(args.config)
    
    # Override config if needed
    if args.target:
        analysis.config.model.target_series = args.target
    if args.output_dir:
        analysis.config.data.output_dir = args.output_dir
    if args.no_timestamp:
        analysis.config.timestamp_outputs = False
        analysis.results.timestamp_outputs = False
    
    # Run analysis
    analysis.run_analysis()


if __name__ == "__main__":
    main()
