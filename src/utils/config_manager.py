"""
Configuration management for Kalman filter analysis.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data-related configuration."""
    input_path: str
    output_dir: str = "output"
    cache_dir: str = "cache"
    end_date: Optional[str] = None
    
    def __post_init__(self):
        # Create directories if they don't exist
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.cache_dir).mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Kalman filter model configuration."""
    target_series: str
    formulas_file: str
    normalize: bool = True
    error_variance_ratio: float = 0.01
    loglikelihood_burn: int = 20
    use_exact_diffuse: bool = False
    transformation: str = "square"
    max_attempts: int = 3


@dataclass
class RiskIndicatorConfig:
    """Risk indicator configuration."""
    # Correlation settings
    min_lag: int = 4
    max_lag: int = 8
    top_n: int = 40
    min_obs: int = 200
    
    # Quality filters
    min_years: int = 30
    min_variation_pct: float = 10.0
    max_crisis_ratio: float = 5.0
    min_abs_correlation: float = 0.5
    
    # Ridge regression
    ridge_window: int = 60
    ridge_alphas: list = field(default_factory=lambda: [0.1, 1.0, 10.0])
    min_data_pct: float = 0.7
    
    # Z-score settings
    zscore_window: int = 20
    zscore_min_periods: int = 4
    zscore_expanding_threshold: int = 40
    zscore_clip_range: list = field(default_factory=lambda: [-3, 3])


@dataclass
class ForecastConfig:
    """Forecast configuration."""
    horizon: int = 4
    confidence_levels: list = field(default_factory=lambda: [0.68, 0.95])
    evaluation_start_pct: float = 0.25


@dataclass
class RecessionConfig:
    """Recession analysis configuration."""
    probability_window: int = 40
    high_risk_threshold: float = 0.7
    risk_on_percentile: float = 0.85
    risk_off_percentile: float = 0.15


@dataclass
class PlotConfig:
    """Plotting configuration."""
    dpi: int = 150
    figure_sizes: Dict[str, list] = field(default_factory=dict)
    color_scheme: Dict[str, str] = field(default_factory=dict)
    save_plots: bool = True
    
    def __post_init__(self):
        # Default figure sizes
        if not self.figure_sizes:
            self.figure_sizes = {
                'standard': [12, 8],
                'large': [16, 12],
                'dashboard': [16, 14]
            }
        
        # Default color scheme
        if not self.color_scheme:
            self.color_scheme = {
                'risk_on': 'green',
                'risk_off': 'red',
                'neutral': 'blue',
                'recession': 'gray'
            }


class ConfigManager:
    """Manages configuration for the entire analysis pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file. If None, uses default values.
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        if config_path and Path(config_path).exists():
            self._load_from_file(config_path)
        else:
            self._set_defaults()
            if config_path:
                self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
    
    def _load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse configuration sections
        self.data = DataConfig(**config_dict.get('data', {}))
        self.model = ModelConfig(**config_dict.get('model', {}))
        
        # Hybrid model settings
        self.use_hybrid_error_model = config_dict.get('use_hybrid_error_model', False)
        if 'hybrid_error_model' in config_dict:
            self.hybrid_error_model = config_dict['hybrid_error_model']        
        
        # Risk indicators need special handling for nested structure
        risk_config = config_dict.get('risk_indicators', {})
        risk_flat = {}
        
        # Flatten the nested structure
        if 'correlation' in risk_config:
            risk_flat.update(risk_config['correlation'])
        if 'quality' in risk_config:
            risk_flat.update(risk_config['quality'])
        if 'ridge' in risk_config:
            ridge = risk_config['ridge']
            risk_flat['ridge_window'] = ridge.get('window', 60)
            risk_flat['ridge_alphas'] = ridge.get('alphas', [0.1, 1.0, 10.0])
            risk_flat['min_data_pct'] = ridge.get('min_data_pct', 0.7)
        if 'zscore' in risk_config:
            zscore = risk_config['zscore']
            risk_flat['zscore_window'] = zscore.get('rolling_window', 20)
            risk_flat['zscore_min_periods'] = zscore.get('min_periods', 4)
            risk_flat['zscore_expanding_threshold'] = zscore.get('expanding_threshold', 40)
            risk_flat['zscore_clip_range'] = zscore.get('clip_range', [-3, 3])
        
        self.risk_indicators = RiskIndicatorConfig(**risk_flat)
        self.forecast = ForecastConfig(**config_dict.get('forecast', {}))
        self.recession = RecessionConfig(**config_dict.get('recession', {}))
        self.plots = PlotConfig(**config_dict.get('plots', {}))
        
        # Output settings
        output = config_dict.get('output', {})
        self.save_plots = output.get('save_plots', True)
        self.save_csv = output.get('save_csv', True)
        self.save_diagnostics = output.get('save_diagnostics', True)
        self.timestamp_outputs = output.get('timestamp_outputs', True)
        
        # Logging settings
        log_config = config_dict.get('logging', {})
        self._setup_logging(log_config.get('level', 'INFO'),
                           log_config.get('format'))
    
    def _set_defaults(self):
        """Set default configuration values."""
        self.data = DataConfig(
            input_path="/home/tesla/Z1/temp/data/z1_quarterly/z1_quarterly_data_filtered.parquet"
        )
        self.model = ModelConfig(
            target_series="FA086902005",
            formulas_file="fof_formulas_extracted.json"
        )
        self.risk_indicators = RiskIndicatorConfig()
        self.forecast = ForecastConfig()
        self.recession = RecessionConfig()
        self.plots = PlotConfig()
        
        self.save_plots = True
        self.save_csv = True
        self.save_diagnostics = True
        self.timestamp_outputs = True
        
        self._setup_logging()
    
    def _setup_logging(self, level: str = "INFO", 
                      format_str: Optional[str] = None):
        """Configure logging."""
        if format_str is None:
            format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_str
        )
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file."""
        save_path = path or self.config_path or "config_saved.yaml"
        
        config_dict = {
            'data': {
                'input_path': self.data.input_path,
                'output_dir': self.data.output_dir,
                'cache_dir': self.data.cache_dir
            },
            'model': {
                'target_series': self.model.target_series,
                'formulas_file': self.model.formulas_file,
                'normalize': self.model.normalize,
                'error_variance_ratio': self.model.error_variance_ratio,
                'loglikelihood_burn': self.model.loglikelihood_burn,
                'use_exact_diffuse': self.model.use_exact_diffuse,
                'transformation': self.model.transformation,
                'max_attempts': self.model.max_attempts
            },
            'risk_indicators': {
                'correlation': {
                    'min_lag': self.risk_indicators.min_lag,
                    'max_lag': self.risk_indicators.max_lag,
                    'top_n': self.risk_indicators.top_n,
                    'min_obs': self.risk_indicators.min_obs
                },
                'quality': {
                    'min_years': self.risk_indicators.min_years,
                    'min_variation_pct': self.risk_indicators.min_variation_pct,
                    'max_crisis_ratio': self.risk_indicators.max_crisis_ratio,
                    'min_abs_correlation': self.risk_indicators.min_abs_correlation
                },
                'ridge': {
                    'window': self.risk_indicators.ridge_window,
                    'alphas': self.risk_indicators.ridge_alphas,
                    'min_data_pct': self.risk_indicators.min_data_pct
                },
                'zscore': {
                    'rolling_window': self.risk_indicators.zscore_window,
                    'min_periods': self.risk_indicators.zscore_min_periods,
                    'expanding_threshold': self.risk_indicators.zscore_expanding_threshold,
                    'clip_range': self.risk_indicators.zscore_clip_range
                }
            },
            'forecast': {
                'horizon': self.forecast.horizon,
                'confidence_levels': self.forecast.confidence_levels,
                'evaluation_start_pct': self.forecast.evaluation_start_pct
            },
            'recession': {
                'probability_window': self.recession.probability_window,
                'high_risk_threshold': self.recession.high_risk_threshold,
                'risk_on_percentile': self.recession.risk_on_percentile,
                'risk_off_percentile': self.recession.risk_off_percentile
            },
            'plots': {
                'dpi': self.plots.dpi,
                'figure_sizes': self.plots.figure_sizes,
                'color_scheme': self.plots.color_scheme,
                'save_plots': self.plots.save_plots
            },
            'output': {
                'save_plots': self.save_plots,
                'save_csv': self.save_csv,
                'save_diagnostics': self.save_diagnostics,
                'timestamp_outputs': self.timestamp_outputs
            }
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        self.logger.info(f"Configuration saved to: {save_path}")
    
    def __repr__(self):
        return f"ConfigManager(config_path='{self.config_path}')"
