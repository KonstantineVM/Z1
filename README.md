# Economic Time Series Analysis Framework

A comprehensive Python framework for analyzing Federal Reserve economic time series data using state-of-the-art decomposition and machine learning techniques.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Cache Management](#cache-management)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides a robust framework for analyzing Federal Reserve economic data, with a particular focus on the Z1 Flow of Funds dataset. It combines advanced time series decomposition methods with modern machine learning approaches to uncover complex economic relationships.

### Key Capabilities
- **Automated Data Collection**: Direct integration with Federal Reserve data sources
- **Advanced Analytics**: State-of-the-art time series decomposition and ML models
- **Production Ready**: Comprehensive caching, error handling, and logging
- **Extensible Design**: Easy to add new data sources and models

## Features

### 🔄 Data Management
- **Federal Reserve Data**: Automated downloading and parsing of Z1, H6, H8, G17, G19, G20, H15 datasets
- **External Sources**: Integration with S&P 500, FRED series, gold prices, and custom data
- **Smart Caching**: Intelligent caching system with configurable expiry and refresh
- **Data Validation**: Automatic validation and preprocessing with missing data handling

### 📊 Modeling Capabilities
- **Unobserved Components**: Extract trend, cycle, seasonal, and irregular components
- **Tree Models**: XGBoost and LightGBM with feature importance and decision paths
- **Gaussian Processes**: Advanced regression with uncertainty quantification
- **Neural Networks**: Deep learning enhanced models for complex patterns

### 📈 Economic Analysis
- **Velocity of Money**: Analyze money velocity relationships across sectors
- **Interest Rate Impact**: Study rate changes on economic indicators
- **Savings Dynamics**: Investigate savings patterns and drivers
- **Leading Indicators**: Identify and validate economic leading indicators

### 📉 Visualization Suite
- **Interactive Dashboards**: Plotly-based interactive visualizations
- **Publication Quality**: Matplotlib charts with NBER recession shading
- **Component Analysis**: Detailed decomposition visualizations
- **Uncertainty Plots**: Confidence intervals and prediction bands

## FWTW Network Analysis

The project now includes Flow of Funds Through Wall Street (FWTW) network analysis capabilities:

### Features
- Download and cache FWTW data from Federal Reserve
- Build directed financial networks from flow data
- Analyze systemic risk and identify SIFIs
- Model shock propagation through the financial system
- Integrate network metrics with time series analysis

### Quick Start
```python
from src.network import FWTWDataLoader, NetworkBuilder, NetworkAnalyzer

# Load FWTW data
loader = FWTWDataLoader()
data = loader.load_fwtw_data()

# Build network
builder = NetworkBuilder(data)
network = builder.build_snapshot(data['Date'].max())

# Analyze
analyzer = NetworkAnalyzer(network)
sifis = analyzer.identify_systemically_important()

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- Git

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/economic-timeseries-analysis.git
cd economic-timeseries-analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Docker Installation

```bash
# Build the Docker image
docker build -t econ-timeseries .

# Run the container
docker run -it -v $(pwd)/data:/app/data econ-timeseries
```

## Quick Start

```python
from src.data import CachedFedDataLoader
from src.models import UnobservedComponentsModel
from src.analysis import EconomicAnalysis
from src.visualization import plot_economic_relationships

# Load data with automatic caching
loader = CachedFedDataLoader()
data = loader.load_source('Z1')

# Decompose time series
model = UnobservedComponentsModel()
components = model.decompose(data)

# Analyze economic relationships
analyzer = EconomicAnalysis(components)
results = analyzer.analyze_velocity_of_money()

# Visualize results
plot_economic_relationships(results, show_recessions=True)
```

## Detailed Usage

### 1. Loading Federal Reserve Data

```python
from src.data import CachedFedDataLoader

# Initialize loader with custom parameters
loader = CachedFedDataLoader(
    cache_directory="./data/cache",
    cache_expiry_days=7,
    start_year=1960,
    end_year=2024
)

# Load single source
z1_data = loader.load_source('Z1')

# Load multiple sources in parallel
sources = ['Z1', 'H6', 'H15']
all_data = loader.load_multiple_sources(sources, n_jobs=4)

# Force refresh (bypass cache)
fresh_data = loader.load_source('Z1', force_refresh=True)
```

### 2. Data Preprocessing

```python
from src.data import DataProcessor

processor = DataProcessor()

# Clean and validate data
cleaned_data = processor.clean_data(z1_data)

# Handle missing values
filled_data = processor.handle_missing_values(
    cleaned_data, 
    method='interpolate',
    limit=3
)

# Normalize series for comparison
normalized_data = processor.normalize_series(
    filled_data,
    method='z-score'
)
```

### 3. Time Series Decomposition

```python
from src.models import UnobservedComponentsModel

# Initialize model
uc_model = UnobservedComponentsModel(
    cycle_period=8*4,  # 8 years in quarters
    damping_factor=0.98
)

# Decompose single series
components = uc_model.decompose(series)

# Decompose multiple series in parallel
all_components = uc_model.decompose_parallel(
    data_dict,
    n_jobs=4,
    show_progress=True
)

# Access components
trend = components['trend']
cycle = components['cycle']
seasonal = components['seasonal']
irregular = components['irregular']
```

### 4. Machine Learning Models

```python
from src.models import TreeEnsembleModel, GaussianProcessModel

# Tree-based models
tree_model = TreeEnsembleModel(
    model_type='xgboost',
    n_estimators=100,
    max_depth=5
)

# Train model
tree_model.fit(X_train, y_train)

# Get predictions with feature importance
predictions, importance = tree_model.predict(
    X_test, 
    return_importance=True
)

# Gaussian Process regression
gp_model = GaussianProcessModel(
    kernel='matern',
    length_scale=1.0,
    nu=2.5
)

# Fit and predict with uncertainty
gp_model.fit(X_train, y_train)
mean, std = gp_model.predict(X_test, return_std=True)
```

### 5. Economic Analysis

```python
from src.analysis import EconomicAnalysis, FeatureEngineer

# Feature engineering
engineer = FeatureEngineer()
features = engineer.create_features(
    data,
    lags=[1, 4, 8],
    rolling_windows=[4, 8, 12],
    include_interactions=True
)

# Economic analysis
analyzer = EconomicAnalysis(data)

# Velocity of money analysis
velocity_results = analyzer.analyze_velocity_of_money(
    money_supply_series='M2',
    gdp_series='GDP'
)

# Interest rate impact
rate_impact = analyzer.analyze_interest_rate_impact(
    rate_series='DFF',
    target_variables=['consumption', 'investment']
)

# Find leading indicators
leading_indicators = analyzer.find_leading_indicators(
    target='GDP',
    max_lag=8,
    significance_level=0.05
)
```

### 6. Visualization

```python
from src.visualization import EconomicVisualizer

viz = EconomicVisualizer()

# Component plots
viz.plot_decomposition(
    components,
    title="GDP Components",
    show_recessions=True
)

# Economic relationships
viz.plot_relationships(
    x=velocity_results['velocity'],
    y=velocity_results['inflation'],
    title="Velocity vs Inflation",
    add_regression=True
)

# Feature importance
viz.plot_feature_importance(
    importance_df,
    top_n=20,
    orientation='horizontal'
)

# Forecast with uncertainty
viz.plot_forecast(
    actual=y_test,
    predicted=predictions,
    lower_bound=predictions - 2*std,
    upper_bound=predictions + 2*std,
    title="GDP Forecast with 95% CI"
)
```

## API Reference

### Data Module

#### `CachedFedDataLoader`
Main class for loading Federal Reserve data with caching support.

**Methods:**
- `load_source(source: str, force_refresh: bool = False) -> pd.DataFrame`
- `load_multiple_sources(sources: List[str], n_jobs: int = 1) -> Dict[str, pd.DataFrame]`
- `clear_cache(source: Optional[str] = None) -> None`
- `get_cache_info() -> Dict[str, Any]`

#### `DataProcessor`
Handles data cleaning and preprocessing.

**Methods:**
- `clean_data(data: pd.DataFrame) -> pd.DataFrame`
- `handle_missing_values(data: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame`
- `normalize_series(data: pd.DataFrame, method: str = 'z-score') -> pd.DataFrame`

### Models Module

#### `UnobservedComponentsModel`
State space model for time series decomposition.

**Parameters:**
- `cycle_period`: int, expected cycle length
- `damping_factor`: float, cycle damping (0-1)
- `stochastic_level`: bool, allow stochastic trend
- `stochastic_cycle`: bool, allow stochastic cycle

**Methods:**
- `decompose(series: pd.Series) -> Dict[str, pd.Series]`
- `decompose_parallel(data: Dict[str, pd.Series], n_jobs: int = 1) -> Dict[str, pd.DataFrame]`

#### `TreeEnsembleModel`
XGBoost and LightGBM wrapper with feature importance.

**Parameters:**
- `model_type`: str, 'xgboost' or 'lightgbm'
- `**kwargs`: model-specific parameters

**Methods:**
- `fit(X: pd.DataFrame, y: pd.Series) -> None`
- `predict(X: pd.DataFrame, return_importance: bool = False) -> Union[np.ndarray, Tuple]`
- `get_feature_importance() -> pd.DataFrame`

### Analysis Module

#### `EconomicAnalysis`
Tools for economic relationship analysis.

**Methods:**
- `analyze_velocity_of_money() -> Dict[str, pd.Series]`
- `analyze_interest_rate_impact(rate_series: str, targets: List[str]) -> pd.DataFrame`
- `find_leading_indicators(target: str, max_lag: int = 12) -> pd.DataFrame`

## Configuration

### config.yaml

```yaml
# Data configuration
data:
  cache_directory: "./data/cache"
  cache_expiry_days: 7
  force_download: false
  start_year: 1960
  end_year: 2024
  
# Model configuration  
models:
  unobserved_components:
    cycle_period: 32  # 8 years for quarterly data
    damping_factor: 0.98
    
  tree_ensemble:
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
    
# Analysis configuration
analysis:
  significance_level: 0.05
  max_lags: 12
  rolling_window_sizes: [4, 8, 12, 20]
  
# Visualization configuration
visualization:
  style: "seaborn"
  figsize: [12, 8]
  dpi: 300
  show_recessions: true
```

## Cache Management

The framework includes intelligent caching to minimize redundant downloads:

### Command Line Interface

```bash
# Check cache status
python -m src.data.cache_manager status

# Clear all cache
python -m src.data.cache_manager clear

# Clear specific source
python -m src.data.cache_manager clear --source Z1

# Refresh cache
python -m src.data.cache_manager refresh --sources Z1 H6

# Set cache expiry
python -m src.data.cache_manager config --expiry-days 14
```

### Programmatic Access

```python
from src.data import CacheManager

manager = CacheManager()

# Get cache statistics
stats = manager.get_stats()
print(f"Total cache size: {stats['total_size_mb']} MB")
print(f"Number of cached sources: {stats['num_sources']}")

# Clear expired cache
manager.clear_expired()

# Force refresh specific source
manager.refresh_source('Z1')
```

## Examples

### Example 1: GDP Decomposition and Forecasting

```python
# Load GDP data
loader = CachedFedDataLoader()
gdp_data = loader.load_source('Z1')['GDP']

# Decompose
model = UnobservedComponentsModel()
components = model.decompose(gdp_data)

# Create features
engineer = FeatureEngineer()
features = engineer.create_features(components['cycle'], lags=[1, 2, 4])

# Split data
train_size = int(0.8 * len(features))
X_train, X_test = features[:train_size], features[train_size:]
y_train, y_test = gdp_data[4:train_size+4], gdp_data[train_size+4:]

# Train model
tree_model = TreeEnsembleModel(model_type='xgboost')
tree_model.fit(X_train, y_train)

# Forecast
predictions = tree_model.predict(X_test)

# Visualize
viz = EconomicVisualizer()
viz.plot_forecast(y_test, predictions, title="GDP Forecast")
```

### Example 2: Cross-Market Analysis

```python
# Load multiple data sources
sources = ['Z1', 'H15', 'FRED:SP500']
data = loader.load_multiple_sources(sources)

# Analyze relationships
analyzer = EconomicAnalysis(data)
correlations = analyzer.compute_rolling_correlations(
    window=52,  # 1 year for weekly data
    min_periods=26
)

# Find regime changes
regimes = analyzer.detect_regime_changes(
    correlations,
    method='hmm',
    n_regimes=3
)

# Visualize regime transitions
viz.plot_regime_transitions(
    data,
    regimes,
    title="Market Regime Analysis"
)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_data_loader.py

# Run integration tests only
pytest -m integration

# Run with verbose output
pytest -v
```

### Test Structure

```
tests/
├── unit/
│   ├── test_data_loader.py
│   ├── test_models.py
│   ├── test_analysis.py
│   └── test_visualization.py
├── integration/
│   ├── test_pipeline.py
│   ├── test_cache.py
│   └── test_parallel.py
└── fixtures/
    ├── sample_data.py
    └── mock_responses.py
```

## Performance Optimization

### Parallel Processing

```python
# Enable parallel decomposition
components = model.decompose_parallel(
    data_dict,
    n_jobs=-1,  # Use all CPU cores
    backend='multiprocessing'
)

# Parallel feature engineering
features = engineer.create_features_parallel(
    data_dict,
    n_jobs=4,
    chunk_size=1000
)
```

### Memory Management

```python
# Process large datasets in chunks
for chunk in loader.load_source_chunked('Z1', chunk_size=10000):
    results = model.process_chunk(chunk)
    save_results(results)
```

## Troubleshooting

### Common Issues

1. **Cache Errors**
   ```python
   # Clear corrupted cache
   loader.clear_cache()
   
   # Disable caching temporarily
   loader = CachedFedDataLoader(use_cache=False)
   ```

2. **Memory Issues**
   ```python
   # Reduce memory usage
   import gc
   
   # Process in smaller batches
   results = []
   for batch in data_batches:
       result = process(batch)
       results.append(result)
       gc.collect()
   ```

3. **Connection Errors**
   ```python
   # Configure retry logic
   loader = CachedFedDataLoader(
       max_retries=5,
       retry_delay=2.0,
       timeout=30
   )
   ```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
mypy src/
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{economic_timeseries_analysis,
  title = {Economic Time Series Analysis Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/economic-timeseries-analysis}
}
```

## Acknowledgments

- Federal Reserve Board for providing public access to economic data
- Contributors to statsmodels, scikit-learn, and other dependencies
- NBER for recession dating

## Contact

- Issues: [GitHub Issues](https://github.com/yourusername/economic-timeseries-analysis/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/economic-timeseries-analysis/discussions)
- Email: your.email@example.com
