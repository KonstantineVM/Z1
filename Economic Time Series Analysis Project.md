# Economic Time Series Analysis Project

## Project Structure
```
economic_timeseries_analysis/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── data_sources.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fed_data_loader.py
│   │   ├── external_data_loader.py
│   │   └── data_processor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unobserved_components.py
│   │   ├── tree_models.py
│   │   ├── gaussian_process.py
│   │   └── neural_gp.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── decomposition.py
│   │   ├── feature_engineering.py
│   │   └── economic_analysis.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── component_plots.py
│   │   └── economic_plots.py
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py
│       └── validation.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_decomposition_analysis.ipynb
│   ├── 03_prediction_models.ipynb
│   └── 04_economic_relationships.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_models.py
│   └── test_analysis.py
├── examples/
│   ├── basic_usage.py
│   ├── advanced_analysis.py
│   └── custom_models.py
└── output/
    ├── figures/
    ├── models/
    └── results/
```

## Key Features

### 1. Data Management
- Unified Fed data loader for multiple sources (Z1, H6, H8, G17, G19, G20, H15)
- External data integration (S&P 500, FRED, gold prices, etc.)
- **Intelligent caching system** to minimize redundant downloads
- Automatic data validation and preprocessing

### 2. Cache Management
- Automatic caching of downloaded and processed data
- Configurable cache expiry (default: 7 days)
- Cache status monitoring and management tools
- Parallel data loading with cache support

### 3. Models
- Unobserved Components Models with parallel processing
- Tree-based models (XGBoost, LightGBM) with decision path extraction
- Gaussian Process regression with custom Matérn kernels
- Neural network-enhanced GP models

### 4. Analysis Capabilities
- Trend and cyclical component extraction
- Feature engineering with lagged variables
- Economic relationship analysis (velocity of money, savings dynamics)
- Cross-series amplitude normalization for zero-crossing series

### 5. Visualization
- Component decomposition plots
- Economic relationship visualizations with recession shading
- Feature importance analysis
- Uncertainty quantification plots

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/economic_timeseries_analysis.git
cd economic_timeseries_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
from src.data import CachedFedDataLoader, ExternalDataLoader
from src.models import UnobservedComponentsModel
from src.analysis import EconomicAnalysis
from src.visualization import plot_components, plot_economic_relationships

# Load data with caching
fed_loader = CachedFedDataLoader()
data = fed_loader.load_multiple_sources(['Z1', 'H6', 'H15'])

# Decompose series
uc_model = UnobservedComponentsModel()
components = uc_model.decompose_parallel(data)

# Analyze relationships
analyzer = EconomicAnalysis(components)
results = analyzer.analyze_velocity_of_money()

# Visualize
plot_economic_relationships(results, include_recessions=True)
```

## Cache Management

The project includes intelligent caching to avoid redundant downloads:

```bash
# Check cache status
python cache_manager.py status

# Clear all cache
python cache_manager.py clear

# Clear specific cache
python cache_manager.py clear --source Z1 --type fed

# Refresh cache with fresh data
python cache_manager.py refresh

# Refresh specific sources
python cache_manager.py refresh --sources Z1 H6 sp500
```

### Cache Configuration

Edit `config/config.yaml` to customize cache behavior:

```yaml
data:
  cache_directory: "./data/cache"
  cache_expiry_days: 7  # Days before cache expires
  force_download: false  # Set to true to bypass cache
```

## Configuration

Edit `config/config.yaml` to customize:
- Data sources and paths
- Model parameters
- Analysis settings
- Output directories

## Documentation

See the `notebooks/` directory for detailed examples and tutorials.