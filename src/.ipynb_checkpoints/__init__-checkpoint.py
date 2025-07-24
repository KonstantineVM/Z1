# src/__init__.py
"""
Economic Time Series Analysis Package
"""

__version__ = "0.1.0"

# src/data/__init__.py
"""
Data loading and processing modules
"""

from .fed_data_loader import FedDataLoader
from .external_data_loader import ExternalDataLoader
from .data_processor import DataProcessor

__all__ = ['FedDataLoader', 'ExternalDataLoader', 'DataProcessor']

# src/models/__init__.py
"""
Time series models
"""

from .unobserved_components import UnobservedComponentsModel
from .tree_models import TreeModelAnalyzer
from .gaussian_process import GaussianProcessModel

__all__ = ['UnobservedComponentsModel', 'TreeModelAnalyzer', 'GaussianProcessModel']

# src/analysis/__init__.py
"""
Analysis modules
"""

from .feature_engineering import FeatureEngineer
from .economic_analysis import EconomicAnalysis

__all__ = ['FeatureEngineer', 'EconomicAnalysis']

# src/visualization/__init__.py
"""
Visualization modules
"""

from .economic_plots import EconomicVisualizer

__all__ = ['EconomicVisualizer']

# src/utils/__init__.py
"""
Utility functions
"""

# tests/__init__.py
"""
Test suite for Economic Time Series Analysis
"""