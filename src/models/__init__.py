"""Model implementations for Z1 project."""

try:
    from .hierarchical_kalman_filter import HierarchicalKalmanFilter
except ImportError:
    HierarchicalKalmanFilter = None

try:
    from .sfc_kalman_filter_extended import SFCKalmanFilter
except ImportError:
    SFCKalmanFilter = None

__all__ = ['HierarchicalKalmanFilter', 'SFCKalmanFilter']
