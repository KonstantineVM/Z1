"""
Gaussian Process Models
Implementation of GP regression with custom kernels
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ConstantKernel as C,
    WhiteKernel, DotProduct
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class GaussianProcessModel:
    """
    Gaussian Process regression with custom kernels and optimization
    """
    
    def __init__(self, kernel_type: str = 'matern',
                 nu: float = 2.5,
                 alpha: float = 1e-3,
                 n_restarts_optimizer: int = 10,
                 normalize_y: bool = True):
        """
        Initialize GP model
        
        Parameters:
        -----------
        kernel_type : str
            Type of kernel ('rbf', 'matern', 'rational_quadratic')
        nu : float
            Nu parameter for Matern kernel
        alpha : float
            Regularization parameter (noise level)
        n_restarts_optimizer : int
            Number of restarts for kernel optimization
        normalize_y : bool
            Whether to normalize target values
        """
        self.kernel_type = kernel_type
        self.nu = nu
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        
        self.kernel = None
        self.gp = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler() if normalize_y else None
        self.is_fitted = False
        
    def _create_kernel(self, n_features: int):
        """Create kernel based on type and parameters"""
        if self.kernel_type == 'rbf':
            base_kernel = RBF(length_scale=[1.0] * n_features, 
                             length_scale_bounds=(1e-2, 1e3))
        elif self.kernel_type == 'matern':
            base_kernel = Matern(length_scale=[1.0] * n_features,
                                length_scale_bounds=(1e-2, 1e3),
                                nu=self.nu)
        elif self.kernel_type == 'rational_quadratic':
            base_kernel = RationalQuadratic(length_scale=1.0,
                                          alpha=1.0,
                                          length_scale_bounds=(1e-2, 1e3),
                                          alpha_bounds=(1e-5, 1e2))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # Add constant kernel for scaling
        kernel = C(1.0, (1e-5, 1e5)) * base_kernel
        
        # Optionally add white noise kernel
        if self.alpha > 0:
            kernel += WhiteKernel(noise_level=self.alpha, 
                                 noise_level_bounds=(1e-10, 1e-1))
        
        return kernel
    
    def _custom_optimizer(self, obj_func, initial_theta, bounds):
        """
        Custom optimizer with increased iterations
        """
        result = minimize(
            obj_func, 
            initial_theta, 
            method="L-BFGS-B", 
            jac=True, 
            bounds=bounds, 
            options={'maxiter': 10000}
        )
        return result.x, result.fun
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray]):
        """
        Fit the Gaussian Process model
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Training features
        y : pd.Series or np.ndarray
            Training target
        """
        logger.info("Fitting Gaussian Process model...")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Standardize features
        X_scaled = self.scaler_X.fit_transform(X)
        
        # Standardize target if requested
        if self.normalize_y:
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            y_scaled = y
        
        # Create kernel
        n_features = X.shape[1]
        self.kernel = self._create_kernel(n_features)
        
        # Initialize GP
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            alpha=self.alpha if self.alpha == 0 else 0,  # Alpha handled by WhiteKernel
            optimizer=self._custom_optimizer
        )
        
        # Fit model
        self.gp.fit(X_scaled, y_scaled)
        self.is_fitted = True
        
        # Log kernel parameters
        logger.info(f"Optimized kernel: {self.gp.kernel_}")
        logger.info(f"Log marginal likelihood: {self.gp.log_marginal_likelihood_value_:.3f}")
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
        return_std : bool
            Whether to return uncertainty estimates
            
        Returns:
        --------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Predictions and optionally standard deviations
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Standardize features
        X_scaled = self.scaler_X.transform(X)
        
        # Make predictions
        if return_std:
            y_pred_scaled, y_std_scaled = self.gp.predict(X_scaled, return_std=True)
        else:
            y_pred_scaled = self.gp.predict(X_scaled)
            
        # Inverse transform predictions
        if self.normalize_y:
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            if return_std:
                # Scale standard deviation
                y_std = y_std_scaled * self.scaler_y.scale_[0]
        else:
            y_pred = y_pred_scaled
            if return_std:
                y_std = y_std_scaled
                
        if return_std:
            return y_pred, y_std
        else:
            return y_pred
    
    def evaluate(self, X_test: Union[pd.DataFrame, np.ndarray],
                y_test: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : pd.DataFrame or np.ndarray
            Test features
        y_test : pd.Series or np.ndarray
            Test target
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of metrics
        """
        # Make predictions
        y_pred, y_std = self.predict(X_test, return_std=True)
        
        # Convert to numpy if needed
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
            
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mean_std': np.mean(y_std),
            'std_std': np.std(y_std)
        }
        
        # Calculate coverage (percentage within 2 std)
        lower = y_pred - 2 * y_std
        upper = y_pred + 2 * y_std
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        metrics['coverage_2std'] = coverage
        
        # Calculate negative log likelihood
        nll = -np.mean(
            -0.5 * np.log(2 * np.pi * y_std**2) - 
            0.5 * ((y_test - y_pred) / y_std)**2
        )
        metrics['nll'] = nll
        
        return metrics
    
    def plot_predictions(self, X_test: Union[pd.DataFrame, np.ndarray],
                        y_test: Union[pd.Series, np.ndarray],
                        sample_indices: Optional[np.ndarray] = None,
                        figsize: Tuple[int, int] = (12, 6)):
        """
        Plot predictions with uncertainty bands
        
        Parameters:
        -----------
        X_test : pd.DataFrame or np.ndarray
            Test features
        y_test : pd.Series or np.ndarray
            Test target
        sample_indices : np.ndarray, optional
            Indices to plot (if None, plots all)
        figsize : Tuple[int, int]
            Figure size
        """
        # Make predictions
        y_pred, y_std = self.predict(X_test, return_std=True)
        
        # Convert to numpy if needed
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
            
        # Select samples to plot
        if sample_indices is None:
            sample_indices = np.arange(len(y_test))
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot predictions vs actual
        ax1.plot(sample_indices, y_test[sample_indices], 'b-', 
                label='Actual', linewidth=2)
        ax1.plot(sample_indices, y_pred[sample_indices], 'r--', 
                label='Predicted', linewidth=2)
        
        # Add uncertainty bands
        ax1.fill_between(
            sample_indices,
            y_pred[sample_indices] - 2 * y_std[sample_indices],
            y_pred[sample_indices] + 2 * y_std[sample_indices],
            color='red', alpha=0.2, label='95% CI'
        )
        
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Value')
        ax1.set_title('GP Predictions vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot uncertainty over samples
        ax2.plot(sample_indices, y_std[sample_indices], 'g-', linewidth=2)
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Prediction Uncertainty (Std)')
        ax2.set_title('Prediction Uncertainty')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_kernel_parameters(self) -> Dict[str, any]:
        """
        Get optimized kernel parameters
        
        Returns:
        --------
        Dict[str, any]
            Kernel parameters
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        params = {}
        kernel_params = self.gp.kernel_.get_params()
        
        # Extract key parameters
        for key, value in kernel_params.items():
            if 'length_scale' in key:
                params['length_scales'] = value
            elif 'noise_level' in key:
                params['noise_level'] = value
            elif key.endswith('__constant_value'):
                params['kernel_scale'] = value
                
        params['log_marginal_likelihood'] = self.gp.log_marginal_likelihood_value_
        
        return params
    
    def sample_posterior(self, X: Union[pd.DataFrame, np.ndarray],
                        n_samples: int = 10) -> np.ndarray:
        """
        Sample from the posterior distribution
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Points at which to sample
        n_samples : int
            Number of samples to draw
            
        Returns:
        --------
        np.ndarray
            Samples of shape (n_samples, n_points)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Standardize features
        X_scaled = self.scaler_X.transform(X)
        
        # Sample from posterior
        samples_scaled = self.gp.sample_y(X_scaled, n_samples=n_samples).T
        
        # Inverse transform if needed
        if self.normalize_y:
            samples = np.array([
                self.scaler_y.inverse_transform(sample.reshape(-1, 1)).ravel()
                for sample in samples_scaled
            ])
        else:
            samples = samples_scaled
            
        return samples