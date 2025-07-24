"""
Tree-based Models
Implementation of XGBoost and LightGBM with decision path extraction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import optuna
from optuna.samplers import TPESampler
import shap

logger = logging.getLogger(__name__)


class TreeModelAnalyzer:
    """
    Tree-based model analysis with decision path extraction
    """
    
    DEFAULT_XGBOOST_PARAMS = {
        'objective': 'reg:squarederror',
        'n_estimators': 247,
        'max_depth': 7,
        'learning_rate': 0.5425888631443009,
        'subsample': 0.9963309038163909,
        'colsample_bytree': 0.42837148287574744,
        'gamma': 5.876091524926203e-07,
        'alpha': 0.017699244494402626,
        'lambda': 0.05147617406122128,
        'min_child_weight': 42
    }
    
    DEFAULT_LIGHTGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 100,
        'num_leaves': 15,
        'feature_fraction': 0.5783658631194919,
        'bagging_fraction': 0.9280051724697139,
        'bagging_freq': 7,
        'min_child_samples': 80,
        'lambda_l1': 8.845493410839486e-07,
        'lambda_l2': 1.0517737839042705
    }
    
    def __init__(self, model_type: str = 'xgboost', params: Optional[Dict] = None):
        """
        Initialize tree model analyzer
        
        Parameters:
        -----------
        model_type : str
            Type of model ('xgboost' or 'lightgbm')
        params : Dict, optional
            Model parameters
        """
        self.model_type = model_type.lower()
        
        if self.model_type == 'xgboost':
            self.params = self.DEFAULT_XGBOOST_PARAMS.copy()
        elif self.model_type == 'lightgbm':
            self.params = self.DEFAULT_LIGHTGBM_PARAMS.copy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        if params:
            self.params.update(params)
            
        self.model = None
        self.feature_importance = None
        self.decision_paths = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        """
        Fit the tree model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
        """
        logger.info(f"Training {self.model_type} model...")
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**self.params)
            
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
                
        else:  # lightgbm
            self.model = lgb.LGBMRegressor(**self.params)
            
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.log_evaluation(period=0)]
                )
            else:
                self.model.fit(X_train, y_train)
        
        # Extract feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Model trained. Top features: {self.feature_importance.head()['feature'].tolist()}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def extract_decision_paths(self, X: pd.DataFrame) -> np.ndarray:
        """
        Extract decision paths from the tree model
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        np.ndarray
            Binary matrix of decision paths
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
            
        logger.info("Extracting decision paths...")
        
        if self.model_type == 'xgboost':
            return self._extract_xgboost_paths(X)
        else:
            return self._extract_lightgbm_paths(X)
            
    def _extract_xgboost_paths(self, X: pd.DataFrame) -> np.ndarray:
        """Extract decision paths from XGBoost"""
        # Get leaf indices for all trees
        leaf_indices = self.model.apply(X).astype(int)
        
        # Initialize binary path matrix
        num_trees = leaf_indices.shape[1]
        max_leaves = np.max(leaf_indices) + 1
        paths = np.zeros((X.shape[0], num_trees * max_leaves), dtype=int)
        
        # Populate the matrix
        for sample_idx, sample_leaf_indices in enumerate(leaf_indices):
            for tree_idx, leaf_idx in enumerate(sample_leaf_indices):
                col_idx = tree_idx * max_leaves + leaf_idx
                paths[sample_idx, col_idx] = 1
                
        return paths
    
    def _extract_lightgbm_paths(self, X: pd.DataFrame) -> np.ndarray:
        """Extract decision paths from LightGBM"""
        # Get leaf indices using predict with pred_leaf=True
        leaf_indices = self.model.predict(X, pred_leaf=True)
        
        # Initialize binary path matrix
        num_trees = leaf_indices.shape[1]
        max_leaves = np.max(leaf_indices) + 1
        paths = np.zeros((X.shape[0], num_trees * max_leaves), dtype=int)
        
        # Populate the matrix
        for sample_idx, sample_leaf_indices in enumerate(leaf_indices):
            for tree_idx, leaf_idx in enumerate(sample_leaf_indices):
                col_idx = tree_idx * max_leaves + leaf_idx
                paths[sample_idx, col_idx] = 1
                
        return paths
    
    def fit_linear_on_paths(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           use_lasso: bool = True,
                           lasso_alpha: float = 0.002) -> Dict:
        """
        Fit linear model on decision paths
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
        use_lasso : bool
            Whether to use LASSO (True) or regular linear regression
        lasso_alpha : float
            LASSO regularization parameter
            
        Returns:
        --------
        Dict
            Results including predictions and coefficients
        """
        # Extract decision paths
        paths_train = self.extract_decision_paths(X_train)
        paths_test = self.extract_decision_paths(X_test)
        
        logger.info(f"Decision paths shape: {paths_train.shape}")
        
        # Fit linear model
        if use_lasso:
            linear_model = Lasso(alpha=lasso_alpha, max_iter=10000)
        else:
            linear_model = LinearRegression()
            
        linear_model.fit(paths_train, y_train)
        
        # Make predictions
        y_pred = linear_model.predict(paths_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Extract non-zero coefficients (for LASSO)
        if use_lasso:
            coef_df = pd.DataFrame({
                'path_idx': range(len(linear_model.coef_)),
                'coefficient': linear_model.coef_
            })
            coef_df = coef_df[coef_df['coefficient'] != 0].sort_values(
                'coefficient', key=abs, ascending=False
            )
        else:
            coef_df = pd.DataFrame({
                'path_idx': range(len(linear_model.coef_)),
                'coefficient': linear_model.coef_
            }).sort_values('coefficient', key=abs, ascending=False).head(50)
        
        results = {
            'predictions': y_pred,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'coefficients': coef_df,
            'linear_model': linear_model
        }
        
        logger.info(f"Linear model on paths - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        return results
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               n_trials: int = 100,
                               n_jobs: int = 1) -> Dict:
        """
        Optimize hyperparameters using Optuna
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target
        n_trials : int
            Number of optimization trials
        n_jobs : int
            Number of parallel jobs
            
        Returns:
        --------
        Dict
            Best parameters and optimization history
        """
        logger.info(f"Optimizing {self.model_type} hyperparameters...")
        
        def objective(trial):
            if self.model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                    'max_depth': trial.suggest_int('max_depth', 1, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 10.0, log=True),
                    'subsample': trial.suggest_float('subsample', 0.01, 1.0, log=True),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0, log=True),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                    'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
                    'objective': 'reg:squarederror'
                }
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
            else:  # lightgbm
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1
                }
                
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.log_evaluation(period=0)]
                )
            
            predictions = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            
            return rmse
        
        # Create study
        study = optuna.create_study(direction='minimize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best RMSE: {study.best_value:.4f}")
        
        # Update model with best parameters
        self.params.update(study.best_params)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def get_shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, shap.Explainer]:
        """
        Calculate SHAP values for model interpretation
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to explain
            
        Returns:
        --------
        Tuple[np.ndarray, shap.Explainer]
            SHAP values and explainer object
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
            
        logger.info("Calculating SHAP values...")
        
        if self.model_type == 'xgboost':
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.TreeExplainer(self.model)
            
        shap_values = explainer.shap_values(X)
        
        return shap_values, explainer
    
    def plot_feature_importance(self, top_n: int = 30, 
                               descriptions: Optional[pd.DataFrame] = None):
        """
        Plot feature importance
        
        Parameters:
        -----------
        top_n : int
            Number of top features to show
        descriptions : pd.DataFrame, optional
            DataFrame with 'SERIES_NAME' and 'Long Description' for labels
        """
        if self.feature_importance is None:
            raise ValueError("Model not fitted yet")
            
        # Get top features
        top_features = self.feature_importance.head(top_n).copy()
        
        # Add descriptions if provided
        if descriptions is not None:
            # Extract series names from feature names
            top_features['SERIES_NAME'] = top_features['feature'].str.extract(
                r'([A-Z]+[0-9]+\.[A-Z])'
            )[0]
            
            # Merge with descriptions
            top_features = top_features.merge(
                descriptions, 
                on='SERIES_NAME', 
                how='left'
            )
            
            # Create display labels
            top_features['display'] = top_features.apply(
                lambda row: f"{row['feature']}\n{row.get('Long Description', '')[:50]}...",
                axis=1
            )
        else:
            top_features['display'] = top_features['feature']
        
        # Create plot
        plt.figure(figsize=(12, max(8, top_n * 0.3)))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['display'])
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {self.model_type.upper()}')
        plt.tight_layout()
        
        return plt.gcf()