# tests/test_data_loader.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import shutil
import os

from src.data import CachedFedDataLoader, DataProcessor


class TestCachedFedDataLoader(unittest.TestCase):
    """Test cases for CachedFedDataLoader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = CachedFedDataLoader(
            cache_directory=self.temp_dir,
            cache_expiry_days=7,
            start_year=2020,
            end_year=2023
        )
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test loader initialization"""
        self.assertEqual(self.loader.cache_directory, self.temp_dir)
        self.assertEqual(self.loader.cache_expiry_days, 7)
        self.assertEqual(self.loader.start_year, 2020)
        self.assertEqual(self.loader.end_year, 2023)
        
    @patch('requests.get')
    def test_download_data(self, mock_get):
        """Test data download functionality"""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<?xml version="1.0"?><data>test</data>'
        mock_get.return_value = mock_response
        
        # Test download
        result = self.loader._download_data('Z1')
        self.assertIsNotNone(result)
        mock_get.assert_called_once()
        
    def test_cache_key_generation(self):
        """Test cache key generation"""
        key = self.loader._get_cache_key('Z1', 'processed')
        self.assertIsInstance(key, str)
        self.assertTrue(len(key) > 0)
        
    def test_cache_path_generation(self):
        """Test cache path generation"""
        path = self.loader._get_cache_path('Z1', 'processed')
        self.assertTrue(str(path).startswith(self.temp_dir))
        self.assertTrue(str(path).endswith('.parquet'))
        
    def test_cache_validity_check(self):
        """Test cache validity checking"""
        # No cache exists
        self.assertFalse(self.loader._is_cache_valid('Z1'))
        
        # Create mock cache
        cache_path = self.loader._get_cache_path('Z1')
        metadata_path = self.loader._get_metadata_path('Z1')
        
        # Create directories
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create mock data
        pd.DataFrame({'test': [1, 2, 3]}).to_parquet(cache_path)
        
        # Create metadata
        import json
        metadata = {
            'cached_at': datetime.now().isoformat(),
            'start_year': 2020,
            'end_year': 2023,
            'source': 'Z1'
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        # Should be valid
        self.assertTrue(self.loader._is_cache_valid('Z1'))
        
        # Test expired cache
        metadata['cached_at'] = (datetime.now() - timedelta(days=10)).isoformat()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        self.assertFalse(self.loader._is_cache_valid('Z1'))
        
    def test_save_to_cache(self):
        """Test saving data to cache"""
        data = pd.DataFrame({
            'series1': np.random.randn(100),
            'series2': np.random.randn(100)
        })
        
        self.loader._save_to_cache(data, 'test_source')
        
        # Check files exist
        cache_path = self.loader._get_cache_path('test_source')
        metadata_path = self.loader._get_metadata_path('test_source')
        
        self.assertTrue(cache_path.exists())
        self.assertTrue(metadata_path.exists())
        
        # Load and verify
        loaded_data = pd.read_parquet(cache_path)
        pd.testing.assert_frame_equal(data, loaded_data)
        
    def test_load_from_cache(self):
        """Test loading data from cache"""
        # Save test data
        data = pd.DataFrame({
            'series1': np.random.randn(100),
            'series2': np.random.randn(100)
        })
        self.loader._save_to_cache(data, 'test_source')
        
        # Load from cache
        loaded_data = self.loader._load_from_cache('test_source')
        pd.testing.assert_frame_equal(data, loaded_data)
        
    def test_clear_cache(self):
        """Test cache clearing"""
        # Create some cache files
        self.loader._save_to_cache(pd.DataFrame({'test': [1, 2, 3]}), 'source1')
        self.loader._save_to_cache(pd.DataFrame({'test': [4, 5, 6]}), 'source2')
        
        # Clear specific source
        self.loader.clear_cache('source1')
        self.assertFalse(self.loader._get_cache_path('source1').exists())
        self.assertTrue(self.loader._get_cache_path('source2').exists())
        
        # Clear all
        self.loader.clear_cache()
        self.assertFalse(self.loader._get_cache_path('source2').exists())
        
    @patch.object(CachedFedDataLoader, '_download_and_parse')
    def test_load_source_with_cache(self, mock_download):
        """Test loading source with caching"""
        mock_data = pd.DataFrame({
            'SERIES_NAME': ['GDP', 'CPI'],
            'VALUE': [100, 200]
        })
        mock_download.return_value = mock_data
        
        # First load - should download
        data1 = self.loader.load_source('Z1')
        mock_download.assert_called_once()
        pd.testing.assert_frame_equal(data1, mock_data)
        
        # Second load - should use cache
        mock_download.reset_mock()
        data2 = self.loader.load_source('Z1')
        mock_download.assert_not_called()
        pd.testing.assert_frame_equal(data2, mock_data)
        
        # Force refresh
        mock_download.reset_mock()
        data3 = self.loader.load_source('Z1', force_refresh=True)
        mock_download.assert_called_once()
        pd.testing.assert_frame_equal(data3, mock_data)


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor()
        
        # Create test data with various issues
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'clean_series': np.random.randn(100),
            'missing_series': np.random.randn(100),
            'outlier_series': np.random.randn(100),
            'string_series': ['1.5'] * 50 + ['invalid'] * 50
        }, index=dates)
        
        # Add missing values
        self.test_data.loc[self.test_data.index[10:20], 'missing_series'] = np.nan
        
        # Add outliers
        self.test_data.loc[self.test_data.index[50], 'outlier_series'] = 100
        
    def test_clean_data(self):
        """Test data cleaning"""
        cleaned = self.processor.clean_data(self.test_data.copy())
        
        # Check numeric conversion
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned['string_series']))
        
        # Check outlier handling
        self.assertLess(cleaned['outlier_series'].max(), 100)
        
    def test_handle_missing_values_interpolate(self):
        """Test missing value handling with interpolation"""
        filled = self.processor.handle_missing_values(
            self.test_data.copy(),
            method='interpolate'
        )
        
        # Check no missing values
        self.assertFalse(filled['missing_series'].isna().any())
        
    def test_handle_missing_values_forward_fill(self):
        """Test missing value handling with forward fill"""
        filled = self.processor.handle_missing_values(
            self.test_data.copy(),
            method='ffill'
        )
        
        # Check no missing values
        self.assertFalse(filled['missing_series'].isna().any())
        
    def test_handle_missing_values_mean(self):
        """Test missing value handling with mean imputation"""
        filled = self.processor.handle_missing_values(
            self.test_data.copy(),
            method='mean'
        )
        
        # Check no missing values
        self.assertFalse(filled['missing_series'].isna().any())
        
        # Check mean preserved (approximately)
        original_mean = self.test_data['missing_series'].mean()
        filled_mean = filled['missing_series'].mean()
        self.assertAlmostEqual(original_mean, filled_mean, places=1)
        
    def test_normalize_series_zscore(self):
        """Test z-score normalization"""
        normalized = self.processor.normalize_series(
            self.test_data[['clean_series']].copy(),
            method='z-score'
        )
        
        # Check mean ≈ 0 and std ≈ 1
        self.assertAlmostEqual(normalized['clean_series'].mean(), 0, places=5)
        self.assertAlmostEqual(normalized['clean_series'].std(), 1, places=5)
        
    def test_normalize_series_minmax(self):
        """Test min-max normalization"""
        normalized = self.processor.normalize_series(
            self.test_data[['clean_series']].copy(),
            method='min-max'
        )
        
        # Check range [0, 1]
        self.assertAlmostEqual(normalized['clean_series'].min(), 0, places=5)
        self.assertAlmostEqual(normalized['clean_series'].max(), 1, places=5)
        
    def test_normalize_series_robust(self):
        """Test robust normalization"""
        normalized = self.processor.normalize_series(
            self.test_data[['outlier_series']].copy(),
            method='robust'
        )
        
        # Check that outliers don't dominate
        self.assertLess(normalized['outlier_series'].std(), 
                       self.test_data['outlier_series'].std())
        
    def test_detect_outliers(self):
        """Test outlier detection"""
        outliers = self.processor.detect_outliers(
            self.test_data['outlier_series'],
            method='iqr'
        )
        
        # Should detect the artificial outlier
        self.assertTrue(outliers[50])
        
    def test_remove_outliers(self):
        """Test outlier removal"""
        cleaned = self.processor.remove_outliers(
            self.test_data[['outlier_series']].copy(),
            method='iqr'
        )
        
        # Outlier should be NaN or replaced
        self.assertNotEqual(cleaned.loc[self.test_data.index[50], 'outlier_series'], 100)