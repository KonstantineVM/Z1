"""
Test to verify that Fed data has been downloaded and contains actual data
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cached_fed_data_loader import CachedFedDataLoader


class TestDataVerification(unittest.TestCase):
    """Test to verify downloaded data integrity"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.loader = CachedFedDataLoader(
            cache_directory='./data/cache',
            base_directory='./data/fed_data'
        )
    
    def test_cache_directories_exist(self):
        """Test that cache directories have been created"""
        cache_dirs = [
            './data/cache',
            './data/cache/processed',
            './data/cache/metadata',
        ]
        
        for dir_path in cache_dirs:
            path = Path(dir_path)
            self.assertTrue(path.exists(), f"Directory {dir_path} does not exist")
            self.assertTrue(path.is_dir(), f"{dir_path} is not a directory")
    
    def test_cache_info_available(self):
        """Test that cache info can be retrieved"""
        cache_info = self.loader.get_cache_info()
        
        # Check if any data is cached
        self.assertIsNotNone(cache_info, "Cache info returned None")
        
        if cache_info:
            print(f"\nFound {len(cache_info)} cached sources")
            for source, info in cache_info.items():
                print(f"  - {source}: {info['shape']} shape, cached at {info['cached_at']}")
    
    def test_cached_files_exist(self):
        """Test that cached files actually exist"""
        # Check for parquet files
        parquet_files = list(Path('./data/cache/processed').glob('*.parquet'))
        
        if not parquet_files:
            self.skipTest("No cached parquet files found - run main.py first to download data")
        
        self.assertGreater(len(parquet_files), 0, "No parquet files found in cache")
        
        # Check file sizes
        for f in parquet_files:
            size = f.stat().st_size
            self.assertGreater(size, 0, f"File {f.name} is empty")
            print(f"\n  ✓ {f.name}: {size / 1024 / 1024:.2f} MB")
    
    def test_metadata_files_valid(self):
        """Test that metadata files are valid JSON"""
        metadata_files = list(Path('./data/cache/metadata').glob('*.json'))
        
        if not metadata_files:
            self.skipTest("No metadata files found")
        
        for f in metadata_files:
            with open(f, 'r') as mf:
                try:
                    metadata = json.load(mf)
                    # Check required fields
                    self.assertIn('source', metadata)
                    self.assertIn('cached_at', metadata)
                    self.assertIn('shape', metadata)
                    print(f"\n  ✓ {f.name}: valid metadata for {metadata['source']}")
                except json.JSONDecodeError:
                    self.fail(f"Invalid JSON in {f.name}")
    
    def test_load_z1_data(self):
        """Test loading Z1 data from cache"""
        cache_info = self.loader.get_cache_info()
        
        if not cache_info or 'Z1' not in cache_info:
            self.skipTest("Z1 data not cached - run main.py first to download")
        
        # Try to load Z1 data - FIXED METHOD NAME
        z1_data = self.loader.load_single_source('Z1')
        
        # Basic checks
        self.assertIsNotNone(z1_data, "Z1 data is None")
        self.assertIsInstance(z1_data, pd.DataFrame, "Z1 data is not a DataFrame")
        self.assertFalse(z1_data.empty, "Z1 data is empty")
        
        # Check shape
        self.assertGreater(z1_data.shape[0], 0, "No rows in Z1 data")
        self.assertGreater(z1_data.shape[1], 0, "No columns in Z1 data")
        
        print(f"\n✓ Z1 data loaded successfully:")
        print(f"  Shape: {z1_data.shape}")
        print(f"  Date range: {z1_data.index.min()} to {z1_data.index.max()}")
        print(f"  Columns sample: {list(z1_data.columns[:5])}")
    
    def test_data_quality(self):
        """Test data quality metrics"""
        # FIXED METHOD NAME
        z1_data = self.loader.load_single_source('Z1')
        
        if z1_data is None:
            self.skipTest("No Z1 data available")
        
        # Check data types
        numeric_cols = z1_data.select_dtypes(include=[np.number]).columns
        self.assertGreater(len(numeric_cols), 0, "No numeric columns found")
        
        # Check for reasonable date index
        self.assertIsInstance(z1_data.index, pd.DatetimeIndex, "Index is not DatetimeIndex")
        
        # Check data completeness
        total_values = z1_data.shape[0] * z1_data.shape[1]
        non_null_values = z1_data.notna().sum().sum()
        completeness = non_null_values / total_values * 100
        
        print(f"\n✓ Data quality metrics:")
        print(f"  Completeness: {completeness:.1f}%")
        print(f"  Numeric columns: {len(numeric_cols)}/{len(z1_data.columns)}")
        
        # Should have reasonable completeness (adjust threshold as needed)
        self.assertGreater(completeness, 10, "Data completeness is too low")
    
    def test_data_contains_values(self):
        """Test that data contains actual numeric values"""
        # FIXED METHOD NAME
        z1_data = self.loader.load_single_source('Z1')
        
        if z1_data is None:
            self.skipTest("No Z1 data available")
        
        # Check that not all values are zero or null
        non_zero_cols = (~(z1_data == 0).all()).sum()
        non_null_cols = (~z1_data.isnull().all()).sum()
        
        self.assertGreater(non_zero_cols, 0, "All columns contain only zeros")
        self.assertGreater(non_null_cols, 0, "All columns contain only nulls")
        
        # Check for reasonable variation in data
        # At least some columns should have standard deviation > 0
        std_devs = z1_data.std()
        varying_cols = (std_devs > 0.01).sum()
        self.assertGreater(varying_cols, 0, "No columns show variation")
        
        print(f"\n✓ Data validation:")
        print(f"  Non-zero columns: {non_zero_cols}")
        print(f"  Non-null columns: {non_null_cols}")
        print(f"  Columns with variation: {varying_cols}")
    
    def test_specific_series_exists(self):
        """Test that we can find specific types of series"""
        # FIXED METHOD NAME
        z1_data = self.loader.load_single_source('Z1')
        
        if z1_data is None:
            self.skipTest("No Z1 data available")
        
        # Look for series containing common terms
        household_series = [col for col in z1_data.columns if 'household' in col.lower()]
        
        if household_series:
            print(f"\n✓ Found {len(household_series)} household-related series")
            
            # Check a sample series
            sample = z1_data[household_series[0]].dropna()
            if len(sample) > 0:
                self.assertGreater(sample.max(), sample.min(), 
                                 "Series shows no variation")
                print(f"  Sample series {household_series[0]}:")
                print(f"    Range: {sample.min():.2f} to {sample.max():.2f}")
                print(f"    Mean: {sample.mean():.2f}")


def suite():
    """Create test suite"""
    return unittest.TestLoader().loadTestsFromTestCase(TestDataVerification)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
