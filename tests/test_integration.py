# tests/test_integration.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import shutil
import sys
import os
from pathlib import Path

# Add the project root to the Python path to resolve the ModuleNotFoundError
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CachedFedDataLoader
from src.models import UnobservedComponentsModel
from src.analysis import FeatureEngineer, EconomicAnalysis

class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for the complete analysis pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    @patch('requests.get')
    def test_complete_pipeline(self, mock_get):
        """Test the complete analysis pipeline from data loading to economic analysis."""
        print("\n--- Starting Integration Test: Complete Pipeline ---")

        # Mock data download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_zip_content = b'PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' # Empty zip file
        with open(Path(self.temp_dir) / "FRB_Z1.zip", "wb") as f:
            f.write(mock_zip_content)

        # Create the directory for the XML file
        os.makedirs(Path(self.temp_dir) / "FRB_Z1", exist_ok=True)
        mock_xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<message:CompactData xmlns:message="http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message" xmlns:z1="http://www.federalreserve.gov/structure/compact/Z1_Z1">
  <z1:DataSet>
    <z1:Series SERIES_NAME="GDP" />
    <z1:Series SERIES_NAME="M2" />
  </z1:DataSet>
</message:CompactData>
'''
        with open(Path(self.temp_dir) / "FRB_Z1/FRB_Z1.xml", "w") as f:
            f.write(mock_xml_content)

        mock_response.content = mock_zip_content
        mock_get.return_value = mock_response

        # Create synthetic data
        dates = pd.date_range('2000-01-01', periods=100, freq='QE')
        test_data = pd.DataFrame({
            'GDP': 15000 + 50 * np.arange(100) + np.random.randn(100) * 100,
            'M2': 5000 + 20 * np.arange(100) + np.random.randn(100) * 50,
        }, index=dates)

        # 1. Data Loading
        print("1. 🚚 Loading data with mocked download...")
        loader = CachedFedDataLoader(cache_directory=self.temp_dir, base_directory=self.temp_dir)
        with patch.object(loader, 'parse_xml_data', return_value=test_data):
            data = loader.load_single_source('Z1')
        self.assertIsInstance(data, pd.DataFrame)
        print("   ✅ Data loaded successfully.")

        # 2. Decomposition
        print("2. 📉 Decomposing time series into trend and cycle components...")
        uc_model = UnobservedComponentsModel()
        components = {}
        if 'GDP' in data.columns:
            # The result here is a dictionary like {'trend': ..., 'cycle': ...}
            components['GDP'] = uc_model.decompose_single_series(data['GDP'], 'GDP')
        self.assertIn('trend', components['GDP'])
        self.assertIn('cycle', components['GDP'])
        print("   ✅ Time series decomposed successfully.")

        # 3. Feature Engineering
        print("3. 🛠️  Engineering features from decomposed cycle data...")
        engineer = FeatureEngineer()
        features = engineer.create_lagged_features(
            pd.DataFrame(components['GDP']['cycle']),
            max_lags=8,
            min_lag=1
        )
        self.assertTrue(len(features.columns) > 1)
        print("   ✅ Lagged features created successfully.")

        # 4. Economic Analysis
        print("4. 💹 Analyzing the velocity of money...")
        # Pass the inner dictionary containing the 'trend' and 'cycle' components directly
        analyzer = EconomicAnalysis(components=components['GDP'], original_data=data)
        velocity_results = analyzer.analyze_velocity_of_money()
        self.assertIn('velocity_change', velocity_results)
        print("   ✅ Velocity of money analyzed successfully.")

        # 5. Check cache was used
        print("5. 🗄️ Verifying that data was cached...")
        cache_info = loader.get_cache_info()
        self.assertIn('Z1', cache_info)
        self.assertTrue(cache_info['Z1']['valid'])
        print("   ✅ Cache verified successfully.")
        print("--- Integration Test Passed --- ✅")


if __name__ == '__main__':
    unittest.main()