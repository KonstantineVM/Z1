# tests/test_integration.py
class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
        
    @patch('requests.get')
    def test_complete_pipeline(self, mock_get):
        """Test complete analysis pipeline"""
        from src.data import CachedFedDataLoader
        from src.models import UnobservedComponentsModel
        from src.analysis import FeatureEngineer, EconomicAnalysis
        
        # Mock data download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<?xml version="1.0"?><data>test</data>'
        mock_get.return_value = mock_response
        
        # Create synthetic data
        dates = pd.date_range('2000-01-01', periods=100, freq='Q')
        test_data = pd.DataFrame({
            'GDP': 15000 + 50 * np.arange(100) + np.random.randn(100) * 100,
            'M2': 5000 + 20 * np.arange(100) + np.random.randn(100) * 50,
        }, index=dates)
        
        # 1. Data Loading (mocked)
        loader = CachedFedDataLoader(cache_directory=self.temp_dir)
        with patch.object(loader, '_download_and_parse', return_value=test_data):
            data = loader.load_source('Z1')
            
        self.assertIsInstance(data, pd.DataFrame)
        
        # 2. Decomposition
        uc_model = UnobservedComponentsModel(cycle_period=32)
        components = uc_model.decompose(data['GDP'])
        
        self.assertIn('trend', components)
        self.assertIn('cycle', components)
        
        # 3. Feature Engineering
        engineer = FeatureEngineer()
        features = engineer.create_features(
            pd.DataFrame(components['cycle']),
            lags=[1, 4],
            rolling_windows=[4, 8]
        )
        
        self.assertTrue(len(features.columns) > 1)
        
        # 4. Economic Analysis
        analyzer = EconomicAnalysis(data)
        velocity_results = analyzer.analyze_velocity_of_money('M2', 'GDP')
        
        self.assertIn('velocity', velocity_results)
        
        # 5. Check cache was used
        cache_info = loader.get_cache_info()
        self.assertIn('Z1', cache_info)
        self.assertTrue(cache_info['Z1']['valid'])


if __name__ == '__main__':
    unittest.main()