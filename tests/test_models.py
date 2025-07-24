# tests/test_models.py
class TestUnobservedComponentsModel(unittest.TestCase):
    """Test cases for UnobservedComponentsModel"""
    
    def setUp(self):
        """Set up test fixtures"""
        from src.models import UnobservedComponentsModel
        
        self.model = UnobservedComponentsModel(
            cycle_period=8,
            damping_factor=0.98
        )
        
        # Create synthetic time series
        np.random.seed(42)
        t = np.arange(100)
        trend = 0.5 * t
        cycle = 10 * np.sin(2 * np.pi * t / 8)
        seasonal = 5 * np.sin(2 * np.pi * t / 4)
        noise = np.random.normal(0, 1, 100)
        
        self.test_series = pd.Series(
            trend + cycle + seasonal + noise,
            index=pd.date_range('2020-01-01', periods=100, freq='Q')
        )
        
    def test_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.cycle_period, 8)
        self.assertEqual(self.model.damping_factor, 0.98)
        
    def test_decompose(self):
        """Test time series decomposition"""
        components = self.model.decompose(self.test_series)
        
        # Check all components present
        expected_components = ['observed', 'trend', 'cycle', 'seasonal', 'irregular']
        for comp in expected_components:
            self.assertIn(comp, components)
            self.assertIsInstance(components[comp], pd.Series)
            self.assertEqual(len(components[comp]), len(self.test_series))
            
        # Check reconstruction
        reconstructed = (components['trend'] + components['cycle'] + 
                        components['seasonal'] + components['irregular'])
        np.testing.assert_allclose(
            components['observed'].values,
            reconstructed.values,
            rtol=1e-5
        )
        
    def test_decompose_no_seasonal(self):
        """Test decomposition without seasonal component"""
        model = UnobservedComponentsModel(
            cycle_period=8,
            seasonal=False
        )
        
        components = model.decompose(self.test_series)
        
        # Seasonal should be None or zeros
        self.assertTrue(
            components['seasonal'] is None or 
            (components['seasonal'] == 0).all()
        )
        
    def test_decompose_parallel(self):
        """Test parallel decomposition"""
        # Create multiple series
        data_dict = {
            f'series_{i}': self.test_series + np.random.randn(100) * i
            for i in range(5)
        }
        
        # Decompose in parallel
        results = self.model.decompose_parallel(data_dict, n_jobs=2)
        
        # Check results structure
        for component in ['trend', 'cycle', 'seasonal', 'irregular']:
            self.assertIn(component, results)
            self.assertIsInstance(results[component], pd.DataFrame)
            self.assertEqual(results[component].shape, (100, 5))
            
    def test_forecast(self):
        """Test forecasting functionality"""
        components = self.model.decompose(self.test_series)
        
        # Forecast 10 periods ahead
        forecast = self.model.forecast(components, steps=10)
        
        self.assertIsInstance(forecast, pd.Series)
        self.assertEqual(len(forecast), 10)
        
        # Check forecast index
        expected_start = self.test_series.index[-1] + pd.Timedelta(days=90)
        self.assertEqual(forecast.index[0], expected_start)


class TestTreeEnsembleModel(unittest.TestCase):
    """Test cases for TreeEnsembleModel"""
    
    def setUp(self):
        """Set up test fixtures"""
        from src.models import TreeEnsembleModel
        
        # Create synthetic features and target
        np.random.seed(42)
        self.X = pd.DataFrame({
            f'feature_{i}': np.random.randn(1000)
            for i in range(10)
        })
        
        # Create target with known relationships
        self.y = (
            2 * self.X['feature_0'] + 
            3 * self.X['feature_1'] + 
            0.5 * self.X['feature_0'] * self.X['feature_1'] +
            np.random.normal(0, 0.1, 1000)
        )
        
        # Split data
        split = int(0.8 * len(self.X))
        self.X_train, self.X_test = self.X[:split], self.X[split:]
        self.y_train, self.y_test = self.y[:split], self.y[split:]
        
    def test_xgboost_model(self):
        """Test XGBoost model"""
        from src.models import TreeEnsembleModel
        
        model = TreeEnsembleModel(
            model_type='xgboost',
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1
        )
        
        # Train
        model.fit(self.X_train, self.y_train)
        
        # Predict
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Check reasonable accuracy
        from sklearn.metrics import r2_score
        r2 = r2_score(self.y_test, predictions)
        self.assertGreater(r2, 0.8)  # Should capture most variance
        
    def test_lightgbm_model(self):
        """Test LightGBM model"""
        from src.models import TreeEnsembleModel
        
        model = TreeEnsembleModel(
            model_type='lightgbm',
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1
        )
        
        # Train
        model.fit(self.X_train, self.y_train)
        
        # Predict
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Check reasonable accuracy
        from sklearn.metrics import r2_score
        r2 = r2_score(self.y_test, predictions)
        self.assertGreater(r2, 0.8)
        
    def test_feature_importance(self):
        """Test feature importance extraction"""
        from src.models import TreeEnsembleModel
        
        model = TreeEnsembleModel(model_type='xgboost')
        model.fit(self.X_train, self.y_train)
        
        # Get importance
        importance = model.get_feature_importance()
        
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)
        
        # Check most important features
        top_features = importance.nlargest(2, 'importance')['feature'].tolist()
        self.assertIn('feature_0', top_features)
        self.assertIn('feature_1', top_features)
        
    def test_predict_with_importance(self):
        """Test prediction with importance return"""
        from src.models import TreeEnsembleModel
        
        model = TreeEnsembleModel(model_type='xgboost')
        model.fit(self.X_train, self.y_train)
        
        # Predict with importance
        predictions, importance = model.predict(self.X_test, return_importance=True)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertIsInstance(importance, pd.DataFrame)