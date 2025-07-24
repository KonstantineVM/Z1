# tests/test_analysis.py  
class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer"""
    
    def setUp(self):
        """Set up test fixtures"""
        from src.analysis import FeatureEngineer
        
        self.engineer = FeatureEngineer()
        
        # Create test time series
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'series_1': np.random.randn(100).cumsum(),
            'series_2': np.random.randn(100).cumsum()
        }, index=dates)
        
    def test_create_lags(self):
        """Test lag feature creation"""
        lagged = self.engineer.create_lags(
            self.test_data,
            lags=[1, 2, 5]
        )
        
        # Check columns created
        expected_cols = ['series_1', 'series_2',
                        'series_1_lag_1', 'series_1_lag_2', 'series_1_lag_5',
                        'series_2_lag_1', 'series_2_lag_2', 'series_2_lag_5']
        
        for col in expected_cols:
            self.assertIn(col, lagged.columns)
            
        # Check lag values
        pd.testing.assert_series_equal(
            lagged['series_1_lag_1'].iloc[1:],
            self.test_data['series_1'].iloc[:-1],
            check_names=False
        )
        
    def test_create_rolling_features(self):
        """Test rolling window feature creation"""
        rolling = self.engineer.create_rolling_features(
            self.test_data,
            windows=[5, 10],
            functions=['mean', 'std']
        )
        
        # Check columns created
        self.assertIn('series_1_rolling_5_mean', rolling.columns)
        self.assertIn('series_1_rolling_5_std', rolling.columns)
        self.assertIn('series_1_rolling_10_mean', rolling.columns)
        self.assertIn('series_2_rolling_10_std', rolling.columns)
        
        # Check calculations
        expected_mean = self.test_data['series_1'].rolling(5).mean()
        pd.testing.assert_series_equal(
            rolling['series_1_rolling_5_mean'],
            expected_mean,
            check_names=False
        )
        
    def test_create_interaction_features(self):
        """Test interaction feature creation"""
        interactions = self.engineer.create_interaction_features(
            self.test_data,
            method='multiply'
        )
        
        # Check interaction created
        self.assertIn('series_1_x_series_2', interactions.columns)
        
        # Check calculation
        expected = self.test_data['series_1'] * self.test_data['series_2']
        pd.testing.assert_series_equal(
            interactions['series_1_x_series_2'],
            expected,
            check_names=False
        )
        
    def test_create_features_comprehensive(self):
        """Test comprehensive feature creation"""
        features = self.engineer.create_features(
            self.test_data,
            lags=[1, 2],
            rolling_windows=[5],
            include_interactions=True
        )
        
        # Check all feature types present
        self.assertIn('series_1_lag_1', features.columns)  # Lags
        self.assertIn('series_1_rolling_5_mean', features.columns)  # Rolling
        self.assertIn('series_1_x_series_2', features.columns)  # Interactions
        
        # Check no NaN in features (after dropping)
        features_clean = features.dropna()
        self.assertFalse(features_clean.isna().any().any())


class TestEconomicAnalysis(unittest.TestCase):
    """Test cases for EconomicAnalysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        from src.analysis import EconomicAnalysis
        
        # Create synthetic economic data
        dates = pd.date_range('2000-01-01', periods=200, freq='Q')
        
        # GDP with trend and cycle
        t = np.arange(200)
        gdp_trend = 15000 + 50 * t
        gdp_cycle = 500 * np.sin(2 * np.pi * t / 32)  # 8 year cycle
        gdp = gdp_trend + gdp_cycle + np.random.normal(0, 100, 200)
        
        # Money supply (correlated with GDP)
        m2 = 5000 + 0.3 * gdp + np.random.normal(0, 50, 200)
        
        # Interest rate (inversely related to money growth)
        m2_growth = pd.Series(m2).pct_change().fillna(0)
        interest_rate = 5 - 20 * m2_growth + np.random.normal(0, 0.5, 200)
        
        self.test_data = pd.DataFrame({
            'GDP': gdp,
            'M2': m2,
            'DFF': interest_rate,
            'CONSUMPTION': 0.7 * gdp + np.random.normal(0, 50, 200),
            'INVESTMENT': 0.2 * gdp + np.random.normal(0, 30, 200)
        }, index=dates)
        
        self.analyzer = EconomicAnalysis(self.test_data)
        
    def test_analyze_velocity_of_money(self):
        """Test velocity of money analysis"""
        results = self.analyzer.analyze_velocity_of_money(
            money_supply_series='M2',
            gdp_series='GDP'
        )
        
        # Check results structure
        self.assertIn('velocity', results)
        self.assertIn('velocity_change', results)
        self.assertIn('velocity_trend', results)
        
        # Check velocity calculation
        expected_velocity = self.test_data['GDP'] / self.test_data['M2']
        pd.testing.assert_series_equal(
            results['velocity'],
            expected_velocity,
            check_names=False
        )
        
    def test_analyze_interest_rate_impact(self):
        """Test interest rate impact analysis"""
        results = self.analyzer.analyze_interest_rate_impact(
            rate_series='DFF',
            target_variables=['CONSUMPTION', 'INVESTMENT'],
            max_lag=4
        )
        
        # Check results structure
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('variable', results.columns)
        self.assertIn('lag', results.columns)
        self.assertIn('correlation', results.columns)
        self.assertIn('p_value', results.columns)
        
        # Check we have results for both targets
        self.assertIn('CONSUMPTION', results['variable'].values)
        self.assertIn('INVESTMENT', results['variable'].values)
        
    def test_find_leading_indicators(self):
        """Test leading indicator identification"""
        # Add a leading indicator
        self.test_data['LEADING'] = self.test_data['GDP'].shift(-2)  # Leads by 2 periods
        self.analyzer = EconomicAnalysis(self.test_data)
        
        results = self.analyzer.find_leading_indicators(
            target='GDP',
            max_lag=4,
            significance_level=0.05
        )
        
        # Check results structure
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('indicator', results.columns)
        self.assertIn('lag', results.columns)
        self.assertIn('correlation', results.columns)
        self.assertIn('p_value', results.columns)
        
        # LEADING should be identified as leading indicator
        leading_results = results[results['indicator'] == 'LEADING']
        self.assertTrue(len(leading_results) > 0)
        self.assertTrue((leading_results['p_value'] < 0.05).any())
        
    def test_compute_rolling_correlations(self):
        """Test rolling correlation computation"""
        correlations = self.analyzer.compute_rolling_correlations(
            variables=['GDP', 'M2'],
            window=20,
            min_periods=10
        )
        
        # Check results
        self.assertIsInstance(correlations, pd.Series)
        self.assertEqual(correlations.name, 'GDP_M2_correlation')
        
        # Check values in reasonable range
        self.assertTrue((correlations >= -1).all())
        self.assertTrue((correlations <= 1).all())
        
    def test_detect_regime_changes(self):
        """Test regime change detection"""
        # Create data with clear regime change
        data1 = pd.DataFrame({
            'series': np.random.normal(0, 1, 100)
        })
        data2 = pd.DataFrame({
            'series': np.random.normal(5, 1, 100)  # Different mean
        })
        data = pd.concat([data1, data2], ignore_index=True)
        
        analyzer = EconomicAnalysis(data)
        regimes = analyzer.detect_regime_changes(
            data['series'],
            method='cusum',
            threshold=0.05
        )
        
        # Should detect change around index 100
        self.assertIsInstance(regimes, pd.Series)
        self.assertTrue(regimes.dtype == bool)
        self.assertTrue(regimes[95:105].any())  # Change detected near transition