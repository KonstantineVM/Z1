import unittest
import pandas as pd
import networkx as nx
from src.network import FWTWDataLoader, NetworkBuilder, NetworkAnalyzer

class TestFWTWIntegration(unittest.TestCase):
    
    def setUp(self):
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Date': pd.to_datetime(['2024-03-31'] * 5),
            'Holder Name': ['Bank A', 'Bank B', 'Fund C', 'Bank A', 'Fund C'],
            'Issuer Name': ['Corp X', 'Corp Y', 'Bank A', 'Fund C', 'Corp X'],
            'Instrument Name': ['Bonds'] * 5,
            'Level': [1000, 2000, 1500, 800, 1200]
        })
    
    def test_network_builder(self):
        builder = NetworkBuilder(self.sample_data)
        network = builder.build_snapshot(pd.to_datetime('2024-03-31'))
        
        self.assertEqual(network.number_of_nodes(), 5)
        self.assertEqual(network.number_of_edges(), 5)
    
    def test_network_analyzer(self):
        builder = NetworkBuilder(self.sample_data)
        network = builder.build_snapshot(pd.to_datetime('2024-03-31'))
        analyzer = NetworkAnalyzer(network)
        
        metrics = analyzer.calculate_network_risk_metrics()
        self.assertIn('density', metrics)
        self.assertIn('global_efficiency', metrics)

if __name__ == '__main__':
    unittest.main()
