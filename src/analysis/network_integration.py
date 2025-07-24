"""
Integration of FWTW network analysis with Z1 time series
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from ..network import FWTWDataLoader, NetworkBuilder, NetworkAnalyzer
from .economic_analysis import EconomicAnalysis

class NetworkIntegratedAnalysis(EconomicAnalysis):
    """Combine network and time series analysis"""
    
    def __init__(self, z1_components: Dict, fwtw_data: pd.DataFrame):
        super().__init__(z1_components)
        self.fwtw_data = fwtw_data
        self.network_builder = NetworkBuilder(fwtw_data)
        
    def create_network_features(self, dates: List[pd.Timestamp]) -> pd.DataFrame:
        """Extract network features for time series analysis"""
        features = []
        
        for date in dates:
            network = self.network_builder.build_snapshot(date)
            analyzer = NetworkAnalyzer(network)
            
            # Extract key metrics
            risk_metrics = analyzer.calculate_network_risk_metrics()
            centrality = analyzer.compute_centrality_metrics()
            
            # Create feature vector
            feature_dict = {
                'date': date,
                'network_density': risk_metrics['density'],
                'herfindahl_index': risk_metrics['herfindahl_index'],
                'global_efficiency': risk_metrics['global_efficiency'],
                'avg_clustering': risk_metrics['avg_clustering'],
                'num_nodes': network.number_of_nodes(),
                'num_edges': network.number_of_edges(),
                'total_volume': network.graph.get('total_volume', 0)
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features).set_index('date')
    
    def validate_flows(self) -> pd.DataFrame:
        """Cross-validate FWTW flows with Z1 aggregates"""
        # Implementation for flow validation
        pass
