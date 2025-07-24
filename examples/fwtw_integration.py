#!/usr/bin/env python
"""
Example: Integrating FWTW network analysis with Z1 time series
"""

from src.data import CachedFedDataLoader
from src.network import FWTWDataLoader, NetworkBuilder, NetworkAnalyzer
from src.analysis import NetworkIntegratedAnalysis
from src.models import UnobservedComponentsModel

def main():
    # Load Z1 data
    print("Loading Z1 data...")
    fed_loader = CachedFedDataLoader()
    z1_data = fed_loader.load_single_source('Z1')
    
    # Load FWTW data
    print("Loading FWTW data...")
    fwtw_loader = FWTWDataLoader()
    fwtw_data = fwtw_loader.load_fwtw_data()
    
    # Build and analyze network
    print("Building financial network...")
    builder = NetworkBuilder(fwtw_data)
    latest_date = fwtw_data['Date'].max()
    network = builder.build_snapshot(latest_date)
    
    # Analyze network
    analyzer = NetworkAnalyzer(network)
    sifis = analyzer.identify_systemically_important()
    
    print(f"\nTop 5 Systemically Important Institutions:")
    for entity, score in sifis[:5]:
        print(f"  {entity}: {score:.4f}")
    
    # Integrate with time series analysis
    print("\nIntegrating with Z1 time series...")
    # ... additional integration code ...

if __name__ == "__main__":
    main()
