#!/usr/bin/env python
"""
Check the actual structure and content of cached Z1 data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cached_fed_data_loader import CachedFedDataLoader
from src.data.data_processor import DataProcessor


def main():
    print("Checking Z1 Data Structure...")
    print("=" * 60)
    
    # Load data
    loader = CachedFedDataLoader()
    z1_raw = loader.load_single_source('Z1')
    
    if z1_raw is None:
        print("❌ No Z1 data found!")
        return
    
    print(f"✓ Loaded raw Z1 data: {z1_raw.shape}")
    print(f"\nFirst 10 columns:")
    for i, col in enumerate(z1_raw.columns[:10]):
        print(f"  {i}: {col}")
    
    print(f"\nData types of first 10 columns:")
    for col in z1_raw.columns[:10]:
        print(f"  {col}: {z1_raw[col].dtype}")
    
    # Show sample of raw data
    print(f"\nSample of raw data (first 3 rows, first 5 columns):")
    print(z1_raw.iloc[:3, :5])
    
    # Process the data
    print("\n" + "-" * 60)
    print("Processing Z1 data...")
    
    processor = DataProcessor()
    z1_processed = processor.process_fed_data(z1_raw, 'Z1')
    
    print(f"✓ Processed Z1 data: {z1_processed.shape}")
    
    # Check if we have time series data now
    if isinstance(z1_processed.index, pd.DatetimeIndex):
        print(f"✓ Data has DatetimeIndex")
        print(f"  Date range: {z1_processed.index.min()} to {z1_processed.index.max()}")
    
    # Check numeric columns
    numeric_cols = z1_processed.select_dtypes(include=[np.number]).columns
    print(f"\n✓ Numeric columns: {len(numeric_cols)} / {len(z1_processed.columns)}")
    
    if len(numeric_cols) > 0:
        # Show some statistics
        print(f"\nSample series statistics:")
        sample_col = numeric_cols[0]
        sample_data = z1_processed[sample_col].dropna()
        if len(sample_data) > 0:
            print(f"  Series: {sample_col}")
            print(f"  Non-null values: {len(sample_data)}")
            print(f"  Mean: {sample_data.mean():.2f}")
            print(f"  Std: {sample_data.std():.2f}")
            print(f"  Min: {sample_data.min():.2f}")
            print(f"  Max: {sample_data.max():.2f}")
    
    # Look for specific series types
    print("\n" + "-" * 60)
    print("Looking for specific series types...")
    
    # Find series by keywords
    keywords = ['household', 'corporate', 'debt', 'equity', 'gdp', 'mortgage']
    for keyword in keywords:
        matches = [col for col in z1_processed.columns if keyword in col.lower()]
        if matches:
            print(f"\n'{keyword}' series: {len(matches)} found")
            print(f"  Examples: {matches[:3]}")


if __name__ == "__main__":
    main()
