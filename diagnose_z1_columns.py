#!/usr/bin/env python3
"""
Diagnostic script to understand the actual Z1 data column format.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.data import CachedFedDataLoader
import pandas as pd

def diagnose_z1_columns():
    """Check what the actual Z1 columns look like."""
    
    print("=" * 60)
    print("Z1 COLUMN FORMAT DIAGNOSTIC")
    print("=" * 60)
    
    # Load Z1 data
    print("\nLoading Z1 data...")
    fed_loader = CachedFedDataLoader()
    z1_data = fed_loader.load_single_source('Z1')
    
    if z1_data is None:
        print("ERROR: Could not load Z1 data")
        return
    
    print(f"\nData shape: {z1_data.shape}")
    print(f"Data type: {type(z1_data)}")
    
    # Check columns
    print(f"\nTotal columns: {len(z1_data.columns)}")
    print("\nFirst 20 columns:")
    for i, col in enumerate(z1_data.columns[:20]):
        print(f"  {i:3d}: '{col}' (type: {type(col).__name__})")
    
    # Check if columns contain '.Q' suffix
    q_columns = [col for col in z1_data.columns if '.Q' in str(col)]
    print(f"\nColumns with '.Q': {len(q_columns)}")
    if q_columns:
        print("Examples:", q_columns[:5])
    
    # Check for columns that look like Z1 series codes
    # Z1 codes should be like: FL103064105.Q or FA156902005.Q
    z1_pattern_columns = []
    for col in z1_data.columns:
        col_str = str(col)
        # Remove .Q suffix if present
        if col_str.endswith('.Q'):
            col_clean = col_str[:-2]
        else:
            col_clean = col_str
        
        # Check if it matches Z1 pattern (2 letters + numbers)
        if len(col_clean) >= 10 and col_clean[:2].isalpha() and col_clean[2:].replace('.', '').isdigit():
            z1_pattern_columns.append(col)
    
    print(f"\nColumns matching Z1 pattern: {len(z1_pattern_columns)}")
    if z1_pattern_columns:
        print("\nFirst 10 Z1 pattern columns:")
        for col in z1_pattern_columns[:10]:
            print(f"  {col}")
    
    # Check specific prefixes
    prefixes = ['FL', 'FU', 'FR', 'FV', 'FA', 'FC', 'FG', 'FI']
    print("\nColumns by prefix:")
    for prefix in prefixes:
        matching = [col for col in z1_data.columns if str(col).startswith(prefix)]
        if matching:
            print(f"  {prefix}: {len(matching)} columns")
            print(f"    Examples: {matching[:3]}")
    
    # Check index
    print(f"\nIndex type: {type(z1_data.index)}")
    print(f"Index name: {z1_data.index.name}")
    if len(z1_data) > 0:
        print(f"First index value: {z1_data.index[0]}")
        print(f"Last index value: {z1_data.index[-1]}")
    
    # Check data content
    print("\nData sample (first 5 rows, first 5 columns):")
    print(z1_data.iloc[:5, :5])
    
    return z1_data


def test_column_cleaning():
    """Test how to properly clean column names."""
    
    print("\n" + "=" * 60)
    print("TESTING COLUMN CLEANING")
    print("=" * 60)
    
    fed_loader = CachedFedDataLoader()
    z1_data = fed_loader.load_single_source('Z1')
    
    if z1_data is None:
        return
    
    # Clean column names
    cleaned_columns = []
    for col in z1_data.columns:
        col_str = str(col)
        # Remove .Q suffix
        if col_str.endswith('.Q'):
            col_clean = col_str[:-2]
        else:
            col_clean = col_str
        cleaned_columns.append(col_clean)
    
    # Create new dataframe with cleaned columns
    z1_clean = z1_data.copy()
    z1_clean.columns = cleaned_columns
    
    # Now check for Z1 patterns
    z1_patterns = ['FL', 'FU', 'FR', 'FV', 'FA']
    z1_cols = []
    
    for col in z1_clean.columns:
        if any(col.startswith(pattern) for pattern in z1_patterns):
            z1_cols.append(col)
    
    print(f"\nAfter cleaning column names:")
    print(f"  Found {len(z1_cols)} Z1 series")
    
    if z1_cols:
        # Check composition
        composition = {}
        for pattern in z1_patterns:
            count = len([c for c in z1_cols if c.startswith(pattern)])
            if count > 0:
                composition[pattern] = count
        
        print("\nComposition:")
        for prefix, count in composition.items():
            print(f"  {prefix}: {count}")
        
        # Check for stock-flow pairs
        fl_series = [c for c in z1_cols if c.startswith('FL')]
        pairs = []
        
        for fl in fl_series[:20]:  # Check first 20
            if len(fl) >= 9:
                sector = fl[2:4]
                instrument = fl[4:9]
                fu = f"FU{sector}{instrument}"
                
                fu_matches = [c for c in z1_cols if c.startswith(fu)]
                if fu_matches:
                    pairs.append((fl, fu_matches[0]))
        
        print(f"\nFound {len(pairs)} stock-flow pairs (checking first 20 FL series)")
        if pairs:
            print("Examples:")
            for stock, flow in pairs[:3]:
                print(f"  {stock} <-> {flow}")


if __name__ == "__main__":
    z1_data = diagnose_z1_columns()
    test_column_cleaning()