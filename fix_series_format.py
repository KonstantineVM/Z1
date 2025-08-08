#!/usr/bin/env python3
"""
Fix the series format mismatch between Z1 and FWTW.
The issue: Z1 series have .Q suffix and different format than FWTW expects.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from src.data.cached_fed_data_loader import CachedFedDataLoader
from src.data.data_processor import DataProcessor
from src.network.fwtw_loader import FWTWDataLoader
from src.network.fwtw_z1_mapper import FWTWtoZ1Mapper


def fix_z1_series_format():
    """Fix Z1 series to match FWTW mapping expectations."""
    
    print("=" * 80)
    print("FIXING Z1 SERIES FORMAT")
    print("=" * 80)
    
    # Load Z1 data
    print("\n1. Loading Z1 data...")
    loader = CachedFedDataLoader()
    z1_raw = loader.load_single_source('Z1')
    processor = DataProcessor()
    z1_data = processor.process_fed_data(z1_raw, 'Z1')
    
    print(f"Original Z1 shape: {z1_data.shape}")
    print(f"Sample columns before fix: {list(z1_data.columns)[:5]}")
    
    # Fix 1: Remove .Q suffix from all columns
    print("\n2. Removing .Q suffix from series names...")
    fixed_columns = []
    for col in z1_data.columns:
        col_str = str(col)
        if col_str.endswith('.Q'):
            fixed_columns.append(col_str[:-2])
        else:
            fixed_columns.append(col_str)
    
    z1_data.columns = fixed_columns
    print(f"Sample columns after removing .Q: {list(z1_data.columns)[:5]}")
    
    # Analyze the cleaned series
    print("\n3. Analyzing cleaned series format...")
    series_lengths = {}
    for col in z1_data.columns:
        length = len(str(col))
        if length not in series_lengths:
            series_lengths[length] = []
        series_lengths[length].append(col)
    
    print("Series length distribution:")
    for length, series_list in sorted(series_lengths.items()):
        print(f"  Length {length}: {len(series_list)} series")
        if len(series_list) <= 3:
            for s in series_list:
                print(f"    - {s}")
    
    return z1_data


def test_fwtw_mapping_with_fixed_data(z1_data):
    """Test if FWTW mapping works with fixed Z1 data."""
    
    print("\n" + "=" * 80)
    print("TESTING FWTW MAPPING WITH FIXED DATA")
    print("=" * 80)
    
    # Load FWTW data
    print("\n1. Loading FWTW data...")
    fwtw_loader = FWTWDataLoader()
    fwtw_raw = fwtw_loader.load_fwtw_data()
    print(f"FWTW shape: {fwtw_raw.shape}")
    
    # Map FWTW to Z1
    print("\n2. Mapping FWTW to fixed Z1 series...")
    mapper = FWTWtoZ1Mapper()
    
    # Get Z1 series set
    z1_series_set = set(z1_data.columns)
    
    # Try mapping with validation
    fwtw_mapped = mapper.map_to_z1_series(
        fwtw_raw,
        available_z1_series=z1_series_set,
        include_all=False
    )
    
    if not fwtw_mapped.empty:
        print(f"Mapped data shape: {fwtw_mapped.shape}")
        
        # Check overlaps
        asset_series = set(fwtw_mapped['asset_series'].dropna())
        liability_series = set(fwtw_mapped['liability_series'].dropna())
        
        asset_overlap = asset_series & z1_series_set
        liability_overlap = liability_series & z1_series_set
        
        print(f"\nOverlap Results:")
        print(f"  Asset series in Z1: {len(asset_overlap)}")
        if asset_overlap:
            print(f"    Examples: {list(asset_overlap)[:5]}")
        
        print(f"  Liability series in Z1: {len(liability_overlap)}")
        if liability_overlap:
            print(f"    Examples: {list(liability_overlap)[:5]}")
    else:
        print("❌ Mapping still produces empty result")
    
    # Try generating all possible FWTW series
    print("\n3. Generating all possible FWTW series...")
    fwtw_all = mapper.map_to_z1_series(
        fwtw_raw,
        available_z1_series=None,
        include_all=True
    )
    
    if not fwtw_all.empty:
        all_series = set(fwtw_all['asset_series'].dropna()) | set(fwtw_all['liability_series'].dropna())
        print(f"Total FWTW series generated: {len(all_series)}")
        print(f"Examples: {list(all_series)[:10]}")
        
        # Find pattern matches
        print("\n4. Finding pattern matches...")
        find_pattern_matches(z1_series_set, all_series)


def find_pattern_matches(z1_series, fwtw_series):
    """Find series that match by pattern."""
    
    # Extract patterns from Z1
    z1_patterns = {}
    for series in z1_series:
        s = str(series)
        if len(s) >= 4:
            prefix = s[:2]
            sector = s[2:4] if len(s) > 3 else ""
            
            key = f"{prefix}{sector}"
            if key not in z1_patterns:
                z1_patterns[key] = []
            z1_patterns[key].append(series)
    
    # Extract patterns from FWTW
    fwtw_patterns = {}
    for series in fwtw_series:
        s = str(series)
        if len(s) >= 4:
            prefix = s[:2]
            sector = s[2:4] if len(s) > 3 else ""
            
            key = f"{prefix}{sector}"
            if key not in fwtw_patterns:
                fwtw_patterns[key] = []
            fwtw_patterns[key].append(series)
    
    # Find common patterns
    common_patterns = set(z1_patterns.keys()) & set(fwtw_patterns.keys())
    
    print(f"Common prefix+sector patterns: {len(common_patterns)}")
    if common_patterns:
        for pattern in list(common_patterns)[:10]:
            z1_example = z1_patterns[pattern][0] if z1_patterns[pattern] else "None"
            fwtw_example = fwtw_patterns[pattern][0] if fwtw_patterns[pattern] else "None"
            print(f"  Pattern {pattern}:")
            print(f"    Z1 example: {z1_example}")
            print(f"    FWTW example: {fwtw_example}")


def create_compatibility_mapping():
    """Create a mapping between Z1 and FWTW series formats."""
    
    print("\n" + "=" * 80)
    print("CREATING COMPATIBILITY MAPPING")
    print("=" * 80)
    
    # Common instrument codes from FWTW
    fwtw_instruments = {
        '20500': 'Federal funds and repos',
        '30110': 'Treasury securities',
        '30200': 'Agency securities',
        '30300': 'Municipal securities',
        '30400': 'Corporate bonds',
        '30500': 'Loans',
        '30611': 'Corporate equities',
        '31000': 'Mutual fund shares',
        '40000': 'Trade credit',
        '50000': 'Life insurance reserves',
        '60000': 'Pension entitlements'
    }
    
    # Common sectors
    sectors = {
        '10': 'Nonfinancial corporate',
        '11': 'Nonfinancial noncorporate',
        '15': 'Households',
        '21': 'State and local govt',
        '26': 'Rest of world',
        '31': 'Federal government',
        '42': 'GSEs',
        '47': 'Credit unions',
        '50': 'Other financial',
        '51': 'Property-casualty insurance',
        '54': 'Life insurance',
        '59': 'Pension funds',
        '63': 'Money market funds',
        '65': 'Mutual funds',
        '66': 'Security brokers',
        '70': 'Private depository',
        '71': 'Monetary authority',
        '75': 'Foreign banking',
        '76': 'US chartered depositories'
    }
    
    print("\nExpected Z1 series format examples:")
    for prefix in ['FL', 'FU', 'FA', 'FR']:
        for sector_code in ['10', '15', '31']:
            for inst_code in ['20500', '30400']:
                # Z1 format variants
                z1_format1 = f"{prefix}{sector_code}{inst_code[:-2]}{inst_code[-2:]}"  # e.g., FL1020505
                z1_format2 = f"{prefix}{sector_code}{inst_code}05"  # e.g., FL102050005
                z1_format3 = f"{prefix}{sector_code}{inst_code}005"  # e.g., FL1020500005
                
                print(f"  {z1_format1} or {z1_format2} or {z1_format3}")
                
                if len([z1_format1, z1_format2, z1_format3]) > 10:
                    break
            if len([z1_format1, z1_format2, z1_format3]) > 10:
                break
        break


def save_fixed_data(z1_data):
    """Save the fixed Z1 data for use in SFC analysis."""
    
    print("\n" + "=" * 80)
    print("SAVING FIXED DATA")
    print("=" * 80)
    
    output_path = Path("data/z1_fixed.parquet")
    z1_data.to_parquet(output_path, compression='snappy')
    
    print(f"✓ Fixed Z1 data saved to: {output_path}")
    print(f"  Shape: {z1_data.shape}")
    print(f"  Series count: {len(z1_data.columns)}")
    
    # Also save a CSV sample for inspection
    sample_path = Path("data/z1_fixed_sample.csv")
    z1_data.iloc[:10, :20].to_csv(sample_path)
    print(f"✓ Sample saved to: {sample_path}")
    
    return output_path


def main():
    """Main execution."""
    
    # Fix Z1 series format
    z1_fixed = fix_z1_series_format()
    
    # Test FWTW mapping
    test_fwtw_mapping_with_fixed_data(z1_fixed)
    
    # Create mapping guide
    create_compatibility_mapping()
    
    # Save fixed data
    save_fixed_data(z1_fixed)
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("""
1. The main issue is the .Q suffix in Z1 series - now removed.

2. Check if series codes match expected FWTW format:
   - FWTW expects: FA{2-digit-sector}{5-digit-instrument}05
   - Your Z1 has: FL{2-digit-sector}{various-lengths}

3. To use the fixed data in your SFC analysis:
   - Modify the DataProcessor to return the fixed format
   - Or load the saved z1_fixed.parquet directly

4. If overlap is still low, it's normal because:
   - FWTW tracks bilateral positions between sectors
   - Z1 contains aggregate series
   - Not all bilateral positions have corresponding Z1 aggregates

5. You can still run SFC with:
   - Stock-flow constraints from Z1 pairs (working)
   - Formula constraints (if you add formulas file)
   - FWTW bilateral constraints (even without overlap)
""")


if __name__ == "__main__":
    main()