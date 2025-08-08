#!/usr/bin/env python3
"""
Diagnostic script to identify why FWTW and Z1 series don't match.
This will help fix the overlap issue in the SFC Kalman Filter.
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


def main():
    print("=" * 80)
    print("FWTW-Z1 SERIES MISMATCH DIAGNOSTIC")
    print("=" * 80)
    
    # Step 1: Load and process Z1 data
    print("\n1. Loading Z1 Data...")
    print("-" * 40)
    
    loader = CachedFedDataLoader()
    z1_raw = loader.load_single_source('Z1')
    
    if z1_raw is None:
        print("ERROR: Could not load Z1 data")
        return
    
    print(f"Raw Z1 data shape: {z1_raw.shape}")
    
    # Process to get time series format
    processor = DataProcessor()
    z1_data = processor.process_fed_data(z1_raw, 'Z1')
    print(f"Processed Z1 data shape: {z1_data.shape}")
    
    # Analyze Z1 series format
    print("\nZ1 Series Analysis:")
    z1_series = list(z1_data.columns)
    
    # Check series patterns
    patterns = {'FL': [], 'FU': [], 'FA': [], 'FR': [], 'FV': [], 'Other': []}
    for series in z1_series:
        series_str = str(series)
        if series_str.startswith('FL'):
            patterns['FL'].append(series)
        elif series_str.startswith('FU'):
            patterns['FU'].append(series)
        elif series_str.startswith('FA'):
            patterns['FA'].append(series)
        elif series_str.startswith('FR'):
            patterns['FR'].append(series)
        elif series_str.startswith('FV'):
            patterns['FV'].append(series)
        else:
            patterns['Other'].append(series)
    
    for pattern, series_list in patterns.items():
        if series_list:
            print(f"  {pattern}: {len(series_list)} series")
            if len(series_list) <= 3:
                for s in series_list:
                    print(f"    - {s}")
            else:
                print(f"    Examples: {series_list[0]}, {series_list[1]}, {series_list[2]}")
    
    # Step 2: Load FWTW data
    print("\n2. Loading FWTW Data...")
    print("-" * 40)
    
    fwtw_loader = FWTWDataLoader()
    fwtw_raw = fwtw_loader.load_fwtw_data()
    
    if fwtw_raw is None:
        print("ERROR: Could not load FWTW data")
        return
    
    print(f"FWTW data shape: {fwtw_raw.shape}")
    print(f"FWTW columns: {list(fwtw_raw.columns)}")
    
    # Analyze FWTW codes
    print("\nFWTW Code Analysis:")
    if 'Holder Code' in fwtw_raw.columns:
        unique_holders = fwtw_raw['Holder Code'].unique()
        print(f"  Unique holder codes: {len(unique_holders)}")
        print(f"    Examples: {sorted(unique_holders)[:10]}")
    
    if 'Issuer Code' in fwtw_raw.columns:
        unique_issuers = fwtw_raw['Issuer Code'].unique()
        print(f"  Unique issuer codes: {len(unique_issuers)}")
        print(f"    Examples: {sorted(unique_issuers)[:10]}")
    
    if 'Instrument Code' in fwtw_raw.columns:
        unique_instruments = fwtw_raw['Instrument Code'].unique()
        print(f"  Unique instrument codes: {len(unique_instruments)}")
        print(f"    Examples: {sorted(unique_instruments)[:10]}")
    
    # Step 3: Map FWTW to Z1
    print("\n3. Mapping FWTW to Z1 Series Codes...")
    print("-" * 40)
    
    mapper = FWTWtoZ1Mapper()
    
    # First try with validation against Z1 series
    fwtw_mapped = mapper.map_to_z1_series(
        fwtw_raw, 
        available_z1_series=set(z1_series),
        include_all=False  # Only include if in Z1
    )
    
    print(f"Mapped FWTW data shape: {fwtw_mapped.shape}")
    
    # Check mapped series
    if not fwtw_mapped.empty:
        asset_series = fwtw_mapped['asset_series'].dropna().unique()
        liability_series = fwtw_mapped['liability_series'].dropna().unique()
        
        print(f"  Unique asset series: {len(asset_series)}")
        print(f"  Unique liability series: {len(liability_series)}")
        
        # Find overlaps
        z1_set = set(z1_series)
        asset_overlap = set(asset_series) & z1_set
        liability_overlap = set(liability_series) & z1_set
        
        print(f"\n  Asset series in Z1: {len(asset_overlap)}")
        if asset_overlap:
            print(f"    Examples: {list(asset_overlap)[:5]}")
        
        print(f"  Liability series in Z1: {len(liability_overlap)}")
        if liability_overlap:
            print(f"    Examples: {list(liability_overlap)[:5]}")
    
    # Step 4: Try mapping with all=True to see what's generated
    print("\n4. Mapping FWTW Without Validation...")
    print("-" * 40)
    
    fwtw_all = mapper.map_to_z1_series(
        fwtw_raw,
        available_z1_series=None,
        include_all=True
    )
    
    if not fwtw_all.empty:
        all_asset_series = fwtw_all['asset_series'].dropna().unique()
        all_liability_series = fwtw_all['liability_series'].dropna().unique()
        
        print(f"Generated asset series: {len(all_asset_series)}")
        print(f"Examples of generated series:")
        for s in list(all_asset_series)[:10]:
            in_z1 = "✓" if s in z1_series else "✗"
            print(f"  {s} {in_z1}")
    
    # Step 5: Analyze the mismatch
    print("\n5. MISMATCH ANALYSIS")
    print("=" * 80)
    
    # Check series length differences
    if z1_series and len(all_asset_series) > 0:
        z1_example = z1_series[0] if z1_series else ""
        fwtw_example = list(all_asset_series)[0] if all_asset_series else ""
        
        print(f"Z1 series example: {z1_example}")
        print(f"  Length: {len(z1_example)}")
        print(f"  Format: {analyze_format(z1_example)}")
        
        print(f"\nFWTW mapped example: {fwtw_example}")
        print(f"  Length: {len(fwtw_example)}")
        print(f"  Format: {analyze_format(fwtw_example)}")
    
    # Step 6: Find closest matches
    print("\n6. FINDING CLOSEST MATCHES...")
    print("-" * 40)
    
    if z1_series and len(all_asset_series) > 0:
        # Look for partial matches
        matches = find_partial_matches(z1_series, all_asset_series)
        
        if matches:
            print(f"Found {len(matches)} potential matches:")
            for z1, fwtw in list(matches.items())[:10]:
                print(f"  Z1: {z1}")
                print(f"  FWTW: {fwtw}")
                print()
    
    # Step 7: Recommendations
    print("\n7. RECOMMENDATIONS")
    print("=" * 80)
    
    if len(asset_overlap) == 0 and len(liability_overlap) == 0:
        print("❌ No direct overlap found between FWTW and Z1 series")
        print("\nPossible causes:")
        print("1. Series code format mismatch (check suffix)")
        print("2. Different sector/instrument coding")
        print("3. FWTW uses bilateral codes not in Z1 aggregates")
        print("\nSolutions:")
        print("1. Check if Z1 series have suffixes (e.g., '.Q' removed)")
        print("2. Verify FWTW sector codes match Z1 sectors")
        print("3. Use formula constraints instead of FWTW for test")
    else:
        print(f"✓ Found {len(asset_overlap) + len(liability_overlap)} overlapping series")
        print("The mapping is working but overlap is limited.")
    
    # Save diagnostic data
    print("\n8. SAVING DIAGNOSTIC DATA...")
    print("-" * 40)
    
    output_dir = Path("output/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save series lists
    with open(output_dir / "z1_series.txt", "w") as f:
        for s in sorted(z1_series)[:100]:
            f.write(f"{s}\n")
    
    with open(output_dir / "fwtw_mapped_series.txt", "w") as f:
        for s in sorted(all_asset_series)[:100]:
            f.write(f"{s}\n")
    
    print(f"Diagnostic data saved to {output_dir}")
    print("\nDiagnostic complete!")


def analyze_format(series_code):
    """Analyze the format of a series code."""
    if not series_code:
        return "Empty"
    
    parts = []
    
    # Check prefix
    if series_code[:2].isalpha():
        parts.append(f"Prefix={series_code[:2]}")
    
    # Check for sector (positions 2-4)
    if len(series_code) > 4:
        parts.append(f"Sector={series_code[2:4]}")
    
    # Check for instrument (positions 4-9)
    if len(series_code) > 9:
        parts.append(f"Instrument={series_code[4:9]}")
    
    # Check for suffix
    if len(series_code) > 9:
        parts.append(f"Suffix={series_code[9:]}")
    
    return " | ".join(parts)


def find_partial_matches(z1_series, fwtw_series):
    """Find Z1 and FWTW series that partially match."""
    matches = {}
    
    for z1 in z1_series[:100]:  # Check first 100
        z1_str = str(z1)
        
        # Extract components if possible
        if len(z1_str) >= 4:
            z1_prefix = z1_str[:2]
            z1_sector = z1_str[2:4] if len(z1_str) > 4 else ""
            
            for fwtw in fwtw_series:
                fwtw_str = str(fwtw)
                
                if len(fwtw_str) >= 4:
                    fwtw_prefix = fwtw_str[:2]
                    fwtw_sector = fwtw_str[2:4] if len(fwtw_str) > 4 else ""
                    
                    # Check if prefix and sector match
                    if z1_prefix == fwtw_prefix and z1_sector == fwtw_sector:
                        matches[z1] = fwtw
                        break
    
    return matches


if __name__ == "__main__":
    main()