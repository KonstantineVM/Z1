import pandas as pd
from src.data import CachedFedDataLoader
from src.network import FWTWDataLoader

# Load Z1 data and get series names
fed_loader = CachedFedDataLoader()
z1_data_raw = fed_loader.load_single_source('Z1')

# Get quarterly series names without .Q suffix
quarterly_mask = z1_data_raw['SERIES_NAME'].str.endswith('.Q')
quarterly_series = z1_data_raw[quarterly_mask]['SERIES_NAME'].str.replace('.Q', '', regex=False)
available_series = set(quarterly_series)

print("=" * 60)
print("AVAILABLE Z1 SERIES FORMAT:")
print("=" * 60)
# Show sample of actual series
sample_available = sorted(list(available_series))[:20]
for s in sample_available:
    print(f"  {s}")

# Analyze the format
print("\n" + "=" * 60)
print("Z1 SERIES FORMAT ANALYSIS:")
print("=" * 60)

# Check series lengths and patterns
series_lengths = {}
for s in available_series:
    length = len(s)
    if length not in series_lengths:
        series_lengths[length] = []
    series_lengths[length].append(s)

print("Series lengths distribution:")
for length, series_list in sorted(series_lengths.items()):
    print(f"  Length {length}: {len(series_list)} series")
    if len(series_list) <= 3:
        for s in series_list:
            print(f"    Example: {s}")
    else:
        print(f"    Examples: {series_list[0]}, {series_list[1]}, {series_list[2]}")

# Load FWTW to see what it expects
print("\n" + "=" * 60)
print("FWTW EXPECTED FORMAT:")
print("=" * 60)

fwtw_loader = FWTWDataLoader()
fwtw_data = fwtw_loader.load_fwtw_data()

print(f"FWTW columns: {fwtw_data.columns.tolist()}")
print(f"FWTW shape: {fwtw_data.shape}")

# Check FWTW codes
if 'Instrument Code' in fwtw_data.columns:
    print(f"\nSample Instrument Codes: {fwtw_data['Instrument Code'].head(10).tolist()}")
if 'Holder Code' in fwtw_data.columns:
    print(f"Sample Holder Codes: {fwtw_data['Holder Code'].head(10).tolist()}")
if 'Issuer Code' in fwtw_data.columns:
    print(f"Sample Issuer Codes: {fwtw_data['Issuer Code'].head(10).tolist()}")

# Generate expected Z1 series from FWTW
print("\n" + "=" * 60)
print("EXPECTED Z1 SERIES FROM FWTW:")
print("=" * 60)

expected_from_fwtw = set()
if 'Instrument Code' in fwtw_data.columns and 'Holder Code' in fwtw_data.columns:
    # Take a sample
    for _, row in fwtw_data.head(100).iterrows():
        instrument = str(row['Instrument Code']).zfill(5)
        holder = str(row['Holder Code']).zfill(2)
        issuer = str(row['Issuer Code']).zfill(2) if 'Issuer Code' in fwtw_data.columns else '00'
        
        # Generate expected series codes
        for prefix in ['FA', 'FL', 'FU', 'FR']:
            # Asset side (holder perspective)
            expected_from_fwtw.add(f"{prefix}{holder}{instrument}005")
            # Liability side (issuer perspective)
            expected_from_fwtw.add(f"{prefix}{issuer}{instrument}005")

print("Sample expected series from FWTW:")
for s in sorted(list(expected_from_fwtw))[:20]:
    print(f"  {s}")

# Check overlap
print("\n" + "=" * 60)
print("OVERLAP ANALYSIS:")
print("=" * 60)

overlap = available_series & expected_from_fwtw
print(f"Exact matches: {len(overlap)}")
if overlap:
    print("Sample matches:")
    for s in sorted(list(overlap))[:10]:
        print(f"  {s}")

# Check pattern differences
print("\n" + "=" * 60)
print("PATTERN COMPARISON:")
print("=" * 60)

if len(sample_available) > 0 and len(expected_from_fwtw) > 0:
    avail_example = sample_available[0]
    expect_example = sorted(list(expected_from_fwtw))[0]
    
    print(f"Available format: {avail_example}")
    print(f"  Breakdown: {avail_example[:2]}|{avail_example[2:4]}|{avail_example[4:9]}|{avail_example[9:]}")
    
    print(f"Expected format:  {expect_example}")
    print(f"  Breakdown: {expect_example[:2]}|{expect_example[2:4]}|{expect_example[4:9]}|{expect_example[9:]}")
    
    # Check if the only difference is the suffix
    print("\nChecking suffix patterns:")
    available_without_suffix = set()
    for s in available_series:
        if len(s) >= 9:
            available_without_suffix.add(s[:9])
    
    expected_without_suffix = set()
    for s in expected_from_fwtw:
        if len(s) >= 9:
            expected_without_suffix.add(s[:9])
    
    overlap_without_suffix = available_without_suffix & expected_without_suffix
    print(f"Matches without suffix (first 9 chars): {len(overlap_without_suffix)}")
    
    if overlap_without_suffix:
        print("This suggests the issue is with the suffix!")
        print("Sample matches (first 9 chars):")
        for s in sorted(list(overlap_without_suffix))[:10]:
            avail_with_prefix = [a for a in available_series if a.startswith(s)]
            expect_with_prefix = [e for e in expected_from_fwtw if e.startswith(s)]
            print(f"  {s}* -> Available: {avail_with_prefix[:2]}, Expected: {expect_with_prefix[:2]}")
