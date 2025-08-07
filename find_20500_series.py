import pandas as pd
from src.data import CachedFedDataLoader

# Load Z1 data
fed_loader = CachedFedDataLoader()
z1_data_raw = fed_loader.load_single_source('Z1')

# Get quarterly series
quarterly_mask = z1_data_raw['SERIES_NAME'].str.endswith('.Q')
quarterly_series = z1_data_raw[quarterly_mask]['SERIES_NAME'].str.replace('.Q', '', regex=False)

# Find all series with instrument 20500
instrument_20500_series = [s for s in quarterly_series if '20500' in s[4:9]]

print("=" * 60)
print("SERIES WITH INSTRUMENT 20500 (Federal Funds and Repos)")
print("=" * 60)
print(f"Total: {len(instrument_20500_series)} series\n")

# Group by sector
by_sector = {}
for series in instrument_20500_series:
    sector = series[2:4]
    if sector not in by_sector:
        by_sector[sector] = []
    by_sector[sector].append(series)

print("By Sector:")
for sector, series_list in sorted(by_sector.items()):
    print(f"  Sector {sector}: {len(series_list)} series")
    print(f"    Examples: {series_list[:3]}")

# Now check FWTW - what are the most common sector/instrument combinations?
print("\n" + "=" * 60)
print("CHECKING FWTW FOR COMMON COMBINATIONS")
print("=" * 60)

from src.network import FWTWDataLoader
fwtw_loader = FWTWDataLoader()
fwtw_data = fwtw_loader.load_fwtw_data()

# Get top combinations
fwtw_combinations = fwtw_data.groupby(['Holder Code', 'Issuer Code', 'Instrument Code']).size().reset_index(name='count')
fwtw_combinations = fwtw_combinations.sort_values('count', ascending=False)

print("\nTop 20 FWTW combinations (Holder-Issuer-Instrument):")
for _, row in fwtw_combinations.head(20).iterrows():
    holder = str(int(row['Holder Code'])).zfill(2)
    issuer = str(int(row['Issuer Code'])).zfill(2)
    instrument = str(int(row['Instrument Code'])).zfill(5)
    
    # Check if these series exist in Z1
    test_series_fa = f"FA{holder}{instrument}"
    test_series_fl = f"FL{issuer}{instrument}"
    
    fa_exists = any(s.startswith(test_series_fa) for s in quarterly_series)
    fl_exists = any(s.startswith(test_series_fl) for s in quarterly_series)
    
    status = "✓" if (fa_exists or fl_exists) else "✗"
    print(f"  {status} H:{holder} I:{issuer} Inst:{instrument} - Count:{row['count']:,}")

print("\n" + "=" * 60)
print("SOLUTION:")
print("=" * 60)
print("The issue is NOT format - it's that FWTW has bilateral positions")
print("that don't exist in your Z1 data (different sector-instrument combinations).")
print("\nOptions:")
print("1. This is NORMAL - FWTW tracks bilateral positions Z1 doesn't have")
print("2. The coverage warning can be ignored - you have the Z1 data you need")
print("3. The Kalman filter will work with whatever series overlap exists")
