import pandas as pd
from src.data import CachedFedDataLoader

# Load Z1 data
fed_loader = CachedFedDataLoader()
z1_data_raw = fed_loader.load_single_source('Z1')

# Get quarterly series names without .Q suffix
quarterly_mask = z1_data_raw['SERIES_NAME'].str.endswith('.Q')
quarterly_series = z1_data_raw[quarterly_mask]['SERIES_NAME'].str.replace('.Q', '', regex=False)
available_series = set(quarterly_series)

print("=" * 60)
print("CHECKING FOR MATCHING SERIES")
print("=" * 60)

# FWTW example: Instrument 20500, Holder 10, Issuer 26
# This would create series like:
# FA1020500005 (Holder 10 holds instrument 20500)
# FL2620500005 (Issuer 26 issued instrument 20500)

# Let's check what series you have with these sectors and instruments
print("\nLooking for series with sector 10 (Nonfin Corp Bus):")
sector_10_series = [s for s in available_series if s[2:4] == '10']
print(f"Found {len(sector_10_series)} series with sector 10")
if sector_10_series:
    print("Examples:", sector_10_series[:5])

print("\nLooking for series with sector 26 (Rest of World):")
sector_26_series = [s for s in available_series if s[2:4] == '26']
print(f"Found {len(sector_26_series)} series with sector 26")
if sector_26_series:
    print("Examples:", sector_26_series[:5])

print("\nLooking for series with instrument 20500 (Federal Funds and Repos):")
instrument_20500_series = [s for s in available_series if '20500' in s]
print(f"Found {len(instrument_20500_series)} series with instrument 20500")
if instrument_20500_series:
    print("Examples:", instrument_20500_series[:5])

# Check for the exact series FWTW would expect (with both suffix formats)
print("\n" + "=" * 60)
print("CHECKING EXACT MATCHES:")
print("=" * 60)

test_series = [
    # With 2-digit suffix (your format)
    "FA1020500", "FA102050005", 
    "FL2620500", "FL262050005",
    "FA1020505", "FL2620505",
    # With 3-digit suffix (FWTW format)
    "FA1020500005", "FL2620500005",
    # Check other sectors that might have this instrument
    "FA1120500", "FA1320500", "FA1420500",
]

for series in test_series:
    # Check exact match and with variations
    matches = [s for s in available_series if s.startswith(series)]
    if matches:
        print(f"✓ Found matches for {series}: {matches}")
    else:
        # Check if similar exists
        similar = [s for s in available_series if s[:4] == series[:4]]
        if similar:
            print(f"✗ No match for {series}, but found similar: {similar[:3]}")

# Let's see what instruments sector 10 and 26 actually have
print("\n" + "=" * 60)
print("AVAILABLE INSTRUMENTS FOR KEY SECTORS:")
print("=" * 60)

def extract_instrument(series):
    if len(series) >= 9:
        return series[4:9]
    return None

print("\nInstruments for sector 10:")
sector_10_instruments = set(extract_instrument(s) for s in sector_10_series if extract_instrument(s))
print(f"Found {len(sector_10_instruments)} unique instruments")
print("Sample instruments:", sorted(list(sector_10_instruments))[:10])

print("\nInstruments for sector 26:")
sector_26_instruments = set(extract_instrument(s) for s in sector_26_series if extract_instrument(s))
print(f"Found {len(sector_26_instruments)} unique instruments")
print("Sample instruments:", sorted(list(sector_26_instruments))[:10])

# Check if 20500 is among them
if '20500' in sector_10_instruments:
    print("\n✓ Sector 10 HAS instrument 20500")
else:
    print("\n✗ Sector 10 does NOT have instrument 20500")
    
if '20500' in sector_26_instruments:
    print("✓ Sector 26 HAS instrument 20500")
else:
    print("✗ Sector 26 does NOT have instrument 20500")
