import pandas as pd
import matplotlib.pyplot as plt
import sys

# --- Configuration ---
# Path to your Parquet file
file_path = "/home/tesla/Z1/temp/data/z1_quarterly/z1_quarterly_data_filtered.parquet"
# The series you want to extract and plot
series_code = "FL344190045"

# --- Main Script ---
try:
    # 1. Read the Z.1 Parquet file into a pandas DataFrame
    print(f"Reading data from: {file_path}")
    data = pd.read_parquet(file_path)
    print("Data loaded successfully.")

    # Ensure the index is a DatetimeIndex for proper plotting
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # 2. Check if the series exists and extract it
    if series_code in data.columns:
        print(f"Extracting series: {series_code}")
        series_to_plot = data[series_code]
        
        # 3. Plot the extracted series
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(series_to_plot.index, series_to_plot.values, label=series_code, color='royalblue')
        
        # Formatting the plot
        ax.set_title(f'Time Series Plot for {series_code}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        print("Displaying plot...")
        plt.show()

    else:
        print(f"Error: Series '{series_code}' not found in the Parquet file.", file=sys.stderr)
        sys.exit(1)

except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {file_path}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    sys.exit(1)
