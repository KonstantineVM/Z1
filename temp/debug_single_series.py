import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt

def diagnose_model_structure():
    """Diagnose why the UCM model structure fails for this data"""
    
    # Load data
    z1_data = pd.read_parquet('/home/tesla/Z1/temp/data/z1_quarterly/z1_quarterly_data_filtered.parquet')
    series_name = 'FL104090005'
    series = z1_data[series_name]
    
    print(f"\n=== DIAGNOSING MODEL STRUCTURE ISSUE ===\n")
    
    T = len(series)
    series_values = series.values
    
    # 1. Analyze the growth pattern
    print("1. GROWTH PATTERN ANALYSIS:")
    
    # Calculate log values to check for exponential growth
    log_values = np.log(series_values)
    
    # Fit linear regression to log values
    from scipy import stats
    time_index = np.arange(T)
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, log_values)
    
    print(f"   Log-linear regression:")
    print(f"   - R-squared: {r_value**2:.4f}")
    print(f"   - Implied growth rate: {(np.exp(slope) - 1) * 100:.2f}% per period")
    print(f"   - Annual growth (4 periods): {(np.exp(slope * 4) - 1) * 100:.2f}%")
    
    # 2. Show why the additive model fails
    print("\n2. WHY ADDITIVE MODEL FAILS:")
    
    # The UCM model assumes: Y_t = μ_t + ε_t
    # Where μ_t = μ_{t-1} + β_t (integrated slope)
    # This is ADDITIVE growth
    
    # But the data appears to have MULTIPLICATIVE growth
    # Better model would be: Y_t = Y_{t-1} * (1 + g_t)
    # Or in logs: log(Y_t) = log(Y_{t-1}) + log(1 + g_t)
    
    # Calculate period-over-period growth rates
    growth_rates = series_values[1:] / series_values[:-1] - 1
    avg_growth = np.mean(growth_rates)
    
    print(f"   Average period growth rate: {avg_growth * 100:.2f}%")
    print(f"   If we compound this: (1 + {avg_growth:.4f})^{T-1} = {(1 + avg_growth)**(T-1):.1f}")
    print(f"   Actual growth multiple: {series_values[-1] / series_values[0]:.1f}")
    
    # 3. Compare additive vs multiplicative predictions
    print("\n3. ADDITIVE VS MULTIPLICATIVE MODEL:")
    
    # Additive model: constant slope
    additive_slope = (series_values[-1] - series_values[0]) / (T - 1)
    additive_pred = series_values[0] + additive_slope * time_index
    
    # Multiplicative model: constant growth rate
    multiplicative_growth = (series_values[-1] / series_values[0]) ** (1 / (T - 1)) - 1
    multiplicative_pred = series_values[0] * (1 + multiplicative_growth) ** time_index
    
    # Log-linear model
    log_linear_pred = np.exp(intercept + slope * time_index)
    
    # Calculate errors
    additive_rmse = np.sqrt(np.mean((series_values - additive_pred)**2))
    multiplicative_rmse = np.sqrt(np.mean((series_values - multiplicative_pred)**2))
    log_linear_rmse = np.sqrt(np.mean((series_values - log_linear_pred)**2))
    
    print(f"   RMSE comparison:")
    print(f"   - Additive (constant slope): {additive_rmse:.2e}")
    print(f"   - Multiplicative (constant growth): {multiplicative_rmse:.2e}")
    print(f"   - Log-linear: {log_linear_rmse:.2e}")
    
    # 4. Show the UCM limitation
    print("\n4. UCM MODEL LIMITATION:")
    print("   The UCM model with integrated random walk slope assumes:")
    print("   Y_t = μ_0 + Σ(β_s) for s=1 to t")
    print("   Where β_t = β_{t-1} + η_t (random walk)")
    print("\n   This gives POLYNOMIAL growth at best (quadratic if β increases linearly)")
    print("   But the data shows EXPONENTIAL growth!")
    print(f"\n   Required final level: {series_values[-1]:.2e}")
    print(f"   Max achievable with slope ~{np.std(np.diff(series_values)):.2e}: ~{series_values[0] + np.std(np.diff(series_values)) * T:.2e}")
    
    # 5. Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Raw data with different models
    axes[0, 0].plot(series.index, series_values, 'k-', linewidth=2, label='Actual')
    axes[0, 0].plot(series.index, additive_pred, 'r--', label='Additive model')
    axes[0, 0].plot(series.index, multiplicative_pred, 'g--', label='Multiplicative model')
    axes[0, 0].plot(series.index, log_linear_pred, 'b--', label='Log-linear model')
    axes[0, 0].set_title('Different Model Types')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Log scale
    axes[0, 1].semilogy(series.index, series_values, 'k-', linewidth=2, label='Actual')
    axes[0, 1].semilogy(series.index, multiplicative_pred, 'g--', label='Multiplicative')
    axes[0, 1].semilogy(series.index, log_linear_pred, 'b--', label='Log-linear')
    axes[0, 1].set_title('Log Scale (shows exponential growth)')
    axes[0, 1].set_ylabel('Value (log scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Growth rates
    axes[1, 0].plot(series.index[1:], growth_rates * 100, 'k-', alpha=0.5)
    axes[1, 0].axhline(y=avg_growth * 100, color='r', linestyle='--', 
                       label=f'Average: {avg_growth * 100:.2f}%')
    axes[1, 0].set_title('Period-over-period Growth Rates')
    axes[1, 0].set_ylabel('Growth Rate (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Why UCM fails
    max_slope = np.std(np.diff(series_values)) * 3  # 3 sigma
    ucm_optimistic = series_values[0] + np.cumsum(np.linspace(0, max_slope, T))
    
    axes[1, 1].plot(series.index, series_values, 'k-', linewidth=2, label='Actual')
    axes[1, 1].plot(series.index, ucm_optimistic, 'r--', 
                    label=f'UCM best case\n(slope up to {max_slope:.0f})')
    axes[1, 1].set_title('Why UCM Cannot Fit Exponential Growth')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('model_structure_diagnosis.png')
    plt.show()
    
    # 6. Suggest solution
    print("\n5. SOLUTION:")
    print("   For exponentially growing series like household net worth:")
    print("   1. Work in log space: log(Y_t) = level_t + seasonal_t + ε_t")
    print("   2. Use multiplicative model: Y_t = Y_{t-1} * (1 + growth_t)")
    print("   3. Or allow time-varying observation variance proportional to level")
    print("\n   The current UCM assumes additive components which fundamentally")
    print("   cannot capture exponential growth, no matter how you tune the parameters!")

if __name__ == "__main__":
    diagnose_model_structure()
