# Stock-Flow Consistent (SFC) Kalman Filter

A production-ready implementation of a Stock-Flow Consistent Kalman Filter that enforces exact accounting identities while filtering Federal Reserve Z.1 economic data with bilateral constraints from From-Whom-to-Whom (FWTW) positions.

## Overview

This system implements a theoretically rigorous approach to filtering economic time series data while maintaining stock-flow consistency—a fundamental requirement in macroeconomic accounting where changes in stocks must equal flows plus revaluations. The implementation uses optimal projection methods to enforce constraints exactly while minimizing information loss from the statistical filtering process.

## Key Features

### Exact Constraint Enforcement
- **Stock-flow identities**: Enforced exactly (within machine precision) so that Stock[t] - Stock[t-1] = Flow[t]
- **Market clearing**: Total assets equal total liabilities for each financial instrument
- **Bilateral consistency**: FWTW bilateral positions sum to Z.1 aggregate totals
- **Optimal projection**: Minimizes adjustment to unconstrained estimates while satisfying all constraints

### Comprehensive Data Integration
- **Z.1 Federal Reserve data**: Stocks (FA/FL series) and flows (FU/FR series)
- **FWTW bilateral positions**: Who-owes-what-to-whom relationships
- **Bilateral flows**: Automatically calculated from FWTW stock changes
- **Formula relationships**: Optional support for Z.1 accounting formulas

### Production-Ready Design
- **Memory efficient**: Smart series selection and limiting for large datasets
- **Robust error handling**: Gracefully handles missing data and constraint conflicts
- **Comprehensive logging**: Detailed diagnostics and validation metrics
- **Flexible configuration**: Extensive YAML-based configuration system

## System Architecture

### 1. SFC Projection Module (`sfc_projection.py`)
The mathematical core that enforces accounting constraints through optimal projection:
- Automatically identifies all stock-flow pairs in the data
- Builds sparse constraint matrices for efficient computation
- Projects unconstrained Kalman estimates onto the constraint manifold
- Supports both hard (exact) and soft (approximate) constraints
- Uses closed-form solution: x* = x̂ - P·A'(A·P·A')⁻¹(A·x̂ - b)

### 2. SFC Kalman Filter (`sfc_kalman_filter.py`)
Extended state-space model that integrates stocks, flows, and trends:
- State vector includes: [Stock_Level, Stock_Trend, Flow_Level]
- State transition enforces: Stock[t] = Stock[t-1] + Flow[t]
- Applies constraint projection at each filtering step
- Validates constraint satisfaction throughout
- Returns filtered series with exact SFC consistency

### 3. Unified SFC Kalman (`unified_sfc_kalman.py`)
Orchestration layer that manages the complete filtering pipeline:
- Processes and maps FWTW bilateral positions to Z.1 series codes
- Calculates implied bilateral flows from FWTW stock changes
- Integrates all constraint types into the projection system
- Performs comprehensive validation of results
- Generates detailed diagnostics and metrics

### 4. Driver Script (`run_complete_sfc.py`)
User-facing interface with intelligent defaults:
- Smart series selection prioritizing complete stock-flow pairs
- Automatic memory management through configurable series limits
- Multiple run modes (test, development, production, large)
- Comprehensive output including filtered, smoothed, and bilateral flows

## Installation

### Prerequisites
```bash
# Required Python packages
pip install numpy scipy pandas statsmodels pyyaml
```

### File Structure
```
Z1/
├── src/
│   └── models/
│       ├── sfc_projection.py       # NEW: Constraint projection
│       ├── sfc_kalman_filter.py    # NEW: Extended Kalman filter
│       └── unified_sfc_kalman.py   # REPLACE: Updated orchestrator
├── examples/
│   └── run_complete_sfc.py         # REPLACE: Updated driver
├── config/
│   └── sfc_config.yaml            # REPLACE: Configuration
└── output/                         # Generated results
```

### Quick Start

1. **Test mode** (50 series, ~1 minute):
```bash
python -m examples.run_complete_sfc test
```

2. **Development mode** (200 series, soft constraints, ~5 minutes):
```bash
python -m examples.run_complete_sfc development
```

3. **Production mode** (500 series, exact constraints, ~15 minutes):
```bash
python -m examples.run_complete_sfc production
```

## Configuration

The system is controlled through `config/sfc_config.yaml`:

### Core Settings

```yaml
sfc:
  enforcement:
    enforce_sfc: true          # Master switch for SFC constraints
    stock_flow_consistency:
      enforce: true
      method: 'exact'          # 'exact' or 'soft'
      weight: 1.0             # Weight for soft constraints
    bilateral_consistency:
      enforce: true
      method: 'soft'           # FWTW has measurement error
      weight: 0.3
    market_clearing:
      enforce: true
      method: 'soft'           # Allow small discrepancies
      weight: 0.1
```

### Performance Tuning

```yaml
performance:
  max_series: 500              # Memory limit (adjust based on RAM)
  prioritize_pairs: true       # Select complete stock-flow pairs first
  priority_sectors:            # Important sectors to include
    - '10'                     # Nonfinancial corporate
    - '15'                     # Households
    - '26'                     # Rest of world
  memory:
    use_sparse_matrices: true  # Essential for large systems
```

### Validation Tolerances

```yaml
validation:
  stock_flow:
    relative_tolerance: 1e-6   # Very tight for accounting identity
    absolute_tolerance: 1.0    # $1M USD
  bilateral:
    relative_tolerance: 0.05   # 5% for FWTW measurement error
    absolute_tolerance: 100    # $100M USD
  market_clearing:
    relative_tolerance: 0.02   # 2% acceptable
    absolute_tolerance: 1000   # $1B USD
```

## Expected Output

### Console Output
```
2025-08-07 12:00:00 - INFO - STOCK-FLOW CONSISTENT KALMAN FILTER
2025-08-07 12:00:01 - INFO - Mode: production
2025-08-07 12:00:02 - INFO - Loading Z.1 data...
2025-08-07 12:00:05 - INFO - Found 19,804 quarterly series
2025-08-07 12:00:06 - INFO - Selected 250 stock-flow pairs
2025-08-07 12:00:07 - INFO - Loading FWTW data...
2025-08-07 12:00:10 - INFO - Calculated 50,000 bilateral flows
2025-08-07 12:00:11 - INFO - Running SFC Kalman filter...
2025-08-07 12:05:00 - INFO - Applying SFC constraint projection...
2025-08-07 12:10:00 - INFO - SFC Validation Results:
2025-08-07 12:10:00 - INFO -   stock_flow_max_violation: 0.0000
2025-08-07 12:10:00 - INFO -   market_clearing_mean: 0.0001
2025-08-07 12:10:00 - INFO -   n_constraints_satisfied: 498/500
2025-08-07 12:10:01 - INFO - SFC Kalman filter completed successfully!
```

### Output Files
```
output/production/
├── sfc_filtered.csv           # Filtered series with constraints
├── sfc_smoothed.csv           # Smoothed series (full sample)
├── bilateral_flows.csv        # Calculated FWTW flows
└── constraint_diagnostics.json # Validation metrics
```

## Troubleshooting

### Memory Issues
**Problem**: Process killed or out of memory errors

**Solutions**:
- Reduce `max_series` in config (try 100-200)
- Enable sparse matrices: `use_sparse_matrices: true`
- Use test mode first: `python -m examples.run_complete_sfc test`

### Constraint Violations
**Problem**: Stock-flow constraints not satisfied

**Solutions**:
- Increase `stock_flow_weight` (try 0.7-0.9)
- Switch to exact method: `method: 'exact'`
- Check data quality for series with violations

### Missing Data
**Problem**: No FWTW data or formulas file

**Solutions**:
```yaml
# In config/sfc_config.yaml:
formulas:
  validation_required: false
  allow_missing: true
enforcement:
  formulas:
    enforce: false
  fwtw:
    require_overlap: false
```

### Slow Performance
**Problem**: Filter takes too long to run

**Solutions**:
- Use development mode with soft constraints
- Reduce `max_series` to 100-200
- Set `diagnostic_level: 'summary'`
- Disable intermediate outputs

## Mathematical Foundation

### Constraint Projection
The system solves the constrained optimization problem:

```
min ||x - x̂||²_P⁻¹
s.t. Ax = b
```

Where:
- x̂ = Unconstrained Kalman estimate
- P = State covariance matrix
- A = Constraint matrix
- b = Constraint values

Solution: x* = x̂ - P·A'(APA')⁻¹(Ax̂ - b)

### State Space Model
```
State Evolution:
Stock[t] = Stock[t-1] + Flow[t] + η
Flow[t] = ρ·Flow[t-1] + Trend[t] + ε
Trend[t] = Trend[t-1] + ω

Observation:
Y[t] = Z·State[t] + v
```

## Advanced Usage

### Custom Constraints
Add custom constraints in `sfc_projection.py`:

```python
def add_custom_constraint(self):
    constraint = SFCConstraint(
        name='custom_balance',
        type='custom',
        indices=[idx1, idx2, idx3],
        coefficients=[1.0, -1.0, -1.0],
        rhs=0.0,
        is_hard=True,
        weight=1.0
    )
    self.constraints.append(constraint)
```

### Parallel Processing
For very large datasets, enable parallel processing:

```yaml
performance:
  parallel:
    enable: true
    n_jobs: -1  # Use all CPU cores
```

## Citation

If you use this software in your research, please cite:

```
Stock-Flow Consistent Kalman Filter
Federal Reserve Z.1 Data Analysis System
https://github.com/[your-repo]/sfc-kalman-filter
```

## License

[Specify your license here]

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: [your contact information]

## Acknowledgments

- Federal Reserve Board for Z.1 and FWTW data
- Contributors to the statsmodels library
- [Other acknowledgments]

---

*This system implements true Stock-Flow Consistency with bilateral constraints from FWTW data, providing the most rigorous approach to filtering macroeconomic time series while preserving fundamental accounting identities.*