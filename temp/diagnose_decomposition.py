import json
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan
import hashlib
import logging
from statsmodels.tsa.filters.hp_filter import hpfilter
import matplotlib.pyplot as plt
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

print("="*80)
print("COMPLETE TRACE: LOGIC → STRUCTURE → REALIZATION")
print("="*80)

# Load all data
with open('fof_formulas_extracted.json', 'r') as f:
    all_data = json.load(f)

original_data = pd.read_parquet('/home/tesla/Z1/temp/data/z1_quarterly/z1_quarterly_data_filtered.parquet')

# Target series
series_name = 'FL343073045'
print(f"\nTarget series: {series_name}")

# ============================================================================
# PART 1: LOGIC - What should happen to this series?
# ============================================================================
print("\n" + "="*80)
print("PART 1: LOGIC - Formulas and Identities")
print("="*80)

# 1A. Formula information
formula_info = all_data.get('formulas', {}).get(series_name, {})
if formula_info:
    print(f"\n1A. Series Formula:")
    print(f"  Formula: {formula_info.get('formula', 'None')}")
    print(f"  Type: {formula_info.get('data_type', 'Source')}")
    if formula_info.get('derived_from'):
        print(f"  Computed from:")
        for comp in formula_info['derived_from']:
            print(f"    {comp['operator']} {comp['code']}")
else:
    print("\n1A. No formula - this is source data")

# 1B. Identities involving this series
print(f"\n1B. Identities:")
identities_with_series = []
for identity in all_data.get('identities', []):
    all_identity_series = set(identity['left_side'] + identity['right_side'])
    all_identity_series = {s.lstrip('Δ') for s in all_identity_series}
    
    if series_name in all_identity_series:
        identities_with_series.append(identity)
        print(f"\n  Identity: {identity['identity_name']}")
        print(f"    Type: {identity['identity_type']}")
        print(f"    Equation: {identity['left_side']} = {identity['right_side']}")

# ============================================================================
# PART 2: STRUCTURE - How is it grouped and constrained?
# ============================================================================
print("\n" + "="*80)
print("PART 2: STRUCTURE - Grouping and Constraints")
print("="*80)

# 2A. Find constraint group (from your output)
group_name = 'Formula_FL593073045'
group_series = ['FL343073045', 'FL573073005', 'FL223073045']

print(f"\n2A. Constraint Group: {group_name}")
print(f"  Series in group: {group_series}")

# 2B. Why these series are grouped together
if 'FL593073045' in all_data.get('formulas', {}):
    parent_formula = all_data['formulas']['FL593073045']
    print(f"\n2B. Grouped because of formula for FL593073045:")
    print(f"  Formula: {parent_formula.get('formula', '')}")
    print(f"  These series are its components")

# 2C. Constraints that will be applied
applicable_constraints = []
for identity in all_data.get('identities', []):
    identity_series = {s.lstrip('Δ') for s in identity['left_side'] + identity['right_side']}
    if identity_series.intersection(group_series):
        applicable_constraints.append(identity)

print(f"\n2C. Constraints applied to this group: {len(applicable_constraints)}")
for i, constraint in enumerate(applicable_constraints[:3]):
    print(f"  {i+1}. {constraint['identity_name']}")

# ============================================================================
# PART 3: REALIZATION - What actually happens in PyMC?
# ============================================================================
print("\n" + "="*80)
print("PART 3: REALIZATION - PyMC Model Building")
print("="*80)

# 3A. Model setup
T = len(original_data)
model_hash = hashlib.md5(f"{group_name}_{series_name}".encode()).hexdigest()[:8]
vname = f"{series_name}_{model_hash}"

print(f"\n3A. Model Setup:")
print(f"  Variable prefix: {vname}")
print(f"  Time periods: {T}")

# 3B. Data preparation
series_data = original_data[series_name]
series_values = series_data.dropna().values

print(f"\n3B. Data Stats:")
print(f"  Mean: {series_values.mean():.2f}")
print(f"  Std: {series_values.std():.2f}")
print(f"  Range: [{series_values.min():.2f}, {series_values.max():.2f}]")

# 3C. Variance estimation (as in create_components_for_series)
cycle1, trend1 = hpfilter(series_values, lamb=1600)
var_resid = np.var(cycle1, ddof=1)
_, trend2 = hpfilter(trend1, lamb=1600)
trend_var_estimate = np.var(np.diff(trend2), ddof=1)

print(f"\n3C. Variance Estimates:")
print(f"  Residual variance: {var_resid:.2f}")
print(f"  Trend variance: {trend_var_estimate:.6f}")

# 3D. Build simplified model to show what happens
print(f"\n3D. Building PyMC Model...")

with pm.Model() as model:
    # Initial values
    init_level = float(series_values[0])
    init_trend = float(np.mean(np.diff(series_values[:5])))
    
    print(f"  Initial level: {init_level:.2f}")
    print(f"  Initial trend: {init_trend:.2f}")
    
    # Create components
    level_init = pm.Normal(f'{vname}_level_init', mu=init_level, sigma=float(np.sqrt(var_resid)))
    trend_init = pm.Normal(f'{vname}_trend_init', mu=init_trend, sigma=float(np.sqrt(trend_var_estimate)))
    sigma_trend = pm.HalfNormal(f'{vname}_sigma_trend', sigma=float(np.sqrt(trend_var_estimate)))
    trend_innovations = pm.Normal(f'{vname}_trend_innovations', mu=0, sigma=sigma_trend, shape=T-1)
    
    # Trend = initial + cumsum(innovations)
    trend = pt.concatenate([pt.atleast_1d(trend_init), trend_init + pt.cumsum(trend_innovations)])
    trend_det = pm.Deterministic(f'{vname}_trend', trend)
    
    # Level = integrated trend
    def level_step(beta_prev, mu_prev): 
        return mu_prev + beta_prev
    
    level_values, _ = scan(fn=level_step, sequences=[trend[:-1]], outputs_info=[level_init])
    level = pm.Deterministic(f'{vname}_level', pt.concatenate([pt.atleast_1d(level_init), level_values]))
    
    print(f"\n3E. Model Components Created:")
    print(f"  Trend: cumsum of {T-1} innovations")
    print(f"  Level: integrated trend (scan operation)")

# ============================================================================
# PART 4: CONSTRAINT REALIZATION
# ============================================================================
print("\n" + "="*80)
print("PART 4: CONSTRAINT REALIZATION")
print("="*80)

# Show how constraints would be applied
if applicable_constraints:
    constraint = applicable_constraints[0]  # Show first constraint
    print(f"\n4A. Example Constraint: {constraint['identity_name']}")
    print(f"  Left side: {constraint['left_side']}")
    print(f"  Right side: {constraint['right_side']}")
    
    print(f"\n4B. In PyMC, this creates a penalty term:")
    print(f"  identity_violation = left_sum - right_sum")
    print(f"  pm.Potential('penalty', StudentT.logp(violation))")
    print(f"  This forces the decomposed components to satisfy the identity")

# ============================================================================
# PART 5: DIAGNOSIS - Why does it fail?
# ============================================================================
print("\n" + "="*80)
print("PART 5: DIAGNOSIS - Why Decomposition Fails")
print("="*80)

print("\n5A. The Problem Chain:")
print("  1. Multiple series share constraints (3 series, multiple identities)")
print("  2. Each series has its own level/trend/cycle/seasonal components")
print("  3. Constraints force these components to satisfy accounting identities")
print("  4. ADVI tries to find parameters that satisfy all constraints")
print("  5. The optimization is complex with many local minima")
print("  6. ADVI converges poorly, trend innovations → 0 on average")
print("  7. Result: constant trend, but varying level (mathematical inconsistency)")

print("\n5B. Evidence from your output:")
print("  - Trend is constant: 47.21")
print("  - But level varies: std=201543.40")
print("  - This is impossible if level = ∫trend")
print("  - Shows ADVI failed to find consistent solution")

# ============================================================================
# PART 6: QUICK VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("PART 6: VERIFICATION - Load Actual Results")
print("="*80)

level_df = pd.read_csv('decomposition_level.csv', index_col=0)
trend_df = pd.read_csv('decomposition_trend.csv', index_col=0)

if series_name in level_df.columns:
    saved_level = level_df[series_name].values
    saved_trend = trend_df[series_name].values
    
    print(f"\n6A. Saved Results:")
    print(f"  Level: mean={saved_level.mean():.2f}, std={saved_level.std():.2f}")
    print(f"  Trend: mean={saved_trend.mean():.2f}, std={saved_trend.std():.6f}")
    print(f"  Trend unique values: {len(np.unique(saved_trend))}")
    
    # Check mathematical consistency
    level_diff = np.diff(saved_level)
    expected_trend = saved_trend[:-1]
    
    print(f"\n6B. Mathematical Consistency Check:")
    print(f"  diff(level) should equal trend")
    print(f"  Actual correlation: {np.corrcoef(level_diff, expected_trend)[0,1]:.6f}")
    print(f"  Mean absolute error: {np.abs(level_diff - expected_trend).mean():.2f}")
    
    if np.abs(level_diff - expected_trend).mean() > 1.0:
        print(f"  ❌ FAILED: Level is NOT integrated trend!")

# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("PART 7: GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(4, 2, figsize=(15, 12))
fig.suptitle(f'Complete Analysis: {series_name}', fontsize=16)

# 7A. Original series
ax = axes[0, 0]
series_data.plot(ax=ax, color='black', linewidth=2)
ax.set_title('Original Series')
ax.set_ylabel('Value')
ax.grid(True, alpha=0.3)

# 7B. Decomposition results
ax = axes[0, 1]
if series_name in level_df.columns:
    saved_level[:100].plot(ax=ax, label='Level', color='blue')
    ax2 = ax.twinx()
    saved_trend[:100].plot(ax=ax2, label='Trend', color='red', alpha=0.7)
    ax.set_title('Decomposition: Level & Trend (first 100 obs)')
    ax.set_ylabel('Level', color='blue')
    ax2.set_ylabel('Trend', color='red')
    ax.grid(True, alpha=0.3)

# 7C. HP filter decomposition
ax = axes[1, 0]
ax.plot(trend1[:100], label='HP Trend', color='green')
ax.plot(cycle1[:100], label='HP Cycle', color='orange', alpha=0.7)
ax.set_title('HP Filter Decomposition (first 100 obs)')
ax.legend()
ax.grid(True, alpha=0.3)

# 7D. Level differences vs Trend
ax = axes[1, 1]
if series_name in level_df.columns:
    level_diff = np.diff(saved_level)
    ax.scatter(level_diff[:100], saved_trend[:100], alpha=0.5, s=10)
    ax.plot([level_diff.min(), level_diff.max()], [level_diff.min(), level_diff.max()], 'r--', label='y=x')
    ax.set_xlabel('diff(Level)')
    ax.set_ylabel('Trend')
    ax.set_title('Consistency Check: diff(Level) vs Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 7E. Constraint network
ax = axes[2, 0]
import networkx as nx
G = nx.Graph()
G.add_node(series_name, color='red')
for i, s in enumerate(group_series):
    if s != series_name:
        G.add_node(s, color='lightblue')
        G.add_edge(series_name, s)

pos = nx.spring_layout(G)
colors = [G.nodes[node]['color'] for node in G.nodes()]
nx.draw(G, pos, ax=ax, node_color=colors, with_labels=True, node_size=500, font_size=8)
ax.set_title(f'Constraint Group: {group_name}')

# 7F. Identity relationships
ax = axes[2, 1]
if identities_with_series:
    identity_text = ""
    for i, identity in enumerate(identities_with_series[:5]):
        identity_text += f"{i+1}. {identity['identity_name']}\n"
        identity_text += f"   {' + '.join(identity['left_side'])} = {' + '.join(identity['right_side'])}\n\n"
    ax.text(0.05, 0.95, identity_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', fontfamily='monospace')
    ax.set_title('Key Identities')
    ax.axis('off')

# 7G. Variance analysis
ax = axes[3, 0]
components = ['Original', 'HP Trend', 'HP Cycle', 'Level', 'Trend']
variances = [
    series_values.std()**2,
    trend1.std()**2,
    cycle1.std()**2,
    saved_level.std()**2 if series_name in level_df.columns else 0,
    saved_trend.std()**2 if series_name in trend_df.columns else 0
]
bars = ax.bar(components, variances)
bars[0].set_color('black')
bars[1].set_color('green')
bars[2].set_color('orange')
bars[3].set_color('blue')
bars[4].set_color('red')
ax.set_ylabel('Variance')
ax.set_title('Variance Decomposition')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# 7H. Diagnostics summary
ax = axes[3, 1]
diagnostics_text = f"""
DIAGNOSTICS SUMMARY:
==================
Series: {series_name}
Group: {group_name}
Identities: {len(identities_with_series)}
Constraints: {len(applicable_constraints)}

Variance Analysis:
- Original: {series_values.std():.2f}
- Level: {saved_level.std():.2f}
- Trend: {saved_trend.std():.6f}

Mathematical Consistency:
- diff(Level) ≈ Trend? {np.abs(level_diff - saved_trend[:-1]).mean() < 1.0}
- Trend constant? {saved_trend.std() < 1e-6}
- Reconstruction error: {np.abs(series_data.values - (saved_level + saved_trend)).mean():.2f}
"""
ax.text(0.05, 0.95, diagnostics_text, transform=ax.transAxes, 
        fontsize=9, verticalalignment='top', fontfamily='monospace')
ax.set_title('Diagnostics Summary')
ax.axis('off')

plt.tight_layout()
plt.savefig(f'complete_analysis_{series_name}.png', dpi=150, bbox_inches='tight')
print(f"Saved visualization to complete_analysis_{series_name}.png")

# ============================================================================
# SAVE STRUCTURED REPORT
# ============================================================================
print("\n" + "="*80)
print("SAVING STRUCTURED REPORT")
print("="*80)

report = {
    'series': series_name,
    'group': group_name,
    'group_members': group_series,
    'formula': formula_info,
    'identities': [{'name': i['identity_name'], 
                   'type': i['identity_type'],
                   'equation': f"{i['left_side']} = {i['right_side']}"} 
                  for i in identities_with_series],
    'constraints_count': len(applicable_constraints),
    'data_stats': {
        'mean': float(series_values.mean()),
        'std': float(series_values.std()),
        'min': float(series_values.min()),
        'max': float(series_values.max())
    },
    'decomposition_stats': {
        'level_mean': float(saved_level.mean()) if series_name in level_df.columns else None,
        'level_std': float(saved_level.std()) if series_name in level_df.columns else None,
        'trend_mean': float(saved_trend.mean()) if series_name in trend_df.columns else None,
        'trend_std': float(saved_trend.std()) if series_name in trend_df.columns else None,
        'trend_unique_values': int(len(np.unique(saved_trend))) if series_name in trend_df.columns else None
    },
    'consistency_check': {
        'diff_level_equals_trend': bool(np.abs(level_diff - saved_trend[:-1]).mean() < 1.0) if series_name in level_df.columns else None,
        'trend_is_constant': bool(saved_trend.std() < 1e-6) if series_name in trend_df.columns else None,
        'mean_absolute_error': float(np.abs(level_diff - saved_trend[:-1]).mean()) if series_name in level_df.columns else None
    }
}

with open(f'analysis_report_{series_name}.json', 'w') as f:
    json.dump(report, f, indent=2)
print(f"Saved structured report to analysis_report_{series_name}.json")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Series {series_name}:
- Is part of {len(identities_with_series)} identities
- Processed with {len(group_series)} series in group '{group_name}'
- Has {len(applicable_constraints)} constraints applied
- PyMC model has {T-1} trend innovations to estimate
- ADVI optimization failed to converge properly
- Result: mathematically inconsistent decomposition

📊 Generated outputs:
- Console output (copy all text above)
- Visualization: complete_analysis_{series_name}.png
- Structured report: analysis_report_{series_name}.json
""")