# notebooks/05_fwtw_network_analysis.ipynb
"""
FWTW Network Analysis Integration Example
This notebook demonstrates how to use the FWTW network analysis
capabilities integrated with the Z1 time series analysis.
"""

#%% [markdown]
# # Flow of Funds Through Wall Street (FWTW) Network Analysis
# 
# This notebook demonstrates the integration of FWTW network analysis with the existing Z1 time series analysis framework.

#%% Import required libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.data import CachedFedDataLoader
from src.network import FWTWDataLoader, NetworkBuilder, NetworkAnalyzer
from src.analysis import EconomicAnalysis
from src.visualization import EconomicVisualizer

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

#%% [markdown]
# ## 1. Load FWTW Data

#%% Load FWTW data with caching
print("Loading FWTW data...")
fwtw_loader = FWTWDataLoader(cache_dir="./data/cache/fwtw")

# Load data (will use cache if available)
fwtw_data = fwtw_loader.load_fwtw_data(force_download=False)

print(f"Loaded {len(fwtw_data)} records")
print(f"Date range: {fwtw_data['Date'].min()} to {fwtw_data['Date'].max()}")
print(f"Unique holders: {fwtw_data['Holder Name'].nunique()}")
print(f"Unique issuers: {fwtw_data['Issuer Name'].nunique()}")
print(f"Unique instruments: {fwtw_data['Instrument Name'].nunique()}")

#%% Display sample data
print("\nSample FWTW data:")
fwtw_data.head(10)

#%% [markdown]
# ## 2. Build Financial Networks

#%% Build network for latest date
# Get latest date
latest_date = fwtw_data['Date'].max()
print(f"Building network for {latest_date}")

# Initialize network builder
builder = NetworkBuilder(fwtw_data)

# Build network snapshot
network = builder.build_snapshot(latest_date, min_flow=100)

print(f"Network statistics:")
print(f"  Nodes: {network.number_of_nodes()}")
print(f"  Edges: {network.number_of_edges()}")
print(f"  Density: {nx.density(network):.4f}")
print(f"  Total volume: ${network.graph['total_volume']:,.0f}")

#%% Visualize network
plt.figure(figsize=(15, 10))

# Use spring layout for positioning
pos = nx.spring_layout(network, k=2, iterations=50, seed=42)

# Node sizes based on total flow
node_sizes = [network.nodes[node]['total_degree'] / 1000 for node in network.nodes()]

# Node colors based on net position
node_colors = [network.nodes[node]['net_flow'] for node in network.nodes()]

# Draw network
nx.draw_networkx_nodes(network, pos, 
                      node_size=node_sizes,
                      node_color=node_colors,
                      cmap='RdBu_r',
                      alpha=0.8)

# Draw edges with varying widths
edges = network.edges()
weights = [network[u][v]['weight'] / 10000 for u, v in edges]
nx.draw_networkx_edges(network, pos,
                      width=weights,
                      alpha=0.3,
                      edge_color='gray',
                      arrows=True,
                      arrowsize=10)

# Add labels for major nodes
major_nodes = [node for node, data in network.nodes(data=True) 
               if data['total_degree'] > np.percentile([d['total_degree'] 
                   for n, d in network.nodes(data=True)], 90)]
labels = {node: node.split(';')[0][:20] + '...' if len(node) > 20 else node 
          for node in major_nodes}
nx.draw_networkx_labels(network, pos, labels, font_size=8)

plt.title(f"Financial Network Structure - {latest_date.strftime('%Y Q%m')}",
          fontsize=16, pad=20)
plt.axis('off')
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 3. Network Analysis

#%% Initialize analyzer and compute metrics
analyzer = NetworkAnalyzer(network)

# Compute centrality metrics
print("Computing centrality metrics...")
centrality_metrics = analyzer.compute_centrality_metrics()

# Create centrality dataframe
centrality_df = pd.DataFrame(centrality_metrics)
centrality_df['entity'] = centrality_df.index

# Display top entities by different metrics
print("\nTop 10 entities by PageRank:")
print(centrality_df.nlargest(10, 'pagerank')[['entity', 'pagerank']])

print("\nTop 10 entities by betweenness centrality:")
print(centrality_df.nlargest(10, 'betweenness')[['entity', 'betweenness']])

#%% Identify systemically important financial institutions
sifis = analyzer.identify_systemically_important(method='composite', threshold=0.9)

print(f"\nSystemically Important Financial Institutions (top {len(sifis)}):")
for entity, score in sifis[:10]:
    print(f"  {entity[:50]}... : {score:.4f}")

#%% Calculate network risk metrics
risk_metrics = analyzer.calculate_network_risk_metrics()

print("\nNetwork Risk Metrics:")
for metric, value in risk_metrics.items():
    print(f"  {metric}: {value:.4f}")

#%% [markdown]
# ## 4. Shock Propagation Analysis

#%% Analyze contagion risk
# Select a major institution for shock analysis
shock_node = sifis[0][0]  # Most systemically important institution
print(f"Analyzing shock propagation from: {shock_node[:50]}...")

contagion_results = analyzer.analyze_contagion_risk(
    shock_node=shock_node,
    shock_magnitude=0.15,  # 15% shock
    propagation_threshold=0.01
)

print(f"\nContagion Analysis Results:")
print(f"  Initial shock: {contagion_results['shock_magnitude']:.1%}")
print(f"  Affected nodes: {contagion_results['total_affected_nodes']}")
print(f"  Affected fraction: {contagion_results['affected_fraction']:.1%}")
print(f"  Amplification factor: {contagion_results['amplification_factor']:.2f}")
print(f"  Rounds to convergence: {contagion_results['rounds_to_convergence']}")

# Plot shock propagation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Propagation over rounds
propagation_history = contagion_results['propagation_history']
rounds = range(len(propagation_history))
total_shock = [sum(shocks.values()) for shocks in propagation_history]

ax1.plot(rounds, total_shock, 'o-', linewidth=2, markersize=8)
ax1.set_xlabel('Propagation Round')
ax1.set_ylabel('Total System Shock')
ax1.set_title('Shock Propagation Over Time')
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution of final shocks
final_shocks = propagation_history[-1]
affected_entities = [(entity, shock) for entity, shock in final_shocks.items() 
                    if shock > 0.001]
affected_entities.sort(key=lambda x: x[1], reverse=True)

entities = [e[0][:30] + '...' if len(e[0]) > 30 else e[0] 
           for e in affected_entities[:15]]
shocks = [e[1] for e in affected_entities[:15]]

ax2.barh(entities, shocks)
ax2.set_xlabel('Shock Magnitude')
ax2.set_title('Top 15 Affected Entities')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

#%% [markdown]
# ## 5. Network Evolution Analysis

#%% Build time series of networks
# Get quarterly dates for last 2 years
all_dates = sorted(fwtw_data['Date'].unique())
recent_dates = all_dates[-8:]  # Last 8 quarters

print("Building network time series...")
network_series = builder.build_time_series(recent_dates)

# Calculate network metrics over time
metrics_evolution = []
for date, net in network_series.items():
    analyzer_temp = NetworkAnalyzer(net)
    metrics = analyzer_temp.calculate_network_risk_metrics()
    metrics['date'] = date
    metrics['num_nodes'] = net.number_of_nodes()
    metrics['num_edges'] = net.number_of_edges()
    metrics['total_volume'] = net.graph.get('total_volume', 0)
    metrics_evolution.append(metrics)

evolution_df = pd.DataFrame(metrics_evolution)

# Plot network evolution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

# Plot different metrics
metrics_to_plot = ['density', 'herfindahl_index', 'global_efficiency', 'num_edges']
titles = ['Network Density', 'Concentration (HHI)', 'Global Efficiency', 'Number of Connections']

for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[i]
    ax.plot(evolution_df['date'], evolution_df[metric], 'o-', linewidth=2, markersize=8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Financial Network Evolution Over Time', fontsize=16)
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 6. Integration with Z1 Data

#%% Load Z1 data for comparison
print("Loading Z1 data for integration...")
fed_loader = CachedFedDataLoader()
z1_data = fed_loader.load_single_source('Z1')

# Filter for relevant flow series
flow_series = z1_data[z1_data['SERIES_NAME'].str.contains('FA', na=False)]
print(f"Found {len(flow_series)} flow series in Z1 data")

#%% Compare FWTW flows with Z1 aggregates
# Aggregate FWTW flows by date
fwtw_aggregates = fwtw_data.groupby('Date')['Level'].agg(['sum', 'mean', 'count'])
fwtw_aggregates.columns = ['total_flow', 'avg_flow', 'num_transactions']

# Plot comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: Total flows
ax1.plot(fwtw_aggregates.index, fwtw_aggregates['total_flow'] / 1e6, 
         'o-', label='FWTW Total Flows', linewidth=2)
ax1.set_ylabel('Total Flow (Trillions $)')
ax1.set_title('FWTW Network Flows Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Network complexity
ax2.plot(evolution_df['date'], evolution_df['num_edges'], 
         'o-', color='green', label='Network Connections', linewidth=2)
ax2_twin = ax2.twinx()
ax2_twin.plot(evolution_df['date'], evolution_df['density'], 
              'o-', color='orange', label='Network Density', linewidth=2)

ax2.set_xlabel('Date')
ax2.set_ylabel('Number of Connections', color='green')
ax2_twin.set_ylabel('Network Density', color='orange')
ax2.set_title('Network Complexity Metrics')
ax2.grid(True, alpha=0.3)

# Combine legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()

#%% [markdown]
# ## 7. Sector-Level Analysis

#%% Analyze flows by sector type
# Classify nodes by type
sector_flows = fwtw_data.copy()
sector_flows['holder_type'] = sector_flows['Holder Name'].apply(
    lambda x: builder._classify_node(x)
)
sector_flows['issuer_type'] = sector_flows['Issuer Name'].apply(
    lambda x: builder._classify_node(x)
)

# Aggregate by sector pairs
sector_matrix = sector_flows.groupby(
    ['Date', 'holder_type', 'issuer_type']
)['Level'].sum().reset_index()

# Create sector flow matrix for latest date
latest_sector_matrix = sector_matrix[sector_matrix['Date'] == latest_date].pivot(
    index='holder_type',
    columns='issuer_type',
    values='Level'
).fillna(0)

# Plot sector flow heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(latest_sector_matrix / 1e9,  # Convert to billions
            annot=True, 
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Flow Amount (Billions $)'})
plt.title(f'Inter-Sector Financial Flows - {latest_date.strftime("%Y Q%m")}', 
          fontsize=14, pad=20)
plt.xlabel('Issuer Sector')
plt.ylabel('Holder Sector')
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 8. Key Findings Summary

#%% Generate summary statistics
print("=" * 60)
print("FWTW NETWORK ANALYSIS SUMMARY")
print("=" * 60)

print(f"\nData Coverage:")
print(f"  Period: {fwtw_data['Date'].min()} to {fwtw_data['Date'].max()}")
print(f"  Total transactions: {len(fwtw_data):,}")
print(f"  Unique entities: {fwtw_data['Holder Name'].nunique() + fwtw_data['Issuer Name'].nunique()}")

print(f"\nNetwork Structure (Latest Quarter):")
print(f"  Network density: {risk_metrics['density']:.4f}")
print(f"  Concentration (HHI): {risk_metrics['herfindahl_index']:.4f}")
print(f"  Global efficiency: {risk_metrics['global_efficiency']:.4f}")
print(f"  Average clustering: {risk_metrics['avg_clustering']:.4f}")

print(f"\nSystemic Risk Indicators:")
print(f"  Number of SIFIs: {len(sifis)}")
print(f"  Network assortativity: {risk_metrics['assortativity']:.4f}")
print(f"  Rich club coefficient: {risk_metrics['rich_club_coeff']:.4f}")

print(f"\nContagion Analysis:")
print(f"  15% shock amplification: {contagion_results['amplification_factor']:.2f}x")
print(f"  System-wide impact: {contagion_results['affected_fraction']:.1%} of nodes affected")

print("\nTop 5 Systemically Important Institutions:")
for i, (entity, score) in enumerate(sifis[:5], 1):
    print(f"  {i}. {entity[:60]}...")

#%% Save key results
# Create results directory
import os
os.makedirs('./output/fwtw_analysis', exist_ok=True)

# Save centrality metrics
centrality_df.to_csv('./output/fwtw_analysis/centrality_metrics.csv', index=False)

# Save network evolution
evolution_df.to_csv('./output/fwtw_analysis/network_evolution.csv', index=False)

# Save sector flow matrix
latest_sector_matrix.to_csv('./output/fwtw_analysis/sector_flows_latest.csv')

print("\nResults saved to ./output/fwtw_analysis/")

#%% [markdown]
# ## Next Steps
# 
# 1. **Integrate with UC Models**: Use network metrics as features in time series decomposition
# 2. **Stress Testing**: Run comprehensive stress tests on different shock scenarios
# 3. **Real-time Monitoring**: Set up automated analysis of new FWTW data releases
# 4. **Cross-validation**: Validate network-implied flows against Z1 aggregates
# 5. **Predictive Modeling**: Use network topology to predict future flow patterns