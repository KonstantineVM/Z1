#!/usr/bin/env python3
"""
SFC Visualization Module
Complete implementation of visualization methods for Stock-Flow Consistent Kalman Filter results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SFCVisualizationManager:
    """
    Comprehensive visualization manager for SFC Kalman Filter results.
    """
    
    def __init__(self, output_dir: Path = None, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualization manager.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save plots
        style : str
            Matplotlib style to use
        """
        self.output_dir = Path(output_dir) if output_dir else Path('output/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-darkgrid')
        
        # Color palette
        self.colors = sns.color_palette("husl", 10)
        self.violation_cmap = plt.cm.RdYlGn_r  # Red for violations, green for compliance
        
    def create_all_visualizations(self, 
                                  z1_data: pd.DataFrame,
                                  filtered_states: pd.DataFrame,
                                  smoothed_states: pd.DataFrame,
                                  model_diagnostics: Dict,
                                  constraint_violations: Optional[pd.DataFrame] = None,
                                  fwtw_data: Optional[pd.DataFrame] = None,
                                  stock_flow_pairs: Optional[List] = None):
        """
        Create all visualizations for the SFC analysis.
        
        Parameters
        ----------
        z1_data : pd.DataFrame
            Original Z1 data
        filtered_states : pd.DataFrame
            Filtered state estimates
        smoothed_states : pd.DataFrame
            Smoothed state estimates
        model_diagnostics : Dict
            Model diagnostic information
        constraint_violations : pd.DataFrame
            Time series of constraint violations
        fwtw_data : pd.DataFrame
            FWTW bilateral position data
        stock_flow_pairs : List
            List of stock-flow pairs
        """
        print("\nCreating SFC visualizations...")
        
        # 1. Filtered vs Actual comparison
        self.plot_filtered_vs_actual(z1_data, filtered_states, smoothed_states)
        
        # 2. Constraint violations over time
        if constraint_violations is not None:
            self.plot_constraint_violations(constraint_violations)
        
        # 3. Stock-flow consistency check
        if stock_flow_pairs:
            self.plot_stock_flow_consistency(z1_data, smoothed_states, stock_flow_pairs)
        
        # 4. Shock decomposition
        self.plot_shock_decomposition(filtered_states, smoothed_states)
        
        # 5. FWTW network structure
        if fwtw_data is not None:
            self.plot_fwtw_network(fwtw_data)
        
        # 6. Model diagnostics summary
        self.plot_model_diagnostics(model_diagnostics)
        
        # 7. Series composition
        self.plot_series_composition(z1_data)
        
        # 8. Filtering improvement metrics
        self.plot_filtering_improvement(z1_data, filtered_states, smoothed_states)
        
        print(f"  ✓ Saved all visualizations to {self.output_dir}")
    
    def plot_filtered_vs_actual(self, 
                                actual: pd.DataFrame, 
                                filtered: pd.DataFrame, 
                                smoothed: pd.DataFrame,
                                n_series: int = 6):
        """
        Plot filtered and smoothed estimates vs actual data for selected series.
        """
        fig, axes = plt.subplots(n_series, 1, figsize=(14, 3*n_series))
        if n_series == 1:
            axes = [axes]
        
        # Select series to plot (prioritize those with most data)
        data_coverage = actual.notna().sum().sort_values(ascending=False)
        series_to_plot = data_coverage.head(n_series).index
        
        for idx, (ax, series) in enumerate(zip(axes, series_to_plot)):
            if series in actual.columns and series in filtered.columns:
                # Plot actual data
                ax.plot(actual.index, actual[series], 
                       'o', alpha=0.3, color='gray', markersize=2, label='Actual')
                
                # Plot filtered estimate
                if series in filtered.columns:
                    ax.plot(filtered.index, filtered[series], 
                           '-', color=self.colors[0], linewidth=1.5, label='Filtered')
                
                # Plot smoothed estimate
                if series in smoothed.columns:
                    ax.plot(smoothed.index, smoothed[series], 
                           '-', color=self.colors[1], linewidth=1.5, label='Smoothed')
                
                ax.set_title(f'Series: {series}', fontsize=10)
                ax.set_xlabel('')
                ax.grid(True, alpha=0.3)
                
                if idx == 0:
                    ax.legend(loc='upper right', fontsize=9)
                
                # Add shaded regions for missing data
                is_missing = actual[series].isna()
                if is_missing.any():
                    missing_regions = self._get_missing_regions(is_missing)
                    for start, end in missing_regions:
                        ax.axvspan(actual.index[start], actual.index[min(end, len(actual)-1)], 
                                  alpha=0.1, color='red')
        
        plt.suptitle('Filtered vs Actual Series Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'filtered_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_constraint_violations(self, violations: pd.DataFrame):
        """
        Plot constraint violations over time.
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Stock-flow violations
        ax = axes[0]
        if 'stock_flow_violation' in violations.columns:
            ax.plot(violations.index, violations['stock_flow_violation'], 
                   color=self.colors[2], linewidth=1.5)
            ax.fill_between(violations.index, 0, violations['stock_flow_violation'], 
                           alpha=0.3, color=self.colors[2])
            ax.axhline(y=1e-6, color='red', linestyle='--', alpha=0.5, label='Tolerance')
            ax.set_yscale('log')
            ax.set_ylabel('Violation (log scale)')
            ax.set_title('Stock-Flow Consistency Violations')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 2. Market clearing violations
        ax = axes[1]
        if 'market_clearing_violation' in violations.columns:
            ax.plot(violations.index, violations['market_clearing_violation'],
                   color=self.colors[3], linewidth=1.5)
            ax.fill_between(violations.index, 0, violations['market_clearing_violation'],
                           alpha=0.3, color=self.colors[3])
            ax.axhline(y=1e-3, color='red', linestyle='--', alpha=0.5, label='Tolerance')
            ax.set_yscale('log')
            ax.set_ylabel('Violation (log scale)')
            ax.set_title('Market Clearing Violations')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 3. Bilateral consistency violations
        ax = axes[2]
        if 'bilateral_violation' in violations.columns:
            ax.plot(violations.index, violations['bilateral_violation'],
                   color=self.colors[4], linewidth=1.5)
            ax.fill_between(violations.index, 0, violations['bilateral_violation'],
                           alpha=0.3, color=self.colors[4])
            ax.axhline(y=1e-2, color='red', linestyle='--', alpha=0.5, label='Tolerance')
            ax.set_yscale('log')
            ax.set_ylabel('Violation (log scale)')
            ax.set_xlabel('Date')
            ax.set_title('Bilateral Consistency Violations')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('SFC Constraint Violations Over Time', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'constraint_violations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_stock_flow_consistency(self, 
                                   actual: pd.DataFrame,
                                   smoothed: pd.DataFrame,
                                   stock_flow_pairs: List,
                                   n_pairs: int = 4):
        """
        Plot stock-flow consistency for selected pairs.
        Shows: Stock[t] - Stock[t-1] vs Flow[t] + Revaluation[t]
        """
        # Select pairs to plot
        pairs_to_plot = stock_flow_pairs[:min(n_pairs, len(stock_flow_pairs))]
        
        fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 3*n_pairs))
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for idx, pair in enumerate(pairs_to_plot):
            if idx >= n_pairs:
                break
                
            # Get series names
            stock = pair.stock_series if hasattr(pair, 'stock_series') else pair.get('stock')
            flow = pair.flow_series if hasattr(pair, 'flow_series') else pair.get('flow')
            
            if stock in smoothed.columns and flow in smoothed.columns:
                # Left plot: Stock change vs flow
                ax = axes[idx, 0]
                stock_change = smoothed[stock].diff()
                ax.scatter(smoothed[flow], stock_change, alpha=0.5, s=10)
                
                # Add 45-degree line
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),
                    np.max([ax.get_xlim(), ax.get_ylim()]),
                ]
                ax.plot(lims, lims, 'r--', alpha=0.5, zorder=0)
                
                ax.set_xlabel(f'Flow ({flow})')
                ax.set_ylabel(f'Stock Change ({stock})')
                ax.set_title(f'Stock-Flow Consistency Check: {stock[:10]}', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Right plot: Violation over time
                ax = axes[idx, 1]
                violation = stock_change - smoothed[flow]
                ax.plot(smoothed.index, violation, color=self.colors[idx % 10], linewidth=1)
                ax.fill_between(smoothed.index, 0, violation, alpha=0.3, color=self.colors[idx % 10])
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_xlabel('Date')
                ax.set_ylabel('Violation')
                ax.set_title(f'Stock-Flow Violation Over Time', fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Stock-Flow Consistency Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stock_flow_consistency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_shock_decomposition(self, filtered: pd.DataFrame, smoothed: pd.DataFrame):
        """
        Plot shock decomposition showing filtering vs smoothing differences.
        """
        # Calculate shocks as difference between filtered and smoothed
        shocks = filtered - smoothed
        
        # Select series with largest shocks
        shock_variance = shocks.var().sort_values(ascending=False)
        top_series = shock_variance.head(10).index
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # 1. Time series of shocks for top series
        ax = axes[0]
        for i, series in enumerate(top_series[:5]):
            if series in shocks.columns:
                ax.plot(shocks.index, shocks[series], 
                       label=series[:20], color=self.colors[i], alpha=0.7, linewidth=1)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Shock (Filtered - Smoothed)')
        ax.set_title('Time Series of Filtering Shocks (Top 5 Series by Variance)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. Distribution of shock magnitudes
        ax = axes[1]
        all_shocks = shocks.values.flatten()
        all_shocks = all_shocks[~np.isnan(all_shocks)]
        
        ax.hist(all_shocks, bins=50, alpha=0.7, color=self.colors[0], edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Shock Magnitude')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of All Filtering Shocks')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {np.mean(all_shocks):.2e}\nStd: {np.std(all_shocks):.2e}\nSkew: {pd.Series(all_shocks).skew():.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Shock Decomposition Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'shock_decomposition.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_fwtw_network(self, fwtw_data: pd.DataFrame, top_n: int = 20):
        """
        Plot FWTW network structure showing bilateral relationships.
        """
        if fwtw_data is None or fwtw_data.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Network graph
        ax = axes[0]
        
        # Create network from FWTW data
        G = nx.DiGraph()
        
        # Aggregate flows by holder-issuer
        if 'Holder Code' in fwtw_data.columns and 'Issuer Code' in fwtw_data.columns:
            flows = fwtw_data.groupby(['Holder Code', 'Issuer Code'])['Level'].sum().reset_index()
            flows = flows.nlargest(top_n, 'Level')
            
            for _, row in flows.iterrows():
                G.add_edge(f"H{int(row['Holder Code'])}", 
                          f"I{int(row['Issuer Code'])}", 
                          weight=row['Level'])
            
            # Draw network
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes
            node_colors = ['lightblue' if node.startswith('H') else 'lightcoral' for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
            
            # Draw edges with width proportional to weight
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            edge_widths = [3 * w / max_weight for w in weights]
            
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, 
                                  edge_color='gray', arrows=True, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
            
            ax.set_title(f'FWTW Network Structure (Top {top_n} Flows)')
            ax.axis('off')
        
        # 2. Heatmap of bilateral positions
        ax = axes[1]
        
        if 'Holder Code' in fwtw_data.columns and 'Issuer Code' in fwtw_data.columns:
            # Create pivot table
            pivot = fwtw_data.pivot_table(
                values='Level',
                index='Holder Code',
                columns='Issuer Code',
                aggfunc='sum',
                fill_value=0
            )
            
            # Select top holders and issuers
            top_holders = pivot.sum(axis=1).nlargest(10).index
            top_issuers = pivot.sum(axis=0).nlargest(10).index
            pivot_subset = pivot.loc[top_holders, top_issuers]
            
            # Plot heatmap
            sns.heatmap(pivot_subset, cmap='YlOrRd', cbar_kws={'label': 'Position Level'},
                       fmt='.0f', ax=ax)
            ax.set_xlabel('Issuer Code')
            ax.set_ylabel('Holder Code')
            ax.set_title('Bilateral Position Heatmap (Top 10x10)')
        
        plt.suptitle('FWTW Bilateral Network Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fwtw_network.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_diagnostics(self, diagnostics: Dict):
        """
        Plot model diagnostic summary.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. State composition
        ax = axes[0, 0]
        if 'state_composition' in diagnostics:
            comp = diagnostics['state_composition']
            labels = list(comp.keys())
            sizes = list(comp.values())
            colors = self.colors[:len(labels)]
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('State Space Composition')
        else:
            # Alternative: show state dimensions
            labels = ['Base States', 'SFC States']
            sizes = [diagnostics.get('base_states', 100), diagnostics.get('sfc_states', 0)]
            ax.pie(sizes, labels=labels, colors=self.colors[:2], autopct='%1.1f%%', startangle=90)
            ax.set_title('State Space Dimensions')
        
        # 2. Constraint counts
        ax = axes[0, 1]
        constraint_types = ['Stock-Flow', 'Market Clearing', 'Bilateral', 'Formula']
        counts = [
            diagnostics.get('n_stock_flow_pairs', 0),
            diagnostics.get('n_market_clearing', 0),
            diagnostics.get('n_bilateral_positions', 0),
            diagnostics.get('n_formula_constraints', 0)
        ]
        
        bars = ax.bar(constraint_types, counts, color=self.colors[:4])
        ax.set_ylabel('Count')
        ax.set_title('Number of Constraints by Type')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom')
        
        # 3. Convergence metrics
        ax = axes[1, 0]
        if 'convergence' in diagnostics:
            conv = diagnostics['convergence']
            iterations = range(1, len(conv['log_likelihood']) + 1)
            ax.plot(iterations, conv['log_likelihood'], 'o-', color=self.colors[5])
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Log Likelihood')
            ax.set_title('Model Convergence')
            ax.grid(True, alpha=0.3)
        else:
            # Show single convergence indicator
            ax.text(0.5, 0.5, 'Model Converged' if diagnostics.get('converged', False) else 'Not Converged',
                   transform=ax.transAxes, ha='center', va='center', fontsize=20,
                   color='green' if diagnostics.get('converged', False) else 'red')
            ax.set_title('Convergence Status')
            ax.axis('off')
        
        # 4. Performance metrics
        ax = axes[1, 1]
        metrics = {
            'Log Likelihood': diagnostics.get('log_likelihood', 0),
            'AIC': diagnostics.get('aic', 0),
            'BIC': diagnostics.get('bic', 0),
            'Parameters': diagnostics.get('n_parameters', 0)
        }
        
        y_pos = np.arange(len(metrics))
        values = list(metrics.values())
        
        bars = ax.barh(y_pos, values, color=self.colors[6])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(metrics.keys()))
        ax.set_xlabel('Value')
        ax.set_title('Model Performance Metrics')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val, bar.get_y() + bar.get_height()/2.,
                   f'{val:.1f}', ha='left', va='center')
        
        plt.suptitle('Model Diagnostics Summary', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_series_composition(self, data: pd.DataFrame):
        """
        Plot composition of series in the dataset.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Series prefix distribution
        ax = axes[0]
        prefixes = {}
        for col in data.columns:
            prefix = str(col)[:2] if len(str(col)) >= 2 else 'Other'
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        # Sort by count
        sorted_prefixes = dict(sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:10])
        
        ax.bar(sorted_prefixes.keys(), sorted_prefixes.values(), color=self.colors[0])
        ax.set_xlabel('Series Prefix')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Series Types')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Data availability over time
        ax = axes[1]
        availability = data.notna().sum(axis=1) / len(data.columns) * 100
        ax.plot(data.index, availability, color=self.colors[1], linewidth=2)
        ax.fill_between(data.index, 0, availability, alpha=0.3, color=self.colors[1])
        ax.set_xlabel('Date')
        ax.set_ylabel('Data Availability (%)')
        ax.set_title('Data Availability Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Dataset Composition Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'series_composition.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_filtering_improvement(self, 
                                  actual: pd.DataFrame,
                                  filtered: pd.DataFrame,
                                  smoothed: pd.DataFrame):
        """
        Plot metrics showing improvement from filtering.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Calculate improvements for common series
        common_series = list(set(actual.columns) & set(filtered.columns) & set(smoothed.columns))
        
        if common_series:
            # 1. RMSE reduction
            ax = axes[0, 0]
            rmse_raw = []
            rmse_filtered = []
            rmse_smoothed = []
            
            for series in common_series[:20]:  # Limit to 20 series for clarity
                # Use first differences to calculate RMSE
                actual_diff = actual[series].diff().dropna()
                filtered_diff = filtered[series].diff().dropna()
                smoothed_diff = smoothed[series].diff().dropna()
                
                if len(actual_diff) > 10:
                    # RMSE of first differences (as proxy for noise)
                    rmse_raw.append(np.sqrt(np.mean(actual_diff**2)))
                    rmse_filtered.append(np.sqrt(np.mean(filtered_diff**2)))
                    rmse_smoothed.append(np.sqrt(np.mean(smoothed_diff**2)))
            
            x = np.arange(len(rmse_raw))
            width = 0.25
            
            ax.bar(x - width, rmse_raw, width, label='Raw', color=self.colors[0])
            ax.bar(x, rmse_filtered, width, label='Filtered', color=self.colors[1])
            ax.bar(x + width, rmse_smoothed, width, label='Smoothed', color=self.colors[2])
            
            ax.set_xlabel('Series Index')
            ax.set_ylabel('RMSE of First Differences')
            ax.set_title('Noise Reduction by Series')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 2. Variance reduction
            ax = axes[0, 1]
            var_reduction_filtered = []
            var_reduction_smoothed = []
            
            for series in common_series:
                var_actual = actual[series].var()
                if var_actual > 0:
                    var_reduction_filtered.append(1 - filtered[series].var() / var_actual)
                    var_reduction_smoothed.append(1 - smoothed[series].var() / var_actual)
            
            ax.hist(var_reduction_filtered, bins=30, alpha=0.5, label='Filtered', color=self.colors[3])
            ax.hist(var_reduction_smoothed, bins=30, alpha=0.5, label='Smoothed', color=self.colors[4])
            ax.set_xlabel('Variance Reduction Ratio')
            ax.set_ylabel('Number of Series')
            ax.set_title('Distribution of Variance Reduction')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. Autocorrelation improvement
            ax = axes[1, 0]
            lags = range(1, 11)
            acf_raw = []
            acf_filtered = []
            acf_smoothed = []
            
            for lag in lags:
                # Average autocorrelation across series
                acf_r = np.mean([actual[s].autocorr(lag) for s in common_series[:20] 
                                if s in actual.columns and actual[s].notna().sum() > lag + 10])
                acf_f = np.mean([filtered[s].autocorr(lag) for s in common_series[:20]
                                if s in filtered.columns and filtered[s].notna().sum() > lag + 10])
                acf_s = np.mean([smoothed[s].autocorr(lag) for s in common_series[:20]
                                if s in smoothed.columns and smoothed[s].notna().sum() > lag + 10])
                
                acf_raw.append(acf_r)
                acf_filtered.append(acf_f)
                acf_smoothed.append(acf_s)
            
            ax.plot(lags, acf_raw, 'o-', label='Raw', color=self.colors[5])
            ax.plot(lags, acf_filtered, 's-', label='Filtered', color=self.colors[6])
            ax.plot(lags, acf_smoothed, '^-', label='Smoothed', color=self.colors[7])
            ax.set_xlabel('Lag')
            ax.set_ylabel('Average Autocorrelation')
            ax.set_title('Autocorrelation Structure')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. Missing data handling
            ax = axes[1, 1]
            missing_raw = actual[common_series].isna().sum()
            missing_filtered = filtered[common_series].isna().sum()
            missing_smoothed = smoothed[common_series].isna().sum()
            
            improvement_filtered = (missing_raw - missing_filtered) / missing_raw * 100
            improvement_smoothed = (missing_raw - missing_smoothed) / missing_raw * 100
            
            # Remove infinities and NaNs
            improvement_filtered = improvement_filtered[np.isfinite(improvement_filtered)]
            improvement_smoothed = improvement_smoothed[np.isfinite(improvement_smoothed)]
            
            ax.hist(improvement_filtered, bins=30, alpha=0.5, label='Filtered', color=self.colors[8])
            ax.hist(improvement_smoothed, bins=30, alpha=0.5, label='Smoothed', color=self.colors[9])
            ax.set_xlabel('Missing Data Reduction (%)')
            ax.set_ylabel('Number of Series')
            ax.set_title('Missing Data Imputation Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Filtering Performance Metrics', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'filtering_improvement.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_missing_regions(self, is_missing: pd.Series) -> List[Tuple[int, int]]:
        """
        Get start and end indices of missing data regions.
        """
        regions = []
        in_region = False
        start = 0
        
        for i, missing in enumerate(is_missing):
            if missing and not in_region:
                start = i
                in_region = True
            elif not missing and in_region:
                regions.append((start, i-1))
                in_region = False
        
        if in_region:
            regions.append((start, len(is_missing)-1))
        
        return regions


def integrate_visualization_into_analysis(analysis_instance):
    """
    Helper function to integrate visualization into ProperSFCAnalysis.
    Replace the stub _create_visualizations method with this implementation.
    """
    def _create_visualizations(self):
        """Create comprehensive visualizations using SFCVisualizationManager."""
        self.logger.info("Creating visualizations...")
        
        try:
            # Initialize visualization manager
            viz_manager = SFCVisualizationManager(
                output_dir=self.results.output_dir / 'figures'
            )
            
            # Prepare data
            filtered_series = None
            smoothed_series = None
            if hasattr(self, 'filtered_results') and self.filtered_results is not None:
                filtered_series = self.filtered_results.get('filtered')
                smoothed_series = self.filtered_results.get('smoothed')
            
            # Prepare diagnostics
            diagnostics = {
                'n_stock_flow_pairs': len(self.stock_flow_pairs) if hasattr(self, 'stock_flow_pairs') else 0,
                'n_bilateral_positions': len(self.bilateral_constraints) if hasattr(self, 'bilateral_constraints') else 0,
                'converged': self.fitted_results.mle_retvals.get('converged', False) if self.fitted_results else False,
                'log_likelihood': float(self.fitted_results.llf) if self.fitted_results else 0,
                'aic': float(self.fitted_results.aic) if self.fitted_results else 0,
                'n_parameters': len(self.fitted_results.params) if self.fitted_results else 0,
                'base_states': self.model.state_space.n_base_states if hasattr(self.model, 'state_space') else 100,
                'sfc_states': self.model.state_space.n_sfc_states if hasattr(self.model, 'state_space') else 0
            }
            
            # Create all visualizations
            viz_manager.create_all_visualizations(
                z1_data=self.z1_data,
                filtered_states=filtered_series,
                smoothed_states=smoothed_series,
                model_diagnostics=diagnostics,
                constraint_violations=getattr(self, 'constraint_violations', None),
                fwtw_data=self.fwtw_data,
                stock_flow_pairs=getattr(self, 'stock_flow_pairs', None)
            )
            
            self.logger.info(f"  ✓ Visualizations saved to {viz_manager.output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Could not create some visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    # Replace the method
    analysis_instance._create_visualizations = _create_visualizations.__get__(analysis_instance)


# Example usage in run_proper_sfc.py:
"""
# In the ProperSFCAnalysis class, replace the stub _create_visualizations with:

from sfc_visualization import SFCVisualizationManager, integrate_visualization_into_analysis

# Then in __init__ or run():
integrate_visualization_into_analysis(self)
"""