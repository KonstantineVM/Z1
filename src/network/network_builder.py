# src/network/network_builder.py
"""
Network Builder
Constructs financial networks from FWTW data
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class NetworkBuilder:
    """
    Build financial networks from FWTW data
    
    This class creates directed networks where:
    - Nodes represent financial sectors/entities
    - Edges represent financial flows
    - Edge weights represent flow amounts
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Network Builder
        
        Parameters:
        -----------
        data : pd.DataFrame
            FWTW data with columns: Date, Holder Name, Issuer Name, 
            Instrument Name, Level
        """
        self.data = data
        self.networks = {}
        self.validate_data()
    
    def validate_data(self):
        """Validate input data has required columns"""
        required_columns = ['Date', 'Holder Name', 'Issuer Name', 
                          'Instrument Name', 'Level']
        missing_columns = set(required_columns) - set(self.data.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Initialized NetworkBuilder with {len(self.data)} records")
    
    def build_snapshot(self, date: pd.Timestamp, 
                      min_flow: float = 0,
                      instrument_filter: Optional[List[str]] = None) -> nx.DiGraph:
        """
        Build network snapshot for a specific date
        
        Parameters:
        -----------
        date : pd.Timestamp
            Date for network snapshot
        min_flow : float
            Minimum flow amount to include (filters out small flows)
        instrument_filter : List[str], optional
            List of instruments to include (if None, includes all)
            
        Returns:
        --------
        nx.DiGraph
            Directed network graph
        """
        # Filter data for specific date
        snapshot = self.data[self.data['Date'] == date]
        
        if instrument_filter:
            snapshot = snapshot[snapshot['Instrument Name'].isin(instrument_filter)]
        
        if min_flow > 0:
            snapshot = snapshot[snapshot['Level'] >= min_flow]
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add metadata
        G.graph['date'] = date
        G.graph['num_transactions'] = len(snapshot)
        G.graph['total_volume'] = snapshot['Level'].sum()
        
        # Add edges with attributes
        for _, row in snapshot.iterrows():
            if row['Level'] > 0:
                G.add_edge(
                    row['Holder Name'],
                    row['Issuer Name'],
                    weight=row['Level'],
                    instrument=row['Instrument Name']
                )
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['type'] = self._classify_node(node)
            G.nodes[node]['in_flow'] = sum(G[u][node].get('weight', 0) 
                                          for u in G.predecessors(node))
            G.nodes[node]['out_flow'] = sum(G[node][v].get('weight', 0) 
                                           for v in G.successors(node))
            G.nodes[node]['net_flow'] = G.nodes[node]['in_flow'] - G.nodes[node]['out_flow']
        
        logger.info(f"Built network for {date}: {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges")
        
        return G
    
    def build_time_series(self, dates: Optional[List[pd.Timestamp]] = None,
                         **kwargs) -> Dict[pd.Timestamp, nx.DiGraph]:
        """
        Build network time series for multiple dates
        
        Parameters:
        -----------
        dates : List[pd.Timestamp], optional
            List of dates (if None, uses all unique dates)
        **kwargs : 
            Additional arguments passed to build_snapshot
            
        Returns:
        --------
        Dict[pd.Timestamp, nx.DiGraph]
            Dictionary mapping dates to networks
        """
        if dates is None:
            dates = sorted(self.data['Date'].unique())
        
        networks = {}
        for date in dates:
            networks[date] = self.build_snapshot(date, **kwargs)
        
        self.networks = networks
        logger.info(f"Built {len(networks)} network snapshots")
        
        return networks
    
    def build_aggregated_network(self, start_date: Optional[pd.Timestamp] = None,
                               end_date: Optional[pd.Timestamp] = None) -> nx.DiGraph:
        """
        Build aggregated network over a time period
        
        Parameters:
        -----------
        start_date : pd.Timestamp, optional
            Start date (if None, uses earliest date)
        end_date : pd.Timestamp, optional
            End date (if None, uses latest date)
            
        Returns:
        --------
        nx.DiGraph
            Aggregated network
        """
        # Filter data by date range
        filtered_data = self.data.copy()
        
        if start_date:
            filtered_data = filtered_data[filtered_data['Date'] >= start_date]
        if end_date:
            filtered_data = filtered_data[filtered_data['Date'] <= end_date]
        
        # Aggregate flows by holder-issuer pairs
        aggregated = filtered_data.groupby(['Holder Name', 'Issuer Name'])['Level'].sum().reset_index()
        
        # Build network
        G = nx.DiGraph()
        G.graph['start_date'] = start_date or filtered_data['Date'].min()
        G.graph['end_date'] = end_date or filtered_data['Date'].max()
        
        for _, row in aggregated.iterrows():
            if row['Level'] > 0:
                G.add_edge(
                    row['Holder Name'],
                    row['Issuer Name'],
                    weight=row['Level']
                )
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['type'] = self._classify_node(node)
            G.nodes[node]['total_in_flow'] = sum(G[u][node].get('weight', 0) 
                                                for u in G.predecessors(node))
            G.nodes[node]['total_out_flow'] = sum(G[node][v].get('weight', 0) 
                                                 for v in G.successors(node))
            G.nodes[node]['net_position'] = G.nodes[node]['total_in_flow'] - \
                                           G.nodes[node]['total_out_flow']
        
        return G
    
    def build_instrument_networks(self, date: pd.Timestamp) -> Dict[str, nx.DiGraph]:
        """
        Build separate networks for each instrument type
        
        Parameters:
        -----------
        date : pd.Timestamp
            Date for network snapshots
            
        Returns:
        --------
        Dict[str, nx.DiGraph]
            Dictionary mapping instrument names to networks
        """
        instrument_networks = {}
        instruments = self.data['Instrument Name'].unique()
        
        for instrument in instruments:
            instrument_data = self.data[self.data['Instrument Name'] == instrument]
            
            # Create temporary builder for instrument-specific data
            temp_builder = NetworkBuilder(instrument_data)
            network = temp_builder.build_snapshot(date)
            
            if network.number_of_edges() > 0:
                instrument_networks[instrument] = network
        
        logger.info(f"Built networks for {len(instrument_networks)} instruments")
        
        return instrument_networks
    
    def build_multiplex_network(self, date: pd.Timestamp) -> nx.MultiDiGraph:
        """
        Build multiplex network with different edge types for instruments
        
        Parameters:
        -----------
        date : pd.Timestamp
            Date for network snapshot
            
        Returns:
        --------
        nx.MultiDiGraph
            Multiplex network
        """
        snapshot = self.data[self.data['Date'] == date]
        G = nx.MultiDiGraph()
        
        G.graph['date'] = date
        
        for _, row in snapshot.iterrows():
            if row['Level'] > 0:
                G.add_edge(
                    row['Holder Name'],
                    row['Issuer Name'],
                    weight=row['Level'],
                    instrument=row['Instrument Name'],
                    key=row['Instrument Name']  # Key for multi-edges
                )
        
        return G
    
    def _classify_node(self, node_name: str) -> str:
        """
        Classify node type based on name patterns
        
        Parameters:
        -----------
        node_name : str
            Name of the node
            
        Returns:
        --------
        str
            Node classification
        """
        name_lower = node_name.lower()
        
        if 'bank' in name_lower:
            return 'bank'
        elif 'insurance' in name_lower:
            return 'insurance'
        elif 'pension' in name_lower:
            return 'pension'
        elif 'fund' in name_lower:
            return 'fund'
        elif 'government' in name_lower:
            return 'government'
        elif 'household' in name_lower:
            return 'household'
        elif 'corporate' in name_lower or 'business' in name_lower:
            return 'corporate'
        elif 'foreign' in name_lower or 'rest of' in name_lower:
            return 'foreign'
        else:
            return 'other'
    
    def get_network_evolution(self, metric: str = 'density') -> pd.DataFrame:
        """
        Calculate network metric evolution over time
        
        Parameters:
        -----------
        metric : str
            Metric to calculate ('density', 'nodes', 'edges', 'avg_degree')
            
        Returns:
        --------
        pd.DataFrame
            Time series of network metric
        """
        if not self.networks:
            self.build_time_series()
        
        metrics_data = []
        
        for date, network in sorted(self.networks.items()):
            metric_value = None
            
            if metric == 'density':
                metric_value = nx.density(network)
            elif metric == 'nodes':
                metric_value = network.number_of_nodes()
            elif metric == 'edges':
                metric_value = network.number_of_edges()
            elif metric == 'avg_degree':
                if network.number_of_nodes() > 0:
                    metric_value = 2 * network.number_of_edges() / network.number_of_nodes()
                else:
                    metric_value = 0
            
            metrics_data.append({
                'Date': date,
                metric: metric_value
            })
        
        return pd.DataFrame(metrics_data)
    
    def find_important_paths(self, G: nx.DiGraph, 
                           source: str, 
                           target: str,
                           k: int = 5) -> List[List[str]]:
        """
        Find k most important paths between source and target
        
        Parameters:
        -----------
        G : nx.DiGraph
            Network graph
        source : str
            Source node
        target : str
            Target node
        k : int
            Number of paths to find
            
        Returns:
        --------
        List[List[str]]
            List of paths (each path is a list of nodes)
        """
        try:
            # Get all simple paths
            all_paths = list(nx.all_simple_paths(G, source, target, cutoff=5))
            
            # Calculate path weights (sum of edge weights)
            path_weights = []
            for path in all_paths:
                weight = 0
                for i in range(len(path) - 1):
                    weight += G[path[i]][path[i+1]].get('weight', 0)
                path_weights.append((path, weight))
            
            # Sort by weight and return top k
            path_weights.sort(key=lambda x: x[1], reverse=True)
            
            return [path for path, _ in path_weights[:k]]
            
        except nx.NetworkXNoPath:
            return []
    
    def detect_triangles(self, G: nx.DiGraph) -> List[Tuple[str, str, str]]:
        """
        Detect triangular relationships in the network
        
        Parameters:
        -----------
        G : nx.DiGraph
            Network graph
            
        Returns:
        --------
        List[Tuple[str, str, str]]
            List of triangles (3-node cycles)
        """
        triangles = []
        
        for node in G.nodes():
            successors = set(G.successors(node))
            for successor in successors:
                # Find common successors that link back
                successor_successors = set(G.successors(successor))
                common = successors & successor_successors
                
                for common_node in common:
                    if node in G.successors(common_node):
                        triangle = tuple(sorted([node, successor, common_node]))
                        if triangle not in triangles:
                            triangles.append(triangle)
        
        return triangles