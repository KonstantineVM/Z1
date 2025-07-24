# src/network/network_analyzer.py
"""
Network Analyzer
Analyzes financial network properties and systemic risk
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """
    Analyze financial network properties
    
    This class provides comprehensive analysis tools for financial networks,
    including centrality measures, systemic risk indicators, and community detection.
    """
    
    def __init__(self, network: nx.DiGraph):
        """
        Initialize Network Analyzer
        
        Parameters:
        -----------
        network : nx.DiGraph
            Financial network to analyze
        """
        self.network = network
        self.centrality_cache = {}
        self._validate_network()
    
    def _validate_network(self):
        """Validate network has required structure"""
        if self.network.number_of_nodes() == 0:
            raise ValueError("Network has no nodes")
        
        # Check if edges have weights
        if self.network.number_of_edges() > 0:
            sample_edge = list(self.network.edges(data=True))[0]
            if 'weight' not in sample_edge[2]:
                logger.warning("Network edges do not have weights")
    
    def compute_centrality_metrics(self, 
                                  weighted: bool = True,
                                  normalize: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Compute comprehensive centrality metrics
        
        Parameters:
        -----------
        weighted : bool
            Whether to use edge weights in calculations
        normalize : bool
            Whether to normalize metrics to [0, 1]
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary of centrality metrics
        """
        metrics = {}
        
        # Degree centrality
        metrics['in_degree'] = dict(self.network.in_degree(weight='weight' if weighted else None))
        metrics['out_degree'] = dict(self.network.out_degree(weight='weight' if weighted else None))
        metrics['total_degree'] = {
            node: metrics['in_degree'][node] + metrics['out_degree'][node]
            for node in self.network.nodes()
        }
        
        # Betweenness centrality
        metrics['betweenness'] = nx.betweenness_centrality(
            self.network, 
            weight='weight' if weighted else None,
            normalized=normalize
        )
        
        # Eigenvector centrality
        try:
            metrics['eigenvector'] = nx.eigenvector_centrality(
                self.network,
                weight='weight' if weighted else None,
                max_iter=1000
            )
        except:
            logger.warning("Eigenvector centrality calculation failed")
            metrics['eigenvector'] = {node: 0 for node in self.network.nodes()}
        
        # PageRank
        metrics['pagerank'] = nx.pagerank(
            self.network,
            weight='weight' if weighted else None
        )
        
        # Closeness centrality
        metrics['closeness'] = nx.closeness_centrality(
            self.network,
            distance='weight' if weighted else None
        )
        
        # Hub and authority scores (HITS algorithm)
        try:
            hubs, authorities = nx.hits(
                self.network,
                max_iter=1000,
                normalized=normalize
            )
            metrics['hub_score'] = hubs
            metrics['authority_score'] = authorities
        except:
            logger.warning("HITS algorithm failed")
            metrics['hub_score'] = {node: 0 for node in self.network.nodes()}
            metrics['authority_score'] = {node: 0 for node in self.network.nodes()}
        
        self.centrality_cache = metrics
        return metrics
    
    def identify_systemically_important(self, 
                                      method: str = 'composite',
                                      threshold: float = 0.9) -> List[Tuple[str, float]]:
        """
        Identify systemically important financial institutions (SIFIs)
        
        Parameters:
        -----------
        method : str
            Method to use ('composite', 'pagerank', 'eigenvector', 'degree')
        threshold : float
            Percentile threshold for importance (0-1)
            
        Returns:
        --------
        List[Tuple[str, float]]
            List of (node, importance_score) tuples
        """
        if not self.centrality_cache:
            self.compute_centrality_metrics()
        
        if method == 'composite':
            # Combine multiple metrics
            importance_scores = {}
            
            # Define weights for different metrics
            weights = {
                'total_degree': 0.25,
                'betweenness': 0.25,
                'eigenvector': 0.20,
                'pagerank': 0.30
            }
            
            for node in self.network.nodes():
                score = 0
                for metric, weight in weights.items():
                    if metric in self.centrality_cache:
                        # Normalize to [0, 1] if not already
                        values = list(self.centrality_cache[metric].values())
                        max_val = max(values) if values else 1
                        normalized_score = self.centrality_cache[metric].get(node, 0) / max_val
                        score += weight * normalized_score
                
                importance_scores[node] = score
        
        else:
            # Use single metric
            if method not in self.centrality_cache:
                raise ValueError(f"Unknown method: {method}")
            
            importance_scores = self.centrality_cache[method]
        
        # Sort by importance
        sorted_nodes = sorted(importance_scores.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        
        # Apply threshold
        threshold_value = np.percentile(list(importance_scores.values()), 
                                      threshold * 100)
        
        sifis = [(node, score) for node, score in sorted_nodes 
                 if score >= threshold_value]
        
        logger.info(f"Identified {len(sifis)} systemically important nodes")
        
        return sifis
    
    def calculate_network_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive network risk metrics
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of risk metrics
        """
        metrics = {}
        
        # Network concentration (Herfindahl index)
        total_volume = sum(self.network[u][v]['weight'] 
                          for u, v in self.network.edges())
        
        if total_volume > 0:
            edge_shares = [self.network[u][v]['weight'] / total_volume 
                          for u, v in self.network.edges()]
            metrics['herfindahl_index'] = sum(share**2 for share in edge_shares)
        else:
            metrics['herfindahl_index'] = 0
        
        # Network density
        metrics['density'] = nx.density(self.network)
        
        # Average clustering coefficient
        metrics['avg_clustering'] = nx.average_clustering(
            self.network.to_undirected()
        )
        
        # Network diameter (longest shortest path)
        if nx.is_strongly_connected(self.network):
            metrics['diameter'] = nx.diameter(self.network)
        else:
            # Use largest strongly connected component
            largest_scc = max(nx.strongly_connected_components(self.network), 
                            key=len)
            subgraph = self.network.subgraph(largest_scc)
            metrics['diameter'] = nx.diameter(subgraph) if len(largest_scc) > 1 else 0
        
        # Degree assortativity (do similar nodes connect?)
        metrics['assortativity'] = nx.degree_assortativity_coefficient(self.network)
        
        # Network efficiency
        metrics['global_efficiency'] = nx.global_efficiency(self.network)
        
        # Rich club coefficient
        try:
            rich_club = nx.rich_club_coefficient(
                self.network.to_undirected(), 
                normalized=False
            )
            metrics['rich_club_coeff'] = np.mean(list(rich_club.values()))
        except:
            metrics['rich_club_coeff'] = 0
        
        return metrics
    
    def analyze_contagion_risk(self, 
                             shock_node: str,
                             shock_magnitude: float = 0.1,
                             propagation_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Analyze contagion risk from shock to specific node
        
        Parameters:
        -----------
        shock_node : str
            Node experiencing initial shock
        shock_magnitude : float
            Size of initial shock (as fraction of node's position)
        propagation_threshold : float
            Minimum shock size to propagate
            
        Returns:
        --------
        Dict[str, Any]
            Contagion analysis results
        """
        if shock_node not in self.network:
            raise ValueError(f"Node {shock_node} not in network")
        
        # Initialize shock propagation
        node_shocks = {node: 0 for node in self.network.nodes()}
        node_shocks[shock_node] = shock_magnitude
        
        # Track propagation rounds
        propagation_history = [node_shocks.copy()]
        affected_nodes = {shock_node}
        
        # Propagate shock through network
        max_rounds = 10
        for round_num in range(max_rounds):
            new_shocks = node_shocks.copy()
            
            for node in affected_nodes:
                if node_shocks[node] >= propagation_threshold:
                    # Propagate to successors
                    total_outflow = sum(self.network[node][v]['weight'] 
                                      for v in self.network.successors(node))
                    
                    if total_outflow > 0:
                        for successor in self.network.successors(node):
                            flow_weight = self.network[node][successor]['weight']
                            propagation = node_shocks[node] * (flow_weight / total_outflow)
                            new_shocks[successor] += propagation * 0.5  # Damping factor
            
            # Check for convergence
            if all(abs(new_shocks[node] - node_shocks[node]) < 0.001 
                   for node in self.network.nodes()):
                break
            
            node_shocks = new_shocks
            affected_nodes = {node for node, shock in node_shocks.items() 
                            if shock >= propagation_threshold}
            propagation_history.append(node_shocks.copy())
        
        # Calculate contagion metrics
        results = {
            'initial_shock_node': shock_node,
            'shock_magnitude': shock_magnitude,
            'rounds_to_convergence': len(propagation_history) - 1,
            'total_affected_nodes': len(affected_nodes),
            'affected_fraction': len(affected_nodes) / self.network.number_of_nodes(),
            'total_loss': sum(node_shocks.values()),
            'amplification_factor': sum(node_shocks.values()) / shock_magnitude,
            'most_affected_nodes': sorted(node_shocks.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True)[:10],
            'propagation_history': propagation_history
        }
        
        return results
    
    def detect_communities(self, 
                         method: str = 'louvain',
                         resolution: float = 1.0) -> Dict[str, int]:
        """
        Detect communities in the network
        
        Parameters:
        -----------
        method : str
            Community detection method ('louvain', 'label_propagation')
        resolution : float
            Resolution parameter for modularity-based methods
            
        Returns:
        --------
        Dict[str, int]
            Dictionary mapping nodes to community IDs
        """
        # Convert to undirected for community detection
        undirected = self.network.to_undirected()
        
        if method == 'louvain':
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(
                    undirected, 
                    resolution=resolution
                )
            except ImportError:
                logger.warning("python-louvain not installed, using label propagation")
                method = 'label_propagation'
        
        if method == 'label_propagation':
            communities_generator = nx.community.label_propagation_communities(undirected)
            communities = {}
            for i, community in enumerate(communities_generator):
                for node in community:
                    communities[node] = i
        
        logger.info(f"Detected {len(set(communities.values()))} communities")
        
        return communities
    
    def calculate_interconnectedness(self) -> pd.DataFrame:
        """
        Calculate pairwise interconnectedness between nodes
        
        Returns:
        --------
        pd.DataFrame
            Matrix of interconnectedness scores
        """
        nodes = list(self.network.nodes())
        n = len(nodes)
        
        # Initialize interconnectedness matrix
        interconnectedness = pd.DataFrame(
            np.zeros((n, n)),
            index=nodes,
            columns=nodes
        )
        
        # Calculate for each pair
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    # Direct exposure
                    direct = 0
                    if self.network.has_edge(node1, node2):
                        direct += self.network[node1][node2]['weight']
                    if self.network.has_edge(node2, node1):
                        direct += self.network[node2][node1]['weight']
                    
                    # Common neighbors (indirect exposure)
                    neighbors1 = set(self.network.neighbors(node1))
                    neighbors2 = set(self.network.neighbors(node2))
                    common = len(neighbors1 & neighbors2)
                    
                    # Combined score
                    interconnectedness.loc[node1, node2] = direct + 0.1 * common
        
        return interconnectedness
    
    def find_critical_edges(self, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Find critical edges whose removal would most impact the network
        
        Parameters:
        -----------
        top_k : int
            Number of critical edges to return
            
        Returns:
        --------
        List[Tuple[str, str, float]]
            List of (source, target, criticality_score) tuples
        """
        edge_criticality = []
        
        # Calculate initial efficiency
        initial_efficiency = nx.global_efficiency(self.network)
        
        for u, v in self.network.edges():
            # Temporarily remove edge
            weight = self.network[u][v]['weight']
            self.network.remove_edge(u, v)
            
            # Calculate new efficiency
            new_efficiency = nx.global_efficiency(self.network)
            
            # Criticality is the efficiency loss
            criticality = (initial_efficiency - new_efficiency) * weight
            
            # Restore edge
            self.network.add_edge(u, v, weight=weight)
            
            edge_criticality.append((u, v, criticality))
        
        # Sort by criticality
        edge_criticality.sort(key=lambda x: x[2], reverse=True)
        
        return edge_criticality[:top_k]
    
    def calculate_node_vulnerability(self) -> Dict[str, float]:
        """
        Calculate vulnerability score for each node
        
        Returns:
        --------
        Dict[str, float]
            Vulnerability scores
        """
        vulnerability = {}
        
        for node in self.network.nodes():
            # Factors contributing to vulnerability
            in_degree = self.network.in_degree(node, weight='weight')
            out_degree = self.network.out_degree(node, weight='weight')
            
            # High leverage (high outflow relative to inflow)
            leverage = out_degree / (in_degree + 1)  # Add 1 to avoid division by zero
            
            # Concentration of exposures
            if self.network.out_degree(node) > 0:
                out_weights = [self.network[node][v]['weight'] 
                             for v in self.network.successors(node)]
                concentration = sum(w**2 for w in out_weights) / sum(out_weights)**2
            else:
                concentration = 0
            
            # Combined vulnerability score
            vulnerability[node] = leverage * concentration
        
        return vulnerability