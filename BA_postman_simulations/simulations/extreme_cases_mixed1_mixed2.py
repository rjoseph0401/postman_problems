import sys
import os
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import copy

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from algorithms.mixed1 import MIXED1
from algorithms.mixed2 import MIXED2

def create_asymmetric_cycles_graph(n_nodes=20, seed=None):
    if seed is not None:
        random.seed(seed)
    
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n_nodes))
    
    # Base connectivity - directed cycle
    for i in range(n_nodes):
        G.add_edge(i, (i + 1) % n_nodes, kind='directed', weight=random.randint(1, 5))
    
    # Asymmetric cycles
    cycle_size = max(3, n_nodes // 4)
    cycle_starts = [i * cycle_size for i in range(n_nodes // cycle_size)]
    
    for start in cycle_starts:
        cycle_nodes = list(range(start, min(start + cycle_size, n_nodes)))
        
        for i, current in enumerate(cycle_nodes):
            next_node = cycle_nodes[(i + 1) % len(cycle_nodes)]
            
            # Add forward edge if not in base cycle
            if not G.has_edge(current, next_node):
                G.add_edge(current, next_node, kind='directed', weight=random.randint(1, 5))
            
            # Add reverse with higher cost
            if random.random() < 0.3 and not G.has_edge(next_node, current):
                G.add_edge(next_node, current, kind='directed', weight=random.randint(15, 25))
    
    # Connect cycles with undirected edges
    def add_cycle_connection(start1, start2, size):
        if start2 < n_nodes:
            node1 = start1 + random.randint(0, min(size - 1, n_nodes - start1 - 1))
            node2 = start2 + random.randint(0, min(size - 1, n_nodes - start2 - 1))
            weight = random.randint(5, 10)
            
            if not G.has_edge(node1, node2) and not G.has_edge(node2, node1):
                G.add_edge(node1, node2, kind='undirected', weight=weight)
                G.add_edge(node2, node1, kind='undirected', weight=weight)
    
    # Connect adjacent cycles
    for i in range(len(cycle_starts) - 1):
        add_cycle_connection(cycle_starts[i], cycle_starts[i + 1], cycle_size)
    
    # Connect first and last cycle
    if len(cycle_starts) > 1:
        add_cycle_connection(cycle_starts[0], cycle_starts[-1], cycle_size)
    return G



def create_long_path_graph(n_nodes=20, seed=None):
    if seed is not None:
        random.seed(seed)
    
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n_nodes))
    
    # Create a long directed path as the main structure
    path_length = n_nodes - 2
    path_nodes = list(range(path_length))
    
    # Main directed path with low weights
    for i in range(path_length - 1):
        G.add_edge(path_nodes[i], path_nodes[i + 1], kind='directed', weight=1)
    
    # Add expensive reverse edges on the path
    for i in range(1, path_length - 1, 2):  
        if random.random() < 0.6:
            G.add_edge(path_nodes[i + 1], path_nodes[i], kind='directed', weight=random.randint(20, 30))
    
    # Create side branches that require backtracking
    side_nodes = list(range(path_length, n_nodes))
    for side_node in side_nodes:
        # Connect side node to a random point on the main path
        attach_point = random.choice(path_nodes[1:-1])
        
        # Directed connection from path to side (cheap)
        G.add_edge(attach_point, side_node, kind='directed', weight=random.randint(1, 3))
        # Expensive return path
        G.add_edge(side_node, attach_point, kind='directed', weight=random.randint(15, 25))
        
        if random.random() < 0.4:
            other_point = random.choice([p for p in path_nodes if p != attach_point])
            weight = random.randint(12, 18)
            G.add_edge(side_node, other_point, kind='undirected', weight=weight)
            G.add_edge(other_point, side_node, kind='undirected', weight=weight)
    
    # Ensure strong connectivity
    if path_length > 2:
        G.add_edge(path_nodes[-1], path_nodes[0], kind='directed', weight=random.randint(5, 10))
    
    return G



def ensure_connectivity(G):
    underlying_graph = nx.Graph()
    underlying_graph.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        underlying_graph.add_edge(u, v)
    
    if not nx.is_connected(underlying_graph):
        components = list(nx.connected_components(underlying_graph))
        for i in range(len(components) - 1):
            node1, node2 = random.choice(list(components[i])), random.choice(list(components[i + 1]))
            G.add_edge(node1, node2, kind='undirected', weight=1)
            G.add_edge(node2, node1, kind='undirected', weight=1)
    
    return G

def validate_graph_solvability(G):
    underlying_graph = nx.Graph()
    underlying_graph.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        underlying_graph.add_edge(u, v)
    
    if not nx.is_connected(underlying_graph):
        return False, "Graph is not connected"
    
    return True, "Graph is valid"

def calculate_cost(circuit, original_graph=None):
    if not circuit:
        return float('inf')
    
    #MIXED1 format
    if isinstance(circuit, tuple) and len(circuit) == 2:
        G_prime, circuit_list = circuit
        if not circuit_list:
            return float('inf')
        
        total_cost = 0
        for u, v in circuit_list:
            edge_data = G_prime.get_edge_data(u, v)
            if edge_data and isinstance(edge_data, dict):
                if 0 in edge_data:
                    weight = edge_data[0].get('weight', 1)
                else:
                    weight = edge_data.get('weight', 1)
            else:
                weight = 1
            total_cost += weight
        return total_cost
    
    #MIXED2 format
    if isinstance(circuit, list) and circuit:
        if len(circuit[0]) >= 4:
            return sum(weight for _, _, _, weight in circuit)
        elif len(circuit[0]) == 2 and original_graph:
            total_cost = 0
            for u, v in circuit:
                weight = 1
                for edge_data in [original_graph.get_edge_data(u, v), original_graph.get_edge_data(v, u)]:
                    if edge_data and isinstance(edge_data, dict):
                        if 0 in edge_data:
                            weight = edge_data[0].get('weight', 1)
                        else:
                            weight = edge_data.get('weight', 1)
                        break
                total_cost += weight
            return total_cost
        else:
            return len(circuit)
    
    return float('inf')

def run_extreme_case_simulations(repetitions=3):
    results = []
    extreme_cases = [
        ("Long Path with Detours", create_long_path_graph),
        ("Asymmetric Directed Cycles", create_asymmetric_cycles_graph)
    ]
    
    node_sizes = [10, 15, 20, 30, 40, 50, 75, 100]
    
    for case_name, graph_generator in tqdm(extreme_cases, desc="Testing extreme cases"):
        for n_nodes in tqdm(node_sizes, desc=f"Testing {case_name}", leave=False):
            for rep in range(repetitions):
                seed = 42 * n_nodes * (rep + 1)
                
                # Generate and ensure solvable graph
                G = graph_generator(n_nodes=n_nodes, seed=seed)
                G = ensure_connectivity(G)
                
                # Calculate graph statistics
                directed_count = sum(1 for _, _, data in G.edges(data=True) if data.get('kind') == 'directed')
                undirected_count = sum(1 for _, _, data in G.edges(data=True) if data.get('kind') == 'undirected') // 2
                total_edges = directed_count + undirected_count
                
                result = {
                    'case_name': case_name,
                    'nodes': n_nodes,
                    'total_edges': total_edges,
                    'rep': rep
                }
                
                # Test both algorithms
                for alg_name, algorithm in [('mixed1', MIXED1), ('mixed2', MIXED2)]:
                    try:
                        G_copy = copy.deepcopy(G)
                        start_time = time.time()
                        circuit = algorithm(G_copy)
                        runtime = time.time() - start_time
                        
                        cost = calculate_cost(circuit, G)
                        if cost != float('inf') and cost is not None:
                            result[f'{alg_name}_time'] = runtime
                            result[f'{alg_name}_cost'] = cost
                        else:
                            result[f'{alg_name}_time'] = None
                            result[f'{alg_name}_cost'] = np.nan
                    except Exception:
                        result[f'{alg_name}_time'] = None
                        result[f'{alg_name}_cost'] = np.nan
                
                results.append(result)
    
    return pd.DataFrame(results)

def plot_extreme_case_results(results):
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results
    df = df.replace([float('inf'), np.inf], np.nan)
    
    available_cases = df['case_name'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, case_name in enumerate(available_cases):
        case_data = df[df['case_name'] == case_name].groupby('total_edges').agg({
            'mixed1_cost': 'mean', 'mixed2_cost': 'mean',
            'mixed1_time': 'mean', 'mixed2_time': 'mean'
        }).reset_index().sort_values('total_edges')
        
        # Cost comparison plot
        cost_data = case_data.dropna(subset=['mixed1_cost', 'mixed2_cost'])
        if not cost_data.empty:
            for j, (alg, color, marker) in enumerate([('mixed1', 'blue', 'o'), ('mixed2', 'red', 's')]):
                axes[0, i].plot(cost_data['total_edges'], cost_data[f'{alg}_cost'], 
                               f'{marker}-', color=color, linewidth=2, markersize=6, 
                               label=f'MIXED{j+1}')
            axes[0, i].legend()
        
        axes[0, i].set(xlabel='Number of Edges', ylabel='Tour Cost', 
                      title=f'Cost: {case_name}')
        axes[0, i].grid(True, alpha=0.3)
        
        # Runtime comparison plot
        time_data = case_data.dropna(subset=['mixed1_time', 'mixed2_time'])
        if not time_data.empty:
            for j, (alg, color, marker) in enumerate([('mixed1', 'blue', 'o'), ('mixed2', 'red', 's')]):
                axes[1, i].plot(time_data['total_edges'], time_data[f'{alg}_time'], 
                               f'{marker}-', color=color, linewidth=2, markersize=6, 
                               label=f'MIXED{j+1}')
            axes[1, i].legend()
        
        axes[1, i].set(xlabel='Number of Edges', ylabel='Runtime (seconds)', 
                      title=f'Runtime: {case_name}')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    print("Testing MIXED1 vs MIXED2 on Long Path and Asymmetric Cycles...")
    print("=" * 60)
    
    # Run simulations
    results = run_extreme_case_simulations(repetitions=3)
    
    # Generate plots
    fig = plot_extreme_case_results(results)
    
    # Print concise summary
    print("\nSummary by case:")
    summary = results.groupby('case_name').agg({
        'mixed1_cost': 'mean', 'mixed2_cost': 'mean',
        'mixed1_time': 'mean', 'mixed2_time': 'mean'
    }).round(3)
    print(summary)
    
    # Performance comparison
    valid_comparisons = results.dropna(subset=['mixed1_cost', 'mixed2_cost'])
    if len(valid_comparisons) > 0:
        cost_ratio = (valid_comparisons['mixed2_cost'] / valid_comparisons['mixed1_cost']).mean()
        time_ratio = (valid_comparisons['mixed2_time'] / valid_comparisons['mixed1_time']).mean()
        
        print(f"\nAverage Performance Ratios (MIXED2/MIXED1):")
        print(f"Cost ratio: {cost_ratio:.3f}")
        print(f"Time ratio: {time_ratio:.3f}")
    
    plt.show()

if __name__ == "__main__":
    main()
