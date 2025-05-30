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

def create_hub_graph(n_nodes=20, seed=None):
    if seed is not None:
        random.seed(seed)
    
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n_nodes))
    
    hub_nodes = random.sample(range(n_nodes), max(2, n_nodes // 8))
    
    for hub in hub_nodes:
        for other in range(n_nodes):
            if other != hub and random.random() < 0.7:
                weight = random.randint(1, 15)
                edge_type = random.choice(['out', 'in', 'undirected'])
                
                if edge_type == 'out':
                    G.add_edge(hub, other, kind='directed', weight=weight)
                elif edge_type == 'in':
                    G.add_edge(other, hub, kind='directed', weight=weight)
                else:
                    G.add_edge(hub, other, kind='undirected', weight=weight)
                    G.add_edge(other, hub, kind='undirected', weight=weight)
    
    non_hubs = [i for i in range(n_nodes) if i not in hub_nodes]
    for i in range(len(non_hubs) - 1):
        node1, node2 = non_hubs[i], non_hubs[i + 1]
        weight = random.randint(1, 10)
        if random.random() < 0.5:
            G.add_edge(node1, node2, kind='directed', weight=weight)
        else:
            G.add_edge(node1, node2, kind='undirected', weight=weight)
            G.add_edge(node2, node1, kind='undirected', weight=weight)
    
    return G

def create_asymmetric_cycles_graph(n_nodes=20, seed=None):
    if seed is not None:
        random.seed(seed)
    
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n_nodes))
    
    cycle_size = max(3, n_nodes // 4)
    cycle_starts = [i * cycle_size for i in range(n_nodes // cycle_size)]
    
    for start in cycle_starts:
        end = min(start + cycle_size, n_nodes)
        cycle_nodes = list(range(start, end))
        
        for i in range(len(cycle_nodes)):
            current = cycle_nodes[i]
            next_node = cycle_nodes[(i + 1) % len(cycle_nodes)]
            
            G.add_edge(current, next_node, kind='directed', weight=random.randint(1, 5))
            
            if random.random() < 0.3:
                G.add_edge(next_node, current, kind='directed', weight=random.randint(15, 25))
    
    for i in range(len(cycle_starts) - 1):
        node1 = cycle_starts[i] + random.randint(0, cycle_size - 1)
        node2 = cycle_starts[i + 1] + random.randint(0, min(cycle_size - 1, n_nodes - cycle_starts[i + 1] - 1))
        weight = random.randint(5, 10)
        G.add_edge(node1, node2, kind='undirected', weight=weight)
        G.add_edge(node2, node1, kind='undirected', weight=weight)
    
    return G

def calculate_directed_ratio(G):
    directed_count = sum(1 for _, _, data in G.edges(data=True) if data.get('kind') == 'directed')
    undirected_count = sum(1 for _, _, data in G.edges(data=True) if data.get('kind') == 'undirected') // 2
    total_edges = directed_count + undirected_count
    return directed_count / total_edges if total_edges > 0 else 0

def calculate_cost(circuit, original_graph=None):
    if not circuit:
        return float('inf')
    
    if isinstance(circuit, tuple) and len(circuit) == 2:
        G_prime, circuit_list = circuit
        if not circuit_list:
            return float('inf')
        
        total_cost = 0
        for u, v in circuit_list:
            weight = 1
            if G_prime.has_edge(u, v):
                edge_data = G_prime.get_edge_data(u, v)
                if isinstance(edge_data, dict):
                    if 0 in edge_data:
                        weight = edge_data[0].get('weight', 1)
                    else:
                        weight = edge_data.get('weight', 1)
            total_cost += weight
        return total_cost
    
    if isinstance(circuit, list) and circuit:
        if len(circuit[0]) >= 4:
            return sum(weight for _, _, _, weight in circuit)
        elif len(circuit[0]) == 2:
            if original_graph is not None:
                total_cost = 0
                for u, v in circuit:
                    weight = 1
                    for graph_edge in [original_graph.get_edge_data(u, v), original_graph.get_edge_data(v, u)]:
                        if graph_edge:
                            if 0 in graph_edge:
                                weight = graph_edge[0].get('weight', 1)
                            else:
                                weight = graph_edge.get('weight', 1)
                            break
                    total_cost += weight
                return total_cost
            else:
                return len(circuit)
    
    return float('inf')

def run_extreme_case_simulations(repetitions=5):
    results = []
    extreme_cases = [
        ("High-Degree Hubs", create_hub_graph),
        ("Asymmetric Directed Cycles", create_asymmetric_cycles_graph)
    ]
    
    node_sizes = [10, 15, 20, 30]
    
    for case_name, graph_generator in tqdm(extreme_cases, desc="Testing extreme cases"):
        for n_nodes in tqdm(node_sizes, desc=f"Testing {case_name}", leave=False):
            for rep in range(repetitions):
                seed = 42 * n_nodes * (rep + 1)
                
                # Generate graph
                G = graph_generator(n_nodes=n_nodes, seed=seed)
                directed_ratio = calculate_directed_ratio(G)
                
                directed_count = sum(1 for _, _, data in G.edges(data=True) if data.get('kind') == 'directed')
                undirected_count = sum(1 for _, _, data in G.edges(data=True) if data.get('kind') == 'undirected') // 2
                total_edge_count = directed_count + undirected_count
                
                result = {
                    'case_name': case_name,
                    'nodes': n_nodes,
                    'directed_edges': directed_count,
                    'undirected_edges': undirected_count,
                    'total_edges': total_edge_count,
                    'directed_ratio': directed_ratio,
                    'rep': rep,
                    'seed': seed,
                }
                  # Test MIXED1
                try:
                    G1 = copy.deepcopy(G)
                    start_time = time.time()
                    circuit1 = MIXED1(G1)
                    mixed1_new_time = time.time() - start_time
                    
                    cost1 = calculate_cost(circuit1, G)
                    if cost1 != float('inf') and cost1 is not None:
                        result['mixed1_new_time'] = mixed1_new_time
                        result['mixed1_new_cost'] = cost1
                    else:
                        result['mixed1_new_time'] = None
                        result['mixed1_new_cost'] = np.nan
                except Exception as e:
                    print(f"MIXED1_NEW error: {case_name}, {n_nodes} nodes: {str(e)}")
                    result['mixed1_new_time'] = None
                    result['mixed1_new_cost'] = np.nan
                
                # Test MIXED2_NEW
                try:
                    G2 = copy.deepcopy(G)
                    start_time = time.time()
                    circuit2 = MIXED2(G2)
                    mixed2_new_time = time.time() - start_time
                    
                    cost2 = calculate_cost(circuit2, G)
                    if cost2 != float('inf') and cost2 is not None:
                        result['mixed2_new_time'] = mixed2_new_time
                        result['mixed2_new_cost'] = cost2
                    else:
                        result['mixed2_new_time'] = None
                        result['mixed2_new_cost'] = np.nan
                except Exception as e:
                    print(f"MIXED2_NEW error: {case_name}, {n_nodes} nodes: {str(e)}")
                    result['mixed2_new_time'] = None
                    result['mixed2_new_cost'] = np.nan
                  # Calculate relative performance
                if (pd.notnull(result.get('mixed1_new_cost')) and 
                    pd.notnull(result.get('mixed2_new_cost')) and
                    result['mixed1_new_cost'] > 0):
                    result['cost_diff_percent'] = ((result['mixed2_new_cost'] - result['mixed1_new_cost']) / result['mixed1_new_cost']) * 100
                
                if (pd.notnull(result.get('mixed1_new_time')) and 
                    pd.notnull(result.get('mixed2_new_time')) and
                    result['mixed1_new_time'] > 0):
                    result['time_diff_percent'] = ((result['mixed2_new_time'] - result['mixed1_new_time']) / result['mixed1_new_time']) * 100
                
                results.append(result)
    
    return pd.DataFrame(results)

def plot_extreme_case_results(results):
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results
    df = df.replace([float('inf'), np.inf], np.nan)
    
    selected_cases = ["High-Degree Hubs", "Asymmetric Directed Cycles"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, case_name in enumerate(selected_cases):
        case_df = df[df['case_name'] == case_name]
        grouped = case_df.groupby('total_edges').agg({
            'mixed1_new_cost': 'mean',
            'mixed2_new_cost': 'mean',
            'mixed1_new_time': 'mean',
            'mixed2_new_time': 'mean'
        }).reset_index().sort_values('total_edges')
        
        valid_costs = grouped.dropna(subset=['mixed1_new_cost', 'mixed2_new_cost'])
        valid_times = grouped.dropna(subset=['mixed1_new_time', 'mixed2_new_time'])
        
        # Plot cost comparisons
        if not valid_costs.empty:
            axes[0, i].plot(valid_costs['total_edges'], valid_costs['mixed1_new_cost'], 'o-', 
                            color='blue', linewidth=2, markersize=8, label='MIXED1')
            axes[0, i].plot(valid_costs['total_edges'], valid_costs['mixed2_new_cost'], 's-', 
                            color='red', linewidth=2, markersize=8, label='MIXED2')
        
        axes[0, i].set_xlabel('Number of Edges', fontsize=12)
        axes[0, i].set_ylabel('Tour Cost', fontsize=12)
        axes[0, i].set_title(f'Cost Comparison: {case_name}', fontsize=14)
        axes[0, i].legend(fontsize=11)
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot runtime comparisons 
        if not valid_times.empty:
            axes[1, i].plot(valid_times['total_edges'], valid_times['mixed1_new_time'], 'o-',
                           color='blue', linewidth=2, markersize=8, label='MIXED1')
            axes[1, i].plot(valid_times['total_edges'], valid_times['mixed2_new_time'], 's-',
                           color='red', linewidth=2, markersize=8, label='MIXED2')
        
        axes[1, i].set_xlabel('Number of Edges', fontsize=12)
        axes[1, i].set_ylabel('Runtime (seconds)', fontsize=12)
        axes[1, i].set_title(f'Runtime Comparison: {case_name}', fontsize=14)
        axes[1, i].legend(fontsize=11)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def main():
    print("Starting extreme case simulations for MIXED1_NEW and MIXED2_NEW algorithms...")
    print("=" * 80)
    
    repetitions = 5

    results = run_extreme_case_simulations(repetitions)
    
    fig = plot_extreme_case_results(results)
    
    print("\nSummary by case:")
    summary = results.groupby('case_name').agg({
        'mixed1_new_cost': ['mean', 'min', 'max'],
        'mixed2_new_cost': ['mean', 'min', 'max'],
        'mixed1_new_time': ['mean', 'min', 'max'],
        'mixed2_new_time': ['mean', 'min', 'max'],
        'cost_diff_percent': ['mean', 'min', 'max'],
    })
    print(summary)
    
    significant_diff = results[abs(results['cost_diff_percent']) > 10].sort_values('cost_diff_percent', ascending=False)
    
    print("\nInstances with significant differences (>10% cost difference):")
    if not significant_diff.empty:
        significant_summary = significant_diff.groupby('case_name').agg({
            'cost_diff_percent': ['count', 'mean', 'min', 'max']
        })
        print(significant_summary)
        
        print("\nTop 5 most extreme differences:")
        print(significant_diff[['case_name', 'nodes', 'directed_ratio', 'mixed1_new_cost', 'mixed2_new_cost', 'cost_diff_percent']].head(5))
    else:
        print("No instances with >10% cost difference found.")
    
    # Performance comparison summary
    print("\nPerformance Comparison Summary:")
    print("-" * 50)
    
    # Success rates
    mixed1_new_success = results['mixed1_new_time'].notna().sum()
    mixed2_new_success = results['mixed2_new_time'].notna().sum()
    total_tests = len(results)
    
    print(f"\nSuccess Rates:")
    print(f"MIXED1: {mixed1_new_success}/{total_tests} ({100*mixed1_new_success/total_tests:.1f}%)")
    print(f"MIXED2: {mixed2_new_success}/{total_tests} ({100*mixed2_new_success/total_tests:.1f}%)")
    
    # Cost and time ratios
    valid_comparisons = results.dropna(subset=['mixed1_new_cost', 'mixed2_new_cost', 'mixed1_new_time', 'mixed2_new_time'])
    if len(valid_comparisons) > 0:
        avg_cost_ratio = (valid_comparisons['mixed2_new_cost'] / valid_comparisons['mixed1_new_cost']).mean()
        avg_time_ratio = (valid_comparisons['mixed2_new_time'] / valid_comparisons['mixed1_new_time']).mean()
        
        print(f"\nAverage Ratios (MIXED2/MIXED1):")
        print(f"Cost ratio: {avg_cost_ratio:.3f}")
        print(f"Time ratio: {avg_time_ratio:.3f}")
    
    plt.show()

if __name__ == "__main__":
    main()
