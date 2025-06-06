import sys
import os
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from algorithms.mixed1 import MIXED1
from algorithms.mixed2 import MIXED2

def generate_mixed_graph(n_nodes=10, p_directed=0.3, p_undirected=0.4, seed=None):
    if seed is not None:
        random.seed(seed)
    
    G = nx.MultiDiGraph()
    
    for i in range(n_nodes):
        G.add_node(i)
    
    # Add edges with a mix of directed and undirected
    for i in range(n_nodes - 1):
        is_directed = random.random() < p_directed
        weight = random.randint(1, 10)
        if is_directed:
            G.add_edge(i, i+1, kind='directed', weight=weight)
        else:
            G.add_edge(i, i+1, kind='undirected', weight=weight)
            G.add_edge(i+1, i, kind='undirected', weight=weight)
    
    is_directed = random.random() < p_directed
    weight = random.randint(1, 10)
    if is_directed:
        G.add_edge(n_nodes-1, 0, kind='directed', weight=weight)
    else:
        G.add_edge(n_nodes-1, 0, kind='undirected', weight=weight)
        G.add_edge(0, n_nodes-1, kind='undirected', weight=weight)
    
    # Add additional random edges
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and not G.has_edge(i, j):
                if random.random() < p_directed:
                    G.add_edge(i, j, kind='directed', weight=random.randint(1, 10))
                
                if random.random() < p_undirected:
                    weight = random.randint(1, 10)
                    G.add_edge(i, j, kind='undirected', weight=weight)
                    G.add_edge(j, i, kind='undirected', weight=weight)
    
    directed_count = sum(1 for _, _, data in G.edges(data=True) if data.get('kind') == 'directed')
    undirected_count = sum(1 for _, _, data in G.edges(data=True) if data.get('kind') == 'undirected') // 2 
    total_edges = directed_count + undirected_count
    directed_ratio = directed_count / total_edges if total_edges > 0 else 0
    
    return G, directed_ratio

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
        if isinstance(circuit[0], tuple) and len(circuit[0]) >= 4:
            return sum(weight for _, _, _, weight in circuit)
        elif isinstance(circuit[0], tuple) and len(circuit[0]) == 2:
            if original_graph is not None:
                total_cost = 0
                for u, v in circuit:
                    weight = 1  
                    if original_graph.has_edge(u, v):
                        edge_data = original_graph.get_edge_data(u, v)
                        if isinstance(edge_data, dict):
                            if 0 in edge_data:  
                                weight = edge_data[0].get('weight', 1)
                            else:
                                weight = edge_data.get('weight', 1)
                    elif original_graph.has_edge(v, u):
                        edge_data = original_graph.get_edge_data(v, u)
                        if isinstance(edge_data, dict):
                            if 0 in edge_data:  
                                weight = edge_data[0].get('weight', 1)
                            else:
                                weight = edge_data.get('weight', 1)
                    total_cost += weight
                return total_cost
            else:
                return len(circuit)  
    
    return float('inf')

def run_simulation(node_sizes, repetitions=3, directed_probs=None, undirected_prob=0.4):
    results = []
    
    if directed_probs is None:
        directed_probs = [0.3] 
    
    for n_nodes in tqdm(node_sizes, desc="Testing graph sizes"):
        for p_directed in directed_probs:
            for rep in range(repetitions):
                # Set a reproducible seed
                seed = 42 * n_nodes * int(p_directed*10) * (rep + 1)
                
                # Generate graph
                G, actual_dir_ratio = generate_mixed_graph(n_nodes, p_directed, undirected_prob, seed)
                
                result = {
                    'nodes': n_nodes,
                    'edges': G.number_of_edges() // 2,  
                    'directed_prob': p_directed,
                    'actual_dir_ratio': actual_dir_ratio,
                    'rep': rep,
                    'seed': seed,
                }
                  # Test MIXED1_NEW
                try:
                    G1 = G.copy()
                    start_time = time.time()
                    circuit1 = MIXED1(G1)
                    mixed1_new_time = time.time() - start_time
                    
                    result['mixed1_new_time'] = mixed1_new_time
                    result['mixed1_new_cost'] = calculate_cost(circuit1, G)
                except Exception as e:
                    print(f"MIXED1_NEW error with {n_nodes} nodes, dir_prob={p_directed}, rep {rep}: {str(e)}")
                    result['mixed1_new_time'] = None
                    result['mixed1_new_cost'] = None
                
                # Test MIXED2_NEW
                try:
                    G2 = G.copy()
                    start_time = time.time()
                    circuit2 = MIXED2(G2)
                    mixed2_new_time = time.time() - start_time
                    
                    result['mixed2_new_time'] = mixed2_new_time
                    result['mixed2_new_cost'] = calculate_cost(circuit2, G)
                except Exception as e:
                    print(f"MIXED2_NEW error with {n_nodes} nodes, dir_prob={p_directed}, rep {rep}: {str(e)}")
                    result['mixed2_new_time'] = None
                    result['mixed2_new_cost'] = None
                
                # Calculate relative performance
                if result['mixed1_new_cost'] and result['mixed2_new_cost']:
                    if result['mixed1_new_cost'] > 0:
                        result['cost_ratio'] = result['mixed2_new_cost'] / result['mixed1_new_cost']
                    if result['mixed1_new_time'] > 0:
                        result['time_ratio'] = result['mixed2_new_time'] / result['mixed1_new_time']
                
                results.append(result)
    
    return pd.DataFrame(results)

def plot_results(results):
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results
    grouped_by_edges = df.groupby('edges').mean().reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot execution times
    axes[0,0].plot(grouped_by_edges['edges'], grouped_by_edges['mixed1_new_time'], 'o-', label='MIXED1', color='blue')
    axes[0,0].plot(grouped_by_edges['edges'], grouped_by_edges['mixed2_new_time'], 's--', label='MIXED2', color='red')
    axes[0,0].set_xlabel('Number of Edges')
    axes[0,0].set_ylabel('Runtime (seconds)')
    axes[0,0].set_title('Runtime Comparison by Number of Edges')    
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Plot solution costs
    axes[0,1].plot(grouped_by_edges['edges'], grouped_by_edges['mixed1_new_cost'], 'o-', label='MIXED1', color='blue')
    axes[0,1].plot(grouped_by_edges['edges'], grouped_by_edges['mixed2_new_cost'], 's--', label='MIXED2', color='red')
    axes[0,1].set_xlabel('Number of Edges')
    axes[0,1].set_ylabel('Tour Cost')
    axes[0,1].set_title('Tour Cost Comparison by Number of Edges')
    axes[0,1].legend()
    axes[0,1].grid(True)
    df['dir_ratio_bucket'] = np.round(df['actual_dir_ratio'] * 10) / 10
    grouped_by_ratio = df.groupby('dir_ratio_bucket').mean().reset_index()
    
    # Plot execution times by directed edge ratio
    axes[1,0].plot(grouped_by_ratio['dir_ratio_bucket'], grouped_by_ratio['mixed1_new_time'], 'o-', label='MIXED1', color='blue')
    axes[1,0].plot(grouped_by_ratio['dir_ratio_bucket'], grouped_by_ratio['mixed2_new_time'], 's--', label='MIXED2', color='red')
    axes[1,0].set_xlabel('Proportion of Directed Edges')
    axes[1,0].set_ylabel('Runtime (seconds)')
    axes[1,0].set_title('Runtime by Proportion of Directed Edges')
    axes[1,0].legend()
    axes[1,0].grid(True)
      # Plot cost ratio by directed edge ratio (MIXED2/MIXED1)
    if 'cost_ratio' in grouped_by_ratio.columns:
        axes[1,1].plot(grouped_by_ratio['dir_ratio_bucket'], grouped_by_ratio['cost_ratio'], 'o-', color='purple')
        axes[1,1].axhline(y=1.0, color='gray', linestyle='--')
        axes[1,1].set_xlabel('Proportion of Directed Edges')
        axes[1,1].set_ylabel('Cost Ratio (MIXED2/MIXED1)')
        axes[1,1].set_title('Tour Cost Ratio by Edge Type')
        axes[1,1].grid(True)
    else:
        axes[1,1].plot(grouped_by_ratio['dir_ratio_bucket'], grouped_by_ratio['mixed1_new_cost'], 'o-', label='MIXED1', color='blue')
        axes[1,1].plot(grouped_by_ratio['dir_ratio_bucket'], grouped_by_ratio['mixed2_new_cost'], 's--', label='MIXED2', color='red')
        axes[1,1].set_xlabel('Proportion of Directed Edges')
        axes[1,1].set_ylabel('Tour Cost')
        axes[1,1].set_title('Tour Cost by Proportion of Directed Edges')
        axes[1,1].legend()
        axes[1,1].grid(True)
    plt.tight_layout()
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'mixed1_new_mixed2_new_comparison_{timestamp}.png'
    plt.show()
    
    return filename

def main():
    print("Starting MIXED1_NEW vs MIXED2_NEW Algorithm Comparison")
    print("=" * 60)
    
    node_sizes = [5, 8, 10, 12, 15, 20, 25, 30, 40 ,50]
    directed_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    repetitions = 3
    
    # Run simulation
    results = run_simulation(node_sizes, repetitions, directed_probs=directed_probs)
    
    # Create plots
    plot_filename = plot_results(results)
    
    # Print summary statistics
    print("\nSummary by edge count:")
    summary_by_edges = results.groupby('edges').agg({
        'mixed1_new_time': ['mean'],
        'mixed2_new_time': ['mean'],
        'mixed1_new_cost': ['mean'],
        'mixed2_new_cost': ['mean'],
    })
    print(summary_by_edges)
    
    print("\nSummary by directed edge ratio:")
    summary_by_ratio = results.groupby('dir_ratio_bucket').agg({
        'mixed1_new_time': ['mean', 'std'],
        'mixed2_new_time': ['mean', 'std'],
        'mixed1_new_cost': ['mean', 'std'],
        'mixed2_new_cost': ['mean', 'std'],
    })
    print(summary_by_ratio)
    
    print("\nPerformance Comparison Summary:")
    print("-" * 40)
    
    # Overall averages
    overall_stats = results.agg({
        'mixed1_new_time': ['mean', 'std'],
        'mixed2_new_time': ['mean', 'std'],
        'mixed1_new_cost': ['mean', 'std'],
        'mixed2_new_cost': ['mean', 'std'],
    })
    print("Overall Statistics:")
    print(overall_stats)
    
      # Cost and time ratios
    valid_comparisons = results.dropna(subset=['mixed1_new_cost', 'mixed2_new_cost', 'mixed1_new_time', 'mixed2_new_time'])
    if len(valid_comparisons) > 0:
        avg_cost_ratio = (valid_comparisons['mixed2_new_cost'] / valid_comparisons['mixed1_new_cost']).mean()
        avg_time_ratio = (valid_comparisons['mixed2_new_time'] / valid_comparisons['mixed1_new_time']).mean()
        
        print(f"\nAverage Ratios (MIXED2/MIXED1):")
        print(f"Cost ratio: {avg_cost_ratio:.3f}")
        print(f"Time ratio: {avg_time_ratio:.3f}")

if __name__ == "__main__":
    main()
