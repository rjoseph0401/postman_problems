import sys
import os
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import algorithms
from algorithms.mixed1 import MIXED1
from algorithms.MCPP_BnB import BnBMCPPSolver

def generate_mixed_graph(n_nodes=10, edge_density=0.5, seed=None):
    if seed is not None:
        random.seed(seed)
    
    G = nx.MultiDiGraph()
    costs = {}
    
    for i in range(n_nodes):
        G.add_node(i)
    
    # Ensure graph is connected
    for i in range(n_nodes - 1):
        weight = random.randint(1, 10)
        # Use 50% directed and 50% undirected edges
        if random.random() < 0.5:
            G.add_edge(i, i+1, directed=True, kind='directed', weight=weight)
            costs[(i, i+1)] = weight
        else:
            G.add_edge(i, i+1, directed=False, kind='undirected', weight=weight)
            G.add_edge(i+1, i, directed=False, kind='undirected', weight=weight)
            costs[(i, i+1)] = weight
            costs[(i+1, i)] = weight
    
    weight = random.randint(1, 10)
    if random.random() < 0.5:
        G.add_edge(n_nodes-1, 0, directed=True, kind='directed', weight=weight)
        costs[(n_nodes-1, 0)] = weight
    else:
        G.add_edge(n_nodes-1, 0, directed=False, kind='undirected', weight=weight)
        G.add_edge(0, n_nodes-1, directed=False, kind='undirected', weight=weight)
        costs[(n_nodes-1, 0)] = weight
        costs[(0, n_nodes-1)] = weight
    
    max_possible_edges = n_nodes * (n_nodes - 1)
    target_edges = int(max_possible_edges * edge_density)
    current_edges = n_nodes  
    
    while current_edges < target_edges:
        i = random.randint(0, n_nodes-1)
        j = random.randint(0, n_nodes-1)
        if i != j and not G.has_edge(i, j):
            weight = random.randint(1, 10)
            if random.random() < 0.5:
                G.add_edge(i, j, directed=True, kind='directed', weight=weight)
                costs[(i, j)] = weight
                current_edges += 1
            else:
                G.add_edge(i, j, directed=False, kind='undirected', weight=weight)
                G.add_edge(j, i, directed=False, kind='undirected', weight=weight)
                costs[(i, j)] = weight
                costs[(j, i)] = weight
                current_edges += 1
    
    if seed is not None:
        random.seed()
    
    return G, costs

def calculate_tour_cost(tour, original_graph, costs):
    total_cost = 0
    for u, v in tour:
        weight = None
        
        if original_graph is not None:
            if original_graph.has_edge(u, v):
                edge_data = original_graph.get_edge_data(u, v)
                if isinstance(edge_data, dict) and 0 in edge_data:
                    weight = edge_data[0].get('weight', 1)
                elif isinstance(edge_data, dict) and 'weight' in edge_data:
                    weight = edge_data.get('weight', 1)
            elif original_graph.has_edge(v, u):
                edge_data = original_graph.get_edge_data(v, u)
                if isinstance(edge_data, dict) and 0 in edge_data:
                    weight = edge_data[0].get('weight', 1)
                elif isinstance(edge_data, dict) and 'weight' in edge_data:
                    weight = edge_data.get('weight', 1)
        

        if weight is None:
            weight = costs.get((u, v), costs.get((v, u), 1))
            
        total_cost += weight
    
    return total_cost

def compare_algorithms(n_nodes_range=range(5, 16), num_trials=3, densities=[0.2, 0.4, 0.6], time_limit=300):
    results = []
    for n_nodes in tqdm(n_nodes_range, desc="Graph sizes"):
        for density in densities:
            for trial in range(num_trials):
                try:
                    # Generate a graph
                    seed = 42 + n_nodes * 100 + int(density * 1000) + trial
                    G, costs = generate_mixed_graph(n_nodes, edge_density=density, seed=seed)
                    
                    # Run MIXED1_NEW
                    start_time = time.time()                    
                    try:
                        G1 = G.copy()
                        mixed1_result = MIXED1(G1)
                        mixed1_time = time.time() - start_time
                        
                        if mixed1_result is not None and len(mixed1_result) == 2:
                            mixed1_graph, mixed1_circuit = mixed1_result
                            if mixed1_circuit is not None and len(mixed1_circuit) > 0:
                                mixed1_cost = calculate_tour_cost(mixed1_circuit, G, costs)
                                mixed1_success = True
                            else:
                                mixed1_cost = float('inf')
                                mixed1_success = False
                        else:
                            mixed1_cost = float('inf')                    
                            mixed1_success = False
                    except Exception as e:
                        print(f"Error with MIXED1_NEW on {n_nodes} nodes, density {density}, trial {trial}: {str(e)}")
                        mixed1_circuit = None
                        mixed1_time = time.time() - start_time
                        mixed1_cost = float('inf')
                        mixed1_success = False
                    
                    # Run SimpleMCPPSolver
                    start_time = time.time()
                    try:
                        bnb_solver = BnBMCPPSolver(G, costs, time_limit=time_limit, verbose=False)
                        bnb_solution, bnb_cost = bnb_solver.solve()
                        bnb_time = time.time() - start_time
                        
                        bnb_nodes_explored = bnb_solver.node_counter if hasattr(bnb_solver, 'node_counter') else 0
                        
                        if bnb_solution is not None:
                            if hasattr(bnb_solver, 'convert_solution_to_tour'):
                                bnb_tour = bnb_solver.convert_solution_to_tour(bnb_solution)
                                bnb_success = True
                            else:
                                bnb_tour = None
                                bnb_success = True
                        else:
                            bnb_tour = None
                            bnb_success = False
                            bnb_cost = float('inf')
                    except Exception as e:
                        print(f"Error with SimpleMCPPSolver on {n_nodes} nodes, density {density}, trial {trial}: {str(e)}")
                        bnb_solution = None
                        bnb_tour = None
                        bnb_time = time.time() - start_time
                        bnb_cost = float('inf')
                        bnb_success = False                    
                        bnb_nodes_explored = 0
                    
                    if mixed1_success and bnb_success:
                        relative_cost = mixed1_cost / bnb_cost if bnb_cost > 0 else float('inf')
                        cost_difference = mixed1_cost - bnb_cost
                        cost_ratio = relative_cost
                    else:
                        relative_cost = None
                        cost_difference = None
                        cost_ratio = None
                    
                    n_edges = G.number_of_edges() // 2  # Divide by 2 because we're counting undirected edges as 1
                    n_edges_directed = sum(1 for _, _, data in G.edges(data=True) if data.get('directed', True))
                    n_edges_undirected = sum(1 for _, _, data in G.edges(data=True) if not data.get('directed', True)) // 2
                    actual_density = (n_edges_directed + n_edges_undirected) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
                    
                    # Record results
                    results.append({
                        'n_nodes': n_nodes,
                        'density_target': density,
                        'density_actual': actual_density,
                        'n_edges_total': n_edges_directed + n_edges_undirected,
                        'n_edges_directed': n_edges_directed,
                        'n_edges_undirected': n_edges_undirected,
                        'trial': trial,
                        
                        'mixed1_new_success': mixed1_success,
                        'mixed1_new_cost': mixed1_cost if mixed1_success else None,
                        'mixed1_new_time': mixed1_time,
                        
                        'bnb_success': bnb_success,
                        'bnb_cost': bnb_cost if bnb_success else None,
                        'bnb_time': bnb_time,
                        'bnb_nodes_explored': bnb_nodes_explored,
                        
                        'relative_cost': relative_cost,
                        'cost_difference': cost_difference,
                        'cost_ratio': cost_ratio
                    })
                    
                   
                except Exception as e:
                    print(f"Error in test case (n_nodes={n_nodes}, density={density}, trial={trial}): {str(e)}")
                    results.append({
                        'n_nodes': n_nodes,
                        'density_target': density,
                        'density_actual': 0,
                        'n_edges_total': 0,
                        'n_edges_directed': 0,
                        'n_edges_undirected': 0,
                        'trial': trial,
                        'mixed1_new_success': False,
                        'mixed1_new_cost': None,
                        'mixed1_new_time': 0,
                        'bnb_success': False,
                        'bnb_cost': None,
                        'bnb_time': 0,
                        'bnb_nodes_explored': 0,
                        'relative_cost': None,
                        'cost_difference': None,
                        'cost_ratio': None,
                        'error': str(e)
                    })
    
    df = pd.DataFrame(results)
   
    return df

def plot_results(results):
    """Create plots comparing the algorithms."""
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results
    
    df_clean = df.dropna(subset=['mixed1_new_cost', 'bnb_cost'])
    df_clean = df_clean.replace([float('inf'), float('-inf')], np.nan)
    df_clean = df_clean.dropna(subset=['mixed1_new_cost', 'bnb_cost'])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Runtime vs. Number of Edges
    grouped_by_edges = df_clean.groupby('n_edges_total').mean().reset_index()
    axes[0].plot(grouped_by_edges['n_edges_total'], grouped_by_edges['mixed1_new_time'], 'o-', label='MIXED1', color='blue', linewidth=2)
    axes[0].plot(grouped_by_edges['n_edges_total'], grouped_by_edges['bnb_time'], 's--', label='MCPP BnB', color='red', linewidth=2)
    axes[0].set_xlabel('Number of Edges')
    axes[0].set_ylabel('Runtime (seconds)')
    axes[0].set_title('Runtime vs. Number of Edges')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Solution Cost vs. Number of Edges
    axes[1].plot(grouped_by_edges['n_edges_total'], grouped_by_edges['mixed1_new_cost'], 'o-', label='MIXED1', color='blue', linewidth=2)
    axes[1].plot(grouped_by_edges['n_edges_total'], grouped_by_edges['bnb_cost'], 's--', label='MCPP BnB', color='red', linewidth=2)
    axes[1].set_xlabel('Number of Edges')
    axes[1].set_ylabel('Tour Cost')
    axes[1].set_title('Tour Cost vs. Number of Edges')
    axes[1].legend()
    axes[1].grid(True)

    # 3. Tour Cost vs. Edge Density
    grouped_by_density = df_clean.groupby('density_target').mean().reset_index()
    axes[2].plot(grouped_by_density['density_target'], grouped_by_density['mixed1_new_cost'], 'o-', label='MIXED1', color='blue', linewidth=2)
    axes[2].plot(grouped_by_density['density_target'], grouped_by_density['bnb_cost'], 's--', label='MCPP BnB', color='red', linewidth=2)
    axes[2].set_xlabel('Edge Density')
    axes[2].set_ylabel('Tour Cost')
    axes[2].set_title('Tour Cost vs. Edge Density')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    output_file = f'mixed1_bnb_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    #plt.savefig(output_file, dpi=300)
    #print(f"Plot saved to {output_file}")
    plt.show()

def plot_extended_results(results):
    """Create extended plots with more analysis - removed to keep code concise."""
    pass

if __name__ == "__main__":
    print("Comparing MIXED1 and MCPP BnB algorithms")
    print("=========================================")
    
    n_nodes_range = range(5, 11
                          )  
    num_trials = 3
    densities = [0.2, 0.4, 0.6]
    time_limit = 300  
      # Run comparison
    results_df = compare_algorithms(
        n_nodes_range=n_nodes_range,
        num_trials=num_trials,
        densities=densities,
        time_limit=time_limit
    )
    
    # Create plots
    plot_results(results_df)
    
    print("\nSummary Statistics:")
    print("===================")
    
    # Success rates
    mixed1_new_success_rate = results_df['mixed1_new_success'].mean() * 100
    bnb_success_rate = results_df['bnb_success'].mean() * 100
    print(f"Success rate: MIXED1: {mixed1_new_success_rate:.1f}%, MCPP BnB: {bnb_success_rate:.1f}%")
    
    # Average runtime for successful instances
    mixed1_new_avg_time = results_df[results_df['mixed1_new_success']]['mixed1_new_time'].mean()
    bnb_avg_time = results_df[results_df['bnb_success']]['bnb_time'].mean()
    
    print(f"Avg runtime (successful only): MIXED1: {mixed1_new_avg_time:.2f}s, MCPP BnB: {bnb_avg_time:.2f}s")
