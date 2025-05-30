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
from algorithms.Win_WPP_General import Win_WPP_General
from algorithms.WPP_BnB import BnBWPPSolver

def generate_windy_graph(n_nodes, seed=None):
    if seed is not None:
        random.seed(seed)
    
    G = nx.Graph()
    costs = {}
    
    for i in range(n_nodes):
        G.add_node(i)
    
    edges = []
    
    # ensure graph is connected
    for i in range(1, n_nodes):
        j = random.randrange(i)
        weight_forward = random.randint(1, 10)
        weight_backward = random.randint(1, 10)
        edges.append((i, j))
        G.add_edge(i, j, cij=weight_forward, cji=weight_backward)
        costs[(i, j)] = weight_forward
        costs[(j, i)] = weight_backward
    
    # add some additional random edges
    possible_edges = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes) if (i, j) not in edges]
    num_extra_edges = min(n_nodes, len(possible_edges))
    for i, j in random.sample(possible_edges, num_extra_edges):
        weight_forward = random.randint(1, 10)
        weight_backward = random.randint(1, 10)
        G.add_edge(i, j, cij=weight_forward, cji=weight_backward)
        costs[(i, j)] = weight_forward
        costs[(j, i)] = weight_backward
    
    if seed is not None:
        random.seed()
    
    return G, costs

def compare_algorithms(n_nodes_range=range(5, 16), num_trials=3, time_limit=300):
    results = []
    
    for n_nodes in tqdm(n_nodes_range, desc="Graph sizes"):
        for trial in range(num_trials):
            # Generate a graph
            seed = 999 + n_nodes * 100 + trial
            G, costs = generate_windy_graph(n_nodes, seed=seed)
            
            # Run Win_WPP_General
            start_time = time.time()
            try:
                G_copy = G.copy()
                for u, v in G_copy.edges():
                    G_copy[u][v]['weight_forward'] = costs.get((u, v), 1)
                    G_copy[u][v]['weight_backward'] = costs.get((v, u), 1)
                win_result = Win_WPP_General(G_copy)
                win_time = time.time() - start_time
                
                if win_result is not None:
                    win_circuit, win_cost = win_result
                    win_success = True
                else:
                    win_circuit = None
                    win_cost = float('inf')
                    win_success = False
            except Exception as e:
                print(f"Error with Win_WPP_General on {n_nodes} nodes, trial {trial}: {str(e)}")
                win_circuit = None
                win_time = time.time() - start_time
                win_cost = float('inf')
                win_success = False
            
            # Run SimpleWPPSolver
            start_time = time.time()
            try:
                bnb_solver = BnBWPPSolver(G, costs, time_limit=time_limit, verbose=False)
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
                print(f"Error with SimpleWPPSolver on {n_nodes} nodes, trial {trial}: {str(e)}")
                bnb_solution = None
                bnb_tour = None
                bnb_time = time.time() - start_time
                bnb_cost = float('inf')
                bnb_success = False
                bnb_nodes_explored = 0
            
            # Compare costs and calculate performance metrics
            if win_success and bnb_success:
                relative_cost = win_cost / bnb_cost if bnb_cost > 0 else float('inf')
                cost_difference = win_cost - bnb_cost
                cost_ratio = relative_cost
            else:
                relative_cost = None
                cost_difference = None
                cost_ratio = None
            
            n_edges = len(G.edges())
            n_nodes_actual = len(G.nodes())
            density = 2 * n_edges / (n_nodes_actual * (n_nodes_actual - 1)) if n_nodes_actual > 1 else 0
            
            # Record results
            results.append({
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'density': density,
                'trial': trial,
                
                'win_success': win_success,
                'win_cost': win_cost if win_success else None,
                'win_time': win_time,
                
                'bnb_success': bnb_success,
                'bnb_cost': bnb_cost if bnb_success else None,
                'bnb_time': bnb_time,
                'bnb_nodes_explored': bnb_nodes_explored,
                
                'relative_cost': relative_cost,
                'cost_difference': cost_difference,
                'cost_ratio': cost_ratio
            })
            
    
    df = pd.DataFrame(results)
    
    return df

def plot_results(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Runtime vs. Number of Edges
    ax1 = axes[0]
    runtime_by_edges = df.groupby('n_edges').agg({
        'win_time': 'mean',
        'bnb_time': 'mean'
    })    
    runtime_by_edges.plot(y=['win_time', 'bnb_time'], 
                         ax=ax1, marker='o', linewidth=2, markersize=6)
    ax1.set_title('Runtime vs. Number of Edges')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_xlabel('Number of Edges')
    ax1.legend(['Win General', 'WPP BnB'])
    ax1.grid(True, alpha=0.3)
    
    # 2. Tour Cost vs. Number of Edges
    ax2 = axes[1]
    cost_data = df[(df['win_success']) & (df['bnb_success'])]
    if not cost_data.empty:
        cost_by_edges = cost_data.groupby('n_edges').agg({
            'win_cost': 'mean',
            'bnb_cost': 'mean'
        })        
        cost_by_edges.plot(y=['win_cost', 'bnb_cost'], 
                          ax=ax2, marker='o', linewidth=2, markersize=6)
        ax2.set_title('Tour Cost vs. Number of Edges')
        ax2.set_ylabel('Tour Cost')
        ax2.set_xlabel('Number of Edges')
        ax2.legend(['Win General', 'WPP BnB'])
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No comparable cost data available", 
                 horizontalalignment='center', verticalalignment='center')
        ax2.set_title('Tour Cost vs. Number of Edges')
        ax2.set_xlabel('Number of Edges')
        ax2.set_ylabel('Tour Cost')
    
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    print("Comparing Win_WPP_General and SimpleWPPSolver algorithms")
    print("========================================================")
    
    n_nodes_range = range(5, 13) 
    num_trials = 3
    time_limit = 300  
    
    # Run comparison
    results_df = compare_algorithms(
        n_nodes_range=n_nodes_range,
        num_trials=num_trials,
        time_limit=time_limit
    )
    
    plot_results(results_df)
    
    print("\nSummary Statistics:")
    print("===================")

    win_success_rate = results_df['win_success'].mean() * 100
    bnb_success_rate = results_df['bnb_success'].mean() * 100
    
    print(f"Success rate: Win_WPP_General: {win_success_rate:.1f}%, SimpleWPPSolver: {bnb_success_rate:.1f}%")
    
    win_avg_time = results_df[results_df['win_success']]['win_time'].mean()
    bnb_avg_time = results_df[results_df['bnb_success']]['bnb_time'].mean()
    
    print(f"Avg runtime (successful only): Win_WPP_General: {win_avg_time:.2f}s, SimpleWPPSolver: {bnb_avg_time:.2f}s")
    

