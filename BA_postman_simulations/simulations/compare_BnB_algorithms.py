import sys
import os
import time
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from algorithms.WPP_BnB import BnBWPPSolver
from algorithms.MCPP_BnB import BnBMCPPSolver
from bnb_comparison_plots import create_four_part_comparison

# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def create_comparable_graphs(n_nodes, edge_factor=1.3, seed=None, max_edges=60):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Scale edges with graph size but cap at max_edges
    edge_scaling = 1.0 + (n_nodes / 15) 
    n_edges = min(max_edges, max(n_nodes, int(n_nodes * edge_factor * edge_scaling)))
    
    # Create a windy graph
    G_windy = nx.Graph()
    G_windy.add_nodes_from(range(1, n_nodes + 1))
    
    wpp_costs = {}
    mcpp_costs = {}
    
    # ensure connectivity
    for i in range(1, n_nodes):
        cij = random.randint(1, 10)
        cji = random.randint(1, 10)
        G_windy.add_edge(i, i+1, cij=cij, cji=cji)
        wpp_costs[(i, i+1)] = cij
        wpp_costs[(i+1, i)] = cji
    
    cij = random.randint(1, 10)
    cji = random.randint(1, 10)
    G_windy.add_edge(n_nodes, 1, cij=cij, cji=cji)
    wpp_costs[(n_nodes, 1)] = cij
    wpp_costs[(1, n_nodes)] = cji
    
    # Add remaining edges
    edges_to_add = n_edges - n_nodes
    attempts = 0
    while edges_to_add > 0 and attempts < edges_to_add * 10:
        u = random.randint(1, n_nodes)
        v = random.randint(1, n_nodes)
        attempts += 1
        
        if u != v and not G_windy.has_edge(u, v):
            cij = random.randint(1, 10)
            cji = random.randint(1, 10)
            G_windy.add_edge(u, v, cij=cij, cji=cji)
            wpp_costs[(u, v)] = cij
            wpp_costs[(v, u)] = cji
            edges_to_add -= 1
    
    # Create equivalent mixed graph with costs
    G_mixed = nx.MultiGraph()
    G_mixed.add_nodes_from(G_windy.nodes())
    
    for u, v, data in G_windy.edges(data=True):
        cost_diff = abs(data['cij'] - data['cji'])
        cost_ratio = cost_diff / (max(data['cij'], data['cji']) + 0.1) 
        
        if cost_ratio > 0.5 and random.random() < 0.7:  # 70% chance if costs are asymmetric
            if data['cij'] < data['cji']:
                G_mixed.add_edge(u, v, directed=True, weight=data['cij'])
                mcpp_costs[(u, v)] = data['cij']
            else:
                G_mixed.add_edge(v, u, directed=True, weight=data['cji'])
                mcpp_costs[(v, u)] = data['cji']
        else:
            avg_cost = (data['cij'] + data['cji']) / 2
            G_mixed.add_edge(u, v, directed=False, weight=avg_cost)
            mcpp_costs[(u, v)] = avg_cost
            mcpp_costs[(v, u)] = avg_cost
    
    # Reset random seed
    if seed is not None:
        random.seed()
        np.random.seed()
        
    return G_windy, G_mixed, wpp_costs, mcpp_costs

def run_comparison(node_sizes=[4, 5, 6, 7,8], time_limit=300, trials=2, max_edges=50):
    """
    Run WPP and MCPP algorithms on graphs of different sizes.
    """
    results = []
    
    for n_nodes in node_sizes:
        print(f"\n{'='*50}")
        print(f"Testing graphs with {n_nodes} nodes")
        print(f"{'='*50}")
        
        for trial in range(trials):
            print(f"\nTrial {trial+1}/{trials}")
              # Create comparable graphs
            seed_value = SEED + n_nodes + trial*100
            edge_factor = 1.5 if n_nodes < 10 else 2.0
            G_windy, G_mixed, wpp_costs, mcpp_costs = create_comparable_graphs(
                n_nodes, edge_factor=edge_factor, seed=seed_value, max_edges=50
            )
            
            n_edges_windy = len(G_windy.edges())
            n_edges_mixed = len(G_mixed.edges())
            n_directed = sum(1 for _, _, data in G_mixed.edges(data=True) if data.get('directed', False))
            n_undirected = n_edges_mixed - n_directed
            
            print(f"Generated graphs with {n_nodes} nodes:")
            print(f"  - Windy graph: {n_edges_windy} edges")
            print(f"  - Mixed graph: {n_edges_mixed} edges ({n_directed} directed, {n_undirected} undirected)")
            
            # Run WPP solver
            print("Running SimpleWPPSolver...")
            wpp_start = time.time()
            try:
                solver_wpp = BnBWPPSolver(G_windy, wpp_costs, time_limit=time_limit, verbose=False)
                wpp_solution, wpp_cost = solver_wpp.solve()
                wpp_time = time.time() - wpp_start
                wpp_success = wpp_solution is not None
                
                # Get node exploration count
                wpp_nodes_explored = solver_wpp.node_counter if hasattr(solver_wpp, 'node_counter') else -1
                
                if wpp_success:
                    print(f"✓ WPP solution found with cost {wpp_cost:.2f} in {wpp_time:.2f}s")
                    print(f"  - Nodes explored: {wpp_nodes_explored}")
                    print(f"  - Exploration rate: {wpp_nodes_explored / wpp_time:.1f} nodes/sec")
                else:
                    print("✗ No WPP solution found within time limit")
                    print(f"  - Time spent: {wpp_time:.2f}s")
                    print(f"  - Nodes explored: {wpp_nodes_explored}")
                    wpp_valid = False
            except Exception as e:
                print(f"✗ WPP error: {str(e)}")
                wpp_solution, wpp_cost = None, float('inf')
                wpp_time = time.time() - wpp_start
                wpp_success = False
                wpp_valid = False
                wpp_nodes_explored = -1
            
            # Run MCPP solver
            print("Running SimpleMCPPSolver...")
            mcpp_start = time.time()
            try:
                solver_mcpp = BnBMCPPSolver(G_mixed, mcpp_costs, time_limit=time_limit, verbose=False)
                mcpp_solution, mcpp_cost = solver_mcpp.solve()
                mcpp_time = time.time() - mcpp_start
                mcpp_success = mcpp_solution is not None
                
                # Get node exploration count
                mcpp_nodes_explored = solver_mcpp.node_counter if hasattr(solver_mcpp, 'node_counter') else -1
                
                if mcpp_success:
                    mcpp_valid = True
                    if hasattr(solver_mcpp, 'convert_solution_to_tour'):
                        mcpp_tour = solver_mcpp.convert_solution_to_tour(mcpp_solution)
                        mcpp_valid = mcpp_tour is not None and len(mcpp_tour) > 0
                    
                    print(f"✓ MCPP solution found with cost {mcpp_cost:.2f} in {mcpp_time:.2f}s")
                    print(f"  - Nodes explored: {mcpp_nodes_explored}")
                    print(f"  - Exploration rate: {mcpp_nodes_explored / mcpp_time:.1f} nodes/sec")
                    print(f"  - Valid solution: {mcpp_valid}")
                else:
                    print("✗ No MCPP solution found within time limit")
                    print(f"  - Time spent: {mcpp_time:.2f}s")
                    print(f"  - Nodes explored: {mcpp_nodes_explored}")
                    mcpp_valid = False
            except Exception as e:
                print(f"✗ MCPP error: {str(e)}")
                mcpp_solution, mcpp_cost = None, float('inf')
                mcpp_time = time.time() - mcpp_start
                mcpp_success = False
                mcpp_valid = False
                mcpp_nodes_explored = -1
            
            # Calculate graph density
            max_possible_edges = n_nodes * (n_nodes - 1) // 2
            wpp_density = n_edges_windy / max_possible_edges if max_possible_edges > 0 else 0
            mcpp_density = n_edges_mixed / max_possible_edges if max_possible_edges > 0 else 0
            
            # Collect result data
            result = {
                'nodes': n_nodes,
                'edges_wpp': n_edges_windy,
                'edges_mcpp': n_edges_mixed,
                'directed_edges': n_directed,
                'undirected_edges': n_undirected,
                'density_wpp': wpp_density,
                'density_mcpp': mcpp_density,
                'trial': trial + 1,
                
                'wpp_success': wpp_success,
                'wpp_cost': wpp_cost if wpp_success else None,
                'wpp_time': wpp_time,
                'wpp_nodes': wpp_nodes_explored,
                'wpp_rate': wpp_nodes_explored / wpp_time if wpp_time > 0 else 0,
                
                'mcpp_success': mcpp_success,
                'mcpp_cost': mcpp_cost if mcpp_success else None,
                'mcpp_time': mcpp_time,
                'mcpp_nodes': mcpp_nodes_explored,
                'mcpp_rate': mcpp_nodes_explored / mcpp_time if mcpp_time > 0 else 0,
                'mcpp_valid': mcpp_valid
            }
            
            results.append(result)
            
    df = pd.DataFrame(results)
    
    # Convert data structure for plotting function
    plot_data = []
    for _, row in df.iterrows():        
        wpp_exploration_rate = row['wpp_nodes'] / row['wpp_time'] if row['wpp_time'] > 0 else 0
        plot_data.append({
            'algorithm': 'WPP BnB',
            'n_nodes': row['nodes'],
            'n_edges': row['edges_wpp'],
            'density': row['density_wpp'],
            'success': row['wpp_success'],
            'cost': row['wpp_cost'],
            'runtime': row['wpp_time'],
            'nodes_explored': row['wpp_nodes'],
            'exploration_rate': wpp_exploration_rate,
            'trial': row['trial']
        })
        
        # MCPP data
        mcpp_exploration_rate = row['mcpp_nodes'] / row['mcpp_time'] if row['mcpp_time'] > 0 else 0
        plot_data.append({
            'algorithm': 'MCPP BnB',
            'n_nodes': row['nodes'],
            'n_edges': row['edges_mcpp'],
            'density': row['density_mcpp'],
            'success': row['mcpp_success'],
            'cost': row['mcpp_cost'],
            'runtime': row['mcpp_time'],
            'nodes_explored': row['mcpp_nodes'],
            'exploration_rate': mcpp_exploration_rate,
            'trial': row['trial']
        })
    plot_df = pd.DataFrame(plot_data)
    
    # Plot results
    fig = create_four_part_comparison(plot_df)
    plt.show()
    
    return df

if __name__ == "__main__":
    print("Running comparison of Simple WPP and MCPP Branch and Bound solvers")
    print("====================================================================")
    
    # Configure test parameters
    node_sizes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    time_limit = 300  
    trials = 3
    
    # Run comparison
    results = run_comparison(
        node_sizes=node_sizes,
        time_limit=time_limit,
        trials=trials
    )
    
    # Calculate stats
    print("\n\nSummary Statistics:")
    print("===================")
    
    # Success rates
    wpp_success_rate = results['wpp_success'].mean() * 100
    mcpp_success_rate = results['mcpp_success'].mean() * 100
    
    print(f"Success rate: SimpleWPPSolver: {wpp_success_rate:.1f}%, SimpleMCPPSolver: {mcpp_success_rate:.1f}%")
    
    # Average time for successful instances
    wpp_avg_time = results[results['wpp_success']]['wpp_time'].mean()
    mcpp_avg_time = results[results['mcpp_success']]['mcpp_time'].mean()
    
    print(f"Avg time (successful only): SimpleWPPSolver: {wpp_avg_time:.2f}s, SimpleMCPPSolver: {mcpp_avg_time:.2f}s")
    
    # Average nodes explored
    wpp_avg_nodes = results[results['wpp_success']]['wpp_nodes'].mean()
    mcpp_avg_nodes = results[results['mcpp_success']]['mcpp_nodes'].mean()
    
    print(f"Avg nodes explored: SimpleWPPSolver: {wpp_avg_nodes:.1f}, SimpleMCPPSolver: {mcpp_avg_nodes:.1f}")
