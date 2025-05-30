import sys, os, time, random
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from algorithms.fleury import fleury
from algorithms.hierholzer import hierholzer

def generate_eulerian_graph(n_nodes, seed=None, density=0.4):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    try:
        G = nx.MultiDiGraph()
        
        # Add nodes and a basic Eulerian cycle
        for i in range(1, n_nodes + 1):
            G.add_node(i)
        
        cycle = list(range(1, n_nodes + 1)) + [1]
        for i in range(len(cycle) - 1):
            G.add_edge(cycle[i], cycle[i+1], kind='directed', weight=random.randint(1, 10))
        
        # Add additional random edges
        edge_factor = density * (0.5 if n_nodes > 20 else 1.0)
        for _ in range(int(n_nodes * n_nodes * edge_factor)):
            u, v = random.randint(1, n_nodes), random.randint(1, n_nodes)
            if u != v:
                G.add_edge(u, v, kind='directed', weight=random.randint(1, 10))
                G.add_edge(v, u, kind='directed', weight=random.randint(1, 10))
        
        # Check Eulerian property
        for node in G.nodes():
            if G.in_degree(node) != G.out_degree(node):
                raise ValueError(f"Graph is not Eulerian: node {node}")
        
        if not nx.is_strongly_connected(G):
            return generate_eulerian_graph(n_nodes, seed + 1 if seed else None)
        
        return G
    except Exception as e:
        print(f"Error generating Eulerian graph: {str(e)}")
        raise

def convert_nx_circuit_format(circuit, graph):
    converted = []
    for u, v in circuit:
        edge_data = graph.get_edge_data(u, v)
        key = list(edge_data.keys())[0]
        data = edge_data[key]
        converted.append((u, v, data.get('kind', 'directed'), data.get('weight', 1)))
    return converted

def validate_tour(tour, graph):
    if not tour:
        return False, "Tour is empty"
    
    for i in range(len(tour) - 1):
        if tour[i][1] != tour[i+1][0]:
            return False, f"Tour not connected at position {i}"
    
    if tour[-1][1] != tour[0][0]:
        return False, "Tour does not close"
    
    if len(tour) != graph.number_of_edges():
        return False, f"Tour contains {len(tour)} edges, but graph has {graph.number_of_edges()} edges"
    
    return True, "Tour is valid"

def run_algorithm(algo_name, graph, repeat=3, timeout=120):
    
    total_time = 0
    tour = None
    success = False
    actual_repeats = repeat
    
    for i in range(repeat):
        try:
            start_time = time.time()
            
            if algo_name == 'fleury':
                # For large graphs, run only once to save time
                if graph.number_of_edges() > 500:
                    actual_repeats = 1
                tour = fleury(graph.copy())
            elif algo_name == 'hierholzer':
                tour = hierholzer(graph.copy())
            elif algo_name == 'networkx':
                nx_circuit = list(nx.eulerian_circuit(graph.copy()))
                elapsed = time.time() - start_time
                
                tour = convert_nx_circuit_format(nx_circuit, graph)
            else:
                tour = None
                
            if algo_name != 'networkx':
                elapsed = time.time() - start_time
            
            if elapsed > timeout:
                print(f"{algo_name} Timeout: {elapsed:.2f}s > {timeout}s")
                return float('inf'), None
                
            total_time += elapsed
            success = True
            
            if elapsed > timeout / 2 and i < repeat - 1:
                actual_repeats = i + 1
                break
                
        except Exception as e:
            print(f"Error with {algo_name}: {str(e)}")
            return float('inf'), None
    
    return (total_time / actual_repeats if success else float('inf')), tour

def run_simulation(node_sizes, repetitions=3, algo_repeat=3):
    results = []
    
    for n_nodes in tqdm(node_sizes, desc="Testing graph sizes"):
        density = 0.2 if n_nodes > 20 else 0.4
            
        for rep in range(repetitions):
            seed = 42 * n_nodes * (rep + 1)
            G = generate_eulerian_graph(n_nodes, seed, density)
            
            result = {
                'nodes': n_nodes,
                'edges': G.number_of_edges(),
                'rep': rep, 'seed': seed,
            }
            
            for algo_name in ['fleury', 'hierholzer', 'networkx']:
                timeout = min(60 + n_nodes * 2, 300) if algo_name == 'fleury' else 60
                avg_time, tour = run_algorithm(algo_name, G, algo_repeat, timeout)
                
                result[f'{algo_name}_time'] = avg_time
                
                if tour:
                    valid, message = validate_tour(tour, G)
                    result[f'{algo_name}_valid'] = valid
                    if not valid:
                        print(f"Invalid tour: {algo_name}, {n_nodes} nodes: {message}")
                else:
                    result[f'{algo_name}_valid'] = False
            
            results.append(result)
    
    return pd.DataFrame(results)

def plot_results(results):
    grouped = results.groupby('edges').agg({
        'nodes': 'mean',
        'fleury_time': 'mean',
        'hierholzer_time': 'mean',
        'networkx_time': 'mean',
    }).reset_index()
    
    std_data = results.groupby('edges').agg({
        'fleury_time': 'std',
        'hierholzer_time': 'std',
        'networkx_time': 'std',
    }).fillna(0).reset_index()
    
    # Runtime comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale
    axes[0].errorbar(grouped['edges'], grouped['fleury_time'], 
                   yerr=std_data['fleury_time'], fmt='o-', label='Fleury')
    axes[0].errorbar(grouped['edges'], grouped['hierholzer_time'], 
                   yerr=std_data['hierholzer_time'], fmt='s-', label='Hierholzer')
    axes[0].errorbar(grouped['edges'], grouped['networkx_time'], 
                   yerr=std_data['networkx_time'], fmt='^-', label='NetworkX')
    axes[0].set_xlabel('Number of Edges')
    axes[0].set_ylabel('Runtime (s)')
    axes[0].set_title('Runtime Comparison (Linear Scale)')
    axes[0].grid(True)
    axes[0].legend()
    
    # Logarithmic scale
    fleury_times = grouped['fleury_time'].replace([float('inf')], np.nan)
    
    axes[1].errorbar(grouped['edges'], fleury_times, 
                   yerr=std_data['fleury_time'], fmt='o-', label='Fleury')
    axes[1].errorbar(grouped['edges'], grouped['hierholzer_time'], 
                   yerr=std_data['hierholzer_time'], fmt='s-', label='Hierholzer')
    axes[1].errorbar(grouped['edges'], grouped['networkx_time'], 
                   yerr=std_data['networkx_time'], fmt='^-', label='NetworkX')
    axes[1].set_xlabel('Number of Edges')
    axes[1].set_ylabel('Runtime (s)')
    axes[1].set_title('Runtime Comparison (Logarithmic Scale)')
    axes[1].set_yscale('log')
    axes[1].grid(True)
    axes[1].legend()
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88) 
    #plt.savefig('euler_algorithms_comparison.png', dpi=300)
    plt.show()

def main():
    print("Starting simulation to compare Eulerian circuit algorithms...")
    
    node_sizes = [5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70 , 80]
    repetitions = 3
    algo_repeat = 3
    
    results = run_simulation(node_sizes, repetitions, algo_repeat)
    
    plot_results(results)
    
    print("\nSummary by graph size (edges):")
    summary = results.groupby('edges').agg({
        'nodes': 'mean',
        'fleury_time': 'mean',
        'hierholzer_time': 'mean',
        'networkx_time': 'mean'
    })
    print(summary)

if __name__ == "__main__":
    main()