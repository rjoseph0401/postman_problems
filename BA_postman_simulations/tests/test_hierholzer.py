import networkx as nx
import os
import sys

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from algorithms.hierholzer import hierholzer


if __name__ == "__main__":
    G = nx.MultiDiGraph()
    
    G.add_nodes_from(range(5))
    
    #ensure G is Eulerian
    G.add_edge(0, 1, kind='directed')
    G.add_edge(1, 2, kind='directed')
    G.add_edge(2, 3, kind='directed')
    G.add_edge(3, 4, kind='directed')
    G.add_edge(4, 0, kind='directed')
    G.add_edge(0, 2, kind='directed')
    G.add_edge(2, 0, kind='directed')
    G.add_edge(1, 3, kind='directed')
    G.add_edge(3, 1, kind='directed')
    G.add_edge(2, 4, kind='directed')
    G.add_edge(4, 2, kind='directed')

    print("\nRunning Hierholzer's Algorithm...")
    try:
        tour = hierholzer(G)
        if tour:
            print("\nFound Eulerian Circuit:")
            print(" -> ".join(f"({u}, {v}, {kind})" for u, v, kind, _ in tour))
        else:
            print("\nNo valid tour found")
    except ValueError as e:
        print(f"\nError: {e}")
