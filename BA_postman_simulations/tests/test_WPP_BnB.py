import networkx as nx
import os
import sys

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from algorithms.WPP_BnB import BnBWPPSolver

if __name__ == "__main__":
    G = nx.Graph()
    costs = {}
    
    G.add_nodes_from(range(3))

    G.add_edge(0, 1)
    costs[(0, 1)] = 2
    costs[(1, 0)] = 3
    
    G.add_edge(1, 2)
    costs[(1, 2)] = 3
    costs[(2, 1)] = 1

    G.add_edge(2, 0)
    costs[(2, 0)] = 4
    costs[(0, 2)] = 4

    
    
    print("\nRunning SimpleWPPSolver Algorithm...")
    try:
        solver = BnBWPPSolver(G, costs, time_limit=60, verbose=True)
        solution, objective = solver.solve()
        
        if solution is not None:
            print(f"\nFound solution with cost: {objective:.2f}")
        else:
            print("\nNo valid solution found")
            print("Test FAILED")
    except Exception as e:
        print(f"\nError: {e}")
        print("Test FAILED")