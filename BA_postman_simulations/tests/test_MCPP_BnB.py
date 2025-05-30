import networkx as nx
import os
import sys

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from algorithms.MCPP_BnB import BnBMCPPSolver


if __name__ == "__main__":
    G = nx.MultiDiGraph()
    costs = {}
    
    G.add_nodes_from(range(5))
    G.add_edge(0, 1, directed=True)
    costs[(0, 1)] = 2
    G.add_edge(1, 2, directed=True)
    costs[(1, 2)] = 3
    G.add_edge(2, 3, directed=True)
    costs[(2, 3)] = 1
    G.add_edge(3, 4, directed=True)
    costs[(3, 4)] = 2
    G.add_edge(4, 0, directed=True)
    costs[(4, 0)] = 1
    
    G.add_edge(0, 2, directed=False)
    costs[(2, 0)] = 4 
    costs[(0, 2)] = 4

    print("\nRunning SimpleMCPPSolver Algorithm...")
    try:
        solver = BnBMCPPSolver(G, costs, time_limit=60, verbose=True)
        solution, objective = solver.solve()
        
        if solution is not None:
            print(f"\nFound optimal solution with cost: {objective:.2f}")
            print("Test PASSED")
        else:
            print("\nNo valid solution found")
            print("Test FAILED")
    except Exception as e:
        print(f"\nError: {e}")
        print("Test FAILED")

