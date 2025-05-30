import networkx as nx
import os
import sys

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from algorithms.Win_WPP_General import Win_WPP_General


if __name__ == "__main__":
    G = nx.Graph()
    
    G.add_edge(0, 1, cij=2, cji=3)
    G.add_edge(1, 2, cij=1, cji=4)
    G.add_edge(2, 3, cij=3, cji=2)
    G.add_edge(3, 0, cij=2, cji=5)
    G.add_edge(1, 3, cij=4, cji=3)
    G.add_edge(0, 2, cij=3, cji=3)
    G.add_edge(2, 4, cij=1, cji=2)
    G.add_edge(4, 0, cij=2, cji=1)
    G.add_edge(4, 1, cij=3, cji=2)
    G.add_edge(4, 3, cij=2, cji=3)
    G.add_edge(5, 6, cij=1, cji=2)
    G.add_edge(4,5, cij=3, cji=4)
    G.add_edge(7, 8, cij=2, cji=3)
    G.add_edge(4, 7, cij=1, cji=2)
    G.add_edge(8, 5, cij=3, cji=1)
    G.add_edge(6, 7, cij=2, cji=4)

    print("\nRunning Win's Algorithm for General Graphs...")
    tour = Win_WPP_General(G)
    
    if tour:
        print("\nFound Tour:", tour)
    else:
        print("\nNo valid tour found")