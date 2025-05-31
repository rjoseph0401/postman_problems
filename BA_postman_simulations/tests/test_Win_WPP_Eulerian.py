import networkx as nx
import os
import sys

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from algorithms.Win_WPP_Eulerian import Win_WPP_Eulerian


if __name__ == "__main__":
     G = nx.MultiGraph()
     #ensure G is Eulerian
     G.add_edge(0, 1, cij=2, cji=3)
     G.add_edge(1, 2, cij=1, cji=4)
     G.add_edge(2, 3, cij=3, cji=2)
     G.add_edge(3, 4, cij=2, cji=5)
     G.add_edge(4,0, cij=2, cji=1)

     print("\nRunning Win's Algorithm for Eulerian Graphs...")
     tour = Win_WPP_Eulerian(G)
     if tour:
         print(f"Found Eulerian tour: {tour}")
     else:
         print("No valid tour found")