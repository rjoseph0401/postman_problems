import networkx as nx
from networkx.algorithms.matching import min_weight_matching
from algorithms.Win_WPP_Eulerian import Win_WPP_Eulerian


def Win_WPP_General(G):
    odd_nodes = [node for node in G.nodes() if G.degree(node) % 2 == 1]
    
    if not odd_nodes:
        if not isinstance(G, nx.MultiGraph):
            multi_G = nx.MultiGraph()
            for u, v, data in G.edges(data=True):
                multi_G.add_edge(u, v, cij=data.get('cij', 1), cji=data.get('cji', 1))
            return Win_WPP_Eulerian(multi_G)
        return Win_WPP_Eulerian(G)
    
    symmetric_costs = {}
    for i, j, data in G.edges(data=True):
        cij = data.get('cij', 1)
        cji = data.get('cji', 1)
        symmetric_costs[(i, j)] = symmetric_costs[(j, i)] = (cij + cji) / 2    
    complete = nx.Graph()
    paths = {}
    
    # Calculate shortest paths between all pairs of odd-degree vertices
    for i, u in enumerate(odd_nodes):
        for v in odd_nodes[i + 1:]:
            try:
                path = nx.shortest_path(G, u, v, weight=lambda x, y, d: symmetric_costs[(x, y)])
                cost = sum(symmetric_costs[(path[k], path[k+1])] 
                          for k in range(len(path)-1))
                complete.add_edge(u, v, weight=cost)
                paths[(u, v)] = path
                paths[(v, u)] = list(reversed(path))
            except nx.NetworkXNoPath:
                print(f"Error: No path between odd nodes {u} and {v}")
                return None

    # Step 5: Find minimum-weight perfect matching on odd-degree vertices
    matching = min_weight_matching(complete, weight='weight')

    # Step 6: Create augmented graph by adding duplicate edges along matching paths
    augmented_G = nx.MultiGraph()
    
    # Add all original edges
    for u, v, data in G.edges(data=True):
        augmented_G.add_edge(u, v, cij=data.get('cij', 1), cji=data.get('cji', 1))
    
    # Add duplicate edges for each matching path
    for u, v in matching:
        path = paths[(u, v)]
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            
            # Find original edge data and add duplicate with same costs
            if G.has_edge(start, end):
                edge_data = G[start][end]
                augmented_G.add_edge(start, end, 
                                    cij=edge_data.get('cij', 1),
                                    cji=edge_data.get('cji', 1))
            elif G.has_edge(end, start):
                edge_data = G[end][start]
                augmented_G.add_edge(start, end,
                                   cij=edge_data.get('cji', 1),
                                   cji=edge_data.get('cij', 1))
      # Verify augmented graph is Eulerian
    
    odd_after = [node for node in augmented_G.nodes() if augmented_G.degree(node) % 2 == 1]
    if odd_after:
        print(f"Error: Augmented graph still has odd degree vertices: {odd_after}")
        return None
    
    print(f"\n✓ Augmented graph is Eulerian, calling win_wpp_eulerian_algorithm...")
    return Win_WPP_Eulerian(augmented_G)

