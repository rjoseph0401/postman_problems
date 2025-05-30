import networkx as nx
from algorithms.in_out_degree import INOUTDEGREE


def LARGECYCLES(G, M, U):
    print("Running LARGECYCLES...")
    
    original_U = set((min(u, v), max(u, v)) for u, v in U)
    U_working = U.copy()
    
    G_prime = nx.Graph()
    G_prime.add_nodes_from(G.nodes())
    
    for u, v in U_working:
        weight = 1  
        if G.has_edge(u, v):
            edge_data = G.get_edge_data(u, v)
            if isinstance(edge_data, dict):
                if 0 in edge_data:  
                    weight = edge_data[0].get('weight', 1)
                else:
                    weight = edge_data.get('weight', 1)
        elif G.has_edge(v, u):
            edge_data = G.get_edge_data(v, u)
            if isinstance(edge_data, dict):
                if 0 in edge_data:
                    weight = edge_data[0].get('weight', 1)
                else:
                    weight = edge_data.get('weight', 1)
        
        G_prime.add_edge(u, v, weight=weight)
    
    odd_vertices = [v for v, d in G_prime.degree() if d % 2 == 1]
    
    if odd_vertices:
        
        complete_graph = nx.Graph()
        complete_graph.add_nodes_from(odd_vertices)
        
        # Build auxiliary graph for shortest paths
        aux_graph = nx.Graph()
        aux_graph.add_nodes_from(G.nodes())
        
        # Add all edges from original graph with preserved weights
        for u, v, data in G.edges(data=True):
            weight = 1  # Default weight
            if isinstance(data, dict):
                weight = data.get('weight', 1)
            elif hasattr(data, 'get'):
                weight = data.get('weight', 1)
            
            if data.get('kind') == 'undirected':
                aux_graph.add_edge(u, v, weight=weight)
        
        # Calculate shortest paths between all pairs of odd vertices
        for i, u in enumerate(odd_vertices):
            for v in odd_vertices[i+1:]:
                try:
                    if nx.has_path(aux_graph, u, v):
                        path_length = nx.shortest_path_length(aux_graph, u, v, weight='weight')
                        complete_graph.add_edge(u, v, weight=path_length)
                except:
                    complete_graph.add_edge(u, v, weight=1000)
        
        # Find minimum weight matching
        if complete_graph.edges():
            from networkx.algorithms.matching import min_weight_matching
            matching_edges = min_weight_matching(complete_graph)
            
            # Add shortest paths for matched pairs to U
            for u, v in matching_edges:
                try:
                    if nx.has_path(aux_graph, u, v):
                        path = nx.shortest_path(aux_graph, u, v, weight='weight')
                        
                        for i in range(len(path) - 1):
                            edge = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                            if edge not in original_U:
                                U_working.append((path[i], path[i+1]))
                except:
                    print(f"Could not find path from {u} to {v}")
    
    tour_graph = nx.MultiDiGraph()
    tour_graph.add_nodes_from(G.nodes())
    
    if isinstance(M, nx.MultiDiGraph):
        for u, v, data in M.edges(data=True):
            weight = data.get('weight', 1)
            tour_graph.add_edge(u, v, kind='directed', weight=weight)
    else:
        for u, v in M:
            weight = 1  
            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v)
                if isinstance(edge_data, dict):
                    if 0 in edge_data:  
                        weight = edge_data[0].get('weight', 1)
                    else:
                        weight = edge_data.get('weight', 1)
            
            tour_graph.add_edge(u, v, kind='directed', weight=weight)
    
    for u, v in U_working:
        weight = 1  
        
        if G.has_edge(u, v):
            edge_data = G.get_edge_data(u, v)
            if isinstance(edge_data, dict):
                if 0 in edge_data:  
                    weight = edge_data[0].get('weight', 1)
                else:
                    weight = edge_data.get('weight', 1)
        elif G.has_edge(v, u):
            edge_data = G.get_edge_data(v, u)
            if isinstance(edge_data, dict):
                if 0 in edge_data:  
                    weight = edge_data[0].get('weight', 1)
                else:
                    weight = edge_data.get('weight', 1)
        
        tour_graph.add_edge(u, v, kind='undirected', weight=weight)
        tour_graph.add_edge(v, u, kind='undirected', weight=weight)
    
    # Verify the graph is Eulerian
    print("Verifying Eulerian properties...")
    for node in tour_graph.nodes():
        in_deg = tour_graph.in_degree(node)
        out_deg = tour_graph.out_degree(node)
        if in_deg != out_deg:
            print(f"Warning: Node {node} is not balanced: in={in_deg}, out={out_deg}")
    
    # Find Eulerian tour
    try:
        print("Finding Eulerian circuit...")
        eulerian_circuit = list(nx.eulerian_circuit(tour_graph))
        
        # Convert to tour format with edge types and weights
        tour = []
        for u, v in eulerian_circuit:
            edge_data = tour_graph.get_edge_data(u, v, 0)
            edge_type = edge_data.get('kind', 'directed')
            weight = edge_data.get('weight', 1)
            tour.append((u, v, edge_type, weight))
        
        print(f"Found Eulerian tour with {len(tour)} edges")
        return tour
        
    except nx.NetworkXError as e:
        print(f"Error finding Eulerian circuit: {e}")
        return []


def MIXED2(G):
    print("Running MIXED2_NEW...")
    
    M, U = INOUTDEGREE(G)
    tour = LARGECYCLES(G, M, U)
    
    print(f"MIXED2_NEW completed. Tour length: {len(tour)}")
    return tour
