import networkx as nx


def EVENDEGREE(G):
    G_prime = G.copy()
    
    #Identify vertices of odd degree without considering direction
    odd_vertices = []
    for v in G_prime.nodes():
        degree = 0
        for u, w, data in G_prime.edges(data=True):
            if u == v or w == v:
                degree += 1
        
        if degree % 2 == 1:
            odd_vertices.append(v)
    
    
    if len(odd_vertices) == 0:
        print("No odd degree vertices found. Graph is already even-degree.")
        return G_prime
    
    # Convert mixed graph to undirected graph, ignoring arc direction
    undirected_graph = nx.Graph()
    
    # Add all vertices
    undirected_graph.add_nodes_from(G_prime.nodes())
    
    # Add all edges as undirected edges with their weights
    for u, v, data in G_prime.edges(data=True):
        weight = data.get('weight', 1)
        if not undirected_graph.has_edge(u, v) or undirected_graph[u][v]['weight'] > weight:
            undirected_graph.add_edge(u, v, weight=weight)
    
    #Find shortest paths between all pairs of odd vertices
    shortest_paths = {}
    path_dict = {}
    
    for source in odd_vertices:
        shortest_paths[source] = {}
        path_dict[source] = {}
        
        # Get both lengths and paths using Dijkstra on undirected graph
        try:
            lengths, paths = nx.single_source_dijkstra(undirected_graph, source, weight='weight')
            
            for target in odd_vertices:
                if target != source and target in lengths:
                    shortest_paths[source][target] = lengths[target]
                    path_dict[source][target] = paths[target]
        except nx.NetworkXNoPath:
            continue
    
    #Create a complete graph of odd vertices with shortest path distances as edge weights
    matching_graph = nx.Graph()
    
    for u in odd_vertices:
        for v in odd_vertices:
            if u != v:
                if v in shortest_paths.get(u, {}):
                    weight = shortest_paths[u][v]
                    matching_graph.add_edge(u, v, weight=weight)
    
    if len(odd_vertices) % 2 == 1:
        dummy_node = max(G_prime.nodes()) + 1
        for v in odd_vertices:
            matching_graph.add_edge(dummy_node, v, weight=0)
        odd_vertices.append(dummy_node)
    
    #Perform minimum-weight matching
    try:
        matching = nx.algorithms.matching.min_weight_matching(matching_graph)
    except Exception as e:
        print(f"Error in min_weight_matching: {str(e)}")
        # Fallback matching, just as test case
        matching = [(odd_vertices[i], odd_vertices[i+1]) for i in range(0, len(odd_vertices), 2)]
    
    for u, v in matching:
        if u in G_prime.nodes() and v in G_prime.nodes():
            if u in path_dict and v in path_dict[u]:
                path = path_dict[u][v]
                
                for i in range(len(path) - 1):
                    curr = path[i]
                    next_node = path[i+1]
                    if G.has_edge(curr, next_node):
                        edge_data = G[curr][next_node][0]
                        is_directed = edge_data.get('directed', True)
                        kind = edge_data.get('kind', 'directed')
                        weight = edge_data.get('weight', 1)
                        
                        G_prime.add_edge(curr, next_node, directed=is_directed, kind=kind, weight=weight)
                        
                        if not is_directed:
                            G_prime.add_edge(next_node, curr, directed=is_directed, kind=kind, weight=weight)
                    else:
                        edge_weight = 1
                        G_prime.add_edge(curr, next_node, directed=True, kind='directed', weight=edge_weight)
    
    
    return G_prime
