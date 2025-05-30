def is_connected(graph, start_node):
    """
    Check if there's a path from start_node to all other nodes with outgoing edges
    """
    # For directed graphs, we need to check if all nodes are reachable
    reachable = set()
    to_visit = [start_node]
    
    while to_visit:
        node = to_visit.pop()
        reachable.add(node)
        
        for neighbor in graph.successors(node):
            if neighbor not in reachable and neighbor not in to_visit:
                to_visit.append(neighbor)
    
    # Check if all nodes with edges are reachable
    nodes_with_edges = set()
    for u, v in graph.edges():
        nodes_with_edges.add(u)
        nodes_with_edges.add(v)
    
    return all(node in reachable for node in nodes_with_edges)

def fleury(G):
    work_graph = G.copy()
    
    if not work_graph.nodes():
        return []
    
    # prove whether graph is Eulerian
    for node in work_graph.nodes():
        if work_graph.in_degree(node) != work_graph.out_degree(node):
            raise ValueError(f"Graph ist nicht eulersch: Knoten {node} hat ungleiche Ein-/Ausgangsgrade")
    
    original_edge_count = work_graph.number_of_edges()
    
    # choose a starting node
    start_node = list(work_graph.nodes())[0]
    current_node = start_node
    
    euler_path = []
    
    while work_graph.number_of_edges() > 0:
        if work_graph.out_degree(current_node) == 0:
            # look for a node with outgoing edges to continue the tour
            for i, (u, v, _, _) in enumerate(euler_path):
                if work_graph.out_degree(u) > 0:
                    euler_path = euler_path[i:] + euler_path[:i]
                    current_node = u
                    break
            else:
                for node in work_graph.nodes():
                    if work_graph.out_degree(node) > 0:
                        raise ValueError(f"No valid starting point found: {node} has outgoing edges.")
        edges = list(work_graph.out_edges(current_node, keys=True, data=True))
        if not edges:
            raise ValueError(f"No outgoing edges found from current node: {current_node}")
        
        non_bridge_edges = []
        bridge_edges = []
        
        for u, v, key, data in edges:
            # Check if removing this edge would disconnect the graph
            work_graph.remove_edge(u, v, key)

            if work_graph.out_degree(u) == 0 or not is_connected(work_graph, u):
                bridge_edges.append((u, v, key, data))
            else:
                non_bridge_edges.append((u, v, key, data))
            work_graph.add_edge(u, v, key=key, **data)
        
        # Choose a non-bridge edge if available, otherwise use a bridge
        if non_bridge_edges:
            u, v, key, data = non_bridge_edges[0]
        else:
            u, v, key, data = bridge_edges[0]
        
        work_graph.remove_edge(u, v, key)
        
        euler_path.append((u, v, data.get('kind', 'directed'), data.get('weight', 1)))
        
        current_node = v
    
    
    # prove whether the tour covers all edges
    if len(euler_path) != original_edge_count:
        raise ValueError(f"Die Tour deckt nicht alle Kanten ab: {len(euler_path)} von {original_edge_count}")
    
    # prove whether the tour is connected
    for i in range(len(euler_path) - 1):
        if euler_path[i][1] != euler_path[i+1][0]:
            print(f"Tour nicht zusammenhängend bei Kante {i}: {euler_path[i]} -> {euler_path[i+1]}")
    
    # prove whether the tour ends at the starting node
    if euler_path and euler_path[-1][1] != euler_path[0][0]:
       print(f"Tour endet nicht am Startknoten: {euler_path[-1][1]} != {euler_path[0][0]}")
    
    return euler_path
