def hierholzer(G):
    work_graph = G.copy()
    
    if not work_graph.nodes():
        return []
    
    #prove that the graph is Eulerian
    for node in work_graph.nodes():
        if work_graph.in_degree(node) != work_graph.out_degree(node):
            raise ValueError(f"Graph ist nicht eulersch: Knoten {node} hat ungleiche Ein-/Ausgangsgrade")
    
    tour = []
    
    start_node = list(work_graph.nodes())[0]
    
    def find_circuit(start):
        circuit = []
        current = start
        while True:
            # as long as there are outgoing edges from the current node
            if work_graph.out_degree(current) > 0:
                next_node = next(work_graph.successors(current))
                
                edge_data_dict = work_graph.get_edge_data(current, next_node)
                key = next(iter(edge_data_dict))
                edge_data = edge_data_dict[key]
            
                work_graph.remove_edge(current, next_node, key)
                
                # add edge to traversal
                circuit.append((current, next_node, edge_data.get('kind', 'undirected'), 
                               edge_data.get('weight', 1)))
                
                current = next_node
            else:
                # if there are no outgoing edges
                break
                
        return circuit
    
    #start traversing from the start node
    stack = [start_node]
    while stack:
        current = stack[-1]
        
        if work_graph.out_degree(current) == 0:
            stack.pop()
            continue
        
        new_circuit = find_circuit(current)
        
        # add circuits to tour
        if not tour:
            tour = new_circuit
        else:
            for i, (u, v, _, _) in enumerate(tour):
                if u == current:
                    tour = tour[:i] + new_circuit + tour[i:]
                    break
        
        #prove every node is visited
        for u, _, _, _ in new_circuit:
            if work_graph.out_degree(u) > 0 and u not in stack:
                stack.append(u)
    
    return tour