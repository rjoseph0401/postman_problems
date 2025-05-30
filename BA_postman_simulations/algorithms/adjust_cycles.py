import networkx as nx

def ADJUSTCYCLES(V, O, M, U, M_double_prime):
    
    M_prime = M.copy()
    U_prime = U.copy()
    O_prime = O.copy()
    
    D = nx.MultiDiGraph()
    D.add_nodes_from(V)
    
    for u, v in M_prime:
        D.add_edge(u, v)
    
    
    iteration = 0
    max_iterations = len(V) * 10  
    
    while O_prime and iteration < max_iterations:
        iteration += 1
        
        v = O_prime.pop(0)
          # Find a cycle C with start and end at v
        cycle = find_cycle_with_vertex(D, v)
        
        if cycle:
            
            # Check if cycle contains any arc from M''
            m_double_prime_arcs_in_cycle = []
            for i in range(len(cycle) - 1):
                arc = (cycle[i], cycle[i+1])
                if arc in M_double_prime:
                    m_double_prime_arcs_in_cycle.append(arc)
            
            if m_double_prime_arcs_in_cycle:
                x, y = m_double_prime_arcs_in_cycle[0]
                
                if (x, y) in M_prime:
                    M_prime.remove((x, y))
                    
                    canonical_edge = (min(x, y), max(x, y))
                    U_prime.add(canonical_edge)
                    
                    if D.has_edge(x, y):
                        D.remove_edge(x, y)
                    
                    # Check if x or y become odd-degree 
                    for vertex in [x, y]:
                        directed_degree = D.in_degree(vertex) + D.out_degree(vertex)
                        undirected_degree = 0
                        for u_edge in U_prime:
                            if vertex in u_edge:
                                undirected_degree += 2 
                        
                        total_degree = directed_degree + undirected_degree
                        
                        if total_degree % 2 == 1 and vertex not in O_prime:
                            O_prime.append(vertex)
                else:
                    print(f"Warning: Arc ({x}, {y}) not found in M' but in M''")
            else:
                if v not in O_prime:
                    O_prime.append(v)
        else:
            # Re-add v to try again later, but avoid infinite loops
            if v not in O_prime and iteration < max_iterations - 1:
                O_prime.append(v)
    
    if iteration >= max_iterations:
        print(f"Warning: ADJUSTCYCLES reached maximum iterations ({max_iterations})")
    
    
    return M_prime, U_prime

def find_cycle_with_vertex(G, v):
    visited = {node: False for node in G.nodes()}
    path = [v]
    cycle = find_cycle_dfs(G, v, v, visited, path, depth=0)
    return cycle

def find_cycle_dfs(G, start, current, visited, path, depth):
    visited[current] = True
    
    # Check all neighbors
    for neighbor in G.neighbors(current):
        if neighbor == start and depth > 0:
            path.append(neighbor)
            return path
        
        # If neighbor not visited, continue 
        if not visited[neighbor]:
            path.append(neighbor)
            result = find_cycle_dfs(G, start, neighbor, visited.copy(), path, depth + 1)
            if result:
                return result
            path.pop() 
    
    return None
