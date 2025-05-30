import networkx as nx
from algorithms.adjust_cycles import ADJUSTCYCLES

def EVENPARITY(G, M, U):
    print("Running EVENPARITY...")
    
    temp_graph = nx.MultiDiGraph()
    temp_graph.add_nodes_from(G.nodes())
    
    # Add directed edges from M
    if isinstance(M, list):
        for u, v in M:
            temp_graph.add_edge(u, v)
        M_edges = M
    else:
        for u, v in M.edges():
            temp_graph.add_edge(u, v)
        M_edges = list(M.edges())
    
    # Add undirected edges from U 
    for u, v in U:
        temp_graph.add_edge(u, v)
        temp_graph.add_edge(v, u)
    problem_vertices = []
    
    for v in temp_graph.nodes():
        in_deg = temp_graph.in_degree(v)
        out_deg = temp_graph.out_degree(v)
        total_degree = in_deg + out_deg
        
        is_odd = total_degree % 2 == 1
        is_unbalanced = in_deg != out_deg
        
        if is_odd or is_unbalanced:
            problem_vertices.append(v)
    

    odd_vertices = [v for v in problem_vertices if (temp_graph.in_degree(v) + temp_graph.out_degree(v)) % 2 == 1]
    
    if isinstance(M, list):
        M_prime = M.copy()
    else:
        M_prime = list(M.edges())
    U_prime = U.copy()

    original_arcs = set()
    for u, v, data in G.edges(data=True):
        if data.get('directed', True) and data.get('kind') == 'directed':
            original_arcs.add((u, v))
    
    M_double_prime = [edge for edge in M_edges if edge not in original_arcs]
    
    if problem_vertices:
        M_prime, U_prime = ADJUSTCYCLES(list(G.nodes()), problem_vertices, M_prime, U_prime, M_double_prime)
    else:
        print("All vertices already have even degree and balanced in/out degree.")
    
    # Verify the result
    final_graph = nx.MultiDiGraph()
    final_graph.add_nodes_from(G.nodes())
    
    # Add edges from final M_prime
    for u, v in M_prime:
        final_graph.add_edge(u, v)
    
    # Add undirected edges from final U_prime  
    for u, v in U_prime:
        final_graph.add_edge(u, v)
        final_graph.add_edge(v, u)
    
    all_balanced = True
    for v in final_graph.nodes():
        in_deg = final_graph.in_degree(v)
        out_deg = final_graph.out_degree(v)
        total_deg = in_deg + out_deg
        is_balanced = in_deg == out_deg
        is_even = total_deg % 2 == 0
        
        if not is_balanced or not is_even:
            all_balanced = False
    
    if all_balanced:
        print("  All vertices have even degree and balanced in/out degree!")
    
    print(f"EVENPARITY completed.")
    
    return M_prime, U_prime
