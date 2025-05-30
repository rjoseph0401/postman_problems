import networkx as nx
from algorithms.even_degree import EVENDEGREE
from algorithms.in_out_degree import INOUTDEGREE
from algorithms.even_parity import EVENPARITY
def MIXED1(G):
    G1 = EVENDEGREE(G)
    
    
    M, U = INOUTDEGREE(G1)   
    M_prime, U_prime = EVENPARITY(G1, M, U)
    
    temp_graph = nx.MultiDiGraph()
    temp_graph.add_nodes_from(G.nodes())
    
    if isinstance(M_prime, nx.MultiDiGraph):
        for u, v in M_prime.edges():
            temp_graph.add_edge(u, v)
    else:
        for u, v in M_prime:
            temp_graph.add_edge(u, v)
    
    for u, v in U_prime:
        temp_graph.add_edge(u, v)
        temp_graph.add_edge(v, u)
    
    for v in temp_graph.nodes():
        in_deg = temp_graph.in_degree(v)
        out_deg = temp_graph.out_degree(v)
        total_deg = in_deg + out_deg
    
    # Check if graph is Eulerian
    is_eulerian = nx.is_eulerian(temp_graph)
    
    G_prime = nx.MultiDiGraph()
    G_prime.add_nodes_from(G.nodes())
    if isinstance(M_prime, nx.MultiDiGraph):    
        for u, v, data in M_prime.edges(data=True):
            weight = data.get('weight', 1)
            kind = data.get('kind', 'directed')
            G_prime.add_edge(u, v, weight=weight, kind=kind)
    else:
        for u, v in M_prime:
            weight = 1
            kind = 'directed'
            
            # Try to get weight from original graph in both directions
            if G.has_edge(u, v):
                data = G.get_edge_data(u, v)[0]
                weight = data.get('weight', 1)
                original_kind = data.get('kind', 'directed')
                if original_kind == 'undirected':
                    kind = 'undirected'
            elif G.has_edge(v, u):
                data = G.get_edge_data(v, u)[0]
                weight = data.get('weight', 1)
                original_kind = data.get('kind', 'directed')
                if original_kind == 'undirected':
                    kind = 'undirected'
            else:
                print(f"DEBUG: Edge ({u}, {v}) not found in original graph")
                
            G_prime.add_edge(u, v, weight=weight, kind=kind)
    for u, v in U_prime:
        weight = 1
        kind = 'undirected'
        
        # Get weight from original graph in canonical form
        if G.has_edge(u, v):
            data = G.get_edge_data(u, v)[0]
            weight = data.get('weight', 1)
        elif G.has_edge(v, u):
            data = G.get_edge_data(v, u)[0]
            weight = data.get('weight', 1)
        
        
        G_prime.add_edge(u, v, weight=weight, kind=kind, directed=False)
        G_prime.add_edge(v, u, weight=weight, kind=kind, directed=False)
    
    # Find and return an Euler circuit
    try:
        circuit = list(nx.eulerian_circuit(G_prime))
        return G_prime, circuit
    except nx.NetworkXError:
        print("Error: Could not find Euler circuit in the resulting graph.")
        return G_prime, []


