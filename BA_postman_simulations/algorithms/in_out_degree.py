import networkx as nx
import numpy as np
from scipy.optimize import linprog

def INOUTDEGREE(G):
    
    U = set()
    
    A = []  
    E = []  
    
    # Extract edges from graph
    for u, v, data in G.edges(data=True):
        if data.get('directed', True) and data.get('kind') == 'directed':
            A.append((u, v))
        else:
            edge = (min(u, v), max(u, v))
            if edge not in E:
                E.append(edge)
    
    
    E1 = []
    E2 = []
    for u, v in E:
        E1.append((u, v))
        E1.append((v, u))
        E2.append((u, v))
        E2.append((v, u))
    
    Et = A + E1 + E2
    
    # compute imbalance
    b = {v: 0 for v in G.nodes()}
    
    for u, v, data in G.edges(data=True):
        if data.get('directed', True) and data.get('kind') == 'directed':
            b[v] += 1 
            b[u] -= 1 
        
    
    # Setup and solve LP problem
    c_obj = []
    
    # Costs for A (directed arcs)
    for u, v in A:
        weight = 1
        if G.has_edge(u, v):
            edge_data = G.get_edge_data(u, v)[0]
            weight = edge_data.get('weight', 1)
        c_obj.append(weight)
    
    for u, v in E1:
        weight = 1
        # Find original undirected edge
        orig_edge = (min(u, v), max(u, v))
        if G.has_edge(u, v) or G.has_edge(v, u):
            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v)[0]
            else:
                edge_data = G.get_edge_data(v, u)[0]
            weight = edge_data.get('weight', 1)
        c_obj.append(weight)
    
    for _ in E2:
        c_obj.append(0)
    
    # Variable bounds
    bounds = []
    bounds.extend([(0, None) for _ in range(len(A))])     
    bounds.extend([(0, None) for _ in range(len(E1))])       
    bounds.extend([(0, 1) for _ in range(len(E2))])        
    A_eq = []
    b_eq = []
    
    for v in G.nodes():
        row = [0] * len(Et)
        
        for i, (u, w) in enumerate(Et):
            if w == v:  
                row[i] = 1
            elif u == v:    
                row[i] = -1
        
        A_eq.append(row)
        b_eq.append(-b[v])  
    
    
    # Solve the linear programming problem
    try:
        res = linprog(c_obj, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method='highs')
        if res.success:
            print("LP problem solved successfully.")
            for i, val in enumerate(res.x):
                if val > 1e-6:
                    if i < len(A):
                        print(f"  A[{i}] = {A[i]}: {val}")
                    elif i < len(A) + len(E1):
                        e_idx = i - len(A)
                    else:
                        e_idx = i - len(A) - len(E1)
        else:
            return [], U
    except Exception as e:
        print(f"Error solving LP: {str(e)}")
        return [], U
    
    M = []
    
    # Add original directed arcs to M
    for i, (u, v) in enumerate(A):
        M.append((u, v))
        
        additional_copies = round(res.x[i])
        for _ in range(additional_copies):
            M.append((u, v))
    
    # Add oriented edges from E1 based on LP solution
    a_offset = len(A)
    for i, (u, v) in enumerate(E1):
        copies = round(res.x[a_offset + i])
        for _ in range(copies):
            M.append((u, v))
    
    e2_offset = len(A) + len(E1)
    
    for i in range(0, len(E2), 2):
        u, v = E2[i]      
        v2, u2 = E2[i+1]  
        
        if u != u2 or v != v2:
            print(f"Warning: E2 edges not properly paired at index {i}")
            continue
        
        x1 = round(res.x[e2_offset + i])    
        x2 = round(res.x[e2_offset + i + 1])  
        
        if x1 + x2 == 1:
            if x1 == 1:
                M.append((u, v))
            else:
                M.append((v, u))
        elif x1 + x2 == 0:
            canonical_edge = (min(u, v), max(u, v))
            U.add(canonical_edge)
        else:
            print(f"Warning: Invalid E2 solution for edge ({u},{v}): x1={x1}, x2={x2}")
            canonical_edge = (min(u, v), max(u, v))
            U.add(canonical_edge)
    
    print(f"INOUTDEGREE completed.")
    
    return M, U
