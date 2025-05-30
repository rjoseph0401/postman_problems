import networkx as nx
from itertools import repeat
from collections import defaultdict

def Win_WPP_Eulerian(G: nx.MultiGraph,
                      start=None,
                      scale: int = 100):
    if not nx.is_eulerian(G):
        raise nx.NetworkXError("Graph is not Eulerian")

    demand = defaultdict(int)   
    edge_info = {}             

    copy_idx = 0
    #orient arcs based on costs
    for u, v, k, d in G.edges(keys=True, data=True):
        cuv, cvu = d["cij"], d["cji"]
        if cuv <= cvu:        
            a, b       = u, v
            c_less     = cuv
            c_more     = cvu
        else:                 
            a, b       = v, u
            c_less     = cvu
            c_more     = cuv

        edge_key = f"copy{copy_idx}"
        copy_idx += 1
        edge_info[edge_key] = dict(a=a, b=b,
                                   c_less=c_less,
                                   c_more=c_more)

        demand[a] += 1
        demand[b] -= 1

    Gp = nx.DiGraph()
    for n in G.nodes:
        Gp.add_node(n, demand=demand[n])

    for key, dat in edge_info.items():
        a, b      = dat["a"], dat["b"]
        c_less    = dat["c_less"]
        c_more    = dat["c_more"]
        aux_cost  = 0.5 * (c_more - c_less)
        aux       = f"aux_{key}"

        Gp.add_edge(a, b, weight=c_less)
        Gp.add_edge(b, a, weight=c_more)

        # artificial Arc  b → aux → a  
        Gp.add_node(aux, demand=0)
        Gp.add_edge(b, aux, weight=aux_cost, capacity=2)
        Gp.add_edge(aux, a, weight=0,       capacity=2)

    for _, _, d in Gp.edges(data=True):
        d["weight"] = int(round(d["weight"] * scale))
    # compute min cost flow
    flow_dict = nx.min_cost_flow(Gp)

    Gpp = nx.MultiDiGraph()
    total_cost = 0

    for key, dat in edge_info.items():
        a, b   = dat["a"], dat["b"]
        cl, cm = dat["c_less"], dat["c_more"]
        aux    = f"aux_{key}"

        n_real     = flow_dict[a][b] if b in flow_dict[a] else 0
        n_reverse  = flow_dict[b][a] if a in flow_dict[b] else 0
        n_aux      = (flow_dict[b][aux] if aux in flow_dict[b] else 0) + \
                     (flow_dict[aux][a] if a  in flow_dict[aux] else 0)

        # add edges according to flow values
        if n_aux > 0:
            orient, unit_cost, copies = (b, a), cm, n_reverse + 1
        else:
            orient, unit_cost, copies = (a, b), cl, n_real   + 1

        Gpp.add_edges_from(repeat((*orient, {"weight": unit_cost}), copies))
        total_cost += copies * unit_cost

    if start is None:
        start = next(iter(Gpp.nodes))

    circuit = []
    for u, v in nx.eulerian_circuit(Gpp, source=start):
        if not circuit:
            circuit.append(u)
        circuit.append(v)

    return circuit, total_cost
