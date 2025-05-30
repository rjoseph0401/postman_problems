import numpy as np
import networkx as nx
import time
from scipy.optimize import linprog

class BnBWPPSolver:
    def __init__(self, graph, costs, time_limit=300, gap_tolerance=0.02, verbose=False):
        """
        Initialize the solver with a graph and edge costs.
        """
        self.G = graph
        self.costs = costs
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance
        self.verbose = verbose
        
        self.E = [] 
        
        for u, v in graph.edges():
            canonical_edge = (min(u, v), max(u, v))
            if canonical_edge not in self.E:
                self.E.append(canonical_edge)
        
        self.E = sorted(self.E)
        
        # Track branch-and-bound progress
        self.node_counter = 0
        self.best_solution = (None, float('inf'))  # (solution, cost)
        self.start_time = None   
    def _get_constraints(self, fixed_vars=None):
        n_nodes = len(self.G.nodes())
        n_edges = len(self.E)
        
        n_vars = 2 * n_edges
        
        # Initialize cost vector
        c = np.zeros(n_vars)
        
        for i, (u, v) in enumerate(self.E):
            c[2*i] = self.costs.get((u, v), 1)
            
            c[2*i + 1] = self.costs.get((v, u), 1)        
        A_eq = np.zeros((n_nodes, n_vars))
        b_eq = np.zeros(n_nodes)
        
        # Map node labels to consecutive indices
        node_to_idx = {node: i for i, node in enumerate(self.G.nodes())}
        
        # Add both directions of each edge to flow conservation
        for i, (u, v) in enumerate(self.E):
            A_eq[node_to_idx[u], 2*i] = -1  
            A_eq[node_to_idx[v], 2*i] = 1   
            
            A_eq[node_to_idx[v], 2*i + 1] = -1  
            A_eq[node_to_idx[u], 2*i + 1] = 1   
        
        #each edge must be traversed at least once in some direction
        A_ub = np.zeros((len(self.E), n_vars))
        b_ub = -np.ones(len(self.E))  
        
        for i in range(len(self.E)):
            A_ub[i, 2*i] = -1     
            A_ub[i, 2*i + 1] = -1
        
        bounds = [(0, None) for _ in range(n_vars)]
        
        if fixed_vars:
            for idx, val in fixed_vars.items():
                bounds[idx] = (val, val)
        
        return c, A_eq, b_eq, A_ub, b_ub, bounds

    def _solve_lp_relaxation(self, fixed_vars=None):
        c, A_eq, b_eq, A_ub, b_ub, bounds = self._get_constraints(fixed_vars)
        
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, 
                      bounds=bounds, method='highs')
        
        if res.success:
            return res.x, res.fun
        else:
            if self.verbose:
                print(f"LP relaxation failed: {res.message}")
            return None, None

    def _is_integer_solution(self, solution, tolerance=1e-6):
        if solution is None:
            return False
        return all(abs(x - round(x)) < tolerance for x in solution)

    def _select_branching_var(self, solution):
        if solution is None:
            return None
            
        best_idx = -1
        best_frac = -1
        
        for i, val in enumerate(solution):
            frac = min(val - np.floor(val), np.ceil(val) - val)
            if frac > 1e-6 and frac > best_frac:
                best_frac = frac
                best_idx = i
        return best_idx
        
    def solve(self):
        self.start_time = time.time()
        self.node_counter = 1  
        
        # Set up branch and bound variables
        fixed_vars = {}  
        solution_stack = [fixed_vars]  
        
        # Get initial lower bound from LP relaxation
        initial_solution, initial_obj = self._solve_lp_relaxation({})
        if initial_solution is None:
            if self.verbose:
                print("Initial LP relaxation is infeasible.")
            return None, float('inf')
        
        
        node_bounds = []  
        if initial_obj is not None:
            node_bounds.append(initial_obj)
        
        LB = initial_obj if initial_obj is not None else float('inf') 
        UB = float('inf')  
        
        if self._is_integer_solution(initial_solution):
            if self.verbose:
                print(f"Initial solution is integer with objective {initial_obj:.2f}")
            return initial_solution, initial_obj
        
        while solution_stack and time.time() - self.start_time < self.time_limit:
            if UB < float('inf') and LB > 0:
                current_gap = (UB - LB) / UB
                if current_gap <= self.gap_tolerance:
                    if self.verbose:
                        print(f"Terminating: reached optimality gap target of {self.gap_tolerance}")
                    break
                    
            current_fixed_vars = solution_stack.pop()
            
            # Solve the LP relaxation 
            solution, objective = self._solve_lp_relaxation(current_fixed_vars)
            self.node_counter += 1
            
            if solution is None:
                continue
            
            
            # If solution is integer, update best solution if better
            if self._is_integer_solution(solution):
                if objective < UB:
                    self.best_solution = (solution.copy(), objective)
                    UB = objective
                    if self.verbose:
                        print(f"Found better solution with objective {UB:.2f}")
                continue
            
            var_to_branch = self._select_branching_var(solution)
            
            if var_to_branch == -1:
                continue
                
            # Create branches
            right_fixed = current_fixed_vars.copy()
            right_fixed[var_to_branch] = int(np.ceil(solution[var_to_branch]))
            
            left_fixed = current_fixed_vars.copy()
            left_fixed[var_to_branch] = int(np.floor(solution[var_to_branch]))
            
            # Add these child nodes to our solution stack
            solution_stack.append(right_fixed)
            solution_stack.append(left_fixed)
            
            node_bounds.append(objective)
            
            if node_bounds:
                LB = min(node_bounds)
                
            if self.verbose and self.node_counter % 10 == 0:
                if UB < float('inf'):
                    gap = ((UB - LB) / UB) * 100 if UB > 0 else 0
                    print(f"Nodes explored: {self.node_counter}, LB={LB:.2f}, UB={UB:.2f}, Gap={gap:.2f}%")
                else:
                    print(f"Nodes explored: {self.node_counter}, LB={LB:.2f}, UB=inf")
        
        if self.verbose:
            if UB < float('inf'):
                final_gap = ((UB - LB) / UB) * 100 if UB > 0 else 0
                print(f"Final bounds: LB={LB:.2f}, UB={UB:.2f}, Gap={final_gap:.2f}%")
            else:
                print(f"Final bounds: LB={LB:.2f}, UB=inf")
            print(f"Total nodes explored: {self.node_counter}")
            print(f"Total solve time: {time.time() - self.start_time:.2f} seconds")
        
        return self.best_solution[0], self.best_solution[1]    
    def construct_tour(self, solution):
        if solution is None:
            return None
            
        tour_graph = nx.MultiDiGraph()
        
        for node in self.G.nodes():
            tour_graph.add_node(node)
        
        for i, (u, v) in enumerate(self.E):
            fw_copies = int(round(solution[2*i]))
            for _ in range(fw_copies):
                tour_graph.add_edge(u, v, weight=self.costs.get((u, v), 1))
                
            bw_copies = int(round(solution[2*i + 1]))
            for _ in range(bw_copies):
                tour_graph.add_edge(v, u, weight=self.costs.get((v, u), 1))
        
        for node in tour_graph.nodes():
            if tour_graph.in_degree(node) != tour_graph.out_degree(node):
                if self.verbose:
                    print(f"Warning: Node {node} has imbalanced degrees: in={tour_graph.in_degree(node)}, out={tour_graph.out_degree(node)}")
                return None
        
        if not nx.is_strongly_connected(tour_graph):
            if self.verbose:
                print("Warning: Tour graph is not strongly connected.")
            return None
            
        try:
            circuit = list(nx.eulerian_circuit(tour_graph))
            tour = [(u, v, self.costs.get((u, v), 1)) for u, v in circuit]
            return tour
        except nx.NetworkXError as e:
            if self.verbose:
                print(f"Failed to find Eulerian circuit: {e}")
            return None
    
    def calculate_solution_cost(self, solution):
        if solution is None:
            return float('inf')
        
        total_cost = 0
        
        # Calculate the cost of each edge traversal
        for i, (u, v) in enumerate(self.E):
            fw_traversals = round(solution[2*i])
            if fw_traversals > 0:
                total_cost += fw_traversals * self.costs.get((u, v), 1)
                
            bw_traversals = round(solution[2*i + 1])
            if bw_traversals > 0:
                total_cost += bw_traversals * self.costs.get((v, u), 1)
        
        return total_cost
