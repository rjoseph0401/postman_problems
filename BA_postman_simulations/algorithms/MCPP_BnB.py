import numpy as np
import time
from scipy.optimize import linprog

class BnBMCPPSolver:
    def __init__(self, graph, costs, time_limit=300, gap_tolerance=0.02, verbose=False):
        self.G = graph
        self.costs = costs
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance
        self.verbose = verbose
        
        self.A = []  # List of directed arcs
        self.E = []  # List of undirected edges

        for u, v, key, data in graph.edges(data=True, keys=True):
            if data.get('directed', False):
                if (u, v) not in self.A:
                    self.A.append((u, v))
            else:
                canonical_edge = (min(u, v), max(u, v))
                if canonical_edge not in self.E:
                    self.E.append(canonical_edge)
        
        # Sort edges for consistency
        self.A = sorted(self.A)
        self.E = sorted(self.E)
        
        # Track branch-and-bound progress
        self.node_counter = 0
        self.best_solution = (None, float('inf'))  # (solution, cost)
        self.start_time = None

    def _get_constraints(self, fixed_vars=None):
        n_nodes = len(self.G.nodes())
        n_arcs = len(self.A)
        n_edges = len(self.E)
        
        n_vars = n_arcs + 2 * n_edges
        
        # Initialize cost vector
        c = np.zeros(n_vars)
        
        for i, (u, v) in enumerate(self.A):
            c[i] = self.costs.get((u, v), 1)
        
        for i, (u, v) in enumerate(self.E):
            c[n_arcs + i] = self.costs.get((u, v), 1)
        
        for i, (u, v) in enumerate(self.E):
            c[n_arcs + n_edges + i] = self.costs.get((v, u), 1)
        
        # Flow conservation constraints: in-degree = out-degree at each node
        A_eq = np.zeros((n_nodes, n_vars))
        b_eq = np.zeros(n_nodes)
        
        # Map node labels to consecutive indices
        node_to_idx = {node: i for i, node in enumerate(self.G.nodes())}
        
        # Add directed arcs to flow conservation
        for i, (u, v) in enumerate(self.A):
            A_eq[node_to_idx[u], i] = -1  
            A_eq[node_to_idx[v], i] = 1   
        
        for i, (u, v) in enumerate(self.E):
            A_eq[node_to_idx[u], n_arcs + i] = -1  
            A_eq[node_to_idx[v], n_arcs + i] = 1   
        
        for i, (u, v) in enumerate(self.E):
            A_eq[node_to_idx[v], n_arcs + n_edges + i] = -1  
            A_eq[node_to_idx[u], n_arcs + n_edges + i] = 1   
        

        A_ub = np.zeros((len(self.E), n_vars))
        b_ub = -np.ones(len(self.E))  
        
        for i in range(len(self.E)):
            A_ub[i, n_arcs + i] = -1  
            A_ub[i, n_arcs + n_edges + i] = -1  
        
        bounds = []
        
        for i in range(n_arcs):
            bounds.append((1, None))
        
        for i in range(2 * n_edges):
            bounds.append((0, None))
        
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
        

        fixed_vars = {}  
        solution_stack = [fixed_vars]  
          
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
        
        # Check if initial solution is already integer
        if self._is_integer_solution(initial_solution):
            if self.verbose:
                print(f"Initial solution is integer with objective {initial_obj:.2f}")
            return initial_solution, initial_obj

        while solution_stack and time.time() - self.start_time < self.time_limit:
            # Check if we've reached the desired optimality gap
            if UB < float('inf') and LB > 0:
                current_gap = (UB - LB) / UB
                if current_gap <= self.gap_tolerance:
                    if self.verbose:
                        print(f"Terminating: reached optimality gap target of {self.gap_tolerance}")
                    break
                    
            current_fixed_vars = solution_stack.pop()
            
            solution, objective = self._solve_lp_relaxation(current_fixed_vars)
            self.node_counter += 1
            
            if solution is None:
                continue
            
            
          
            if self._is_integer_solution(solution):
                if objective < UB:
                    self.best_solution = (solution.copy(), objective)
                    UB = objective
                    if self.verbose:
                        print(f"Found better solution with objective {UB:.2f}")
                continue
            
            # Select variable to branch on (most fractional)
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

