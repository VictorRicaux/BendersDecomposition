import cvxpy as cp
import numpy as np

class BendersDecomposition:

    def __init__(self, c, A, b, q, r, T, W, h, eps = 0.001):
        self.c = c
        self.A = A
        self.b = b
        self.q = q
        self.r = r
        self.T = T
        self.W = W
        self.h = h
        self.eps = eps
        self.n = len(q)
        self.d = len(c)
        self.m = len(h[0])

        self.UPPER_BOUND = 10e15
        self.LOWER_BOUND = -10e15
        self.optimality_cuts = {i: [] for i in range(self.n)}
        self.feasibility_cuts = {i: [] for i in range(self.n)}
        self.x_init = self.initialize_master_problem()
        self.max_iter = 100

    def initialize_master_problem(self):
        x = cp.Variable(self.d)
        intialization_problem = cp.Problem(cp.Minimize(0), [self.A @ x >= self.b])
        intialization_problem.solve()
        return x.value
    
    def solve_problem(self):
        x = self.x_init
        iteration = 0

        while abs((self.UPPER_BOUND - self.LOWER_BOUND)/self.UPPER_BOUND) >= self.eps and iteration <= self.max_iter:
            theta_values = dict()
            for i in range(self.n):
                l, obj_supbroblem, status_subproblem, _ = self.solve_subproblem(i, x)
                if status_subproblem == 'optimal':
                    theta_values[i] = obj_supbroblem
                    self.optimality_cuts[i].append(l)
                else:
                    sigma = self.solve_farkas_subproblem(i, x)
                    self.feasibility_cuts[i].append(sigma)
            
            if len(theta_values) == self.n:
                self.UPPER_BOUND = min(self.UPPER_BOUND, self.c @ x + np.mean(self.r) + np.mean(list(theta_values.values())))
            
            x, self.LOWER_BOUND = self.solve_master_problem()
            iteration += 1
            print("Iteration {} : UB={}, LB={}".format(iteration, self.UPPER_BOUND, self.LOWER_BOUND))
        
        y = []
        for i in range(self.n):
            _, _, _, yi = self.solve_subproblem(i, x)
            y.append(yi)
        
        return x, y, iteration, self.LOWER_BOUND
    
    def solve_subproblem(self, i, x):
        lamb = cp.Variable(self.m)
        subproblem_objective = cp.Maximize(lamb @ (self.h[i] - self.T[i] @ x))
        subproblem = cp.Problem(subproblem_objective, [self.W[i].T @ lamb <= self.q[i]])
        subproblem.solve()
        yi = subproblem.constraints[0].dual_value

        return lamb.value, subproblem.value, subproblem.status, yi
    
    def solve_farkas_subproblem(self, i, x):
        sigma = cp.Variable(self.m)
        subproblem_constraints = [sigma @ (self.h[i] - self.T[i] @ x) == 1, self.W[i].T @ sigma <= 0]
        subproblem = cp.Problem(cp.Minimize(0), subproblem_constraints)
        subproblem.solve()
        return sigma.value
    
    def solve_master_problem(self):
        x = cp.Variable(self.d)
        theta = cp.Variable(self.n)
        objective = cp.Minimize(self.c @ x + ((cp.sum(theta) + sum(self.r)) / self.n))
        constraints = [self.A @ x >= self.b]
        for i in range(self.n):
            constraints += [theta[i] >= -10e15]
        for i,list_of_lambdas in self.optimality_cuts.items():
            for l in list_of_lambdas:
                constraints += [theta[i] >= l @ (self.h[i] - self.T[i] @ x)]
        for i,list_of_sigmas in self.feasibility_cuts.items():
            for sigma in list_of_sigmas:
                constraints += [sigma @ (self.h[i] - self.T[i] @ x) <= 0]
        master_problem = cp.Problem(objective, constraints)
        master_problem.solve()
        return x.value, master_problem.value