import cvxpy as cp
import numpy as np

class ProgressiveHedging:

    def __init__(self, c, A, b, q, r, T, W, h, rho, max_iter=1000):
        self.c = c
        self.A = A
        self.b = b
        self.q = q
        self.r = r
        self.T = T
        self.W = W
        self.h = h
        self.rho = rho
        self.n = len(q)
        self.d = len(c)
        self.m = len(q[0])
        self.max_iter = max_iter
        self.x = self.initialize_problem()

        self.x_scenarios = np.zeros((self.n, self.d))
        self.y_scenarios = np.zeros((self.n, self.m))
        self.lambda_dual = np.zeros((self.n, self.d))

    def initialize_problem(self):
        x = cp.Variable(self.d)
        intialization_problem = cp.Problem(cp.Minimize(0), [self.A @ x >= self.b])
        intialization_problem.solve()
        return x.value
    
    def solve_problem(self):
        for k in range(self.max_iter):
            for i in range(self.n):
                self.x_scenarios[i], self.y_scenarios[i] = self.solve_scenario(i)
            
            self.x = self.solve_augmented_lagrangian()
            objective = self.c @ self.x + np.mean(self.r) + (1 / self.n) * sum(self.q[i] @ self.y_scenarios[i] for i in range(self.n))

            self.lambda_dual += self.rho * (self.x_scenarios - self.x)
            print("Iteration {} : x={}, obj={}".format(k, self.x, objective))
        
        return objective, self.x, self.y_scenarios, k
    
    def solve_scenario(self, i):
        y = cp.Variable(self.m, nonneg=True)
        x = cp.Variable(self.d)
        objective = cp.Minimize(self.q[i] @ y +  self.lambda_dual[i] @ (x - self.x) + self.rho / 2 * cp.sum_squares(x - self.x))
        constraints = [self.T[i] @ x + self.W[i] @ y == self.h[i]]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return x.value, y.value
    
    def solve_augmented_lagrangian(self):
        x = cp.Variable(self.d)
        objective = cp.Minimize(self.c @ x + sum(self.lambda_dual[i] @ (self.x_scenarios[i] - x) + self.rho / 2 * cp.sum_squares(self.x_scenarios[i] - x) for i in range(self.n)))
        constraints = [self.A @ x >= self.b]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return x.value







            


