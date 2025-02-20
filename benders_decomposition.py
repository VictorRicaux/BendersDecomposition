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

        self.UPPER_BOUND = 10e15
        self.LOWER_BOUND = -10e15
        self.optimality_cuts = {i: [] for i in range(self.n)}
        self.feasibility_cuts = {i: [] for i in range(self.n)}
        self.x_init = self.initialize_master_problem()
        self.max_iter = 100