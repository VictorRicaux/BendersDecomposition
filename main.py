from benders_decomposition import BendersDecomposition
import numpy as np


# Define the problem data
c = np.array([50, 100])  # Incremental costs of thermal generators
c_w = 40  # Incremental cost of wind generator
g_min = np.array([0, 0])  # Minimum power output of thermal generators
g_max = np.array([1000, 1000])  # Maximum power output of thermal generators

# Number of scenarios
n = 100

d = np.random.uniform(1300, 1700, n)  # Energy demand scenarios
w_max = np.random.uniform(100, 300, n)  # Max wind power scenarios
r = c_w * w_max

# Define matrices for constraints
A = np.array([[1,0], [-1, 0], [0, 1], [0, -1]])  # Coefficients for power balance constraints
b = np.array([[g_min[0]], [-g_max[0]], [g_min[1]], [-g_max[1]]])  # Upper bounds as right-hand side
b = b.reshape(-1)

# Define second-stage data
T = np.array([np.ones((1, 2)) for _ in range(n)])  # Coefficients for coupling constraints
W = np.array([np.array([[-1]]) for _ in range(n)])  # Coefficients for wind power constraints
h = np.array([[d[i] - w_max[i]] for i in range(n)])  # Right-hand side for each scenario
q = [np.array([-c_w]) for _ in range(n)]  # Cost coefficients for wind power in each scenario, ensure it's 2D

def main():
    benders_decomposition = BendersDecomposition(c, A, b, q, r, T, W, h)
    x_opt, y_opt, iterations, obj_value = benders_decomposition.solve_problem()
    print("Optimal value: ", obj_value)
    print("Optimal solution: ", x_opt)
    print("Iterations: ", iterations)

if __name__ == '__main__':
    main()