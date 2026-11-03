"""Stationary inter-server distance in the light-traffic regime.

Listing from Appendix A of the thesis: discretises the fixed-point integral
equation for the limiting distribution and solves it by power iteration.
"""
import numpy as np
from tqdm import tqdm

# 1. Define the Transition Kernel K(x|y) 
def P1(x, y):
    if x <= y:
        return (np.exp(y) * (1 - np.exp(-x))) / (np.exp(1) - 1)
    else:
        return (1 - np.exp(y - x)) + (np.exp(y) * (1 - np.exp(-x))) / (np.exp(1) - 1)

def P2(x, y):
    if y >= x:
        return np.exp(-y) * (np.exp(x) - 1) + (np.exp(x - y) * (1 - np.exp(-x))) / (np.exp(1) - 1)
    else:
        return (1 - np.exp(-y)) + (np.exp(x - y) * (1 - np.exp(-x))) / (np.exp(1) - 1)

def K_exp(x, y):
    return y * P1(x, y) + (1-y) * P2(x, y)

# 2. Set up Discretization 
N = 10000
dy = 1.0 / N
y_grid = np.linspace(0.5 * dy, 1.0 - 0.5 * dy, N)
x_grid = np.linspace(0, 1.0, N)

# 3. Pre-compute the Transition Matrix 
K_matrix = np.zeros((N, N))
for i in tqdm(range(N)):
    for j in range(N):
        K_matrix[i, j] = K_exp(x_grid[i], y_grid[j])

# 4. Perform Power Iteration 
f_n = np.full(N, 1.0) * dy
for iteration in range(100):
    F_n_plus_1_cdf = K_matrix.dot(f_n)
    F_with_zero = np.insert(F_n_plus_1_cdf, 0, 0)
    f_n_plus_1 = np.diff(F_with_zero)

    if np.allclose(f_n, f_n_plus_1, atol=1e-6):
        print(f"Converged after {iteration} iterations.")
        f_n = f_n_plus_1
        break
    f_n = f_n_plus_1

# 5. Final Numerical Results 
f_stationary_pdf, F_stationary_cdf = f_n / dy, np.cumsum(f_n)
