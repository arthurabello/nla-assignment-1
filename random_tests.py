import numpy as np
import matplotlib.pyplot as plt
from methods.jacobi import jacobi_method
from methods.gauss_seidel import gauss_seidel_method

A_orig = np.array([
    [1.0,  2.0, -2.0], #original matrix and vectors
    [1.0,  1.0,  1.0],
    [2.0,  2.0,  1.0]
])
b = np.array([1.0, 2.0, 3.0])
x0 = np.zeros_like(b)

max_iter = 50
tol = 1e-8
n_experimentos = 10

def run_methods(A, b, x0):

    """
    Executes the Gauss Seidel and JAcobi methods.

    Args:
        1.A (the matrix of the system)
        2.b (The vector b of Ax=b)
        3.x0 (The initial guess of the algorithm)

    Returns:
        1.Jacobi's result
        2.Gauss-Seidel's result
    """

    try:
        x_exact = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None

    x_j, err_j = jacobi_method(A, b, x0, tolerance=tol, 
                               maximum_number_of_iterations=max_iter, 
                               x_exact=x_exact)
    x_g, err_g = gauss_seidel_method(A, b, x0, tolerance=tol, 
                                     maximum_number_of_iterations=max_iter, 
                                     x_exact=x_exact)
    return err_j, err_g

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 18))
axes = axes.flatten()  #more easily indexed

for i in range(n_experimentos):
    perturb = np.random.uniform(-0.2, 0.2, size=A_orig.shape)
    A_pert = A_orig + perturb

    err_j, err_g = run_methods(A_pert, b, x0)
    if err_j is None: #we do not consider singular matrices here
        axes[i].set_title(f'Complete Perturbation {i+1}\n(Singular)')
        axes[i].axis('off')
        continue

    axes[i].plot(err_j, 'o-', label='Jacobi')
    axes[i].plot(err_g, 's-', label='Gauss-Seidel')
    axes[i].set_yscale('log')
    axes[i].set_xlabel('Iteration')
    axes[i].set_ylabel('Error(norm)')
    axes[i].set_title(f'Complete Perturbation {i+1}')
    axes[i].legend()
    axes[i].grid(True)

plt.suptitle('Complete Perturbation - 10 Experiments', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 18))
axes = axes.flatten()

for i in range(n_experimentos):
    perturb = np.zeros_like(A_orig)
    diag_pert = np.random.uniform(-0.2, 0.2, size=A_orig.shape[0])
    np.fill_diagonal(perturb, diag_pert)  #only diagonal experiments
    A_pert = A_orig + perturb

    err_j, err_g = run_methods(A_pert, b, x0)
    if err_j is None:
        axes[i].set_title(f'Diagonal Perturbation {i+1}\n(Singular)')
        axes[i].axis('off')
        continue

    axes[i].plot(err_j, 'o-', label='Jacobi')
    axes[i].plot(err_g, 's-', label='Gauss-Seidel')
    axes[i].set_yscale('log')
    axes[i].set_xlabel('Iteration')
    axes[i].set_ylabel('Error(norm)')
    axes[i].set_title(f'Diagonal Perturbation {i+1}')
    axes[i].legend()
    axes[i].grid(True)

plt.suptitle('Diagonal Perturbation - 10 Experiments', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 18)) #only non-diagonal perturbation
axes = axes.flatten()

for i in range(n_experimentos):
    perturb = np.zeros_like(A_orig)
    mask = np.ones_like(A_orig, dtype=bool) #masks non-diagonal elemtns
    np.fill_diagonal(mask, False)
    perturb[mask] = np.random.uniform(-0.2, 0.2, size=A_orig[mask].shape)
    A_pert = A_orig + perturb

    err_j, err_g = run_methods(A_pert, b, x0)
    if err_j is None:
        axes[i].set_title(f'Non-diagonal perturbation {i+1}\n(Singular)')
        axes[i].axis('off')
        continue

    axes[i].plot(err_j, 'o-', label='Jacobi')
    axes[i].plot(err_g, 's-', label='Gauss-Seidel')
    axes[i].set_yscale('log')
    axes[i].set_xlabel('Iteration')
    axes[i].set_ylabel('Error (norm)')
    axes[i].set_title(f'Non-diagonal perturbation {i+1}')
    axes[i].legend()
    axes[i].grid(True)

plt.suptitle('Non-Diagonal Perturbation - 10 Experiments', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
