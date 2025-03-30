import numpy as np
import matplotlib.pyplot as plt

def jacobi_method(A, b, x0, tolerance=1e-6, maximum_number_of_iterations=100, x_exact=None):

    """
    Jacobi's method to solve a linear system Ax=b.

    Args:
        1.A (the matrix of the system)
        2.b (The vector b of Ax=b)
        3.x0 (The initial guess of the algorithm)
        4.tolerance (The tolerance (distance from the actual solution) for which we accept it to be a solution)
        5.maximum_number_of_iterations (self-explanatory)

    Returns:
        1.The approximate solution to Ax=b after some iterations
        2.The error aftr each iteration
    """

    n = len(b) #dimension of the system
    x = x0.copy()
    errors = [] #we will fill this with the errors later

    if x_exact is None:
        x_exact = np.linalg.solve(A, b) #exact solution to the system using

    for k in range(maximum_number_of_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
            x_new[i] = (b[i] - s) / A[i, i]

        current_error = np.linalg.norm(x_new - x_exact) #distance between what we just got and the actual solution
        errors.append(current_error)

        if current_error< tolerance:
            break #its close enough

        x = x_new.copy()

    return x_new, errors