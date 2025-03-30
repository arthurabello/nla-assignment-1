import numpy as np

def gauss_seidel_method(A, b, x0, tolerance=1e-6, maximum_number_of_iterations=100, x_exact=None):

    """
    Gauss-Seidel's method to solve a linear system Ax=b.

    Args:
        1.A (the matrix of the system)
        2.b (The vector b of Ax=b)
        3.x0 (The initial guess of the algorithm)
        4.tolerance (The tolerance (distance from the actual solution) for which we accept it to be a solution)
        5.maximum_number_of_iterations (self-explanatory)

    Returns:
        1.The approximate solution to Ax=b after some iterations
        2.The error after each iteration
    """

    n = len(b) #dimension
    x = x0.copy()
    list_of_errors = [] #we will fill this with the errors later

    if x_exact is None:
        x_exact = np.linalg.solve(A, b) #actual solution of the system

    for k in range(maximum_number_of_iterations):
        x_old = x.copy()
        for i in range(n):
            s = 0
            for j in range(i):
                s += A[i, j] * x[j]
            for j in range(i+1, n):
                s += A[i, j] * x_old[j]

            x[i] = (b[i] - s) / A[i, i]

        current_error = np.linalg.norm(x - x_exact) #distance between what we just got and the actual solution
        list_of_errors.append(current_error)

        if current_error < tolerance:
            break #it's close enough

    return x, list_of_errors