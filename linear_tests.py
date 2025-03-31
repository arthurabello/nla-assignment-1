import numpy as np
import matplotlib.pyplot as plt
from methods.jacobi import jacobi_method
from methods.gauss_seidel import gauss_seidel_method

def run_test(A, b, x0, tolerance, maximum_number_of_iterations, test_label):

    """
    Executes the Gauss Seidel and JAcobi methods.

    Args:
        1.A (the matrix of the system)
        2.b (The vector b of Ax=b)
        3.x0 (The initial guess of the algorithm)
        4.tolerance (The tolerance (distance from the actual solution) for which we accept it to be a solution)
        5.maximum_number_of_iterations (self-explanatory)
        6.test_label (self-explanatory)

    Returns:
        1.The exact solution to Ax=b using numpy's solver
        2.The list of errors from the Jacobi method
        3.The list of errors from the Gauss-Seidel method
    """

    try:
        x_exact = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print(f"{test_label}: Sigular Matrix/Weird Matrix")
        return None, None, None

    x_j, errors_j = jacobi_method(A, b, x0, tolerance=tolerance, maximum_number_of_iterations=max_iter, x_exact=x_exact)
    x_g, errors_g = gauss_seidel_method(A, b, x0, tolerance=tolerance, maximum_number_of_iterations=max_iter, x_exact=x_exact)
    
    return x_exact, errors_j, errors_g

def plot_subplots(tests, nrows, ncols, overall_title):

    """
     Plots the graphs associated to each test.
    Args:
        1.tests (A list of tuples containing (label, errors_j, errors_g) for each test)
        2.nrows (The number of rows in the subplot grid)
        3.ncols (The number of columns in the subplot grid)
        4.overall_title (The title for the entire figure)
    Returns:
        1.None (Displays the plot using plt.show())
    """

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, (label, errors_j, errors_g) in enumerate(tests):
        ax = axes[i]
        ax.plot(errors_j, 'o-', label='Jacobi')
        ax.plot(errors_g, 's-', label='Gauss-Seidel')
        ax.set_title(label)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error (norm)')
        ax.set_yscale('log')   #log scale for better visualization
        ax.legend()
        ax.grid(True)
    
    for j in range(i+1, len(axes)): #deactivates empty subplots
        axes[j].axis('off')
    
    plt.suptitle(overall_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


max_iter = 50
tolerance = 1e-8 #general config

# ================================
# 2X2 MATRICES TESTING
# ================================

tests_2x2 = []

A_2_1 = np.array([[4.0, 1.0], #diagonally dominant matrix (it should converge faster hopefully)
                  [2.0, 3.0]])
b_2_1 = np.array([1.0, 2.0])
x0_2_1 = np.zeros_like(b_2_1    )
label_2_1 = "2x2 Test 1: Dominant Matrix"
_, errors_j, errors_g = run_test(A_2_1, b_2_1, x0_2_1, tolerance, max_iter, label_2_1)
tests_2x2.append((label_2_1, errors_j, errors_g))

A_2_2 = np.array([[1.0, 2.0], #non diagonally dominant matrix (it can diverge or oscillate hopefully)
                  [2.0, 1.0]])
b_2_2 = np.array([3.0, 3.0])
x0_2_2 = np.zeros_like(b_2_2)
label_2_2 = "2x2 Test 2: Non-Dominant Matrix"
_, errors_j, errors_g = run_test(A_2_2, b_2_2, x0_2_2, tolerance, max_iter, label_2_2)
tests_2x2.append((label_2_2, errors_j, errors_g))

A_2_3 = np.array([[1e-4, 1.0], #nearly singular matrix 
                  [1.0, 1.0]])
b_2_3 = np.array([1.0, 2.0])
x0_2_3 = np.zeros_like(b_2_3)
label_2_3 = "2x2 Test 3: Nearly Singular Matrix"
_, errors_j, errors_g = run_test(A_2_3, b_2_3, x0_2_3, tolerance, max_iter, label_2_3)
tests_2x2.append((label_2_3, errors_j, errors_g))

plot_subplots(tests_2x2, nrows=1, ncols=3, overall_title="2x2 Matrices")


# ================================
# 3X3 MATRICES TESTING
# ================================

tests_3x3 = []

A_3_1 = np.array([[5.0, 1.0, 1.0],
                  [2.0, 6.0, 1.0], #diagonally dominant matrix (it'll converge faster (hopefully))
                  [1.0, 1.0, 7.0]])
b_3_1 = np.array([7.0, 8.0, 9.0])
x0_3_1 = np.zeros_like(b_3_1)
label_3_1 = "3x3 Test 1: Dominant Matrix"
_, errors_j, errors_g = run_test(A_3_1, b_3_1, x0_3_1, tolerance, max_iter, label_3_1)
tests_3x3.append((label_3_1, errors_j, errors_g))

A_3_2 = np.array([
    [1.0, 3.0, 1.0],
    [2.0, 1.0, 2.0],   #non-dominant matrix but invertible
    [1.0, 2.0, 2.0]
])
b_3_2 = np.array([5.0, 6.0, 7.0])
x0_3_2 = np.zeros_like(b_3_2)
label_3_2 = "3x3 Test 2: Non-Dominant Matrix"
_, errors_j, errors_g = run_test(A_3_2, b_3_2, x0_3_2, tolerance, max_iter, label_3_2)
tests_3x3.append((label_3_2, errors_j, errors_g))

A_3_3 = np.array([[1.0, 0.0, 1.0],
                  [-1.0, 1.0, 0.0],
                  [1.0, 2.0, -3.0]])    #hopefully jacobi will converge and Gauss-Seidel will have constant error
b_3_3 = np.array([1.0, 2.0, 3.0])
x0_3_3 = np.zeros_like(b_3_3)
label_3_3 = "3x3 Test 3: Special Event"
_, errors_j, errors_g = run_test(A_3_3, b_3_3, x0_3_3, tolerance, max_iter, label_3_3)
tests_3x3.append((label_3_3, errors_j, errors_g))

plot_subplots(tests_3x3, nrows=1, ncols=3, overall_title="3x3 Matrices")