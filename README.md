# Project 1 – Numerical Linear Algebra

This repository contains the implementation of the iterative methods **Jacobi** and **Gauss-Seidel** for solving linear systems. The project performs a comparative analysis between the two methods through tests on 2×2 and 3×3 matrices, and it investigates the effects of perturbations on the coefficient matrix.


## Project Description

In this project, the **Jacobi** and **Gauss-Seidel** methods are implemented to solve the linear system **Ax = b**. The tests include matrices with different properties:
- **Diagonally Dominant:** Faster convergence.
- **Non-Dominant:** May lead to oscillation or divergence.
- **Nearly Singular (Ill-Conditioned):** Cases where numerical stability is challenging.

Additionally, the project examines the impact of different types of perturbations on the original matrix:
- Perturbation across the entire matrix.
- Perturbation only on the diagonal.
- Perturbation on off-diagonal elements only.

The results of these tests are visualized using graphs that show the evolution of the error (calculated as the norm of the difference between the approximation and the exact solution) on a logarithmic scale.

## Features

- **Iterative Methods Implementation:**
  - *Jacobi:* Updates all components using values from the previous iteration.
  - *Gauss-Seidel:* Uses updated values immediately during the iteration.
- **Testing Suite:**
  - Evaluation with 2×2 and 3×3 matrices, including special cases.
- **Perturbation Analysis:**
  - Investigates how different perturbations affect convergence.
- **Graphical Visualization:**
  - Displays the evolution of the error over iterations for both methods.
- **Method Comparison:**
  - Highlights scenarios where one method converges while the other may exhibit constant error or diverge.


## Project Structure

- **[methods/](https://github.com/arthurabello/nla-assignment-1/tree/main/methods)**
  - [`jacobi.py`](https://github.com/arthurabello/nla-assignment-1/blob/main/methods/jacobi.py): Implementation of the Jacobi method.
  - [`gauss_seidel.py`](https://github.com/arthurabello/nla-assignment-1/blob/main/methods/gauss_seidel.py): Implementation of the Gauss-Seidel method.
- [**linear_tests.py**](https://github.com/arthurabello/nla-assignment-1/blob/main/linear_tests.py)
  - Main script for running tests.
  - Contains functions to execute the methods, compute errors, and plot graphs.
- **Documentation:**
  - **[project_statement.pdf](https://github.com/arthurabello/nla-assignment-1/blob/main/project_statement.pdf)**: The project statement.
  - **[project.pdf](https://github.com/arthurabello/nla-assignment-1/blob/main/project.pdf)**: The full project.
---

## How to Run

### Prerequisites
- **Python 3.x**
- Libraries:
  - [NumPy](https://numpy.org/)
  - [Matplotlib](https://matplotlib.org/)

### Instructions
1. Clone the repository:

   ```bash
   git clone https://github.com/arthurabello/nla-assignment-1.git

2. Go to the directory:

    ```bash
    cd nla-assignment-1
    ```
3. Run whichever **test** script you want to. example:

    ```bash
    python random_tests.py
    ```

## License

See License **[here](https://github.com/arthurabello/nla-assignment-1/blob/main/LICENSE)** for more info.

## Author

This guy **[here](https://github.com/arthurabello)**.