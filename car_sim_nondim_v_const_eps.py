import numpy as np
from scipy.linalg import expm
#import matplotlib.pyplot as plt

# Define the system of differential equations
def car_system(state_vector, n, alpha, beta, v0, epsilon):
    """
    Parameters:
    - state_vector: state vector [d1, v1, ..., dn, vn]
    - n: number of cars
    - alpha, beta: parameters of the system
    - v0: velocity of the front car (constant)
    - epsilon: strength of the nonlinear term
    """
    # Initialize update state vector
    update_statevector = np.zeros(2 * n)
    for i in range(n):
        update_statevector[2 * i] = state_vector[i]      # Position
        update_statevector[2 * i + 1] = state_vector[i + n]  # Velocity

    # Define A and B matrices for the linear system
    A = np.array([[0, -1], [alpha, -(1 + beta)]])
    B = np.array([[0, 1], [0, 1]])

    # Build the system matrix
    system_matrix = np.zeros((2 * n, 2 * n))
    for i in range(n - 1):
        system_matrix[2 * i + 2:2 * i + 4, 2 * i:2 * i + 2] = B
    for i in range(n):
        system_matrix[2 * i:2 * i + 2, 2 * i:2 * i + 2] = A

    # Input vector (driving force for the lead car)
    input_vector = np.zeros(2 * n)
    input_vector[0] = v0  # Position input
    input_vector[1] = v0  # Velocity input

    # Nonlinear term: Avoidance of collisions
    nonlinear_vector = np.zeros(2 * n)
    for i in range(n - 1):  # Apply nonlinear term only to trailing cars
        if np.abs(update_statevector[2 * i]) > 1e-8:  # Avoid division by zero
            nonlinear_vector[2 * i + 1] = epsilon / update_statevector[2 * i]  # Add repulsion

    # Compute derivatives
    derivatives = system_matrix.dot(update_statevector) + input_vector - nonlinear_vector
    return derivatives, system_matrix, input_vector, update_statevector

# Solve the system using the matrix exponential
def solve_with_matrix_exponential(system_matrix, update_statevector, input_vector, nonlinear_term, t_eval):
    """
    Solves the system using the matrix exponential, incorporating a nonlinear term.

    Parameters:
    - system_matrix: Linear dynamics matrix
    - update_statevector: Initial state vector
    - input_vector: External input vector
    - nonlinear_term: Function to compute nonlinear corrections
    - t_eval: Time points for evaluation

    Returns:
    - solution: Solution array (state vectors at each time point)
    """
    solution = []

    for t in t_eval:
        # Compute the matrix exponential solution for the linear part
        exp_system_matrix_t = expm(t * system_matrix)
        z_t = exp_system_matrix_t @ (np.linalg.inv(system_matrix) @ input_vector + update_statevector) - \
              np.linalg.inv(system_matrix) @ input_vector

        # Add the nonlinear term at each step
        nonlinear_correction = nonlinear_term(update_statevector)
        z_t -= nonlinear_correction
        solution.append(z_t)

    return np.array(solution)

# Nonlinear correction function
def nonlinear_term(state_vector, n, epsilon):
    nonlinear_vector = np.zeros(2 * n)
    for i in range(n - 1):  # Nonlinear term applies to trailing cars
        if np.abs(state_vector[2 * i]) > 1e-8:  # Avoid division by zero
            nonlinear_vector[2 * i + 1] = epsilon / state_vector[2 * i]
    return nonlinear_vector