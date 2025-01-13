import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define the system of differential equations
def car_system(t, state_vector, n, b, c, T, v0):
    """
    Parameters:
    - t: time
    - state_vector: state vector [d1, v1, ..., dn, vn]
    - n: number of cars
    - b, c: parameters of the system
    - T: desired time gap
    - v0: velocity of the front car (constant)
    """
    

    update_statevector = np.zeros([2*n])
    # Extract state variables
    for i in range(n):
        update_statevector[2*i] = state_vector[i]  # Index is 0 or divisible by 2
        update_statevector[2*i+1] = state_vector[i+n]
    # Define A matrix
    A = np.array([
    [0, -1],
    [ c, -b - T * c]
    ])

    B= np.array([[0, 1], [0, b]])

    system_matrix = np.zeros([2*n, 2*n])

    input_vector = np.zeros([2*n])
    input_vector[0] = 1*v0
    input_vector[1] = b*v0

    for i in range(n-1):    
        system_matrix[2*i+2:2*i+4, 2*i:2*i+2] = B

    for i in range(n):
        system_matrix[2*i:2*i+2, 2*i:2*i+2] = A


    derivatives = system_matrix.dot(update_statevector) + input_vector
    
    # Combine derivatives
    return derivatives, system_matrix, input_vector, update_statevector

def solve_with_matrix_exponential(system_matrix, update_statevector ,input_vector, t_span, t_eval):
    """
    Solves the system using the matrix exponential.

    Parameters:
    - System_matrix: System matrix
    - B: Input matrix
    - v0: Input velocity (scalar)
    - state_vector: Initial state vector [d1, v1, ..., dn, vn]
    - t_span: Time span (start, end)
    - t_eval: Time points to evaluate the solution

    Returns:
    - t_eval: Time points
    - solution: Solution matrix (state vectors at each time point)
    """
    # Precompute A^-1 B
    system_matrix_inv_B = np.linalg.solve(system_matrix, input_vector)

    # Initialize solution array
    solution = []

    # Solve the system for each time step
    for t in t_eval:
        exp_system_matrix_t = expm(t * system_matrix)  # Compute e^(tC)
        z_t = exp_system_matrix_t @ (np.linalg.inv(system_matrix)@ input_vector+update_statevector)-np.linalg.inv(system_matrix)@input_vector  # Compute e^(tC) A^-1 B
        solution.append(z_t)

    return np.array(solution)