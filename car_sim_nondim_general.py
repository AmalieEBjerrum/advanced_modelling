import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import expm


#Definition of the velocity function
def v0_function(t,omega):
    return  np.exp(1j*omega*t)#2*np.sin(omega * t)+3  # Example: exp(i * omega * t)

# Define the system of differential equations
def car_system(t,state_vector, n, alpha, beta,omega):
    """
    Parameters:
    - t: time
    - state_vector: state vector [d1, v1, ..., dn, vn]
    - n: number of cars
    - alpha, beta: parameters of the system
    - T: desired time gap
    - v0: velocity of the front car (constant)
    """
    # Initialize the update state vector
    update_statevector = np.zeros([2*n])
    # Extract state variables
    for i in range(n):
        update_statevector[2*i] = state_vector[i]       # Position d_i
        update_statevector[2*i+1] = state_vector[i+n]   # Velocity v_i

    
    # Define A matrix
    A = np.array([
    [0, -1],
    [ alpha, -(1+beta)]
    ])

    B= np.array([[0, 1], [0, 1]])

    system_matrix = np.zeros([2*n, 2*n])

    input_vector = np.zeros([2*n])
    input_vector[0] = 1
    input_vector[1] = 1

    for i in range(n-1):    
        system_matrix[2*i+2:2*i+4, 2*i:2*i+2] = B

    for i in range(n):
        system_matrix[2*i:2*i+2, 2*i:2*i+2] = A


    derivatives = system_matrix.dot(update_statevector) + input_vector*v0_function(t,omega)
    
    # Combine derivatives
    return derivatives, system_matrix, input_vector, update_statevector

def solve_with_matrix_exponential(t_eval, alpha, beta, N, omega,system_matrix, input_vector, update_statevector,K):
    """
    Solves the system using the matrix exponential.

    Parameters:
    - t_eval: Array of time points where the solution is evaluated
    - alpha: System parameter
    - beta: System parameter
    - N: Number of cars
    - omega: Frequency of oscillation (default is 1)

    Returns:
    - d_matrix: Distance matrix (time x cars)
    - v_matrix: Velocity matrix (time x cars)
    """
    # Precompute constants
    gamma1 = -(1j * omega + beta) / (omega**2 - 1j * omega - 1j * omega * beta - alpha)
    gamma2 = -(1j * omega + alpha) / (omega**2 - 1j * omega - 1j * omega * beta - alpha)
    # Initialize solution arrays
    num_times = len(t_eval)
    v_matrix = np.zeros((num_times, N), dtype=complex)  # Complex numbers for velocities
    d_matrix = np.zeros((num_times, N), dtype=complex)  # Complex numbers for distances

    # Solve the system for each time step and car
    for l, t in enumerate(t_eval):
        for k in range(N):
            d_t = gamma1* gamma2**(k)* v0_function(t, omega)
            v_t = gamma2**(k+1) * v0_function(t, omega)
            v_matrix[l, k] = v_t
            d_matrix[l, k] = d_t
    
    sol_dist=[]
    sol_vel=[]
    # Initialize x_matrix and y_matrix with the correct shape
    x_matrix = np.zeros((len(t_eval), N))  # For positions
    y_matrix = np.zeros((len(t_eval), N))  # For velocities

    # Iterate over time and cars
    for l, t in enumerate(t_eval):
        exp_system_matrix_t = expm(t * system_matrix)  # Compute e^(tC)
        z_t = exp_system_matrix_t @ (np.linalg.inv(system_matrix) @ input_vector * K + update_statevector) \
            - np.linalg.inv(system_matrix) @ input_vector * K  # Compute the state vector at time t
        
        # Assign position and velocity for each car
        for k in range(N):
            x_matrix[l, k] = z_t[2 * k]      # Position of car k
            y_matrix[l, k] = z_t[2 * k + 1]  # Velocity of car k

    sol_dist=d_matrix+x_matrix
    sol_vel=v_matrix+y_matrix
    return sol_dist, sol_vel