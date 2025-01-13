from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from car_sim_nondim import car_system
from car_sim_nondim import solve_with_matrix_exponential

# Parameters
n = 3  # Number of cars
alpha=0.01
beta= 0.5
omega=5 # Frequency of the velocity function


#random seeed
np.random.seed(42)

# Initial conditions
initial_distance = np.random.randint(0, 11, n)  # Initial distances
initial_velocities = np.zeros(n)  # Initial velocities
initial_conditions = np.concatenate([initial_distance, initial_velocities])

# Time span for simulation
t_span = (0, 20)  # From t=0 to t=20
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Evaluation points

derivatives, system_matrix, input_vector, update_statevector = car_system(0,initial_conditions, n, alpha,beta, omega)
#solve with matrix exponential
d_matrix, v_matrix = solve_with_matrix_exponential(t_eval, alpha, beta, n, omega)
#solution_transposed = np.transpose(solution)

# Assume t_eval, d_matrix, v_matrix, omega, and v0_function are already defined

# Calculate the initial velocity for plotting
initial_velocity = np.real(np.exp(t_eval* omega*1j))  # Take only the real part

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# First subplot: Distance
for i in range(n):
    axs[0].plot(t_eval, np.real(d_matrix[:, i]), label=f"Car {i+1}")  # Real part of distance
axs[0].set_ylabel("Distance (Real Part)")
axs[0].set_title("Real Part of Distances (d_matrix)")
axs[0].legend()
axs[0].grid()

# Second subplot: Velocity
for i in range(n):
    axs[1].plot(t_eval, np.real(v_matrix[:, i]), label=f"Car {i+1}")  # Real part of velocity
# Add the initial velocity (real part)
axs[1].plot(t_eval, initial_velocity, 'k--', label="Initial Velocity (v0)")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Velocity (Real Part)")
axs[1].set_title("Real Part of Velocities (v_matrix) and Initial Velocity")
axs[1].legend()
axs[1].grid()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

