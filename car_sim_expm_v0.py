from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from car_sim_v0 import car_system
from car_sim_v0 import solve_with_matrix_exponential
from car_sim_v0 import v0_function

# Parameters
n = 2  # Number of cars
b = 0.5
c = 1.0
T = 1  # Desired time gap
#v0 = 10.0  # Velocity of the front car (constant)

#random seeed
np.random.seed(42)

# Initial conditions
initial_distance = np.random.randint(30, 40, n)  # Initial distances
initial_velocities = np.zeros(n)  # Initial velocities
initial_conditions = np.concatenate([initial_distance, initial_velocities])

# Time span for simulation
t_span = (0, 100)  # From t=0 to t=20
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Evaluation points


derivatives, system_matrix, input_vector, update_statevector = car_system(0, initial_conditions, n, b, c, T)
#solve with matrix exponential
solution = solve_with_matrix_exponential(system_matrix, update_statevector ,input_vector, t_span, t_eval)


solution_transposed = np.transpose(solution)


# Combined subplot for distances and velocities
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot distances (even indices)
for i in range(0, 2 * n, 2):
    axs[0].plot(t_eval, solution_transposed[i].real, label=f"Car {i // 2 + 1}")  # Plot real part
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Distance")
axs[0].set_title("Distances of Cars Over Time")
axs[0].legend()
axs[0].grid()

# Plot velocities (odd indices)
for i in range(1, 2 * n, 2):
    axs[1].plot(t_eval, solution_transposed[i].real, label=f"Car {i // 2 + 1}")  # Plot real part
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Velocity")
axs[1].set_title("Velocities of Cars Over Time")
axs[1].legend()
axs[1].grid()

# Add v0_function as Car 0
v0_values = [v0_function(t).real for t in t_eval]
axs[1].plot(t_eval, v0_values, label="Car 0 (v0)", linestyle="--")


plt.tight_layout()
plt.show()