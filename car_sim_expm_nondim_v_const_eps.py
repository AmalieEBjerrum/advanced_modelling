#from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import solve_ivp
from car_sim_nondim_v_const_eps import car_system
from car_sim_nondim_v_const_eps import solve_with_matrix_exponential
from car_sim_nondim_v_const_eps import nonlinear_term

# Simulation parameters
n = 3  # Number of cars
alpha = 0.5
beta = 0.3
v0 = 10
epsilon = 0.1

# Random initial conditions
np.random.seed(42)
initial_distance = np.random.randint(0, 11, n)  # Initial distances
sum_initial_distances = np.sum(initial_distance)  # Normalize distances
initial_distance = initial_distance / sum_initial_distances
initial_velocities = np.zeros(n)  # Initial velocities
initial_conditions = np.concatenate([initial_distance, initial_velocities])

# Time span and evaluation points
t_span = (0, 20)  # Simulation from t=0 to t=20
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Evaluation points

# Solve the system
derivatives, system_matrix, input_vector, update_statevector = car_system(initial_conditions, n, alpha, beta, v0, epsilon)
solution = solve_with_matrix_exponential(
    system_matrix,
    update_statevector,
    input_vector,
    lambda sv: nonlinear_term(sv, n, epsilon),
    t_eval
)

# Extract solution for plotting
solution_transposed = np.transpose(solution)

# Plot distances and velocities
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot distances
for i in range(0, 2 * n, 2):
    axs[0].plot(t_eval, solution_transposed[i].real, label=f"Car {i // 2 + 1}")  # Plot real part
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Distance")
axs[0].set_title("Distances of Cars Over Time")
axs[0].legend()
axs[0].grid()

# Plot velocities
for i in range(1, 2 * n, 2):
    axs[1].plot(t_eval, solution_transposed[i].real, label=f"Car {i // 2 + 1}")  # Plot real part
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Velocity")
axs[1].set_title("Velocities of Cars Over Time")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
