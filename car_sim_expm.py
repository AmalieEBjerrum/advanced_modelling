from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from car_sim import car_system
from car_sim import solve_with_matrix_exponential

# Parameters
n = 3  # Number of cars
b = 0.5
c = 1.0
T = 1  # Desired time gap
v0 = 10.0  # Velocity of the front car (constant)

#random seeed
np.random.seed(42)

# Initial conditions
initial_distance = np.random.randint(0, 11, n)  # Initial distances
initial_velocities = np.zeros(n)  # Initial velocities
initial_conditions = np.concatenate([initial_distance, initial_velocities])

# Time span for simulation
t_span = (0, 20)  # From t=0 to t=20
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Evaluation points




derivatives, system_matrix, input_vector, update_statevector = car_system(0, initial_conditions, n, b, c, T, v0)
#solve with matrix exponential
solution = solve_with_matrix_exponential(system_matrix, update_statevector ,input_vector, t_span, t_eval)


solution_transposed = np.transpose(solution)


# Plot even row indices (including 0)
plt.figure(figsize=(10, 6))
for i in range(0, 2 * n, 2):  # Even indices: 0, 2, 4, ...
    plt.plot(t_eval, solution_transposed[i], label=f"Car {i//2}")
plt.xlabel("Time")
plt.ylabel("Distance")
plt.title("Even Row Indices (Including 0)")
plt.legend()
plt.grid()
plt.show()

#plot the odd row indices
plt.figure(figsize=(10, 6))
for i in range(1, 2 * n, 2):  # Odd indices: 1, 3, 5, ...
    plt.plot(t_eval, solution_transposed[i], label=f"Car {i//2}")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Odd Row Indices")
plt.legend()
plt.grid()
plt.show()
