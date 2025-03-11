import numpy as np
import time
from numba import jit

@jit(nopython=True)
def lotka_volterra_vectorized():
    # Parameters
    N = 1000 # Number of simulations
    timestep = 10000 # Total number of time steps
    alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5
    dt = 0.01 # Time step size

    # Initial conditions and stochastic parameters
    x0, y0 = 40.0, 9.0
    sigma_x, sigma_y = 0.1, 0.1 # Standard deviation of noise for prey and predators

    prey_populations = np.zeros((N, timestep), dtype=np.float32)
    predator_populations = np.zeros((N, timestep), dtype=np.float32)

    # Initialize populations
    prey_populations[:, 0] = x0
    predator_populations[:, 0] = y0
    for j in range(timestep - 1):
        # Generate random numbers for noise
        noise_x = (np.random.random(N) - 0.5) * sigma_x * np.sqrt(2.0 / dt)
        noise_y = (np.random.random(N) - 0.5) * sigma_y * np.sqrt(2.0 / dt)

        # Deterministic part of the Lotka-Volterra equations
        prey = prey_populations[:, j]
        predator = predator_populations[:, j]
        dx = (alpha * prey - beta * prey * predator) * dt
        dy = (delta * prey * predator - gamma * predator) * dt

        # Update populations with both deterministic and stochastic parts
        new_prey = prey + dx + noise_x
        new_predator = predator + dy + noise_y

        # Ensure non-negative population sizes
        prey_populations[:, j + 1] = np.maximum(new_prey, 0.0)
        predator_populations[:, j + 1] = np.maximum(new_predator, 0.0)

    # Calculate the mean across all simulations
    mean_prey = np.zeros(timestep, dtype=np.float32)
    mean_predator = np.zeros(timestep, dtype=np.float32)
    for t in range(timestep):
        mean_prey[t] = np.mean(prey_populations[:, t])
        mean_predator[t] = np.mean(predator_populations[:, t])

    return mean_prey, mean_predator

if __name__ == '__main__':
    mprey, mpred = lotka_volterra_vectorized()

    # Benchmarking the function 2000 times and averaging
    num_runs = 2000 
    execution_times = []

    print('Starting benchmarking...')
    for _ in range(num_runs):
        start_time = time.time()
        mprey, mpred = lotka_volterra_vectorized()
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

    average_execution_time = sum(execution_times) / num_runs
    print(f"Average execution time over {num_runs} runs: {average_execution_time:.4f} seconds")
