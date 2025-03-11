import mlx.core as mx
import time
mx.enable_compile()

def update_populations(prey, predator, alpha, beta, delta, gamma, dt, sigma_x, sigma_y):
    # Generate random numbers for noise - works directly with batched input
    noise_x = (mx.random.uniform(shape=prey.shape) - 0.5) * sigma_x * mx.sqrt(2.0 / dt)
    noise_y = (mx.random.uniform(shape=predator.shape) - 0.5) * sigma_y * mx.sqrt(2.0 / dt)

    # Deterministic part of the Lotka-Volterra equations
    dx = (alpha * prey - beta * prey * predator) * dt
    dy = (delta * prey * predator - gamma * predator) * dt

    # Update populations with both deterministic and stochastic parts
    new_prey = prey + dx + noise_x
    new_predator = predator + dy + noise_y

    # Ensure non-negative population sizes
    new_prey = mx.maximum(new_prey, 0.0)
    new_predator = mx.maximum(new_predator, 0.0)

    return new_prey, new_predator

# Compile the function
update_populations_compiled = mx.compile(update_populations)

def lotka_volterra():
    # Parameters
    N = 1000 # Number of simulations
    timestep = 10000 # Total number of time steps
    alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5
    dt = 0.01 # Time step size

    # Initial conditions and stochastic parameters
    x0, y0 = 40.0, 9.0
    sigma_x, sigma_y = 0.1, 0.1 # Standard deviation of noise for prey and predators

    prey_populations = mx.zeros((N, timestep), dtype=mx.float32)
    predator_populations = mx.zeros((N, timestep), dtype=mx.float32)

    # Initialize populations
    prey_populations[:, 0] = x0
    predator_populations[:, 0] = y0

    for j in range(timestep - 1):
        prey_populations[:, j + 1], predator_populations[:, j + 1] = update_populations_compiled(
            prey_populations[:, j], 
            predator_populations[:, j],
            alpha, beta, delta, gamma, dt, sigma_x, sigma_y
        )

    # Calculate the mean across all simulations
    mean_prey = mx.mean(prey_populations, axis=0)
    mean_predator = mx.mean(predator_populations, axis=0)

    return mean_prey, mean_predator

lotka_volterra_compiled = mx.compile(lotka_volterra)

if __name__ == '__main__':
    # Compile the entire function for better performance
    mprey, mpred = lotka_volterra_compiled()
    #mprey, mpred = lotka_volterra()

    # Benchmarking the function 20 times and averaging
    num_runs = 2000
    execution_times = []

    print('Starting benchmarking...')
    for _ in range(num_runs):
        start_time = time.time()
        mprey, mpred = lotka_volterra_compiled()
        #mprey, mpred = lotka_volterra()
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

    average_execution_time = sum(execution_times) / num_runs
    print(f"Average execution time over {num_runs} runs: {average_execution_time:.4f} seconds")