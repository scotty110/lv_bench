import jax
from jax import random, jit, lax, config
import jax.numpy as jnp
import time

def process_timesteps(prey_populations, predator_populations, alpha, beta, delta, gamma, dt, sigma_x, sigma_y, timestep, key):
    # Precompute constants
    noise_scale_x = sigma_x * jnp.sqrt(2.0 / dt)
    noise_scale_y = sigma_y * jnp.sqrt(2.0 / dt)
    
    def body_fun(j, vals):
        prey_pops, pred_pops, curr_key = vals
    
        # Generate both noise terms with a single key split
        curr_key, subkey = random.split(curr_key)
        noise = random.uniform(subkey, (2, prey_pops.shape[0])) - 0.5
        noise_x = noise[0] * noise_scale_x
        noise_y = noise[1] * noise_scale_y

        # Deterministic part (vectorized)
        prey = prey_pops[:, j]
        predator = pred_pops[:, j]
        dx = (alpha * prey - beta * prey * predator) * dt
        dy = (delta * prey * predator - gamma * predator) * dt

        # Update populations
        new_prey = prey + dx + noise_x
        new_predator = predator + dy + noise_y

        # Ensure non-negative population sizes
        prey_pops = prey_pops.at[:, j + 1].set(jnp.maximum(new_prey, 0.0))
        pred_pops = pred_pops.at[:, j + 1].set(jnp.maximum(new_predator, 0.0))
        
        return (prey_pops, pred_pops, curr_key)
    
    # Run the loop with fori_loop for JIT compatibility
    initial_vals = (prey_populations, predator_populations, key)
    final_prey_pops, final_pred_pops, _ = lax.fori_loop(0, timestep - 1, body_fun, initial_vals)
    
    return final_prey_pops, final_pred_pops

process_timesteps_compiled = jit(process_timesteps)

def lotka_volterra_vectorized():
    # Parameters
    N = 1000 # Number of simulations
    timestep = 10000 # Total number of time steps
    alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5
    dt = 0.01 # Time step size

    # Initial conditions and stochastic parameters
    x0, y0 = 40.0, 9.0
    sigma_x, sigma_y = 0.1, 0.1 # Standard deviation of noise for prey and predators

    prey_populations = jnp.zeros((N, timestep), dtype=jnp.float32)
    predator_populations = jnp.zeros((N, timestep), dtype=jnp.float32)

    # Initialize populations
    prey_populations = prey_populations.at[:, 0].set(x0)
    predator_populations = predator_populations.at[:, 0].set(y0)
    
    # Process all timesteps
    key = random.PRNGKey(0)
    prey_populations, predator_populations = process_timesteps_compiled(
        prey_populations, predator_populations, alpha, beta, delta, gamma, dt, sigma_x, sigma_y, timestep, key
    )

    # Calculate the mean across all simulations
    mean_prey = jnp.mean(prey_populations, axis=0)
    mean_predator = jnp.mean(predator_populations, axis=0)

    return mean_prey, mean_predator

lotka_volterra_vectorized_compiled = jit(lotka_volterra_vectorized)

if __name__ == '__main__':
    # Warm up
    mprey, mpred = lotka_volterra_vectorized_compiled()
    #mprey, mpred = lotka_volterra_vectorized()

    num_runs = 2000 
    execution_times = []

    print('Starting benchmarking...')
    for _ in range(num_runs):
        start_time = time.time()

        mprey, mpred = lotka_volterra_vectorized_compiled()
        #mprey, mpred = lotka_volterra_vectorized()

        mprey.block_until_ready()
        mpred.block_until_ready()

        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

    average_execution_time = sum(execution_times) / num_runs
    print(f"Average execution time over {num_runs} runs: {average_execution_time:.4f} seconds")
