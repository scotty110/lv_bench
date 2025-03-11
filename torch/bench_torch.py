import torch
import time

# Set global device and dtype
#torch.set_num_threads(1)
DEVICE = torch.device("cuda")
DTYPE = torch.float32

def update_populations(
    j: int,
    prey_populations: torch.Tensor,
    predator_populations: torch.Tensor,
    noise_x: torch.Tensor,
    noise_y: torch.Tensor,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
    dt: float,
    sigma_x: float,
    sigma_y: float,
    noise_scale: torch.Tensor,
    zero_tensor: torch.Tensor
):
    # Generate random numbers for noise (in-place operations)
    noise_x.uniform_(-0.5, 0.5).mul_(sigma_x * noise_scale)
    noise_y.uniform_(-0.5, 0.5).mul_(sigma_y * noise_scale)
    
    # Current populations
    prey = prey_populations[:, j]
    predator = predator_populations[:, j]
    
    # Deterministic part of the Lotka-Volterra equations
    dx = (alpha * prey - beta * prey * predator) * dt
    dy = (delta * prey * predator - gamma * predator) * dt
    
    # Update populations with both deterministic and stochastic parts
    new_prey = prey + dx + noise_x
    new_predator = predator + dy + noise_y
    
    # Ensure non-negative population sizes (in-place when possible)
    prey_populations[:, j+1] = torch.maximum(new_prey, zero_tensor)
    predator_populations[:, j+1] = torch.maximum(new_predator, zero_tensor)

compiled_update_populations = torch.compile(update_populations, backend="inductor")

@torch.jit.script
def lotka_volterra_vectorized(
    device: torch.device, 
    dtype: torch.dtype = torch.float32
):
    # Parameters
    N = 1000             # Number of simulations
    timestep = 10_000    # Total number of time steps
    alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5
    dt = 0.01            # Time step size

    # Initial conditions and stochastic parameters
    x0, y0 = 40.0, 9.0
    sigma_x, sigma_y = 0.1, 0.1  # Standard deviation of noise

    # Initialize populations
    prey_populations = torch.zeros((N, timestep), device=device, dtype=dtype)
    predator_populations = torch.zeros((N, timestep), device=device, dtype=dtype)
    
    prey_populations[:, 0] = x0
    predator_populations[:, 0] = y0
    
    # Scale factor for noise
    noise_scale = torch.sqrt(torch.tensor(2.0 / dt, device=device, dtype=dtype))
    zero_tensor = torch.tensor(0.0, device=device, dtype=dtype)
    
    # Pre-allocate tensors for in-loop operations
    noise_x = torch.zeros(N, device=device, dtype=dtype)
    noise_y = torch.zeros(N, device=device, dtype=dtype)
    
    for j in range(timestep-1):
        compiled_update_populations(
            j, prey_populations, predator_populations, noise_x, noise_y,
            alpha, beta, delta, gamma, dt, sigma_x, sigma_y, noise_scale, zero_tensor
        )
    
    # Calculate the mean across all simulations
    mean_prey = torch.mean(prey_populations, dim=0)
    mean_predator = torch.mean(predator_populations, dim=0)
    
    return mean_prey, mean_predator

def benchmark():
    execution_times = []
    # Warmup and compile
    with torch.no_grad():
        mean_prey, mean_predator = lotka_volterra_vectorized(DEVICE, DTYPE)
    
    print('Starting Benchmark...')
    n_runs = 2000
    for i in range(n_runs):
        start_time = time.time()
        with torch.no_grad():
            mean_prey, mean_predator = lotka_volterra_vectorized(DEVICE, DTYPE)
        end_time = time.time()
        execution_time = (end_time - start_time)
        execution_times.append(execution_time)
    
    average_execution_time = sum(execution_times) / len(execution_times)
    print(f"Average execution time over {n_runs} runs: {average_execution_time:.4f} ")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    benchmark()
