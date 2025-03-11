# Simple Lotka Volterra Monte Carlo Simulation
Why:
1. Monte Carlo was easy to Vecotrize (really were benchmarking the compilers?).
2. LV was some actual work to do (note the actual implementation might not be correct, but it is doing work).

## Results

### 9950x
| Library | Time (sec) |
|---|---|
| numpy | 0.1086 |
| fortran | 0.0922 |

#### m2
| Library | Time (sec) |
|---|---|
| mlx | 0.1077 |
| numpy | 0.1321 |
| fortran | 0.0502 |

### cuda (3090ti)
| Library | Time (sec) |
|---|---|
| jax | 0.1063 |
| torch | 0.2943 |
