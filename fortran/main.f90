program lotka_volterra_vectorized
    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64

    implicit none

    ! Parameters
    integer, parameter :: N = 1000          ! Number of simulations
    integer, parameter :: timestep = 10000   ! Total number of time steps
    real(dp) :: alpha, beta, delta, gamma
    real(dp), parameter :: dt = 0.01_dp      ! Time step size

    ! Initial conditions and stochastic parameters
    real(dp) :: x0, y0, sigma_x, sigma_y
    real(dp) :: dx(N), dy(N)
    real(dp) :: noise_x(N), noise_y(N)

    real(dp), allocatable :: prey_populations(:,:), predator_populations(:,:)

    ! Variables for loops
    integer :: i, j

    ! Allocate and initialize variables for mean populations
    real(dp), dimension(timestep) :: mean_prey, mean_predator

    ! Initialize parameters and initial populations
    alpha = 1.0_dp
    beta = 0.1_dp
    delta = 0.075_dp
    gamma = 1.5_dp
    x0 = 40.0_dp
    y0 = 9.0_dp
    sigma_x = 0.1_dp   ! Standard deviation of noise for prey
    sigma_y = 0.1_dp   ! Standard deviation of noise for predators

    allocate(prey_populations(N, timestep))
    allocate(predator_populations(N, timestep))

    ! Initialize populations and results arrays
    prey_populations(:, 1) = x0
    predator_populations(:, 1) = y0

    call random_seed()

    do j = 1, timestep - 1
        ! Generate random numbers for noise
        call random_number(noise_x)
        call random_number(noise_y)

        ! Calculate stochastic components
        noise_x = (noise_x - 0.5_dp) * sigma_x * sqrt(2.0_dp / dt)
        noise_y = (noise_y - 0.5_dp) * sigma_y * sqrt(2.0_dp / dt)

        ! Deterministic part of the Lotka-Volterra equations
        dx = (alpha * prey_populations(:, j) - beta * prey_populations(:, j) * predator_populations(:, j)) * dt
        dy = (delta * prey_populations(:, j) * predator_populations(:, j) - gamma * predator_populations(:, j)) * dt

        ! Update populations with both deterministic and stochastic parts
        prey_populations(:, j+1) = prey_populations(:, j) + dx + noise_x
        predator_populations(:, j+1) = predator_populations(:, j) + dy + noise_y

        ! Ensure non-negative population sizes
        where (prey_populations(:, j+1) < 0.0_dp)
            prey_populations(:, j+1) = 0.0_dp
        end where

        where (predator_populations(:, j+1) < 0.0_dp)
            predator_populations(:, j+1) = 0.0_dp
        end where
    end do

    ! Calculate the mean across all simulations
    mean_prey = sum(prey_populations, dim=1) / N
    mean_predator = sum(predator_populations, dim=1) / N

end program lotka_volterra_vectorized