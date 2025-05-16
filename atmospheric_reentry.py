import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Constants and parameters for the simulation
c_d = 10      # Drag coefficient
m = 10        # Spacecraft mass (kg)
mu = 398600   # Earth's gravitational parameter (km^3/s^2)
rho_0 = 1.225 # Sea level atmospheric density (kg/m^3)
h_scale = 8   # Atmospheric scale height (km)
R_E = 6371    # Earth's radius in km
dt = 1        # Time step (s)
steps = 100   # Number of time steps
d = 5         # Horizontal radar distance (km)
shape_weibull = 1.2 # Weibull shape parameter
scale_weibull = 0.08 # Weibull scale parameter
mean_gauss = 0      # Mean of Gaussian measurement noise
std_gauss = 6       # Standard deviation of Gaussian measurement noise

# Initial conditions for altitude and velocity
x0 = np.array([200, -3])  

# Initialize random number generator for reproducibility
rng = np.random.default_rng(seed=2024)


def weibull_with_scale(shape, scale, size):
    """
    Generates Weibull-distributed process noise scaled by a factor.
    """
    noise = scale * rng.weibull(shape, size)
    return noise if size > 1 else noise[0]


def multinomial_resample(particles,weights):
    """
    Performs multinomial resampling to select particle indices based on weights.
    """
    N = len(weights)
    return particles[rng.choice(np.arange(N), size=N, p=weights), :]


def spacecraft_dynamics(state, dt, process_noise):
    """
    Computes spacecraft dynamics for one time step.
    Includes atmospheric drag and gravitational effects.
    """
    h = state[0]  # Altitude (km)
    v = state[1]  # Velocity (km/s)
    r = R_E + h   # Distance from Earth's center (km)
    rho = rho_0 * np.exp(-h / h_scale)  # Atmospheric density

    # Compute accelerations due to drag and gravity
    a_drag = -c_d * rho * v * abs(v) / m
    a_grav = -mu / r**2

    # Update altitude and velocity for the next time step
    h_next = h + v * dt
    v_next = v + (a_drag + a_grav) * dt + process_noise

    return np.array([h_next, v_next])


def radar_measurement(state, measurement_noise):
    """
    Simulates radar measurement of altitude with added Gaussian noise.
    """
    h = state[0]  # Altitude (km)
    return np.sqrt(h**2 + d**2) + measurement_noise


def getdfdX(X, fun):
    """
    Computes the numerical Jacobian of a function using central differences.

    Args:
        X: Current state.
        fun: Function for which to compute the Jacobian.
    Returns:
        dfdX: Jacobian matrix.
    """
    dx = 0.001  # Small perturbation for numerical differentiation
    f0 = fun(X)
    n = X.shape[0]
    m = f0.shape[0] if not np.isscalar(f0) else 1
    dfdX = np.zeros((m, n))

    # Compute partial derivatives with respect to each state variable
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = dx
        f1 = fun(X + ei)
        f0 = fun(X - ei)
        dfdX[:, i] = (f1 - f0) / (2 * dx)

    return dfdX


if __name__ == "__main__":
    # Simulate true dynamics and noise-free dynamics reference

    # Initialize the parameters for simulation
    N = 100  # Number of particles and weights
    problem = 3

    # Prepare containers for results
    true_states = [x0]  # Ground truth states
    measurements = []  # Noisy measurements
    noise_free_dynamics = [x0]  # Noise-free dynamics for reference

    # Sample 100 initial particles
    particles = rng.uniform(x0 + np.array([-20, -2]), x0 + np.array([20, 2]), size=(N, 2))

    # Initial weights for particles
    weights = np.ones(N) / N

    # Loop (steps) times
    for k in range(steps):
        # Sample 100 realizations of the process noise
        process_noise = weibull_with_scale(shape_weibull, scale_weibull, 100)

        # Propagate each particle by the spacecraft dynamics
        for i in range(N):
            particles[i] = spacecraft_dynamics(particles[i], dt, process_noise[i])

        # Gaussian distribution of the measurement noise
        y_k = []  # Radar measurement container
        for i in range(N):
            measurement_noise = rng.normal(mean_gauss, std_gauss)  # Gaussian noise with mean = 0, std = 6
            y_k = radar_measurement(particles[i], measurement_noise)  # Radar measurement

        # Compute predicted measurements for all particles
        predicted_measurements = np.sqrt(particles[:, 0]**2 + d**2)  # Predicted measurement (sqrt(h^2 + d^2))

        # Compute the conditional likelihoods for each particle
        likelihoods = np.exp(-0.5 * ((y_k - predicted_measurements) / std_gauss)**2) / (std_gauss * np.sqrt(2 * np.pi))

        # Update the weights
        for i in range(N):
            if problem == 2:
                weights[i] = weights[i] * likelihoods[i]  # For problem 2
            elif problem == 3:
                weights[i] = likelihoods[i]  # For problem 3

        # Normalize the weights
        weights = weights / np.sum(weights)

        # Estimate the state
        estimated_state = particles.T @ weights

        # Resample particles
        particles = multinomial_resample(particles, weights)

        # Simulate noise-free dynamics for comparison
        noise_free_x = spacecraft_dynamics(noise_free_dynamics[-1], dt, process_noise=0)

        # Store results for plotting
        true_states.append(estimated_state)
        measurements.append(y_k)
        noise_free_dynamics.append(noise_free_x)

    # Convert results to arrays for easier manipulation
    true_states = np.array(true_states)
    measurements = np.array(measurements)
    noise_free_dynamics = np.array(noise_free_dynamics)

    # Extract altitude data for visualization
    true_altitudes = true_states[:, 0]
    noise_free_altitudes = noise_free_dynamics[:, 0]

    # Extract velocity data for visualization
    true_velocities = true_states[:, 1]
    noise_free_velocities = noise_free_dynamics[:, 1]

    # Plot altitude comparison
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(steps + 1) * dt, true_altitudes, label="True Altitude", linewidth=2)
    plt.plot(np.arange(steps + 1) * dt, noise_free_altitudes, label="Noise-Free Dynamics", linestyle='-', linewidth=1.5)
    plt.plot(np.arange(steps) * dt, measurements, '.', label="Radar Measurements", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (km)")
    plt.title("Altitude: True Dynamics and Noise-Free Dynamics")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot velocity comparison
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(steps + 1) * dt, true_velocities, label="True Velocity", linewidth=2)
    plt.plot(np.arange(steps + 1) * dt, noise_free_velocities, label="Noise-Free Dynamics", linestyle='-',
             linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (km/s)")
    plt.title("Velocity: True Dynamics and Noise-Free Dynamics")
    plt.legend()
    plt.grid()
    plt.show()
