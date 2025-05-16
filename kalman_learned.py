from unknown_planet import *
import numpy as np
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Load the trained neural network and normalization parameters
    Ninputs, Noutputs, Nunits, Nlayers = 7, 5, 100, 5  # Network architecture parameters
    net_trained = NeuralNet(Ninputs, Noutputs, Nunits, Nlayers)  # Initialize the network
    net_trained.load_state_dict(torch.load("../../Downloads/unknown_planet (1)/unknown_planet/data/net_data.pth", weights_only=False))  # Load trained weights
    net_trained.eval()  # Set the network to evaluation mode
    Xmaxes = np.load("../../Downloads/unknown_planet (1)/unknown_planet/data/input_maxes.npy")  # Load state normalization factors
    ymaxes = np.load("../../Downloads/unknown_planet (1)/unknown_planet/data/output_maxes.npy")  # Load residual dynamics normalization factors
    up = UnknownPlanet(net_trained, Xmaxes, ymaxes)  # Create UnknownPlanet instance

    # Load true state trajectory and measurements
    Xtrue = np.load("../../Downloads/unknown_planet (1)/unknown_planet/data/Xtrue.npy")  # True states
    ytrue = np.load("../../Downloads/unknown_planet (1)/unknown_planet/data/ytrue.npy")  # Measurements

    # Spacecraft initial conditions
    r0_mr = 0.1  # Approximate radius
    r0 = r0_mr + 0.1  # Initial orbital radius
    vx0 = 0  # Initial radial velocity
    vth0 = np.sqrt(up.mu / r0)  # Tangential velocity for circular orbit
    om0 = vth0 / r0  # Angular velocity
    h0 = om0 * r0 ** 2  # Angular momentum
    Om0, i0, th0 = np.pi / 3, np.pi / 3, 0  # Initial orientation and angle
    Xc0 = np.array([r0, vx0, h0, Om0, i0, th0])  # Initial state vector

    # Simulation setup
    T = 2 * np.pi / om0 * 10  # Simulate for 10 orbital periods
    dt = up.h  # Time step
    Nint = int(T / dt)  # Number of integration steps
    t = 0  # Start time
    this = np.zeros(Nint + 1)  # Time history array]

    # Initialize state histories for nominal, Kalman, and Kalman+Learned dynamics
    Xcn = Xc0.copy()  # Initial state for nominal dynamics
    Xhat = Xc0.copy()  # Initial state for Kalman dynamics
    Xhat_kalman_learned = Xc0.copy()  # Initial state for Kalman + Learned dynamics
    Xchisn = np.zeros((Nint + 1, up.nX))  # History for nominal dynamics
    Xchisn[0, :] = Xcn
    Xchisk = np.zeros((Nint + 1, up.nX))  # History for Kalman dynamics
    Xchisk[0, :] = Xhat
    Xchis_kl = np.zeros((Nint + 1, up.nX))  # History for Kalman + Learned dynamics
    Xchis_kl[0, :] = Xhat_kalman_learned

    # Simulation loop
    for i in range(Nint):
        yc = ytrue[i, :]  # Current measurement

        # Integrate nominal dynamics
        _, Xcn = up.rk4(t, Xcn, None, lambda t, X, U: up.dynamicspolar3d_nominal(t, X))
        Xchisn[i + 1, :] = Xcn

        # Integrate Kalman dynamics
        dfun_kalman = lambda t, X, U: up.dynamicspolar3d_nominal(t, X) + kalman_correction(X, yc, up)
        _, Xhat = up.rk4(t, Xhat, None, dfun_kalman)
        Xchisk[i + 1, :] = Xhat

        # Integrate Kalman + Learned dynamics
        learned_correction = lambda X: up.res_dyn(X)
        dfun_kalman_learned = lambda t, X, U: up.dynamicspolar3d_nominal(t, X) + kalman_correction(X, yc, up) + learned_correction(X)
        _, Xhat_kalman_learned = up.rk4(t, Xhat_kalman_learned, None, dfun_kalman_learned)
        Xchis_kl[i + 1, :] = Xhat_kalman_learned

        # Increment time
        this[i + 1] = t

    # Visualization
    alp = 0.7

    # Plot true vs. nominal trajectories
    plt.figure()
    plt.plot(Xtrue[:, 0] * np.cos(Xtrue[:, 5]), Xtrue[:, 0] * np.sin(Xtrue[:, 5]), label="True")
    plt.plot(Xchisn[:, 0] * np.cos(Xchisn[:, 5]), Xchisn[:, 0] * np.sin(Xchisn[:, 5]), label="Nominal")
    plt.grid()
    plt.xlabel("Horizontal Position")
    plt.ylabel("Vertical Position")
    plt.legend()
    plt.title("True vs Nominal 2D Trajectories")

    # Plot true vs. Kalman-corrected trajectories
    plt.figure()
    plt.plot(Xtrue[:, 0] * np.cos(Xtrue[:, 5]), Xtrue[:, 0] * np.sin(Xtrue[:, 5]), label="True")
    plt.plot(Xchisk[:, 0] * np.cos(Xchisk[:, 5]), Xchisk[:, 0] * np.sin(Xchisk[:, 5]), label="Kalman Corrected")
    plt.grid()
    plt.xlabel("Horizontal Position")
    plt.ylabel("Vertical Position")
    plt.legend()
    plt.title("True vs Kalman Corrected 2D Trajectories")

    # Plot true vs. Kalman + Learned trajectories
    plt.figure()
    plt.plot(Xtrue[:, 0] * np.cos(Xtrue[:, 5]), Xtrue[:, 0] * np.sin(Xtrue[:, 5]), label="True")
    plt.plot(Xchis_kl[:, 0] * np.cos(Xchis_kl[:, 5]), Xchis_kl[:, 0] * np.sin(Xchis_kl[:, 5]),
             label="Kalman + Learned")
    plt.grid()
    plt.xlabel("Horizontal Position")
    plt.ylabel("Vertical Position")
    plt.legend()
    plt.title("True vs Kalman + Learned 2D Trajectories")

    plt.show()
