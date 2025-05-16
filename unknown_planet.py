import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import control
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OMP: Error #15: Initializing libiomp5md.dll, but found mk2iomp5md.dll already initialized.

# Set up the device (use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize random number generator with a seed for reproducibility
rng = np.random.default_rng(seed=2024)


def get_kalman_gain(x_hat, planet):
    """
    Finds the Kalman gain.
    Args:
        x_hat: Current state estimate
        planet: UnknownPLanet instance for dynamics and measurements.
    Returns:
        Kalman gain
    """
    # Define system matrices
    dyn = lambda X: planet.dynamicspolar3d_nominal(None, X)
    M_1 = planet.getdfdX(x_hat, dyn)  # Jacobian of dynamics
    M_2 = planet.Hc  # Measurement matrix

    # Define LQR tuning matrices
    Q = np.eye(planet.nX)  # State weighting matrix
    R = np.eye(planet.nY)  # Measurement noise weighting matrix

    # Compute LQR gain
    return control.lqr(M_1.T, M_2.T, Q, R)[0].T


# Kalman correction function
def kalman_correction(x_hat, y, planet):
    """
    Compute the Kalman Gain correction term for the estimator dynamics.
    Args:
        x_hat: Current state estimate.
        y: Current measurement vector.
        planet: UnknownPlanet instance for dynamics and measurements.
    Returns:
        Kalman correction vector.
    """
    # Compute the Kalman gain
    L = get_kalman_gain(x_hat, planet)  # Finds the Kalman gain

    # Compute the other factor in the correction term
    correction = y - np.dot(planet.Hc, x_hat)  # y - H * Xhat

    # Compute the Kalman correction vector
    return np.dot(L, correction)



class NeuralNet(nn.Module):
    """Defines a feedforward neural network for learning residual dynamics."""
    def __init__(self, Ninputs, Noutputs, Nunits, Nlayers):
        """
        Initialize the neural network architecture.
        Args:
            Ninputs: Number of input features.
            Noutputs: Number of output features.
            Nunits: Number of units in hidden layers.
            Nlayers: Number of hidden layers.
        """
        super(NeuralNet, self).__init__()
        self.fcI = nn.Linear(Ninputs, Nunits)  # Input layer
        self.fcs = nn.ModuleList([nn.Linear(Nunits, Nunits) for _ in range(Nlayers)])  # Hidden layers
        self.fcO = nn.Linear(Nunits, Noutputs)  # Output layer
        
    def forward(self, X):
        """
        Perform forward propagation through the network.
        Args:
            X: Input tensor.
        Returns:
            Output tensor.
        """
        X = torch.tanh(self.fcI(X))  # Apply tanh activation to input layer
        for fcH in self.fcs:
            X = torch.tanh(fcH(X))  # Apply tanh activation to hidden layers
        X = self.fcO(X)  # Output layer without activation
        return X

class UnknownPlanet:
    """Simulates the dynamics of an unknown planet using nominal and learned models."""
    def __init__(self, net, Xmaxes, ymaxes):
        """
        Initialize the planet dynamics simulator.
        Args:
            net: Neural network for learning residual dynamics.
            Xmaxes: Normalization constants for the state vector.
            ymaxes: Normalization constants for the residual dynamics outputs.
        """
        self.mu = 1  # Gravitational constant
        self.h = 0.001  # Integration step size
        self.nX = 6  # Number of states
        self.Hc = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], 
                            [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])  # Measurement matrix
        self.nY = self.Hc.shape[0]  # Number of measurements
        self.net = net  # Neural network for learned residual dynamics
        self.Xmaxes = Xmaxes  # Normalization factors for state inputs
        self.ymaxes = ymaxes  # Normalization factors for residual outputs

    def dynamicspolar3d_nominal(self, t, X):
        """
        Computes nominal dynamics of the spacecraft in polar 3D coordinates.
        Args:
            t: Time (not used in this simple dynamics).
            X: State vector [radius, velocity, angular momentum, ..., angle].
        Returns:
            dXdt: Rate of change of the state vector.
        """
        r, vx, h, _, _, th = X  # Unpack state variables
        drdt = vx  # Rate of change of radius
        dvxdt = -self.mu / r**2 + h**2 / r**3  # Radial acceleration
        dhdt = 0  # Angular momentum remains constant
        dOmdt = 0  # Rate of change of right ascension
        didt = 0  # Rate of change of inclination
        dthdt = h / r**2  # Angular velocity
        dXdt = np.array([drdt, dvxdt, dhdt, dOmdt, didt, dthdt])  # Combine rates into a vector
        return dXdt

    def res_dyn(self, X):
        """
        Computes residual dynamics using the trained neural network.
        Args:
            X: Current state vector.
        Returns:
            res_dyn_out: Residual dynamics as a vector.
        """
        Xth = np.array([np.cos(X[5]), np.sin(X[5])])  # Represent angle as sin and cos
        Xsincos = np.hstack((X[:5], Xth))  # Extend state with sin/cos of the angle
        Xnn = Xsincos / self.Xmaxes  # Normalize state inputs
        Xnn_torch = self.np2torch(Xnn)  # Convert to torch tensor
        ynn_torch = self.net(Xnn_torch)  # Pass through the neural network
        ynn = ynn_torch.detach().numpy() * self.ymaxes  # Denormalize the output
        res_dyn_out = np.hstack((0, ynn))  # Append zero for radial dynamics
        return res_dyn_out

    def getdfdX(self, X, fun):
        """
        Computes the Jacobian matrix of the given function using finite differences.
        Args:
            X: Current state vector.
            fun: Function for which the Jacobian is computed.
        Returns:
            dfdX: Jacobian matrix.
        """
        dx = 0.001  # Perturbation size for finite difference
        n = X.shape[0]
        dfdX = np.zeros((n, n))  # Initialize Jacobian matrix
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = dx  # Perturb one variable
            f1 = fun(X + ei)  # Forward step
            f0 = fun(X - ei)  # Backward step
            dfdX[:, i] = (f1 - f0) / (2 * dx)  # Central difference formula
        return dfdX
        
    def rk4(self, t, X, U, dynamics):
        """
        Performs 4th-order Runge-Kutta integration.
        Args:
            t: Current time.
            X: Current state vector.
            U: Input (unused here).
            dynamics: Dynamics function to integrate.
        Returns:
            t + h: Updated time.
            X_next: Updated state vector.
        """
        h = self.h  # Integration step size
        k1 = dynamics(t, X, U)
        k2 = dynamics(t + h / 2., X + k1 * h / 2., U)
        k3 = dynamics(t + h / 2., X + k2 * h / 2., U)
        k4 = dynamics(t + h, X + k3 * h, U)
        return t + h, X + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.

    def np2torch(self, x):
        """
        Converts numpy array to torch tensor.
        Args:
            x: Numpy array.
        Returns:
            Torch tensor.
        """
        x = torch.from_numpy(x).to(device)
        x = x.type(torch.FloatTensor).to(device)
        return x
    
if __name__ == "__main__":
    
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
    h0 = om0 * r0**2  # Angular momentum
    Om0, i0, th0 = np.pi / 3, np.pi / 3, 0  # Initial orientation and angle
    Xc0 = np.array([r0, vx0, h0, Om0, i0, th0])  # Initial state vector

    # Simulation setup
    T = 2 * np.pi / om0 * 10  # Simulate for 10 orbital periods
    dt = up.h  # Time step
    Nint = int(T / dt)  # Number of integration steps
    t = 0  # Start time
    this = np.zeros(Nint + 1)  # Time history array

    # Define dynamics for simulation
    dres_net = lambda X: up.res_dyn(X)  # Learned residual dynamics
    dfun_nom = lambda t, X, U: up.dynamicspolar3d_nominal(t, X)  # Nominal dynamics

    # Initialize state histories for nominal dynamics
    Xcn = Xc0.copy()
    Xchisn = np.zeros((Nint + 1, up.nX))  # History of nominal dynamics
    Xchisn[0, :] = Xc0

    # Initialize state histories for nominal + learned dynamics
    dfun_net = lambda t, X, U: dfun_nom(t, X, U) + dres_net(X)
    Xcl = Xc0.copy()
    Xchisl = np.zeros((Nint + 1, up.nX))  # History of nominal + learned dynamics
    Xchisl[0, :] = Xc0

    # Simulation loop
    for i in range(Nint):
        yc = ytrue[i, :]  # Current measurement
        # Integrate nominal dynamics
        _, Xcn = up.rk4(t, Xcn, None, dfun_nom)
        # Integrate nominal + learned dynamics
        t, Xcl = up.rk4(t, Xcl, None, dfun_net)
        # Save state histories
        this[i + 1] = t
        Xchisn[i + 1, :] = Xcn
        Xchisl[i + 1, :] = Xcl

    # Visualization
    alp = 0.7  # Alpha for transparency in plots

    # Plot true vs. nominal trajectories
    plt.figure()
    plt.plot(Xtrue[:, 0] * np.cos(Xtrue[:, 5]), Xtrue[:, 0] * np.sin(Xtrue[:, 5]), label="True")
    plt.plot(Xchisn[:, 0] * np.cos(Xchisn[:, 5]), Xchisn[:, 0] * np.sin(Xchisn[:, 5]), label="Nominal")
    plt.grid()
    plt.xlabel("Horizontal Position")
    plt.ylabel("Vertical Position")
    plt.legend()
    plt.title("True vs. Nominal 2D Trajectories")

    # Plot true vs. measured trajectories
    plt.figure()
    plt.plot(Xtrue[:, 0] * np.cos(Xtrue[:, 5]), Xtrue[:, 0] * np.sin(Xtrue[:, 5]), label="True")
    plt.plot(ytrue[:, 0] * np.cos(ytrue[:, 3]), ytrue[:, 0] * np.sin(ytrue[:, 3]), label="Measured")
    plt.grid()
    plt.xlabel("Horizontal Position")
    plt.ylabel("Vertical Position")
    plt.legend()
    plt.title("True vs. Measured 2D Trajectories")

    # Plot true vs. learned (without estimation)
    plt.figure()
    plt.plot(Xtrue[:, 0] * np.cos(Xtrue[:, 5]), Xtrue[:, 0] * np.sin(Xtrue[:, 5]), label="True", alpha=alp)
    plt.plot(Xchisl[:, 0] * np.cos(Xchisl[:, 5]), Xchisl[:, 0] * np.sin(Xchisl[:, 5]), label="Learned", alpha=alp)
    plt.grid()
    plt.xlabel("Horizontal Position")
    plt.ylabel("Vertical Position")
    plt.legend()
    plt.title("True vs. Learned 2D Trajectories (No Estimation)")

    plt.show()
