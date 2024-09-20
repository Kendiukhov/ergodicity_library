"""
partial_sde Submodule

The `Partial SDE` submodule provides functionality for simulating, visualizing, and analyzing stochastic partial differential equations (SPDEs). The core of this module is the `PSDESimulator` class, which implements numerical solutions to SPDEs, allowing users to model systems with both deterministic drift and stochastic diffusion terms over time and space.

Key Features:

1. **Stochastic PDE Simulation**:

   - The simulator models the evolution of a spatially distributed quantity `u(t, x)` over time with specified drift and diffusion terms. These terms represent the deterministic and stochastic components of the equation, respectively.

   - The simulator supports different boundary conditions including Dirichlet, Neumann, and periodic.

2. **Numerical Solution**:

   - The SPDE is solved using a finite difference approach in both time and space, with the option to include stochastic noise (via a Wiener process) at each time step.

3. **Visualization**:

   - **2D Plots**: The `plot_results` method provides a 2D surface plot of the solution over time and space, along with a time slice at the final time step.

   - **3D Plots**: The `plot_3d` method generates a 3D surface plot to visualize the evolution of the solution.

   - **Animations**: The `create_animation` method generates an animation of the solution's evolution, saved as a video file.

4. **Customizability**:

   - Users can define their own drift and diffusion terms, initial conditions, and boundary conditions to simulate a wide variety of SPDEs.

   - The spatial and temporal resolution can be adjusted through the `nx` (number of spatial points) and `nt` (number of time points) parameters.

Example Usage:

if __name__ == "__main__":

    # Define the drift (deterministic) term of the SPDE (e.g., heat equation)

    def drift(t, x, u, u_x, u_xx):

        return 0.01 * u_xx  # Heat equation term

    # Define the diffusion (stochastic) term of the SPDE

    def diffusion(t, x, u):

        return 0.1 * np.ones_like(x)  # Constant noise term

    # Define the initial condition

    def initial_condition(x):

        return np.sin(np.pi * x)  # Initial sine wave

    # Initialize the simulator

    simulator = PSDESimulator(
        drift=drift,
        diffusion=diffusion,
        initial_condition=initial_condition,
        x_range=(0, 1),
        t_range=(0, 1),
        nx=100,
        nt=1000
    )

    # Run the simulation

    simulator.simulate()

    # Visualize the results

    simulator.plot_results()

    simulator.plot_3d()

    simulator.create_animation("spde_evolution.mp4")

    print("Simulation and visualization complete.")

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple
from ergodicity.process.basic import WienerProcess

def wiener_increment_function(timestep_increment: float):
    """
    Function to generate Wiener process increments.
    This is a default increment function for the PSDE simulator.
    It can be changed to use different increments, such as for example Levy increments.

    :param timestep_increment: The time step of the discrete increment
    :type timestep_increment: float
    :return: The Wiener process increment
    :rtype: float
    """
    WP = WienerProcess()
    dW = WP.increment(timestep_increment=timestep_increment)
    return dW

class PSDESimulator:
    """
    Partial Stochastic Differential Equation (PSDE) Simulator.
    This class provides functionality to simulate and visualize the stochastic evolution of a spatially distributed quantity

    Attributes:

        drift (Callable): The drift function f(t, x, u, u_x, u_xx)

        diffusion (Callable): The diffusion function g(t, x, u)

        initial_condition (Callable): The initial condition u(0, x)

        x_range (tuple): The spatial range (x_min, x_max)

        t_range (tuple): The time range (t_min, t_max)

        nx (int): Number of spatial points

        nt (int): Number of time points

        boundary_type (str): Type of boundary condition (e.g., "dirichlet", "neumann", "periodic")

        boundary_func (Callable): The boundary condition function

        x (ndarray): Spatial grid points

        t (ndarray): Time grid points

        dx (float): Spatial step size

        dt (float): Time step size

        u (ndarray): Array to store the solution u(t, x)
    """
    def __init__(self,
                 drift: Callable,
                 diffusion: Callable,
                 initial_condition: Callable,
                 x_range: tuple,
                 t_range: tuple,
                 nx: int,
                 nt: int,
                 boundary_condition: Tuple[str, Callable] = ("dirichlet", lambda t, x: 0), increment: Callable = wiener_increment_function):
        """
        Initialize the PSDE simulator.

        :param drift: The drift function f(t, x, u, u_x, u_xx)
        :type drift: Callable
        :param diffusion: The diffusion function g(t, x, u)
        :type diffusion: Callable
        :param initial_condition: The initial condition u(0, x)
        :type initial_condition: Callable
        :param x_range: The spatial range (x_min, x_max)
        :type x_range: tuple
        :param t_range: The time range (t_min, t_max)
        :type t_range: tuple
        :param nx: Number of spatial points
        :type nx: int
        :param nt: Number of time points
        :type nt: int
        :param boundary_condition: Type of boundary condition and function. It can be "dirichlet", "neumann", or "periodic"
        :type boundary_condition: Tuple[str, Callable]
        :param increment: The increment function for the stochastic term (default: Wiener process)
        :type increment: Callable
        """
        self.drift = drift
        self.diffusion = diffusion
        self.initial_condition = initial_condition
        self.x_range = x_range
        self.t_range = t_range
        self.nx = nx
        self.nt = nt
        self.boundary_type, self.boundary_func = boundary_condition

        self.x = np.linspace(x_range[0], x_range[1], nx)
        self.t = np.linspace(t_range[0], t_range[1], nt)
        self.dx = (x_range[1] - x_range[0]) / (nx - 1)
        self.dt = (t_range[1] - t_range[0]) / (nt - 1)

        self.u = np.zeros((nt, nx))
        self.u[0, :] = initial_condition(self.x)

        # Apply initial boundary conditions
        self.apply_boundary_condition(0)

        self.increment = increment

    def apply_boundary_condition(self, n):
        """
        Apply boundary conditions at time step n.

        :param n: The time step index
        :type n: int
        :return: None
        :rtype: None
        """
        if self.boundary_type == "dirichlet":
            self.u[n, 0] = self.boundary_func(self.t[n], self.x[0])
            self.u[n, -1] = self.boundary_func(self.t[n], self.x[-1])
        elif self.boundary_type == "neumann":
            # Forward difference for left boundary, backward for right
            self.u[n, 0] = self.u[n, 1] - self.dx * self.boundary_func(self.t[n], self.x[0])
            self.u[n, -1] = self.u[n, -2] + self.dx * self.boundary_func(self.t[n], self.x[-1])
        elif self.boundary_type == "periodic":
            self.u[n, 0] = self.u[n, -2]
            self.u[n, -1] = self.u[n, 1]
        else:
            raise ValueError("Unsupported boundary condition type")

    def simulate(self):
        """
        Run the simulation.

        :return: None
        :rtype: None
        """
        for n in range(1, self.nt):
            # dW = np.sqrt(self.dt) * np.random.normal(0, 1, self.nx)
            dW = [self.increment(timestep_increment=self.dt) for i in range(self.nx)]

            u_x = np.gradient(self.u[n - 1], self.dx)
            u_xx = np.gradient(u_x, self.dx)

            drift_term = self.drift(self.t[n - 1], self.x, self.u[n - 1], u_x, u_xx)
            diffusion_term = self.diffusion(self.t[n - 1], self.x, self.u[n - 1])

            self.u[n] = (self.u[n - 1] +
                         drift_term * self.dt +
                         diffusion_term * dW)

            # Apply boundary conditions after each time step
            self.apply_boundary_condition(n)

    def plot_results(self):
        """
        Plot the results of the simulation. This includes a 2D surface plot of the function evolution and a time slice at the final time step.

        :return: None
        :rtype: None
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Surface plot
        T, X = np.meshgrid(self.t, self.x)
        surf = ax1.contourf(T, X, self.u.T, cmap='viridis')
        ax1.set_title('Surface Plot of u(t,x)')
        ax1.set_xlabel('t')
        ax1.set_ylabel('x')
        plt.colorbar(surf, ax=ax1)

        # Final time slice
        ax2.plot(self.x, self.u[-1, :])
        ax2.set_title(f'u(t,x) at t = {self.t[-1]:.2f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('u')

        plt.tight_layout()
        plt.show()

    def plot_3d(self):
        """
        Create a 3D plot of the function evolution.
        This plot shows the surface of u(t,x) over time and space.

        :return: None
        :rtype: None
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        T, X = np.meshgrid(self.t, self.x)
        surf = ax.plot_surface(T, X, self.u.T, cmap='viridis')

        ax.set_title('3D Evolution of u(t,x)')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('u')

        plt.colorbar(surf)
        plt.show()

    def create_animation(self, filename='psde_evolution.mp4'):
        """
        Create an animation of the function evolution.
        The animation shows the spatial distribution of u at each time step.

        :param filename: The name of the output file (default: 'psde_evolution.mp4')
        :type filename: str
        :return: None
        :rtype: None
        """
        fig, ax = plt.subplots()
        line, = ax.plot([], [])
        ax.set_xlim(self.x_range)
        ax.set_ylim(np.min(self.u), np.max(self.u))
        ax.set_xlabel('x')
        ax.set_ylabel('u')

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            line.set_data(self.x, self.u[i, :])
            ax.set_title(f't = {self.t[i]:.2f}')
            return line,

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=self.nt, interval=50, blit=True)
        anim.save(filename, writer='ffmpeg', fps=30)
        plt.close(fig)

# Example usage: Heat equation with stochastic forcing
if __name__ == "__main__":
    # Define the PSDE components
    def drift(t, x, u, u_x, u_xx):
        return 0.01 * u_xx  # Heat equation term


    def diffusion(t, x, u):
        return 0.1 * np.ones_like(x)  # Constant noise


    def initial_condition(x):
        return np.sin(np.pi * x)  # Initial sine wave


    # Set up the simulation
    simulator = PSDESimulator(
        drift=drift,
        diffusion=diffusion,
        initial_condition=initial_condition,
        x_range=(0, 1),
        t_range=(0, 1),
        nx=100,
        nt=1000
    )

    # Run the simulation
    simulator.simulate()

    # Plot the results
    simulator.plot_results()

    # Create 3D plot
    simulator.plot_3d()

    # Create animation
    simulator.create_animation()

    print("Simulation and visualization complete.")
