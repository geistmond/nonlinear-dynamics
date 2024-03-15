"""
Builds on an implementation of the Kadomtsev-Petviashvili Equation using a Fast Fourier Transform.

The KP equation describes nonlinear wave motion. 
For example, it can model rare phenomena seen in standing waves and solitons.
A surface tension parameter can be added. 

It is also used to model standing wave / soliton type phenomena in other phases of matter like matter waves Bose-Einstein condensates.
Ferromagnetism also exhibits this type of standing wave phenomenon.

Please see README.md for simple intro reading.
"""

import numpy as np


# Imports for graphing
import matplotlib.pyplot as plt
from scipy.integrate import odeint # Evidently now deprecated.
from scipy.integrate import solve_ivp # "For newer code" according to SciPy website.
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})

# Some constants that help discretize for graphing
# Diffusion constant nu, experiment with values like this, original value was 0.0 but this is more fun
nu = 0.0000001 # diffusion constant
L = 20       # Length of domain
N = 1000     # Number of discretization points
dx = L/N
x = np.arange(-L/2, L/2, dx) # Define x domain

# Define discrete wave numbers
kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)

# Initial conditions
u0 = 1 / np.cosh(x)

# Simulate PDE in spatial domain
#dt = 0.025   # tick time since the source material is a continuous function
dt = 0.1 # try a larger tick time instead
t = np.arange(0, 800*dt, dt)

def rhsBurgers(u, t, kappa, nu):
    """
    Variation on the original equation used for finding the higher-order partial derivatives using (I)FFT.
    """
    uhat = np.fft.fft(u)
    d_uhat = (1j)*kappa*uhat
    dd_uhat = -np.power(kappa, 2) * uhat
    d_u = np.fft.ifft(d_uhat)
    dd_u = np.fft.ifft(dd_uhat)
    du_dt = - u * d_u + nu * dd_u
    return du_dt.real # Why is complex domain left out?


# Integrate the Ordinary Differential Equation using the function that was defined using the FFT bin deltas.
u = odeint(rhsBurgers, u0, t, args=(kappa, nu))

# Newer approach from SciPy for integrating ODE but this is harder to make work with the original example.
#tv = [0, 800*dt]
#v = solve_ivp(rhsBurgers, t_span=tv, y0=u0, args=(kappa, nu))


def plot(d=2)->None:
    print("Plotting...")
    if d == 3:
        # Waterfall plot
        print("3d plot.")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Takes a section from the result of the integrated ODE to graph.
        u_plot = u[0:-1:10, :] 
        for j in range(u_plot.shape[0]):
            ys = j*np.ones(u_plot.shape[1])
            ax.plot(x, ys, u_plot[j, :], color=cm.jet(j*20))
        plt.figure()
        plt.show()
    else:
        print("2d plot.")
        plt.imshow(np.flipud(u), aspect = 1) # square aspect is clearer, original value was 8
        plt.title(label='KP limited case:',
                loc="left",
                color='blue',
                fontsize=10,
                )
        plt.xlabel('x')
        plt.ylabel('t')
        # plt.axis('off')
        plt.set_cmap('jet')
        plt.show()
    return None

plot(d=2)