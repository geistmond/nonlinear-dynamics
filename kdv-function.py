"""
Korteweg de Vries equation solution graph based on SciPy docs.
https://scipy-cookbook.readthedocs.io/items/KdV.html

This method also uses a Fast Fourier Transform. It uses the Method of Lines.
https://en.wikipedia.org/wiki/Method_of_lines

Intended to complement the other standing wave equation in this project, the Kadomtsev-Petviashvili Equation.
"""

from functools import lru_cache

import numpy as np

from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff

import matplotlib.pyplot as plt


def kdv_exact(x:float, c:float)->float:
    """
    Profiles the exact solution to the KdV for a single soliton on the real axis.
    """
    u = 0.5*c*np.cosh(0.5*np.sqrt(c)*x)**(-2)
    return u


def kdv(u, t, L):
    """
    Discretized DiffEq for KdV equation.
    """
    ux = psdiff(u, period=L)
    uxxx = psdiff(u, period=L, order=3) # 3rd order differentiator
    # Compute du/dt
    dudt = -6*u*ux - uxxx
    return dudt


def kdv_solution(u0, t, L):
    """
    Integrate ODE on a periodic domain.
    * u0 is the initial condition.
    * t is an array of time values at which to compute a solution.
    * L is the length of the periodic domain.
    """
    sol = odeint(kdv, u0, t, args=(L,), mxstep=5000)
    return sol


def graph()->None:
     # Set the size of the domain, and create the discretized grid.
    L = 50.0
    N = 64
    dx = L / (N - 1.0)
    x = np.linspace(0, (1-1.0/N)*L, N)

    # Set the initial conditions.
    # Not exact for two solitons on a periodic domain, but close enough...
    u0 = kdv_exact(x-0.33*L, 0.75) + kdv_exact(x-0.65*L, 0.4)

    # Set the time sample grid.
    T = 200
    t = np.linspace(0, T, 501)

    print("Computing the solution.")
    sol = kdv_solution(u0, t, L)

    print("Plotting.")

    plt.figure(figsize=(6,5))
    plt.imshow(sol[::-1, :], extent=[0,L,0,T])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    #plt.axis('normal')
    plt.title('Korteweg-de Vries on a Periodic Domain')
    plt.show()


graph()