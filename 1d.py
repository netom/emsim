#!/usr/bin/env python3

# 1D FDTD, Ey/Hx mode

import numpy as np
import matplotlib.pyplot as plt
import time

# TODO:
# * Calculate grid resolution
#   * Hinting sould be possible
# * Calculate timestamp
#   * Hinting should be possible
# * Receive hints and simulation parameters as command-line parameters
# * 

# Simulation parameters

dt = 10**-18 # Time step
steps = 100000

# dt < 1 / sqrt(1/dx**2 + 1/dy**2 + 1/dz**2) - the Courant stability condition
# Rule of thumb: dmin / 2*c0

dz = 1.0 # Spatial step in direction Z

gridsize = 2000

# Basic constants

c0 = 299792458           # m/s
m0 = 4*np.pi*10**-7   # N/A**2
e0 = 8.854187817*10**-12 # F/m

er = np.ones(gridsize) # permittivity, can be diagonally anisotropic
mr = np.ones(gridsize) # permeability, can be diagonally anisotropic

er[500:700] = np.repeat(2.5,200) # Add a slab of plastic

n  = mr / er # refractive index

e = e0 * er
m = m0 * mr

# Magnetic field normalization coeffitient
mfnc = np.sqrt(m0 / e0)

# Update coeffitients

mkey = c0*dt/e/mfnc
mkhx = c0*dt/m*mfnc

# Yee grid scheme

# dx, dy, dz, must be as square as possible, but can be different
# Function values are assigned at the middle of the square
#
# Field components are staggered at different places around
#   the grid unit cube / square
#
# * This helps naturally satisfy the divergence equations
# * Material boundaries are naturally handled
# * Easier to calculate discrete curls
# * WARNING: field components can be in different materials!

E = np.zeros(gridsize) # Electric field
H = np.zeros(gridsize) # Normalized magnetic field

# Display
plt.ion()
fig = plt.figure()
ax = plt.axes(ylim=(-10, 10))
line1, = ax.plot(np.linspace(0, 1, gridsize), np.zeros(gridsize), 'r-')
line2, = ax.plot(np.linspace(0, 1, gridsize), np.zeros(gridsize), 'b-')

# Stupid soft source
source = np.zeros(steps)
source[:5000] = 1.0 * np.sinc(np.linspace(-10,10,5000))

for t in range(steps):
    E[int(gridsize/2)] += source[t] # Stupid source injection

    H[:-1] = H[:-1] + mkhx[:-1] * (E[1:] - E[:-1]) / dz
    H[-1]  = H[-1]  + mkhx[-1]  * (0     - E[-1] ) / dz # Dirichlet numerical boundary conditions

    E[0]  = E[0]  + mkey[0]  * (H[0]  - 0     ) / dz # Dirichlet numerical boundary conditions
    E[1:] = E[1:] + mkey[1:] * (H[1:] - H[:-1]) / dz

    # Dampen at the edges instead of using the messy perfect edge algorithm.
    E[:100] *= np.linspace(0.99,1.0,100)
    H[:100] *= np.linspace(0.99,1.0,100)
    E[-100:] *= np.linspace(1.0,0.99,100)
    H[-100:] *= np.linspace(1.0,0.99,100)

    if t % 100 == 0:
        line1.set_ydata(E)
        line2.set_ydata(H)
        fig.canvas.draw()
        plt.pause(0.001)

while True:
    plt.pause(0.001)
