#!/usr/bin/env python3

# 1D FDTD, Ey/Hx mode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# TODO:
# * Calculate grid resolution
#   * Hinting sould be possible
# * Calculate timestamp
#   * Hinting should be possible
# * Receive hints and simulation parameters as command-line parameters
# * 

# Simulation parameters

dt = 10**-18     # Time step size, seconds
simlen = 10**-13 # Simulation length size, seconds
steps = int(simlen / dt)

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

er[1200:1400] = np.repeat(2.5,200) # Add a slab of plastic

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
ax = plt.axes(ylim=(-15, 15))
line1, = ax.plot(range(gridsize), np.zeros(gridsize), 'r-')
line2, = ax.plot(range(gridsize), np.zeros(gridsize), 'b-')
ax.add_patch(patches.Rectangle((1200, -20), 200, 40, color=(0.9,0.9,0.9))) # Device
ax.add_patch(patches.Rectangle((0, -20), 100, 40, color=(0.6,0.6,0.6))) # PML, left
ax.add_patch(patches.Rectangle((gridsize-100, -20), gridsize, 40, color=(0.6,0.6,0.6))) # PML, right

# Sinc tf/sf source
def sinc_source(t):
    sinc_from  = -10.0
    sinc_to    = 10.0
    sinc_steps = 5000.0
    if t < sinc_steps:
        # TODO: correct for the time and space staggering
        return (
             np.sinc(sinc_from + t * (sinc_to - sinc_from) / sinc_steps), # H, normalized in vacuum
            -np.sinc(sinc_from + t * (sinc_to - sinc_from) / sinc_steps)  # E
        )
    else:
        return (0.0, 0.0)

for t in range(steps):
    src = sinc_source(t)

    H[:-1] = H[:-1] + mkhx[:-1] * (E[1:] - E[:-1]) / dz
    H[-1]  = H[-1]  + mkhx[-1]  * (0     - E[-1] ) / dz # Dirichlet numerical boundary conditions
    H[int(300)] += src[0] # H source injection

    E[0]  = E[0]  + mkey[0]  * (H[0]  - 0     ) / dz # Dirichlet numerical boundary conditions
    E[1:] = E[1:] + mkey[1:] * (H[1:] - H[:-1]) / dz
    E[int(300)] += src[1] # E source injection

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

print("Simulation complete")

while True:
    plt.pause(0.001)
