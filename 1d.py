#!/usr/bin/env python3

# 1D FDTD, Ey/Hx mode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal
import time

# TODO:
# * Calculate grid resolution
#   * Hinting sould be possible
# * Calculate timestamp
#   * Hinting should be possible
# * Receive hints and simulation parameters as command-line parameters
# * 

# Basic constants

m0 = 4*np.pi*10**-7      # N/A**2
e0 = 8.854187817*10**-12 # F/m
c0 = 1.0/np.sqrt(m0*e0)  # The speed of light

# Simulation parameters

GHz = 1000000000.0

# Material layers
# Layers are overwriting each other, last write wins
# First layer should be free space.
# (mr, er, start, width)
layers = [
    (1.0, 1.0, 0.0, 1.0), # Free space
    (1.0, 2.5, 0.6, 0.2) # A 20cm slab of plastic
]

# Calculate maximal refractive index
n_max = 1.0
n_min = 1000000.0 # TODO: init to first layer
for layer in layers:
    n = layer[0] / layer[1]
    if n > n_max:
        n_max = n
    if n < n_min:
        n_min = n

space_size = 1.0                   # meters
freq_max = 100*GHz                 # maximal resolvable frequency
lamb_min = c0 / (freq_max * n_max) # minimal wavelength
dzpmwl = 10                        # delta-z per minimal wavelength, a rule-of-thumb constant
dz = lamb_min / dzpmwl             # Spatial step size, meters
gridsize = round(space_size / dz)  # Size of the grid in cells
simlen = 5 * space_size / c0       # Simulation length, seconds (5 travels back & forth)
dt = n_min * dz / (200000000*c0)   # From the Courant stability confition. This is a rule of thumb TODO: this is wrong.
steps = int(simlen / dt)           # Number of simulation steps

print("simulation length:", simlen)
print("grid size:", gridsize)
print("steps:", steps)
print("dt:", dt)
print("dz:", dz)

mr = np.ones(gridsize) # permeability, can be diagonally anisotropic
er = np.ones(gridsize) # permittivity, can be diagonally anisotropic

for layer in layers:
    # TODO: snap layers to grid / snap grid to layers?
    for i in range(max(0, int(layer[2]/dz)), min(gridsize, int((layer[2]+layer[3])/dz))):
        er[i] = layer[0]
        mr[i] = layer[1]

m = m0 * mr
e = e0 * er

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
line1, = ax.plot(np.linspace(0.0, space_size, gridsize), np.zeros(gridsize), 'r-')
line2, = ax.plot(np.linspace(0.0, space_size, gridsize), np.zeros(gridsize), 'b-')

for layer in layers:
    n = layer[0]/layer[1]
    ax.add_patch(patches.Rectangle((layer[2], -20), layer[3], 40, color=(n, n, n)))

# Sinc tf/sf source
def sinc_source(t):
    sinc_from  = -10*np.pi
    sinc_to    = 10*np.pi
    sinc_steps = 10000.0
    if t < sinc_steps:
        # TODO: correct for the time and space staggering
        return (
             np.sinc(sinc_from + t * (sinc_to - sinc_from) / sinc_steps), # H, normalized in vacuum
            -np.sinc(sinc_from + t * (sinc_to - sinc_from) / sinc_steps)  # E
        )
    else:
        return (0.0, 0.0)

def gausspulse_source(t):
    fr = -0.003
    to = 0.003
    pulse_steps = 1000.0

    if t < pulse_steps:
        return(
             5*signal.gausspulse(fr + t * (to - fr) / pulse_steps, retquad=False, retenv=True)[1],
            -5*signal.gausspulse(fr + t * (to - fr) / pulse_steps, retquad=False, retenv=True)[1]
        )
    else:
        return (0.0, 0.0)

for t in range(steps):
    src = gausspulse_source(t)

    H[:-1] = H[:-1] + mkhx[:-1] * (E[1:] - E[:-1]) / dz
    H[-1]  = H[-1]  + mkhx[-1]  * (0     - E[-1] ) / dz # Dirichlet numerical boundary conditions
    H[int(101)] += src[0] # H source injection
    if np.isnan(H[0]):
        print("NAN")
        break;

    E[0]  = E[0]  + mkey[0]  * (H[0]  - 0     ) / dz # Dirichlet numerical boundary conditions
    E[1:] = E[1:] + mkey[1:] * (H[1:] - H[:-1]) / dz
    E[int(101)] += src[1] # E source injection

    # Dampen at the edges instead of using the messy perfect edge algorithm.
    E[:100] *= np.linspace(0.98,1.0,100)
    H[:100] *= np.linspace(0.98,1.0,100)
    E[-100:] *= np.linspace(1.0,0.98,100)
    H[-100:] *= np.linspace(1.0,0.98,100)

    if t % 100 == 0:
        line1.set_ydata(E)
        line2.set_ydata(H)
        fig.canvas.draw()
        plt.pause(0.001)

print("Simulation complete")

while True:
    plt.pause(0.001)
