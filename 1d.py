#!/usr/bin/env python3

# 1D FDTD, Ey/Hx mode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from scipy import signal
import time
import numba

# TODO:
# * Receive hints and simulation parameters as command-line parameters

# Basic constants

m0 = 4*np.pi*10**-7      # N/A**2
e0 = 8.854187817*10**-12 # F/m
c0 = 1.0/np.sqrt(m0*e0)  # The speed of light

# Simulation parameters

KHz = 1000.0
MHz = 1000000.0
GHz = 1000000000.0

ps = 10**-12
ns = 10**-9
us = 10**-6

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
gridsize = int(space_size / dz)    # Size of the grid in cells
simlen = 5 * space_size / c0       # Simulation length, seconds (5 travels back & forth)
dt = n_min * dz / (2*c0)           # From the Courant-Friedrichs-Lewy condition. This is a rule of thumb
steps = int(simlen / dt)           # Number of simulation steps
bsize = 100                        # Dampening boundary thickness
bcoeff = 1.015                     # Dampening coefficient
batch = 10                         # Number of iterations between drawings

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

# Update coeffitients, using normalized magnetic field
mkhx = c0*dt/mr
mkey = c0*dt/er

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

E  = np.zeros(gridsize) # Electric field
H  = np.zeros(gridsize) # Normalized magnetic field
LB = np.log(np.linspace(np.e/bcoeff, np.e, bsize)) # left boundary
RB = np.log(np.linspace(np.e, np.e/bcoeff, bsize)) # Right boundary

# Display
fig = plt.figure()
ax = plt.axes(ylim=(-5, 5))
line1, = ax.plot(np.linspace(0.0, space_size, gridsize), np.zeros(gridsize), 'r-')
line2, = ax.plot(np.linspace(0.0, space_size, gridsize), np.zeros(gridsize), 'b-')

for layer in layers:
    n = layer[0]/layer[1]
    ax.add_patch(patches.Rectangle((layer[2], -20), layer[3], 40, color=(n, n, n)))

# Sinc function source
@numba.jit()
def sinc_source(er, ur, period, t0, t):
    a_corr = -np.sqrt(er/ur) # amplitude correction term
    t_corr = np.sqrt(er*ur)*dz/(2*c0) + dt/2 # Time correction term
    return (
        # H field
        a_corr * np.sinc((t-t0)*2/period + t_corr),
        # E field
        np.sinc((t-t0)*2/period)  # E
    )

# Gaussian pulse source
@numba.jit()
def gausspulse_source(er, ur, t0, tau, t):
    a_corr = -np.sqrt(er/ur) # amplitude correction term
    t_corr = np.sqrt(er*ur)*dz/(2*c0) + dt/2 # Time correction term
    return (
         a_corr * np.exp(-((t-t0)/tau)**2 + t_corr),
         np.exp(-((t-t0)/tau)**2)
    )

# Outputs 1.0 at time 0
@numba.jit()
def blip_source(t):
    return (0.0, 1.0) if t == 0 else (0.0, 0.0)

def init_animation():
    global line1, line2
    return line1, line2

@numba.jit()
def step(E, H, ifrom, ito):
    for i in range(ifrom, ito):
        t = i*dt
        src = gausspulse_source(1.0, 1.0, 200*ps, 50*ps, t)
        H[1:-1] += mkhx[1:-1] * (E[2:] - E[1:-1]) / dz
        H[int(500)] += src[0] # H source injection

        E[1:-1] += mkey[1:-1] * (H[1:-1] - H[:-2]) / dz
        E[int(500)] += src[1] # E source injection

        # Apply the dampening boundary
        E[:bsize]  *= LB
        H[:bsize]  *= LB
        E[-bsize:] *= RB
        H[-bsize:] *= RB

i = 0
def animate(_):
    global i, ax, line1, line2, H, E, mkhx, mkey

    time1 = time.time()
    step(E, H, i, i+batch)
    time2 = time.time()
    print("step %d took %fms" % (i, time2-time1))

    line1.set_ydata(E)
    line2.set_ydata(H)

    i += batch

    return line1, line2

anim = animation.FuncAnimation(fig, animate, init_func=init_animation, interval=0, blit=True)
plt.show()
