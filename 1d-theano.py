#!/usr/bin/env python3

# 1D FDTD, Ey/Hx mode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from scipy import signal
import time

import theano
import theano.sparse
import theano.tensor as T

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

print("simulation length:", simlen)
print("grid size:", gridsize)
print("steps:", steps)
print("dt:", dt)
print("dz:", dz)

mr = np.ones(gridsize, dtype=np.float32) # permeability, can be diagonally anisotropic
er = np.ones(gridsize, dtype=np.float32) # permittivity, can be diagonally anisotropic

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

E = theano.shared(np.zeros(gridsize, dtype=np.float32)) # Electric field
H = theano.shared(np.zeros(gridsize, dtype=np.float32)) # Normalized magnetic field

# Display
fig = plt.figure()
ax = plt.axes(ylim=(-5, 5))
line1, = ax.plot(np.linspace(0.0, space_size, gridsize), np.zeros(gridsize, dtype=np.float32), 'r-')
line2, = ax.plot(np.linspace(0.0, space_size, gridsize), np.zeros(gridsize, dtype=np.float32), 'b-')

for layer in layers:
    n = layer[0]/layer[1]
    ax.add_patch(patches.Rectangle((layer[2], -20), layer[3], 40, color=(n, n, n)))

# Sinc function source
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
def gausspulse_source(er, ur, t0, tau, t):
    a_corr = -np.sqrt(er/ur) # amplitude correction term
    t_corr = np.sqrt(er*ur)*dz/(2*c0) + dt/2 # Time correction term
    return (
         a_corr * np.exp(-((t-t0)/tau)**2 + t_corr),
         np.exp(-((t-t0)/tau)**2)
    )

# Outputs 1.0 at time 0
def blip_source(t):
    return (0.0, 1.0) if t == 0 else (0.0, 0.0)

def init_animation():
    global line1, line2
    return line1, line2


src = T.fscalar()

step1 = theano.function([src], None, updates=[
    (H, T.inc_subtensor(T.inc_subtensor(H[1:-1], mkhx[1:-1] * (E[2:] - E[1:-1]) / dz)[500:501], [src]))
])
step2 = theano.function([src], None, updates=[
    (E, T.inc_subtensor(T.inc_subtensor(E[1:-1], mkey[1:-1] * (H[1:-1] - H[:-2]) / dz)[500:501], [src]))
])

i = 0
def animate(_):
    global i, ax, line1, line2, H, E, mkhx, mkey
    print(i)
    for i in range(i, i+100):
        t = i*dt
        src_ = gausspulse_source(1.0, 1.0, 200*ps, 50*ps, t)

        step1(np.float32(src_[0]))
        step2(np.float32(src_[1]))

        # Simply dampen at the edges instead of using the messy perfect edge or PML method.
        #E[:100] *= np.linspace(0.985,1.0,100)
        #H[:100] *= np.linspace(0.985,1.0,100)
        #E[-100:] *= np.linspace(1.0,0.985,100)
        #H[-100:] *= np.linspace(1.0,0.985,100)

    line1.set_ydata(E.get_value())
    line2.set_ydata(H.get_value())
    return line1, line2

anim = animation.FuncAnimation(fig, animate, init_func=init_animation, interval=0, blit=True)
plt.show()
