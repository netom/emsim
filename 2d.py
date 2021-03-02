#!/usr/bin/env python3

# 2D FDTD, Hz mode

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

# Material patches
# patches are overwriting each other, last write wins
# First patch should be free space.
# (mr, er, startx, stary, widthx, widthy)
layers = [
    (1.0, 1.0, 0.0, 0.0, 1.0, 1.0), # Free space
    (1.0, 2.5, 0.6, 0,6, 0.2, 0.2) # A 20x20cm slab of plastic
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

space_size_x = 1.0                  # meters
space_size_y = 1.0                  # meters
freq_max = 20*GHz                   # maximal resolvable frequency
lamb_min = c0 / (freq_max * n_max)  # minimal wavelength
dxpmwl = 10                         # delta-x per minimal wavelength, a rule-of-thumb constant
dypmwl = 10                         # delta-y per minimal wavelength, a rule-of-thumb constant
dx = lamb_min / dxpmwl              # Spatial step size, meters
dy = lamb_min / dypmwl              # Set the two spatial steps equal for now
gridsize_x = int(space_size_x / dx) # Size of the grid in cells
gridsize_y = int(space_size_y / dy) # Size of the grid in cells
simlen = 5 * max(space_size_x, space_size_y) / c0   # Simulation length, seconds (5 travels back & forth)
dt = n_min * min(dx, dy) / (2*c0)   # From the Courant-Friedrichs-Lewy condition. This is a rule of thumb
steps = int(simlen / dt)            # Number of simulation steps
bsize = 100                         # Dampening boundary thickness
bcoeff = 1.015                      # Dampening coefficient
batch = 10                          # Number of iterations between drawings


print("simulation length:", simlen)
print("grid size:", gridsize_x, gridsize_y)
print("steps:", steps)
print("dt:", dt)
print("dx:", dx)
print("dy:", dy)

mrx = np.ones((gridsize_x, gridsize_y)) # permeability, can be diagonally anisotropic
mry = np.ones((gridsize_x, gridsize_y))
erz = np.ones((gridsize_x, gridsize_y)) # permittivity

for layer in layers:
    for i in range(max(0, int(layer[2]/dx)), min(gridsize_x, int((layer[2]+layer[4])/dx))):
        for j in range(max(0, int(layer[3]/dy)), min(gridsize_y, int((layer[3]+layer[5])/dy))):
            erz[i][j] = layer[0]
            mrx[i][j] = layer[1]
            mry[i][j] = layer[1]

# Fields

Cex = np.zeros((gridsize_x, gridsize_y)) # Curl of the normalized electric field
Cey = np.zeros((gridsize_x, gridsize_y))
Hx  = np.zeros((gridsize_x, gridsize_y)) # Magnetic field
Hy  = np.zeros((gridsize_x, gridsize_y))
Chz = np.zeros((gridsize_x, gridsize_y)) # Curl of the magnetic field
Dz  = np.zeros((gridsize_x, gridsize_y)) # Normalized D-field
Ez  = np.zeros((gridsize_x, gridsize_y)) # Normalized electric field

for layer in layers:
    n = layer[0]/layer[1]
    #ax.add_patch(patches.Rectangle((layer[2], -20), layer[3], 40, color=(n, n, n))) # TODO.

# Sinc function source
@numba.jit()
def sinc_source(er, ur, period, t0, t):
    a_corr = -np.sqrt(er/ur) # amplitude correction term
    t_corr = np.sqrt(er*ur)*dx/(2*c0) + dt/2 # Time correction term
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
    t_corr = np.sqrt(er*ur)*dx/(2*c0) + dt/2 # Time correction term
    return (
         a_corr * np.exp(-((t-t0)/tau)**2 + t_corr),
         np.exp(-((t-t0)/tau)**2)
    )

# Outputs 1.0 at time 0
@numba.jit()
def blip_source(t):
    return (0.0, 1.0) if t == 0 else (0.0, 0.0)

# Display
fig = plt.figure()
im  = plt.imshow(Ez, animated=True, clim=(-0.002, 0.002), cmap='gray', interpolation='nearest')

def init_animation():
    global im
    return im,

@numba.jit()
def step(Cex, Cey, Hx, Hy, Chz, Dz, Ez, ifrom, ito):
    for i in range(ifrom, ito):
        t = i*dt
        src = gausspulse_source(1.0, 1.0, 300*ps, 100*ps, t)

        Cex[1:-1,1:-1] =  (Ez[1:-1,2:] - Ez[1:-1,1:-1]) / dy
        Cey[1:-1,1:-1] = -(Ez[2:,1:-1] - Ez[1:-1,1:-1]) / dx

        Hx -= c0 * dt / mrx * Cex
        Hy -= c0 * dt / mry * Cey

        Chz[1:-1,1:-1] = (Hy[1:-1,1:-1] - Hy[:-2,1:-1]) / dx - (Hx[1:-1,1:-1] - Hx[1:-1,:-2]) / dy

        Dz += c0 * dt * Chz
        Ez[:] = 1.0 / erz * Dz # Mind the colon

        Ez[200,200] += src[1] # Simple soft source injection

i = 0
def animate(_):
    global i, im, Cex, Cey, Ez, dx, dy, dt, Hx, Hy, Chz, Dz, erz, mrx, mry

    time1 = time.time()
    step(Cex, Cey, Hx, Hy, Chz, Dz, Ez, i, i+batch)
    time2 = time.time()
    print("step %d took %fms" % (i, time2-time1))

    im.set_array(Ez)

    i += batch

    return im,

anim = animation.FuncAnimation(fig, animate, init_func=init_animation, interval=0, blit=True)
plt.show()
