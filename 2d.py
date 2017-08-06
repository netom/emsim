#!/usr/bin/env python3

# 2D FDTD, Hz mode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from scipy import signal
import time

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
freq_max = 10*GHz                   # maximal resolvable frequency
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
    for i in range(max(0, int(layer[2]/dx)), min(gridsize_x, int((layer[2]+layer[3])/dx))):
        #er[i] = layer[0]
        #mr[i] = layer[1]
        # TODO
        pass

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
def gausspulse_source(er, ur, t0, tau, t):
    a_corr = -np.sqrt(er/ur) # amplitude correction term
    t_corr = np.sqrt(er*ur)*dx/(2*c0) + dt/2 # Time correction term
    return (
         a_corr * np.exp(-((t-t0)/tau)**2 + t_corr),
         np.exp(-((t-t0)/tau)**2)
    )

def blip_source(t):
    return (0.0, 1.0) if t == 0 else (0.0, 0.0)

# Display
fig = plt.figure()
im  = plt.imshow(Ez, animated=True, clim=(-0.002, 0.002), cmap='gray', interpolation='nearest')

def init_animation():
    global im
    return im,

i = 0
def animate(_):
    global i, im, Cex, Cey, Ez, dx, dy, dt, Hx, Hy, Chz, Dz, erz, mrx, mry

    if i >= steps:
        return im,

    for i in range(i, i+10):
        t = i*dt
        src = gausspulse_source(1.0, 1.0, 300*ps, 100*ps, t)

        Cex[:,:-1] = (Ez[:,1:] - Ez[:,:-1]) / dy
        Cex[:, -1] = (       0 - Ez[:, -1]) / dy
        Cey[:-1,:] = -(Ez[1:,:] - Ez[:-1,:]) / dx
        Cey[ -1,:] = -(       0 - Ez[-1 ,:]) / dx

        Hx -= c0 * dt / mrx * Cex
        Hy -= c0 * dt / mry * Cey

        Chz[1:,1:] = (Hy[1:,1:] - Hy[:-1,1:]) / dx - (Hx[1:,1:] - Hx[1:,:-1]) / dy
        Chz[0 ,1:] = (Hy[0 ,1:] -          0) / dx - (Hx[0 ,1:] - Hx[0 ,:-1]) / dy
        Chz[1:,0 ] = (Hy[1:,0 ] - Hy[:-1,0 ]) / dx - (Hx[1:,0 ] -          0) / dy
        Chz[0 ,0 ] = (Hy[0 ,0 ] -          0) / dx - (Hx[0 ,0 ] -          0) / dy

        Dz += c0 * dt * Chz

        Ez = 1.0 / erz * Dz

        Ez[100,100] += src[1] # Simple soft source injection

    im.set_array(Ez)
    return im,

anim = animation.FuncAnimation(fig, animate, init_func=init_animation, interval=0, blit=True)
plt.show()
