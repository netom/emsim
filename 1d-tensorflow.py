#!/usr/bin/env python3

# 1D FDTD, Ey/Hx mode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from scipy import signal
import tensorflow as tf

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
freq_max = 100*GHz                # maximal resolvable frequency
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

sess = tf.Session(config=tf.ConfigProto())

mr = np.ones(gridsize) # permeability, can be diagonally anisotropic
er = np.ones(gridsize) # permittivity, can be diagonally anisotropic

for layer in layers:
    # TODO: snap layers to grid / snap grid to layers?
    for i in range(max(0, int(layer[2]/dz)), min(gridsize, int((layer[2]+layer[3])/dz))):
        er[i] = layer[0]
        mr[i] = layer[1]

mr = tf.constant(mr, tf.float32)
er = tf.constant(er, tf.float32)

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

E = tf.Variable(tf.zeros(gridsize)) # Electric field
H = tf.Variable(tf.zeros(gridsize)) # Normalized magnetic field

# Display
fig = plt.figure()
ax = plt.axes(ylim=(-5, 5))
line1, = ax.plot(np.linspace(0.0, space_size, gridsize), np.zeros(gridsize), 'r-')
line2, = ax.plot(np.linspace(0.0, space_size, gridsize), np.zeros(gridsize), 'b-')

for layer in layers:
    n = layer[0]/layer[1]
    ax.add_patch(patches.Rectangle((layer[2], -20), layer[3], 40, color=(n, n, n)))

# Sinc function source
def sinc_source(er, ur, period, t0, t):

    sinc = lambda x: tf.cond(tf.equal(x, 0), lambda: tf.constant(1.0), lambda: tf.sin(np.pi*x)/(np.pi*x))

    a_corr = -tf.sqrt(er/ur)                 # Amplitude correction term
    t_corr = tf.sqrt(er*ur)*dz/(2*c0) + dt/2 # Time correction term

    x = (t-t0)*2/period

    return (
        a_corr * sinc(x + t_corr), # H field
        sinc(x)                    # E field
    )

# Gaussian pulse source
def gausspulse_source(er, ur, t0, tau, t):
    a_corr = -tf.sqrt(er/ur) # amplitude correction term
    t_corr = tf.sqrt(er*ur)*dz/(2*c0) + dt/2 # Time correction term
    return (
         a_corr * tf.exp(-((t-t0)/tau)**2 + t_corr),
         tf.exp(-((t-t0)/tau)**2)
    )

# Outputs 1.0 at time 0
def blip_source(t):
    return (0.0, 1.0) if t == 0 else (0.0, 0.0)

def init_animation():
    global line1, line2
    return line1, line2

i = 0
def animate(_):
    global i, ax, line1, line2, H, E, mkhx, mkey, step, t

    print(i)
    for i in range(i, i+20):
        step.run({t: i*dt})

    line1.set_ydata(E.eval())
    line2.set_ydata(H.eval())

    return line1, line2


t = tf.placeholder(tf.float32, shape=())

gpsrc = gausspulse_source(1.0, 1.0, 200*ps, 50*ps, t)

op1 = H[1:-1].assign(H[1:-1] + mkhx[1:-1] * (E[2:] - E[1:-1]) / dz)

with tf.control_dependencies([op1]):
    op2 = H[int(500)].assign(H[int(500)] + gpsrc[0])

with tf.control_dependencies([op2]):
    op3 = E[1:-1].assign(E[1:-1] + mkey[1:-1] * (H[1:-1] - H[:-2]) / dz)

with tf.control_dependencies([op3]):
    op4 = E[int(500)].assign(E[int(500)] + gpsrc[1])

step = tf.group(op1, op2, op3, op4)

with sess.as_default():
    tf.global_variables_initializer().run()
    #tf.summary.FileWriter('log', sess.graph)
    anim = animation.FuncAnimation(fig, animate, init_func=init_animation, interval=0, blit=True)
    plt.show()
