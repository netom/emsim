# EMSim

EMSim is a FDTD (Finite-Difference Time-Domain) solver for the Maxwell
equations.

The three scripts are standalone simulations — not a library.

- `1d.py` — 1D FDTD, Ey/Hx mode, accelerated with `numba`
- `2d.py` — 2D FDTD, Hz mode, accelerated with `numba`
- `1d-tensorflow.py` — 1D FDTD using the TF2 eager execution API

## Run

```bash
. venv.sh  # creates .venv, activates it, installs requirements.txt
./1d.py    # run a simulation (opens matplotlib animation window)
./2d.py
./1d-tensorflow.py
```

## Code

### Grid and time-step derivation

Simulation parameters (grid size, `dt`, `dz`/`dx`/`dy`, number of steps) are
all derived from physical constants and the material stack at startup. The
Courant-Friedrichs-Lewy (CFL) condition governs `dt`:

```python
dt = n_min * dz / (2 * c0)
```

Grid spatial resolution is set by the minimum wavelength and `dzpmwl` (cells
per wavelength, default 10).

### Material definition

Materials are defined as a list of tuples that **overwrite** each other in
order (last write wins). Free space must be the first entry:

- 1D: `(mr, er, start_m, width_m)`
- 2D: `(mr, er, startx_m, starty_m, widthx_m, widthy_m)`

The material arrays (`mr`, `er`) are filled by iterating layers and snapping to
the grid.

### Update coefficients

Precomputed coefficient arrays `mkhx` and `mkey` are used in the field update
equations to avoid repeated division inside the hot loop:

```python
mkhx = c0 * dt / mr
mkey = c0 * dt / er
```

### Normalized magnetic field

The magnetic field `H` is stored normalized (multiplied by the impedance of
free space). Source injection must account for this with `a_corr =
-sqrt(er/ur)`.

### `@numba.jit()` step functions

The inner simulation loop is wrapped with `@numba.jit()` for performance. The
`step()` function takes field arrays and a range `(ifrom, ito)` rather than
iterating one step at a time. The `batch` variable controls how many steps run
between animation frames.

### Absorbing boundaries

Boundaries use a log-linear dampening ramp (`LB`, `RB`) applied to both E and H
after each update step. Thickness is `bsize` cells.

### Source types

Three source functions are defined in each script (sinc, Gaussian pulse, blip).
Sources are injected as soft sources by adding to a specific grid cell index.
The Gaussian pulse source is used by default.

### TensorFlow variant

`1d-tensorflow.py` uses TF2 eager execution. `E` and `H` are `tf.Variable`s.
The inner loop is compiled with `@tf.function(input_signature=[...])`, which
traces the graph once and reuses it for every call. Sequential field update
ordering (H before E) is achieved by structuring the Python code sequentially
inside the `@tf.function` body — no `tf.control_dependencies` needed.
Point-source injection uses `tf.tensor_scatter_nd_add`; interior field updates
are written back via `tf.concat` + `Variable.assign`. Field values are read back
for display with `.numpy()`.
EMSim is a FDTD solver for the Maxwell equations with focus on simulating antennas.
