# pyTOM — Topology Optimization for Magnetic Actuators in Python

`pyTOM` is an educational, open-source framework for density-based topology
optimization (TO) of magnetic actuators. It accompanies the paper:

> A. Ramadoni, J. Lee, *pyTOM: Topology Optimization for Magnetic Actuators in Python*,
> Structural and Multidisciplinary Optimization.

The framework captures three advanced features: permanent magnets as fixed field
sources, nonlinear (field-dependent) reluctivity for magnetic saturation, and a
multi-position analysis strategy that evaluates the actuator over several plunger
positions. It uses a Helmholtz density filter, a Heaviside projection, and the
Method of Moving Asymptotes (MMA) for the design update.

## Requirements

- Python 3 (developed and tested with Python 3.14)
- [NumPy](https://numpy.org/) — array and linear-algebra operations
- [SciPy](https://scipy.org/) — sparse matrices and sparse linear solvers
- [Matplotlib](https://matplotlib.org/) — figures and field/density plots

Install everything with:

```bash
pip install -r requirements.txt
```

## Repository structure

Each numerical example lives in a self-contained folder that carries its own copy
of the modules it needs, so it can be run in isolation. The folders correspond to
the four numerical examples of Section 5 of the paper (see Fig. 9):

```
1. Numerical ex 1/               IPM motor — electromagnetic field validation
2. Numerical ex 2 - Linear/      Actuator TO, linear material
2. Numerical ex 2 - Nonlinear/   Actuator TO, nonlinear (saturable) material
3. Numerical ex 3/               Actuator TO, multi-position analysis
```

The single-purpose modules are named by workflow stage:

| Module | Role |
|---|---|
| `F1_Pre_Mesh_Import`   | import the Gmsh mesh, build domain identifiers |
| `F2_Pre_FEM_Init`      | element kernels, sparse-matrix indices, boundary conditions |
| `F3_Pre_Opt_Init`      | design variables, Helmholtz filter, MMA setup |
| `F4_Main_Solve_VecPot` | solve the (linear) magnetostatic vector potential |
| `F5_Main_Comp_Flux`    | compute the flux density B = curl A |
| `F6_Main_NR_Jacobian`  | Newton–Raphson tangent for the nonlinear solve |
| `F7_Main_Comp_Force`   | Maxwell-stress-tensor force |
| `F8_Main_Comp_Sens`    | adjoint sensitivity analysis |
| `F0_Main_PM_Source`    | permanent-magnet equivalent source |
| `F0_Main_Mat_Nonlinear`, `F0_Main_Mat_Derivative` | nonlinear B–H model and its derivative |
| `F0_Main_Line_Search`  | energy-based backtracking line search for the damped Newton step |
| `F9_Post_Process_Plot` | field, density, and convergence plots |
| `mma.py`               | third-party MMA optimizer (see note below) |

The driver of each example is its `Main_code_*.py` file
(`Main_code_Ex1.py`, `Main_code_Ex2_Linear.py`, `Main_code_Ex2_Nonlinear.py`,
`Main_code_Mulpos.py`).

Not every folder contains every module. Example 1 performs field validation only
and therefore omits the optimization modules (`F7`, `F8`, `mma.py`), while the
linear case of Example 2 omits the nonlinear ones (`F6`, `F0_Main_Mat_*`,
`F0_Main_Line_Search`).

## How to run

Run an example from inside its own folder so that the modules and the `.msh`
file are found:

```bash
cd "2. Numerical ex 2 - Nonlinear"
python Main_code_Ex2_Nonlinear.py
```

For Example 3, the single driver sweeps several plunger-position counts in one
run and writes a force-profile comparison:

```bash
cd "3. Numerical ex 3"
python Main_code_Mulpos.py
```

The list of cases is set at the top of `Main_code_Mulpos.py`:

```python
NPOS_LIST = [1, 11, 21]   # numbers of plunger positions (counts, not indices)
```

Each entry produces one optimization whose objective is the magnetic force
averaged over that number of plunger positions, and each run writes to its own
`Results/Npos_<n>/` folder.

Figures (field, density, and convergence history) are written to the local
`Figures/` or `Results/` folder of each example as the optimization runs.

## Nonlinear solver: energy-based line search

The nonlinear magnetostatic problem is solved by a damped Newton–Raphson
iteration. The step size is selected by a backtracking line search on the
magnetic energy functional, implemented in `F0_Main_Line_Search.py`: starting
from the full Newton step, the step is halved until the trial update decreases
the energy. This makes the iteration globally convergent while recovering the
full Newton step, and hence the fast local convergence, near the solution.

The linear case of Example 2 contains a single direct linear solve and therefore
no Newton iteration and no line search.

## Output verbosity and timing

Each driver exposes a verbosity switch:

```python
inputs["verbose"] = True    # default: print the Newton-Raphson trace and MMA timing
inputs["verbose"] = False   # print only one compact line per optimization iteration
```

In `Main_code_Mulpos.py` the same switch is set inside `INPUTS_BASE`.

Each optimization iteration also reports the time spent in the MMA design update,
and the end of each run reports the cumulative MMA time together with its share of
the total wall time, so the cost of the optimizer can be compared against the
finite-element solve, the force evaluation, and the adjoint sensitivity analysis.

## Third-party code

`mma.py` is the publicly available Python port (by A. Deetman) of the original
MATLAB MMA code by K. Svanberg. It is included unmodified and retains its
original copyright and GNU General Public License. See
<https://github.com/arjendeetman/GCMMA-MMA-Python>.

Because this GPL-licensed component is distributed together with the framework,
the repository as a whole is released under the GNU General Public License v3.0;
see `LICENSE`.

## Archiving

An archived snapshot of this repository is deposited on Zenodo and is citable via
a permanent DOI; see the Data Availability statement of the accompanying paper.

## Citing pyTOM

If you use this code, please cite the accompanying paper (see the top of this
file).
