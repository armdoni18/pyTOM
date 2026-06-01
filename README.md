# pyTOM — Topology Optimization for Magnetic Actuators in Python

`pyTOM` is an educational, open-source framework for density-based topology
optimization (TO) of magnetic actuators. It accompanies the paper:

> A. Ramadoni, J. Lee, *pyTOM: Topology Optimization for Magnetic Actuators in Python*.

The framework captures three features that are often omitted in educational TO
codes for electromechanical devices: permanent magnets as fixed field sources,
nonlinear (field-dependent) reluctivity for magnetic saturation, and a
multi-position analysis strategy that evaluates the actuator over several
plunger positions. It uses a Helmholtz density filter, a Heaviside projection,
and the Method of Moving Asymptotes (MMA) for the design update.

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

Each numerical example lives in a self-contained folder that carries its own
copy of the modules it needs, so it can be run in isolation:

```
1. pyTOM Numerical ex 1/      IPM motor — electromagnetic field validation
2. pyTOM Numerical ex 2/
   Linear/                    Actuator TO, linear material
   NonLinear/                 Actuator TO, nonlinear (saturable) material
3. pyTOM Numerical ex 3/      Actuator TO, multi-position analysis
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
| `F9_Post_Process_Plot` | field, density, and convergence plots |
| `mma.py`               | third-party MMA optimizer (see note below) |

The driver of each example is its `Main_code_*.py` file
(`Main_code_Ex1.py`, `Main_code_Ex2_Linear.py`, `Main_code_Ex2_Nonlinear.py`,
`Main_code_Mulpos.py`).

## How to run

Run an example from inside its own folder so that the modules and the `.msh`
file are found:

```bash
cd "2. pyTOM Numerical ex 2/NonLinear"
python Main_code_Ex2_Nonlinear.py
```

For Example 3, the single driver sweeps several plunger-position counts in one
run and writes a force-profile comparison:

```bash
cd "3. pyTOM Numerical ex 3"
python Main_code_Mulpos.py
```

Figures (field, density, and convergence history) are written to the local
`Figures/` or `Results/` folder of each example as the optimization runs.

## Third-party code

`mma.py` is the publicly available Python port (by A. Deetman) of the original
MATLAB MMA code by K. Svanberg. It is included unmodified and retains its
original copyright and GNU General Public License.

## Citing pyTOM

If you use this code, please cite the accompanying paper (see the top of this
file).
