# pyTOM

## 📌 Overview
**pyTOM** is a Python-based tool for numerical simulations of Topology optimization for the magnetic actuator problem.

This repository provides multiple numerical examples that demonstrate the complete computational workflow, including:
- Mesh import
- FEM initialization
- Magnetic field computation
- Nonlinear solving
- Force calculation
- Sensitivity analysis
- Post-processing and visualization

---

## 📂 Repository Structure

Each folder represents an independent numerical example.

pyTOM/
1. pyTOM Numerical ex 1/
2. pyTOM Numerical ex 2/
3. pyTOM Numerical ex 3/
---

## ▶️ How to Run 
Each example can be executed independently.
### General Steps
1. Go to the desired example folder  
2. Run the corresponding main script 

---
### 🔹 Example (Example 3)
1. Go to folder:
   `3. pyTOM Numerical ex 3`
3. Run:
   `Main_code_Mulpos.py`

## ⚠️ Important Setup

Before running the code, create a folder named:
`Figures`

This folder is required because all post-processing results (plots and figures) will be saved there.

### How to create:
- Manually create a folder named `Figures` inside the example folder  
---

## ⚙️ Detailed Workflow 

The simulation follows this pipeline:

### 1. Mesh Import
`F1_Pre_Mesh_Import.py`
- Reads mesh data from `.msh` file  
- Defines nodes and element connectivity  

### 2. FEM Initialization
`F2_Pre_FEM_Init.py`
- Sets up degrees of freedom  
- Applies boundary conditions  

### 3. Optimization Initialization
`F3_Pre_Opt_Init.py`
- Defines parameters and variables  

### 4. Solve Magnetic Vector Potential
`F4_Main_Solve_VecPot.py`
- Solves FEM system for vector potential **A**  

### 5. Magnetic Flux Density
`F5_Main_Comp_Flux.py`
- Computes magnetic flux density **B**  

### 6. Nonlinear Solver
`F6_Main_NR_Jacobian.py`
- Uses Newton-Raphson iteration  
- Handles nonlinear material behavior  

### 7. Force Computation
`F7_Main_Comp_Force.py`
- Computes electromagnetic force using Maxwell Stress Tensor (MST)  

### 8. Sensitivity Analysis
`F8_Main_Comp_Sens.py`
- Computes sensitivities for optimization  

### 9. Post Processing
`F9_Post_Process_Plot.py`
- Visualizes:
  - Domain  
  - Vector potential  
  - Magnetic field  

---

## 📁 Input Files

- `.msh` → Mesh file (geometry + discretization)
- Python scripts → Solver and computation modules

---

## 🧪 Notes

- Each example runs independently  
- Ensure mesh file path is correct  
- Results depend on mesh quality and parameters  





