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

## 📁 Input Files

- `.msh` → Mesh file (geometry + discretization)
- Python scripts → Solver and computation modules

---

## 🧪 Notes

- Each example runs independently  
- Ensure mesh file path is correct  
- Results depend on mesh quality and parameters  





