# -*- coding: utf-8 -*-
"""
Generates a refined XDSM diagram for the WDA-ASO framework.

This version simplifies the view by grouping geometry and mesh handling
as internal processes of the main "Evaluator" block, which clarifies
the primary optimization loop. This script is compatible with older
versions of pyXDSM that do not support the 'label' argument in add_process.
"""
from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

# 1. Initialize XDSM object
x = XDSM(use_sfmath=True)

# 2. Define core components
x.add_system("param", FUNC, [r"\text{WDA-Informed}",r"\text{Parameterization}"], stack=False)
x.add_system("DoEs", FUNC, [r"\text{Design of}",r"\text{Experiments (DoEs)}"], stack=False)
x.add_system("opt", OPT, [r"\textbf{Multi-objective}",r"\textbf{Bayesian Optimizer}"])
x.add_system("eval", SOLVER, [r"\text{Geometry, }",r" Mesh \& CFD"], stack=True)
# x.add_system("param", FUNC, r"\text{WDA-Informed Parameterization}", stack=False)

# Internal BO components
x.add_system("surrogate", FUNC, [r"\text{Surrogate Modeling }",r"\text{(Gaussian Process)}"], stack=False)
x.add_system("acqf", FUNC, [r"\text{Acquisition Function}",r"\text{Optimization}"], stack=False)

# 3. Define the main optimization process on the diagonal
# The 'label' argument has been removed for compatibility.
x.add_process(["opt", "eval", "opt"], arrow=True)

# 4. Define data connections
# --- SETUP ---
x.add_input("param", r"\text{Baseline Vehicle, WDA}")
x.connect("param", "DoEs", r"DVs, \text{Bounds}")
x.connect("DoEs", "opt", r"X^0, Y^0")

# --- OPTIMIZATION LOOP ---
x.connect("opt", "eval", r"\text{Updated DVs } (x_k)")
x.add_input("eval", r"\text{Design Condition}")
x.connect("eval", "opt", r"y_k = (\text{Vol.}, L/D, \text{etc.})")

# --- INTERNAL BO LOGIC ---
x.connect("opt", "surrogate", r"X, Y = \{x_i, y_i\}_{i=1}^k")
x.connect("surrogate", "acqf", r"\text{Surrogate Models}")
x.connect("acqf", "opt", r"x_{k+1}")

# 5. Define overall outputs
x.add_output("opt", r"\text{Pareto Frontier } ", side=LEFT)

# 6. Write the XDSM diagram to a file
x.write("WDA_ASO_XDSM_final_layout", build=True)
print("Refined diagram WDA_ASO_XDSM_refined.pdf and .png have been generated.")