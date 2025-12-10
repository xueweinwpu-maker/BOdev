# -*- coding: utf-8 -*-
"""
Generates a corrected and improved Extended Design Structure Matrix (XDSM)
diagram for the WDA-ASO framework. This version removes the non-existent
'add_group' call and uses naming and stacking to create a clear visual hierarchy.
"""
from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT,RIGHT

# 1. Initialize XDSM object with a stacked layout for clarity
x = XDSM(use_sfmath=True)

# 2. Define components. The main OPT block is now the group title.
x.add_system("param", FUNC, [r"\text{1:WDA-Informed}",r"\text{Parameterization}"], stack=False)
x.add_system("opt", OPT, [r"\text{2, 8$\rightarrow$3:}", r"\textbf{2:Bayesian Optimizer}"])
x.add_system("geo_para", FUNC, [r"\text{3: Geometry}",r"\text{Parameterization}"])
x.add_system("volume_grid", FUNC, [r"\text{4: Volume Mesh}",r"\text{Deformation}"])
x.add_system("eval", SOLVER, [r"5:\text{CFD Solver \&}",r"\text{Post-Processing}"],stack=True)
# Internal BO Components (will be stacked under the main optimizer)
x.add_system("surrogate", FUNC, [r"6:\text{Surrogate Modeling }",r"\text{(Gaussian Process)}"], stack=False)
x.add_system("acqf", FUNC, [r"7:\text{Acquisition}",r"\text{Function Optimization}"], stack=False)

# 3. Define the main optimization process on the diagonal
x.add_process(["opt", "geo_para","volume_grid","eval", "opt"], arrow=True, )

# 4. Define data connections (the workflow)
# --- SETUP ---
x.add_input("param", r"\text{1:Baseline Geom., WDA}")
x.add_input("eval", r"\text{5:Fly Condition}")




# --- OPTIMIZATION LOOP ---
# Optimizer sends a new design to be evaluated
x.connect("param", "opt", r"2:D, \text{Bounds}")
x.connect("param", "geo_para", r"\text{3:FFD points}")
x.connect("opt", "geo_para", [r"3:\text{Updated FFD}",r"\text{dispalcement }(x_k)"])
x.connect("param", "volume_grid", [r"4:\text{Volume Mesh}"])
x.connect("geo_para", "volume_grid", [r"4:\text{Updated}",r"\text{Surface Mesh}"])
# x.connect("opt", "eval", r"x_k")
# Evaluator returns the results
x.connect("eval", "opt", r"7:y_k = (\text{Vol}, K,etc.)")

# --- INTERNAL BO LOGIC ---
# Optimizer sends full history to train surrogates
x.connect("opt", "surrogate", r"6:X, Y = \{x_i, y_i\}_{i=1}^k")
# Surrogates are passed to the acquisition function
x.connect("surrogate", "acqf", r"7:\text{GP Models}")
# Acquisition function proposes the *next* point to sample
x.connect("acqf", "opt", r"8:x_{k+1}")

# 5. Define overall inputs and outputs
x.add_output("opt", r"\text{Final Design } x^*", side=LEFT)
# x.add_output("eval", r"\text{Performance History}", side=RIGHT)

# 6. Write the XDSM diagram to a file
x.write("WDA_ASO_XDSM_corrected", build=True)
print("Corrected diagram WDA_ASO_XDSM_corrected.pdf and .png have been generated.")