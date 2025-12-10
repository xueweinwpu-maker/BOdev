# -*- coding: utf-8 -*-
"""
Generates an improved Extended Design Structure Matrix (XDSM) diagram for the
WDA-ASO framework with a clearer, more standard workflow representation.

This script uses the pyXDSM library. To run it, you will need:
1. pyXDSM: pip install pyxdsm
2. A LaTeX distribution (e.g., MiKTeX for Windows).
"""
# Note: This script uses the GROUP feature. If you get an ImportError,
# you may need to update pyXDSM: pip install --upgrade pyxdsm
from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, GROUP

# 1. Initialize XDSM object
x = XDSM()

# 2. Define components, separating the main loop from the optimizer's internals
# Main Process Components
x.add_system("preproc", FUNC, r"\text{1: WDA-Informed Parameterization}")
x.add_system("opt_driver", OPT, r"\text{2, 7$\rightarrow$3: Optimizer}")
x.add_system("geo_para", FUNC, r"\text{3: Geometry Parameterization}")
x.add_system("volume_grid", FUNC, r"\text{4: Volume Mesh Deformation}")

# x.add_system("geo_para", FUNC, r"\text{3：Geometry Parameterization}")

# x.add_system("volume_grid", FUNC, r"\text{3：Volume Mesh Deformation}")
x.add_system("evaluator", SOLVER, r"\text{CFD Analysis}")


# Internal BO Components (to be grouped under the main optimizer)
x.add_system("surrogate", FUNC, r"\text{Surrogate Modeling (GP)}")
x.add_system("acqf", FUNC, r"\text{Acquisition Function Opt.}")

# 3. Group the internal BO components visually to explain how BO works
# x.add_group([ "opt_driver", "surrogate", "acqf"], "Bayesian Optimizer")

# 4. Define the main optimization process on the diagonal
x.add_process(["opt_driver", "geo_para","volume_grid","evaluator", "opt_driver"], arrow=True)

# 5. Define data connections (the workflow)
# --- SETUP ---
# x.add_input("param", r"\text{Baseline Geom., WDA}")
# x.connect("param", "opt_driver", r"D, \text{Bounds}")

# --- OPTIMIZATION LOOP ---
# Optimizer sends a new design to be evaluated
x.connect("opt_driver", "evaluator", r"x_k")

# The Evaluator includes these steps internally
x.add_input("opt_driver", r"\text{2:Baseline design}")
x.add_input("evaluator", r"\text{CFD Solver}")
x.add_input("evaluator", r"\text{Post-Processing}")

# Evaluator returns the results
x.connect("evaluator", "opt_driver", r"y_k = (\text{Vol}, K)")

# --- INTERNAL BO LOGIC ---
# Optimizer sends full history to train surrogates
x.connect("opt_driver", "surrogate", r"X, Y = \{x_i, y_i\}_{i=1}^k")

# Surrogates are passed to the acquisition function
x.connect("surrogate", "acqf", r"\text{GP Models}")

# The acquisition function proposes the next point to sample
x.connect("acqf", "opt_driver", r"x_{k+1}")

# --- FINAL OUTPUTS ---
x.add_output("opt_driver", r"x^*, \text{Plots}")

# 6. Write the XDSM diagram to a file
x.write("WDA_ASO_XDSM_Detailed", build=True)
print("WDA_ASO_XDSM_Detailed.pdf and .png have been generated.")

