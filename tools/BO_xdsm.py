"""
XDSM diagram for CFD-based Shape Optimization with Adjoint Method
Requires: pip install pyxdsm
"""

from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT, RIGHT

def create_cfd_optimization_xdsm():
    """
    Creates XDSM diagram for aerodynamic shape optimization
    with FFD parameterization and adjoint-based gradient computation
    """
    
    # Initialize XDSM
    x = XDSM()
    
    # Add all systems/components with their labels
    x.add_system("preproc", FUNC, r"\text{1: WDA-Informed Parameterization}")
    x.add_system("baseline", FUNC, r"\text{2: Baseline design}")
    x.add_system("ffd", FUNC, r"\text{3: FFD points}")
    x.add_system("mesh", FUNC, r"\text{4: Volume mesh}")
    x.add_system("opt", OPT, r"\text{2, 7$\rightarrow$3: Optimizer}")
    x.add_system("geom", FUNC, r"\text{3: Geometry parameterization}")
    x.add_system("meshdef", FUNC, r"\text{4: Volume mesh deformation}")
    x.add_system("cfd", SOLVER, r"\text{5: CFD solver}")
    x.add_system("adjoint", SOLVER, r"\text{6: Adjoint solver}")
    x.add_system("output", FUNC, r"\text{8: Optimized design}")
    
    # Initial setup connections (top horizontal flow)
    x.connect("baseline", "preproc", "")
    x.connect("preproc", "ffd", "")
    x.connect("ffd", "mesh", "")
    
    # Optimizer connections
    x.connect("baseline", "opt", "")
    x.connect("opt", "geom", r"\text{3: Updated FFD displacement}")
    x.connect("opt", "ffd", "")
    
    # Geometry parameterization
    x.connect("geom", "meshdef", r"\text{4: Updated design}" + "\n" + r"\text{surface coordinates}")
    x.connect("geom", "opt", r"\text{7: Geometric constraints}" + "\n" + r"\text{and derivatives}")
    
    # Mesh deformation
    x.connect("mesh", "meshdef", "")
    x.connect("meshdef", "cfd", r"\text{5: Updated mesh}")
    
    # CFD solver
    x.connect("cfd", "adjoint", r"\text{6: State variables}")
    x.connect("cfd", "opt", r"\text{7: Values of objectives}" + "\n" + r"\text{and constraints}")
    
    # Adjoint solver feedback
    x.connect("adjoint", "opt", r"\text{7: Derivatives of objectives}" + "\n" + r"\text{and constraints}")
    
    # Final output
    x.connect("opt", "output", "")
    
    # Define the main process flow with arrows
    x.add_process(["preproc", "baseline", "ffd", "mesh", "opt", 
                   "geom", "meshdef", "cfd", "adjoint", "opt", "output"], 
                  arrow=True)
    
    # Write output files
    x.write("cfd_shape_optimization", cleanup=False)
    print("Generated: cfd_shape_optimization.tex")
    print("\nXDSM diagram created successfully!")




if __name__ == "__main__":
    print("Creating XDSM diagrams for CFD-based shape optimization...\n")
    
    try:
        # Create the main diagram
        create_cfd_optimization_xdsm()
        

        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure pyxdsm is installed: pip install pyxdsm")