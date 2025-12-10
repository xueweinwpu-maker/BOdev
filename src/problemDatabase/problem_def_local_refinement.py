# -*- coding: utf-8 -*-
"""
Problem definition for a Phase 2, local refinement optimization run.
This file takes the best point found during a global search and creates
a new, tightly-bounded problem to intensely search its local vicinity.
"""
import torch
import pandas as pd
from typing import List
import numpy as np

from src.problem_structures import (
    OptimizationConfig, DesignPoint, Objective, GlobalConstraint, InnerOptConfig
)
from surrogates.pipelines import SurrogateBackedPipeline

def get_full_data_bounds(data_paths: List[str], offline_dim: int):
    """
    Helper function to get the absolute bounds from the full dataset.
    """
    all_dfs = []
    for path in data_paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df.dropna(inplace=True)
        all_dfs.append(df)
    full_df = pd.concat(all_dfs, ignore_index=True)
    var_names = [f'ratio[{i}]' for i in range(offline_dim)]
    lower_bounds = full_df[var_names].min().values
    upper_bounds = full_df[var_names].max().values
    return lower_bounds, upper_bounds

def define_problem() -> OptimizationConfig:
    """
    Defines the local refinement optimization problem.
    """
    data_files = ["data/train_dataM04.csv", "data/verify_dataM04.csv"]

    low_speed_pipeline = SurrogateBackedPipeline(
        mach_str="0.4",
        train_paths=data_files,
        inner_loop_objective="CL"
    )

    design_points = [
        DesignPoint(
            name="Subsonic_Ma0.4",
            pipeline=low_speed_pipeline,
            inner_opt_config=InnerOptConfig(
                bounds=[(-50.0, -20.0)], x0=[-35.0]
            )
        ),
    ]

    objectives = [
        Objective(
            name="Max CL (Ma 0.4)", 
            eval_function=lambda res_map: res_map["Subsonic_Ma0.4"]["CL"]
        ),
    ]

    constraints = [
        GlobalConstraint(
            name="K_min_subsonic",
            description="K @ Ma 0.4 >= 3.95",
            eval_function=lambda res_map: res_map["Subsonic_Ma0.4"]["K"] - 3.95
        )
    ]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.double
    
    # --- Phase 2: Local Refinement Configuration ---
    
    # 1. Define the center point (best result from the global search)
    center_point = np.array([
        0.0499, 0.0495, 0.0500, 0.0495, 0.0499, 0.0496, -0.1200, 0.0197, 
        0.0200, 0.0199, 0.0200, 0.0198, 0.0198, 0.0200, 0.0198
    ])
    
    # 2. Define the size of the refinement box (e.g., +/- 0.02)
    box_half_width = 0.02
    
    # 3. Create the new, tight bounds
    new_lower_bounds = center_point - box_half_width
    new_upper_bounds = center_point + box_half_width
    
    # 4. Clip the new bounds to ensure they don't exceed the original data's absolute limits
    original_lower, original_upper = get_full_data_bounds(data_files, 15)
    final_lower_bounds = np.maximum(new_lower_bounds, original_lower)
    final_upper_bounds = np.minimum(new_upper_bounds, original_upper)

    offline_bounds = torch.tensor([final_lower_bounds, final_upper_bounds], dtype=DTYPE, device=DEVICE)
    
    # 5. Use the center point as the initial guiding point for the search
    guiding_point = torch.tensor(center_point, dtype=DTYPE, device=DEVICE).unsqueeze(0)

    problem_config = OptimizationConfig(
        offline_dim=15,
        online_dim=1,
        offline_bounds=offline_bounds,
        design_points=design_points,
        objectives=objectives,
        constraints=constraints,
        n_initial_samples=5,   # Fewer initial samples needed
        n_bo_iterations=30,  # Focused run with fewer iterations
        batch_size=1,
        user_initial_points=guiding_point,
        num_restarts=20,       # More intense local search
        raw_samples=1024
    )
    
    # Print a detailed summary of this new problem
    print("\n" + "="*60)
    print(" " * 12 + "Phase 2: Local Refinement Problem Summary")
    print("="*60)
    print(f"Problem Definition File: {__name__}")
    
    print("\n----- Objectives & Constraints -----")
    print(f"  - Objective: {objectives[0].name}")
    print(f"  - Constraint: {constraints[0].description}")
        
    print("\n----- Search Center Point -----")
    print(f"  - Values (first 3): {np.round(guiding_point.numpy().flatten()[:3], 4)}...")

    print("\n----- TIGHT Offline Variable Bounds -----")
    for i in range(problem_config.offline_dim):
        lb = offline_bounds[0, i].item()
        ub = offline_bounds[1, i].item()
        print(f"  - Var {i:<2} (ratio[{i}]):  [{lb:8.4f}, {ub:8.4f}]")

    print("\n----- Run Settings -----")
    print(f"Total Evaluations: {problem_config.n_initial_samples} (initial) + {problem_config.n_bo_iterations} (BO)")
    print("="*60 + "\n")

    return problem_config
# ```

### Your Next Step

# You can now launch this focused, Phase 2 optimization run with the following command:

# ```bash
# python main.py --problem src.problem_def_local_refinement
