# -*- coding: utf-8 -*-
"""
Problem definition for a single-point, single-objective optimization case.
This version is refactored into a more direct, single-level optimization,
where all 16 design variables (shape + sweep) are optimized simultaneously.
"""
import torch
import pandas as pd
from typing import List
import numpy as np

from src.problem_structures import (
    OptimizationConfig, DesignPoint, Objective, GlobalConstraint
)
from surrogates.pipelines import SurrogateBackedPipeline

def get_full_16d_bounds_from_data(data_paths: List[str], device, dtype):
    """
    Reads all training data files to determine the min/max bounds for all 16
    variables (15 offline + 1 online), creating a precise search space.
    """
    print("Deriving full 16-variable bounds from the provided data files...")
    all_dfs = []
    for path in data_paths:
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df.dropna(inplace=True)
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: Could not find data file at {path} for bounds calculation.")

    if not all_dfs:
        raise ValueError("No data files found to determine bounds.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    # Get bounds for all 16 variables
    var_names = [f'ratio[{i}]' for i in range(16)]
    
    lower_bounds = full_df[var_names].min().values
    upper_bounds = full_df[var_names].max().values
    
    print("Full 16-variable bounds successfully derived.")
    return torch.tensor([lower_bounds, upper_bounds], dtype=dtype, device=device)

def get_guiding_point(data_path: str, bounds: torch.Tensor):
    """
    Finds a good starting point from the dataset that is both feasible
    and respects the optimization bounds.
    """
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    
    lower_bounds = bounds[0].cpu().numpy()
    upper_bounds = bounds[1].cpu().numpy()
    
    bounded_df = df.copy()
    for i in range(16): # Check all 16 variables against the bounds
        var_name = f'ratio[{i}]'
        bounded_df = bounded_df[
            (bounded_df[var_name] >= lower_bounds[i]) & 
            (bounded_df[var_name] <= upper_bounds[i])
        ]

    if bounded_df.empty:
        print("Warning: No points in the dataset fall within the specified bounds.")
        return None

    alpha = torch.tensor(4.0 * torch.pi / 180.0, dtype=torch.double)
    
    cy_tensor = torch.tensor(bounded_df['CY'].values, dtype=torch.double)
    cx_tensor = torch.tensor(bounded_df['Cx'].values, dtype=torch.double)

    cl = cy_tensor * torch.cos(alpha) - cx_tensor * torch.sin(alpha)
    cd = cy_tensor * torch.sin(alpha) + cx_tensor * torch.cos(alpha)
    k = cl / cd
    
    feasible_mask_tensor = k > 4.0
    
    if not feasible_mask_tensor.any():
        print("Warning: No truly feasible (K > 4.0) guiding points found.")
        return None
        
    feasible_cl = cl[feasible_mask_tensor]
    best_local_idx = torch.argmax(feasible_cl)
    
    feasible_mask_numpy = feasible_mask_tensor.cpu().numpy()
    feasible_indices = bounded_df.index[feasible_mask_numpy]
    
    original_idx = feasible_indices[best_local_idx.item()]
    best_point_series = df.loc[original_idx]
    
    # Get all 16 variables for the guiding point
    all_vars = best_point_series[[f'ratio[{i}]' for i in range(16)]].values
    
    print(f"Selected a guiding point with true CL = {cl[feasible_mask_tensor][best_local_idx]:.4f} and true K = {k[feasible_mask_tensor][best_local_idx]:.4f}")
    return torch.tensor(all_vars, dtype=torch.double).unsqueeze(0)


def define_problem() -> OptimizationConfig:
    """
    Defines and assembles the complete single-level optimization problem.
    """
    data_files = ["data/train_dataM04.csv", "data/verify_dataM04.csv"]

    # The pipeline is now a simple evaluator, no inner-loop objective needed
    low_speed_pipeline = SurrogateBackedPipeline(
        mach_str="0.4",
        train_paths=data_files,
    )

    # In a single-level problem, we only have one "DesignPoint" which is the problem itself
    design_points = [
        DesignPoint(
            name="Subsonic_Ma0.4",
            pipeline=low_speed_pipeline,
            # No inner_opt_config is needed for a single-level problem
            inner_opt_config=None 
        ),
    ]

    objectives = [
        Objective(
            name="Max CL (Ma 0.4)", 
            eval_function=lambda res_map: res_map["Subsonic_Ma0.4"]["CL"]
        ),
    ]

    # The constraint is now a global constraint for the single optimizer
    constraints = [
        GlobalConstraint(
            name="K_min_subsonic",
            description="K @ Ma 0.4 >= 4.8",
            eval_function=lambda res_map: res_map["Subsonic_Ma0.4"]["K"] - 4.8
        )
    ]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.double
    
    # Get bounds for the full 16-dimensional space
    full_bounds = get_full_16d_bounds_from_data(
        data_paths=data_files,
        device=DEVICE,
        dtype=DTYPE
    )
    
    guiding_point = get_guiding_point(
        "data/train_dataM04.csv", 
        bounds=full_bounds
    )

    problem_config = OptimizationConfig(
        # The problem is now a single 16-dimensional problem
        offline_dim=16,
        online_dim=0,
        offline_bounds=full_bounds,
        design_points=design_points,
        objectives=objectives,
        constraints=constraints,
        n_initial_samples=17,
        n_bo_iterations=100,
        batch_size=1,
        user_initial_points=guiding_point,
        num_restarts=10,
        raw_samples=500
    )
    
    # ... (Summary printout would be updated to reflect the new structure) ...
    print("\n" + "="*60)
    print(" " * 15 + "Single-Level Optimization Problem Summary")
    print("="*60)
    # ...
    return problem_config