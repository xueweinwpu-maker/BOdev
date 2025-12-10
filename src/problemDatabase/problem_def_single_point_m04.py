# -*- coding: utf-8 -*-
"""
Problem definition for a single-point, single-objective optimization case.
This version is updated to use the final, more sophisticated architecture
where local constraints are handled by the inner optimization loop.
"""
import torch
import pandas as pd
from typing import List
import numpy as np

from src.problem_structures import (
    OptimizationConfig, DesignPoint, Objective, GlobalConstraint, InnerOptConfig, InnerConstraint
)
from surrogates.pipelines import SurrogateBackedPipeline

def get_offline_bounds_from_data(data_paths: List[str], offline_dim: int, device, dtype):
    """
    Reads all training data files to determine the min/max bounds for each
    offline variable, creating a precise search space.
    """
    print("Deriving offline variable bounds from the provided data files...")
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
        raise ValueError("No data files found to determine offline bounds.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    var_names = [f'ratio[{i}]' for i in range(offline_dim)]
    
    lower_bounds = full_df[var_names].min().values
    upper_bounds = full_df[var_names].max().values
    
    print("Offline bounds successfully derived.")
    return torch.tensor([lower_bounds, upper_bounds], dtype=dtype, device=device)

def get_guiding_point(data_path: str, offline_dim: int, bounds: torch.Tensor):
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
    for i in range(offline_dim):
        var_name = f'ratio[{i}]'
        bounded_df = bounded_df[
            (bounded_df[var_name] >= lower_bounds[i]) & 
            (bounded_df[var_name] <= upper_bounds[i])
        ]

    if bounded_df.empty:
        print("Warning: No points in the dataset fall within the specified offline_bounds.")
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
    
    offline_vars = best_point_series[[f'ratio[{i}]' for i in range(offline_dim)]].values
    
    print(f"Selected a guiding point with true CL = {cl[feasible_mask_tensor][best_local_idx]:.4f} and true K = {k[feasible_mask_tensor][best_local_idx]:.4f}")
    return torch.tensor(offline_vars, dtype=torch.double).unsqueeze(0)


def define_problem() -> OptimizationConfig:
    """
    Defines and assembles the complete optimization problem configuration.
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
                bounds=[(-50.0, -20.0)], 
                x0=[-35.0],
                # ** REFACTOR: The local constraint is now correctly defined here **
                constraints=[
                    InnerConstraint(
                        name="K_min_subsonic",
                        description="K @ Ma 0.4 >= 3.95",
                        eval_function=lambda res: res["K"] - 3.95
                    )
                ]
            )
        ),
    ]

    objectives = [
        Objective(
            name="Max CL (Ma 0.4)", 
            eval_function=lambda res_map: res_map["Subsonic_Ma0.4"]["CL"]
        ),
    ]

    # ** REFACTOR: No global constraints are needed for this single-point problem **
    constraints = []

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.double
    
    offline_bounds = get_offline_bounds_from_data(
        data_paths=data_files,
        offline_dim=15,
        device=DEVICE,
        dtype=DTYPE
    )
    
    guiding_point = get_guiding_point(
        "data/train_dataM04.csv", 
        offline_dim=15,
        bounds=offline_bounds
    )

    problem_config = OptimizationConfig(
        offline_dim=15,
        online_dim=1,
        offline_bounds=offline_bounds,
        design_points=design_points,
        objectives=objectives,
        constraints=constraints,
        n_initial_samples=10,
        n_bo_iterations=100,
        batch_size=1,
        user_initial_points=guiding_point,
        num_restarts=20,
        raw_samples=1024
    )
    
    # Print a detailed summary of the problem configuration
    print("\n" + "="*60)
    print(" " * 15 + "Optimization Problem Summary")
    print("="*60)
    print(f"Problem Definition File: {__name__}")
    
    print("\n----- Objectives -----")
    for obj in objectives:
        print(f"  - {obj.name}")
        
    print("\n----- Global Constraints -----")
    if not constraints:
        print("  - None")
    else:
        for con in constraints:
            print(f"  - {con.name}: {con.description}")
        
    print("\n----- Design Points & Inner-Loop Constraints -----")
    for dp in design_points:
        print(f"  - {dp.name}:")
        print(f"    - Online Variable Bounds: {dp.inner_opt_config.bounds}")
        print(f"    - Inner Loop Objective:   Maximize '{dp.pipeline.inner_loop_objective}'")
        if dp.inner_opt_config.constraints:
            for inner_con in dp.inner_opt_config.constraints:
                print(f"    - Inner Loop Constraint:  {inner_con.description}")

    if guiding_point is not None:
        print("\n----- Guiding Point -----")
        print("  - An initial point will be used from the dataset.")
        print(f"  - Values (first 3): {np.round(guiding_point.numpy().flatten()[:3], 4)}...")
    else:
        print("\n----- Guiding Point -----")
        print("  - No guiding point found. Starting with a fully random sample.")

    print("\n----- Offline Variable Bounds (Data-Driven) -----")
    for i in range(problem_config.offline_dim):
        lb = offline_bounds[0, i].item()
        ub = offline_bounds[1, i].item()
        print(f"  - Var {i:<2} (ratio[{i}]):  [{lb:8.4f}, {ub:8.4f}]")

    print("\n----- Run Settings -----")
    print(f"Total Evaluations: {problem_config.n_initial_samples} (initial) + {problem_config.n_bo_iterations} (BO)")
    print(f"Acqf Restarts: {problem_config.num_restarts}, Raw Samples: {problem_config.raw_samples}")
    print("="*60 + "\n")

    return problem_config

