# -*- coding: utf-8 -*-
"""
Problem definition for a single-point optimization of the 30 DV case.
Objective: Maximize 'volume'
Constraint: Lift-to-drag ratio 'K' must not be less than the baseline (all DVs=0).
"""
import torch
import pickle
import numpy as np
import os
import pandas as pd

from src.problem_structures import (
    OptimizationConfig, DesignPoint, Objective, GlobalConstraint
)
from surrogates.pipelines import SurrogateBackedPipeline

def define_problem() -> OptimizationConfig:
    """Defines the 30 DV optimization problem."""
    print("--- Defining 30 DV Optimization Problem ---")
    
    model_name = "30_DV_Model"
    model_path = f"trained_models/pipeline_{model_name}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find pre-trained model at '{model_path}'.\n"
            "Please run 'tools/run_validation.py' with this run_name and 'save_model: True' first."
        )
        
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)

    # --- Get baseline performance (all DVs at zero) for the constraint ---
    baseline_dvs = np.zeros(30)
    baseline_results = pipeline.run(baseline_dvs, None)
    # baseline_k = baseline_results.get("K", 0.0)
    baseline_k = 4.53 #keep constant to match the 38 DV case
    print(f"Baseline (all DVs=0) Lift-to-Drag Ratio (K): {baseline_k:.4f}")

    # --- Define the single Design Point ---
    design_points = [
        DesignPoint(name=model_name, pipeline=pipeline)
    ]

    # --- Define Objective and Constraints ---
    objectives = [
        Objective(
            name="Maximize Volume",
            eval_function=lambda res_map: res_map[model_name]["Volume"]
        )
    ]
    
    constraints = [
        GlobalConstraint(
            name="Maintain L/D Ratio",
            description=f"K >= {baseline_k:.4f}",
            eval_function=lambda res_map: res_map[model_name]["K"] - baseline_k*0.99  # Slightly relaxed to avoid numerical issues
        )
    ]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.double
    
    # --- MODIFICATION: Set bounds from the surrogate's training data ---
    print("Setting optimization bounds from surrogate's training data...")
    train_data_path = pipeline.train_paths[0]
    train_df = pd.read_csv(train_data_path)
    # Ensure column names match the format used by the pipeline
    if "ratio[0]" in train_df.columns:
        rename_dict = {f'ratio[{i}]': f'dv_{i+1}' for i in range(30)}
        train_df.rename(columns=rename_dict, inplace=True)
    
    lower_bounds = train_df[pipeline.input_cols].min().values
    upper_bounds = train_df[pipeline.input_cols].max().values
    offline_bounds = torch.tensor([lower_bounds, upper_bounds], dtype=DTYPE, device=DEVICE)
    print("Bounds set successfully.")
    
    user_initial_point = torch.zeros(1, 30, device=DEVICE, dtype=DTYPE)
    
    problem_config = OptimizationConfig(
        offline_dim=30,
        online_dim=0, # No online variables in this case
        offline_bounds=offline_bounds,
        design_points=design_points,
        objectives=objectives,
        constraints=constraints,
        n_initial_samples=41,  # At least as many as dimensions
        n_bo_iterations=320,
        batch_size=1,
        user_initial_points=user_initial_point,
        num_restarts=10,
        raw_samples=512
    )
    
    return problem_config

# python main.py --problem src.problem_def_30dvs 