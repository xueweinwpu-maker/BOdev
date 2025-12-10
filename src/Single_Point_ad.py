# -*- coding: utf-8 -*-
"""
Problem definition for a SINGLE-POINT, BI-LEVEL optimization.

This file serves as an example of the framework's flexibility. It defines a
problem with only one design point (Ma=6.0) and a simple objective: to
maximize the lift-to-drag ratio (K) at that point.

The bi-level structure is still used, with the inner loop optimizing the
sweep angle and the outer loop optimizing the 15 offline shape variables.
"""
import torch
import pickle
from typing import List
import numpy as np

from src.problem_structures import (
    OptimizationConfig, DesignPoint, Objective, InnerOptConfig, InnerConstraint
)
from surrogates.pipelines import SurrogateBackedPipeline

def define_problem() -> OptimizationConfig:
    """Defines a single-point problem to maximize K at Ma=6.0."""
    print("--- Defining Single-Point (Ma=6.0) Bi-Level Problem ---")
    
    # --- Load the required pre-trained surrogate model ---
    with open("trained_models/pipeline_ma6.0.pkl", 'rb') as f:
        supersonic_pipeline = pickle.load(f)
    
    # Re-configure the loaded pipeline with the correct inner-loop objective
    supersonic_pipeline.inner_loop_objective = "K"
    
    # --- Performance values from paper for constraints ---
    CL_MA6_ORI, XCP_MA6_ORI = 0.08886, 0.7197

    # --- Define the single Design Point ---
    design_points = [
        DesignPoint(
            name="Ma_6.0", 
            pipeline=supersonic_pipeline,
            inner_opt_config=InnerOptConfig(
                bounds=[(0.0, 20.0)], 
                x0=[0.0], 
                maxiter=15,
                constraints=[
                    InnerConstraint(name="CL_min_supersonic", description=f"CL >= {0.9 * CL_MA6_ORI:.4f}",
                                    eval_function=lambda res: res["CL"] - (0.9 * CL_MA6_ORI)),
                    InnerConstraint(name="Xcp_Stability_Ma6", description=f"|Xcp - {XCP_MA6_ORI}| <= 0.02",
                                    eval_function=lambda res: 0.02 - abs(res["Xcp"] - XCP_MA6_ORI))
                ]
            )
        )
    ]

    # --- Simple Maximization Objective (Outer Loop) ---
    # The objective is simply the final K value from the single design point's results.
    objectives = [
        Objective(
            name="Maximize Ma=6.0 K",
            eval_function=lambda res_map: res_map["Ma_6.0"]["K"]
        )
    ]
    
    # --- No global constraints in this simple example ---
    constraints = []

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.double
    
    # Define bounds for the 15 offline shape variables
    # For simplicity, we define them manually here. A real run might load them from data.
    lower_bounds = [-0.2] * 6 + [-0.12] * 3 + [-0.08] * 3 + [-0.06] * 3
    upper_bounds = [0.05] * 6 + [0.02] * 9
    offline_bounds = torch.tensor([lower_bounds, upper_bounds], dtype=DTYPE, device=DEVICE)
    
    # A single, all-zero vector as the initial starting point
    user_initial_point = torch.zeros(1, 15, device=DEVICE, dtype=DTYPE)
    
    problem_config = OptimizationConfig(
        offline_dim=15,
        online_dim=1,
        offline_bounds=offline_bounds,
        design_points=design_points,
        objectives=objectives,
        constraints=constraints,
        n_initial_samples=17,
        n_bo_iterations=85,
        batch_size=1,
        user_initial_points=user_initial_point,
        num_restarts=10,
        raw_samples=256
    )
    
    return problem_config
