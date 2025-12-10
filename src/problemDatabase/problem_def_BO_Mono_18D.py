# Monolithic Optimization Baseline (18-Dimensional):src/problem_def_BO_Mono_18D.py
# -*- coding: utf-8 -*-
"""
Problem definition for a MONOLITHIC (single-level) optimization.

This script sets up an 18-dimensional problem to serve as a direct baseline
for comparison against the bi-level optimization approach. The 18 design
variables consist of:
- 15 offline shape variables
- 3 independent 'online' sweep angle variables (one for each Mach number)

**MODIFIED**: This version loads pre-trained surrogate models from the
'trained_models/' directory to ensure a fair comparison.
"""
import torch
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np
import pickle
import os
import sys

from src.problem_structures import (
    OptimizationConfig, DesignPoint, Objective, GlobalConstraint
)
from src.base_pipeline import AeroPipeline
from surrogates.pipelines import SurrogateBackedPipeline

# Helper functions (copied from bi-level definition for consistency)
def get_offline_bounds_from_data(data_paths: List[str], offline_dim: int, device, dtype):
    """Derives offline variable bounds from data."""
    print("Deriving offline variable bounds from data files...")
    all_dfs = [pd.read_csv(p) for p in data_paths]
    full_df = pd.concat(all_dfs, ignore_index=True)
    var_names = [f'ratio[{i}]' for i in range(offline_dim)]
    lower_bounds = full_df[var_names].min().values
    upper_bounds = full_df[var_names].max().values
    return torch.tensor([lower_bounds, upper_bounds], dtype=dtype, device=device)

def load_surrogate_pipeline(model_path: str) -> SurrogateBackedPipeline:
    """Loads a pre-trained surrogate pipeline from a pickle file."""
    if not os.path.exists(model_path):
        print(f"FATAL: Could not find the pre-trained model file at '{model_path}'.")
        print("Please run 'python train_and_save_surrogates.py' first to generate the models.")
        sys.exit(1)
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"Successfully loaded pre-trained surrogate from: {model_path}")
        return pipeline
    except Exception as e:
        print(f"FATAL: Error loading surrogate model from {model_path}: {e}")
        sys.exit(1)

# Custom pipeline to handle the 18D -> 16D variable slicing
class SlicerPipeline(AeroPipeline):
    """
    A wrapper pipeline that takes an 18D vector, slices it, and passes the
    correct 16D subset (15 shape + 1 sweep) to an underlying surrogate pipeline.
    This is the key component that enables the monolithic optimization.
    """
    def __init__(self, underlying_pipeline: SurrogateBackedPipeline, sweep_var_index: int):
        self.underlying_pipeline = underlying_pipeline
        self.sweep_var_index = sweep_var_index # Index (0, 1, or 2) for which sweep variable to use

    def run(self, offline_vars: np.ndarray, online_vars: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        In the monolithic setup, the BO loop passes the full 18D vector as
        'offline_vars' and 'online_vars' is None.
        """
        if offline_vars.shape[0] != 18:
            raise ValueError(f"SlicerPipeline expects 18 variables, but received {offline_vars.shape[0]}")

        shape_vars = offline_vars[:15]
        # Select the correct sweep variable from the last 3 elements of the input vector
        sweep_var = np.array([offline_vars[15 + self.sweep_var_index]])

        # The underlying SurrogateBackedPipeline expects offline (shape) and online (sweep) vars separately
        return self.underlying_pipeline.run(shape_vars, sweep_var)


def define_problem() -> OptimizationConfig:
    """Defines the 18-dimensional monolithic version of the Chinese paper's problem."""
    # --- MODIFICATION: Load pre-trained models instead of training ---
    model_dir = "trained_models"
    low_speed_pipeline = load_surrogate_pipeline(os.path.join(model_dir, "pipeline_ma0.4.pkl"))
    supersonic_pipeline = load_surrogate_pipeline(os.path.join(model_dir, "pipeline_ma3.0.pkl"))
    high_speed_pipeline = load_surrogate_pipeline(os.path.join(model_dir, "pipeline_ma10.0.pkl"))

    # --- Original Performance Values from Paper (for objective and constraints) ---
    CL_MA04_ORI, K_MA04_ORI = 0.3780, 4.8323
    K_MA6_ORI, CL_MA6_ORI, XCP_MA6_ORI = 3.58, 0.08886, 0.7197
    K_MA10_ORI, CL_MA10_ORI, XCP_MA10_ORI = 3.9058, 0.0989, 0.7033

    # --- Design Points (without inner loops) ---
    design_points = [
        DesignPoint(
            name="Ma_0.4",
            pipeline=SlicerPipeline(low_speed_pipeline, sweep_var_index=0),
            inner_opt_config=None
        ),
        DesignPoint(
            name="Ma_6.0",
            pipeline=SlicerPipeline(supersonic_pipeline, sweep_var_index=1),
            inner_opt_config=None
        ),
        DesignPoint(
            name="Ma_10.0",
            pipeline=SlicerPipeline(high_speed_pipeline, sweep_var_index=2),
            inner_opt_config=None
        ),
    ]

    # --- Weighted-Sum Objective (Identical to bi-level problem) ---
    def weighted_sum_objective(res_map):
        cl_ma04 = res_map["Ma_0.4"]["CL"]
        k_ma6 = res_map["Ma_6.0"]["K"]
        k_ma10 = res_map["Ma_10.0"]["K"]
        return 0.2 * (cl_ma04 / CL_MA04_ORI) + 0.4 * (k_ma6 / K_MA6_ORI) + 0.4 * (k_ma10 / K_MA10_ORI)

    objectives = [Objective(name="Chinese Paper Weighted Sum", eval_function=weighted_sum_objective)]

    # --- Global Constraints (Replicating the bi-level problem's inner constraints) ---
    constraints = [
        GlobalConstraint(name="K_min_subsonic", description=f"K >= {K_MA04_ORI}",
                         eval_function=lambda res: res["Ma_0.4"]["K"] - K_MA04_ORI),
        GlobalConstraint(name="CL_min_supersonic", description=f"CL >= {0.9 * CL_MA6_ORI:.4f}",
                         eval_function=lambda res: res["Ma_6.0"]["CL"] - (0.9 * CL_MA6_ORI)),
        GlobalConstraint(name="Xcp_Stability_Ma6", description=f"|Xcp - {XCP_MA6_ORI}| <= 0.02",
                         eval_function=lambda res: 0.02 - abs(res["Ma_6.0"]["Xcp"] - XCP_MA6_ORI)),
        GlobalConstraint(name="CL_min_hypersonic", description=f"CL >= {0.9 * CL_MA10_ORI:.4f}",
                         eval_function=lambda res: res["Ma_10.0"]["CL"] - (0.9 * CL_MA10_ORI)),
        GlobalConstraint(name="Xcp_Stability_Ma10", description=f"|Xcp - {XCP_MA10_ORI}| <= 0.02",
                         eval_function=lambda res: 0.02 - abs(res["Ma_10.0"]["Xcp"] - XCP_MA10_ORI)),
    ]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.double

    # --- Define Bounds for all 18 variables ---
    low_speed_data = ["data/train_dataM04.csv", "data/verify_dataM04.csv"]
    supersonic_data = ["data/train_dataM6.csv", "data/verify_dataM6.csv"]
    high_speed_data = ["data/train_dataM10.csv", "data/verify_dataM10.csv"]
    shape_bounds = get_offline_bounds_from_data(
        data_paths=low_speed_data + supersonic_data + high_speed_data,
        offline_dim=15, device=DEVICE, dtype=DTYPE
    )
    sweep_bounds = torch.tensor([
        [-50.0, 0.0, 10.0],
        [-20.0, 20.0, 25.0]
    ], device=DEVICE, dtype=DTYPE)
    full_bounds = torch.cat([shape_bounds, sweep_bounds], dim=1)

    user_initial_point = torch.zeros(1, 18, device=DEVICE, dtype=DTYPE)
    
    # --- Set total evaluations to match the bi-level run for a fair comparison ---
    n_total_evals = 102
    n_initial = 37 # A reasonable number for an 18D problem (~2*dim + 1)
    n_bo_iter = n_total_evals - n_initial # Will be 65
    
    problem_config = OptimizationConfig(
        offline_dim=18,
        online_dim=0,
        offline_bounds=full_bounds,
        design_points=design_points,
        objectives=objectives,
        constraints=constraints,
        n_initial_samples=n_initial,
        n_bo_iterations=n_bo_iter,
        batch_size=1,
        user_initial_points=user_initial_point,
        num_restarts=10,
        raw_samples=512
    )

    return problem_config
