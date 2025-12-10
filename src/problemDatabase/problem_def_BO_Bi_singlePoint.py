# Bi-Level Optimization (Loads Pre-Trained Models):src/problem_def_BO_Bi_singlePoint.py
# -*- coding: utf-8 -*-
"""
Problem definition for a BI-LEVEL optimization that accurately replicates the
weighted-sum, single-objective problem from the original Chinese paper.
This version is updated to load pre-trained surrogate models from .pkl files.
"""
import torch
import pandas as pd
import numpy as np
import pickle
import os
from typing import List

from src.problem_structures import (
    OptimizationConfig, DesignPoint, Objective, InnerOptConfig, InnerConstraint
)
from surrogates.pipelines import SurrogateBackedPipeline

def get_offline_bounds_from_data(data_paths: List[str], offline_dim: int, device, dtype):
    """Derives offline variable bounds from data."""
    print("Deriving offline variable bounds from data files...")
    all_dfs = [pd.read_csv(p) for p in data_paths]
    full_df = pd.concat(all_dfs, ignore_index=True)
    var_names = [f'ratio[{i}]' for i in range(offline_dim)]
    lower_bounds = full_df[var_names].min().values
    upper_bounds = full_df[var_names].max().values
    
    bounds_np = np.array([lower_bounds, upper_bounds])
    return torch.tensor(bounds_np, dtype=dtype, device=device)

def define_problem() -> OptimizationConfig:
    """Defines the bi-level version of the Chinese paper's problem."""
    low_speed_data = ["data/train_dataM04.csv", "data/verify_dataM04.csv"]
    supersonic_data = ["data/train_dataM6.csv", "data/verify_dataM6.csv"]
    high_speed_data = ["data/train_dataM10.csv", "data/verify_dataM10.csv"]

    # --- Load Pre-trained Surrogate Models ---
    print("Loading pre-trained surrogate models...")
    model_dir = "trained_models"
    try:
        with open(os.path.join(model_dir, "pipeline_ma0.4.pkl"), 'rb') as f:
            low_speed_pipeline = pickle.load(f)
        with open(os.path.join(model_dir, "pipeline_ma6.0.pkl"), 'rb') as f: # Ma 6.0 uses Ma 3.0 model
            supersonic_pipeline = pickle.load(f)
        with open(os.path.join(model_dir, "pipeline_ma10.0.pkl"), 'rb') as f:
            high_speed_pipeline = pickle.load(f)
        print("Surrogate models loaded successfully.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find pre-trained model file: {e}. Please run train_and_save_surrogates.py first.") from e

    # --- FIX: Re-configure loaded pipelines with the correct inner-loop objectives ---
    # This is crucial because the saved .pkl files were trained without this specific setting.
    low_speed_pipeline.inner_loop_objective = "CL"
    supersonic_pipeline.inner_loop_objective = "K"
    high_speed_pipeline.inner_loop_objective = "K"
    print("Loaded pipelines re-configured for inner-loop optimization.")

    # --- Original Performance Values from Paper ---
    CL_MA04_ORI, K_MA04_ORI = 0.3780, 4.8323
    # K_MA6_ORI, CL_MA6_ORI, XCP_MA6_ORI = 3.945928, 0.123825, 0.703858
    # K_MA10_ORI, CL_MA10_ORI, XCP_MA10_ORI = 3.9058, 0.0989, 0.7033

    # --- Design Points with Inner-Loop Constraints ---
    design_points = [
        DesignPoint(
            name="Ma_0.4", pipeline=low_speed_pipeline,
            inner_opt_config=InnerOptConfig(
                bounds=[(-50.0, -20.0)], x0=[-20.0], maxiter=15,
                constraints=[InnerConstraint(name="K_min_subsonic", description=f"K >= {K_MA04_ORI}",
                                             eval_function=lambda res: res["K"] - K_MA04_ORI)]
            )
        ),
        # DesignPoint(
        #     name="Ma_6.0", pipeline=supersonic_pipeline,
        #     inner_opt_config=InnerOptConfig(
        #         bounds=[(0.0, 20.0)], x0=[0.0], maxiter=15,
        #         constraints=[
        #             InnerConstraint(name="CL_min_supersonic", description=f"CL >= {0.9 * CL_MA6_ORI:.4f}",
        #                             eval_function=lambda res: res["CL"] - (0.9 * CL_MA6_ORI)),
        #             InnerConstraint(name="Xcp_Stability_Ma6", description=f"Xcp - {XCP_MA6_ORI} <= 0.02",
        #                             eval_function=lambda res: 0.02 - (res["Xcp"] - XCP_MA6_ORI))
        #         ]
        #     )
        # ),
        # DesignPoint(
        #     name="Ma_10.0", pipeline=high_speed_pipeline,
        #     inner_opt_config=InnerOptConfig(
        #         bounds=[(10.0, 25.0)], x0=[10.0], maxiter=15,
        #         constraints=[
        #             InnerConstraint(name="CL_min_hypersonic", description=f"CL >= {0.9 * CL_MA10_ORI:.4f}",
        #                             eval_function=lambda res: res["CL"] - (0.9 * CL_MA10_ORI)),
        #             InnerConstraint(name="Xcp_Stability_Ma10", description=f"|Xcp - {XCP_MA10_ORI}| <= 0.02",
        #                             eval_function=lambda res: 0.02 - abs(res["Xcp"] - XCP_MA10_ORI))
        #         ]
        #     )
        # ),
    ]

    # --- Weighted-Sum Objective (Outer Loop) ---
    def weighted_sum_objective(res_map):
        cl_ma04 = res_map["Ma_0.4"]["CL"]
        # k_ma6 = res_map["Ma_6.0"]["K"]
        # k_ma10 = res_map["Ma_10.0"]["K"]
        # return 0.333 * (cl_ma04 / CL_MA04_ORI) + 0.667 * (k_ma6 / K_MA6_ORI) 
        return 1.0 * (cl_ma04 / CL_MA04_ORI) 
    objectives = [Objective(name="Chinese Paper Weighted Sum", eval_function=weighted_sum_objective)]

    # --- The Outer Loop has no global constraints in this setup ---
    constraints = []

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.double
    
    offline_bounds = get_offline_bounds_from_data(
        data_paths=low_speed_data + supersonic_data + high_speed_data,
        offline_dim=15, device=DEVICE, dtype=DTYPE
    )
    
    user_initial_point = torch.zeros(1, 15, device=DEVICE, dtype=DTYPE)
    
    problem_config = OptimizationConfig(
        offline_dim=15, online_dim=1,
        offline_bounds=offline_bounds,
        design_points=design_points,
        objectives=objectives,
        constraints=constraints,
        n_initial_samples=17,
        n_bo_iterations=80,
        batch_size=1,
        user_initial_points=user_initial_point,
        num_restarts=10,
        raw_samples=256
    )
    
    return problem_config

