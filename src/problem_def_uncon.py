# src/problem_def_uncon.py
import torch
import pandas as pd
import numpy as np
import os
import sys

# Parallel structure imports
from src.problem_structures import OptimizationConfig, DesignPoint, Objective
from surrogates.pipelines import TruthPipeline

def define_problem(use_mtgp: bool = False) -> OptimizationConfig:
    print(f"--- Defining Ma6 + Ma10 Benchmark (MTGP: {use_mtgp}) ---")
    
    # 1. Initialize Truth Pipelines
    pipeline_ma6 = TruthPipeline(pipeline_name="Ma6_Truth", fixed_ratio_15=1.0)
    pipeline_ma10 = TruthPipeline(pipeline_name="Ma10_Truth", fixed_ratio_15=1.0)
    
    dp_ma6 = DesignPoint(name="Ma6", pipeline=pipeline_ma6)
    dp_ma10 = DesignPoint(name="Ma10", pipeline=pipeline_ma10)

    # 2. Dynamic Bounds Setting
    print("Setting bounds from Ma10 training data...")
    internal_pipe = pipeline_ma10.internal_pipeline
    train_path = internal_pipe.train_paths[0] if internal_pipe.train_paths else "data/train_dataM10_0_extended.csv"
    if not os.path.exists(train_path):
        train_path = "data/train_dataM10_0_extended.csv"
        
    df = pd.read_csv(train_path)
    df.columns = df.columns.str.strip()
    input_cols = [f'ratio[{i}]' for i in range(15)]
    
    lower = df[input_cols].min().values
    upper = df[input_cols].max().values
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bounds = torch.tensor([lower, upper], dtype=torch.double, device=DEVICE)

    # 3. Two separate objectives for MTGP to correlate
    objectives = [
        Objective(name="Ma6_K", eval_function=lambda r: r["Ma6"]["K"]),
        Objective(name="Ma10_K", eval_function=lambda r: r["Ma10"]["K"])
    ]
    
    user_initial_point = torch.zeros(1, 15, device=DEVICE, dtype=torch.double)
    
    return OptimizationConfig(
        offline_dim=15, 
        online_dim=0,
        offline_bounds=bounds,
        design_points=[dp_ma6, dp_ma10],
        objectives=objectives,
        constraints=[], 
        user_initial_points=user_initial_point,
        n_initial_samples=10, 
        n_bo_iterations=30,
        batch_size=1,
        
        # KEY FLAGS
        model_type="mtgp" if use_mtgp else "standard",
        mtgp_rank=1
    )