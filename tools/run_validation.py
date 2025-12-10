# -*- coding: utf-8 -*-
"""
A general-purpose, configurable script to train and validate surrogate models
for any number of design variables.
MODIFIED: Saves all outputs to a centralized 'validation_results' directory.
"""
import pandas as pd
import torch
import os
import sys
import pickle
from datetime import datetime
from typing import List

# Add project root to path to allow for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surrogates.model import GPyTorchSurrogateModel
from surrogates.pipelines import SurrogateBackedPipeline

def run_validation(train_file: str, valid_file: str, num_dvs: int, outputs: List[str], dv_format: str, run_name: str, save_model: bool):
    """
    The core logic for training and validating a surrogate model.
    """
    print(f"--- Starting Configurable Validation for '{run_name}' ---")

    # 1. Generate Input Column Names
    if dv_format == "underscore":
        input_cols = [f'dv_{i}' for i in range(num_dvs)]
    elif dv_format == "bracket":
        input_cols = [f'ratio[{i}]' for i in range(num_dvs)]
    else:
        raise ValueError(f"Invalid dv_format specified: {dv_format}")

    # 2. Create a timestamped output directory inside 'validation_results'
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join('validation_results', f"{run_name}_{timestamp}")
    # Model_dir = os.path.join('trained_models')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving all outputs to: {output_dir}")

    # 3. Load Data
    print(f"Loading training data from: {train_file}")
    # handle possible BOM and trailing spaces in headers
    train_df = pd.read_csv(train_file, encoding='utf-8-sig')
    train_df.columns = train_df.columns.str.strip()
    print("train columns:", train_df.columns.tolist())
    train_x_df = train_df[input_cols]        
    print(f"Loading validation data from: {valid_file}")
    valid_df = pd.read_csv(valid_file, encoding='utf-8-sig')
    valid_df.columns = valid_df.columns.str.strip()
    print("valid columns:", valid_df.columns.tolist())
    valid_x_df = valid_df[input_cols]
    
    trained_models = {}

    # 4. Train and Validate a model for each output
    for coeff_name in outputs:
        print(f"\n--- Processing model for '{coeff_name}' ---")
        
        # tolerant access: strip whitespace from requested name and try case-insensitive match
        coeff_clean = coeff_name.strip()
        if coeff_clean not in train_df.columns:
            # try case-insensitive match
            cols_lower = {c.lower(): c for c in train_df.columns}
            if coeff_clean.lower() in cols_lower:
                coeff_clean = cols_lower[coeff_clean.lower()]
            else:
                raise KeyError(f"Output column '{coeff_name}' not found. Available: {train_df.columns.tolist()}")
        train_y_df = train_df[[coeff_clean]]

        valid_y_df = valid_df[[coeff_name]]
        
        train_x_torch = torch.tensor(train_x_df.values, dtype=torch.double)
        train_y_torch = torch.tensor(train_y_df.values, dtype=torch.double)
        valid_x_torch = torch.tensor(valid_x_df.values, dtype=torch.double)
        
        model = GPyTorchSurrogateModel(train_x_torch, train_y_torch)
        model.train_model()
        
        print(f"--- Validating model for '{coeff_name}' ---")
        metrics = model.validate(valid_x_torch, valid_y_df.values)
        trained_models[coeff_name] = model

        print(f"--- Validation Metrics for {coeff_name} ({run_name}) ---")
        print(f"R-squared (R2):             {metrics['R2']:.6f}")
        print(f"Root Mean Squared Error (RMSE):   {metrics['RMSE']:.6f}")
        print(f"Mean Absolute Error (MAE):        {metrics['MAE']:.6f}")
        print(f"Maximum Absolute Error (MaxAE):   {metrics['MaxAE']:.6f}\n")
        
        with open(os.path.join(output_dir, f'metrics_{coeff_name}.txt'), 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

        model.plot_validation(name=f"{coeff_name}_{run_name}", valid_x=valid_x_torch, valid_y=valid_y_df.values, output_dir=output_dir)
        model.plot_ard_relevance(name=f"{coeff_name}_{run_name}", var_names=input_cols, output_dir=output_dir)

    # 5. Save the final pipeline object if requested
    if save_model:
        print("\n--- Re-training models on combined dataset and saving final pipeline ---")
        combined_df = pd.concat([train_df, valid_df], ignore_index=True)
        combined_x_df = combined_df[input_cols]
        final_models = {}

        for coeff_name in outputs:
            print(f"Re-training final model for '{coeff_name}'...")
            combined_y_df = combined_df[[coeff_name]]
            combined_x_torch = torch.tensor(combined_x_df.values, dtype=torch.double)
            combined_y_torch = torch.tensor(combined_y_df.values, dtype=torch.double)
            
            final_model = GPyTorchSurrogateModel(combined_x_torch, combined_y_torch)
            final_model.train_model()
            final_models[coeff_name] = final_model
        
        pipeline = SurrogateBackedPipeline(
            train_paths=[train_file, valid_file],
            input_cols=input_cols,
            output_cols=outputs,
            train_on_init=False
        )
        pipeline.models = final_models

        pipeline_path = os.path.join('trained_models', f"{run_name}.pkl")
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"{run_name}.pkl saved successfully to: {pipeline_path}")

    print("\n--- Validation Run Finished ---")


if __name__ == '__main__':
    # ===========================================
    # CONFIGURATION
    # ===========================================
    CONFIG = {
        "train_file": "data/train_noWDA_Ma5_AoA4.csv",
        "valid_file": "data/valid_noWDA_Ma5_AoA4.csv",
        "num_dvs": 38,
        "outputs": ["cl", "cd", "Volume"],
        # "train_file": "data/train_dataM04.csv",
        # "valid_file": "data/verify_dataM04.csv",
        # "num_dvs": 16,
        # "outputs": [ "CY"],
        "dv_format": "bracket",
        "run_name": "38_DV_Model_Ma04_ARD",
        "save_model": True
    }

    run_validation(**CONFIG)

