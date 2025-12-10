# -*- coding: utf-8 -*-
"""
A general-purpose, configurable script to train and validate surrogate models
for any number of design variables.
MODIFIED: Now re-trains the model on the combined training and validation
dataset before saving the final pipeline for improved accuracy. Also queries
and prints the baseline performance after initial validation.
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

    # 1. Generate Input Column Names based on configuration
    if dv_format == "underscore":
        input_cols = [f'dv_{i+1}' for i in range(num_dvs)]
    elif dv_format == "bracket":
        input_cols = [f'ratio[{i}]' for i in range(num_dvs)]
    else:
        raise ValueError(f"Invalid dv_format specified: {dv_format}")

    print(f"Expecting {num_dvs} input variables: {input_cols[0]}, {input_cols[1]}, ...")
    print(f"Modeling {len(outputs)} output variables: {outputs}")

    # 2. Create timestamped directory for results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("validation_results", f"run_{run_name.replace(' ', '_')}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Validation results will be saved to: {output_dir}")

    # 3. Load and Clean Data
    try:
        train_df = pd.read_csv(train_file)
        valid_df = pd.read_csv(valid_file)
        
        train_df.columns = train_df.columns.str.strip()
        valid_df.columns = valid_df.columns.str.strip()
        
        if dv_format == "underscore":
             rename_dict = {f'ratio[{i}]': f'dv_{i+1}' for i in range(num_dvs)}
             train_df.rename(columns=rename_dict, inplace=True)
             valid_df.rename(columns=rename_dict, inplace=True)

        required_cols = input_cols + outputs
        
        train_df.dropna(subset=required_cols, inplace=True)
        valid_df.dropna(subset=required_cols, inplace=True)
        
        print(f"Loaded and cleaned data. Training points: {len(train_df)}, Validation points: {len(valid_df)}")

    except FileNotFoundError as e:
        print(f"FATAL: Could not find data file. {e}")
        return
    except KeyError as e:
        print(f"FATAL: A required column was not found in the CSV files: {e}")
        print("Please check your num_dvs, dv_format, and outputs arguments.")
        return
        
    if train_df.empty or valid_df.empty:
        print("FATAL: No valid data found after cleaning. Aborting.")
        return

    # 4. Prepare Data Tensors for initial validation
    train_x_torch = torch.tensor(train_df[input_cols].values)
    valid_x_torch = torch.tensor(valid_df[input_cols].values)
    
    # 5. Loop Through Outputs, Train on training set, and Validate on validation set
    initial_models = {}
    for coeff_name in outputs:
        print(f"\n--- Validating model for '{coeff_name}' ---")
        train_y_torch = torch.tensor(train_df[coeff_name].values).unsqueeze(-1)
        valid_y_torch = torch.tensor(valid_df[coeff_name].values).unsqueeze(-1)
        
        model = GPyTorchSurrogateModel(train_x_torch, train_y_torch)
        model.train_model()
        initial_models[coeff_name] = model
        
        model_name = f"{coeff_name} ({run_name})"
        
        GPyTorchSurrogateModel.validate_model(
            model, valid_x_torch, valid_y_torch, 
            name=model_name, output_dir=output_dir
        )
        
        GPyTorchSurrogateModel.plot_ard_relevance(
            model, var_names=input_cols, 
            name=model_name, output_dir=output_dir
        )

    # 6. Query Baseline Performance using the initial models
    print("\n--- Querying Baseline Performance (all DVs = 0) ---")
    baseline_x = torch.zeros(1, num_dvs, dtype=torch.double)
    baseline_results = {}
    for name, model in initial_models.items():
        pred, _ = model.predict(baseline_x)
        baseline_results[name] = pred.item()

    print("Predicted baseline values:")
    for name, value in baseline_results.items():
        print(f"  - {name}: {value:.6f}")

    if 'cl' in baseline_results and 'cd' in baseline_results:
        k_baseline = baseline_results['cl'] / baseline_results['cd'] if abs(baseline_results['cd']) > 1e-9 else 0
        print(f"  - K (L/D): {k_baseline:.6f}")
    print("\nUse these values to set constraints in your problem definition files.")


    # 7. Save the complete pipeline object if requested
    if save_model:
        print(f"\n--- Re-training final models on all available data for '{run_name}' ---")
        
        # Combine training and validation data for the final, most accurate model
        combined_df = pd.concat([train_df, valid_df], ignore_index=True)
        combined_x_torch = torch.tensor(combined_df[input_cols].values)
        
        final_models = {}
        for coeff_name in outputs:
            print(f"Re-training final model for '{coeff_name}'...")
            combined_y_torch = torch.tensor(combined_df[coeff_name].values).unsqueeze(-1)
            final_model = GPyTorchSurrogateModel(combined_x_torch, combined_y_torch)
            final_model.train_model()
            final_models[coeff_name] = final_model

        print(f"\n--- Saving trained pipeline for '{run_name}' ---")
        pipeline = SurrogateBackedPipeline(
            train_paths=[train_file], # Path is only for metadata now
            input_cols=input_cols,
            output_cols=outputs,
            calculate_derived_metrics=False,
            mach_str=run_name,
            train_on_init=False 
        )
        pipeline.models = final_models # Manually insert our final, re-trained models
        
        save_dir = "trained_models"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"pipeline_{run_name.replace(' ', '_')}.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"Successfully saved pipeline to '{save_path}'")

    print(f"\n--- Validation testing complete. All results saved in '{output_dir}' ---")


if __name__ == '__main__':
    # =========================================================================
    # --- MAIN CONFIGURATION SECTION ---
    # =========================================================================
    # INSTRUCTIONS:
    # 1. Edit the CONFIG dictionary below for your specific use case.
    # 2. Set 'save_model' to True if you want to save the final pipeline for optimization.
    # 3. Run the script: python tools/run_validation.py
    # 4. Check the printed baseline values to set constraints in your problem definition files.
    # 5. make sure all output variable names match those in your CSV files exactly.
    # =========================================================================
    
    # Example for the 30 DV case
    CONFIG = {
        "train_file": "data/train_WDA_Ma5_AoA4.csv",
        "valid_file": "data/valid_WDA_Ma5_AoA4.csv",
        "num_dvs": 30,
        "outputs": ["cl", "cd", 'Volume'],
        "dv_format": "underscore",  # 'underscore' for dv_1, or 'bracket' for ratio[0]
        "run_name": "30_DV_Model",
        "save_model": True
    }

    # # Example for the 38 DV case
    # CONFIG = {
    #     "train_file": "data/train_noWDA_Ma5_AoA4.xlsx - 2025-09-16 20.04.csv",
    #     "valid_file": "data/valid_noWDA_Ma5_AoA4.xlsx - 2025-09-17 14.30.csv",
    #     "num_dvs": 38,
    #     "outputs": ["cl", "cd", "volume"],
    #     "dv_format": "bracket",
    #     "run_name": "38_DV_Model_Ma5",
    #     "save_model": True
    # }
    # =========================================================================

    run_validation(
        train_file=CONFIG["train_file"],
        valid_file=CONFIG["valid_file"],
        num_dvs=CONFIG["num_dvs"],
        outputs=CONFIG["outputs"],
        dv_format=CONFIG["dv_format"],
        run_name=CONFIG["run_name"],
        save_model=CONFIG["save_model"]
    )

