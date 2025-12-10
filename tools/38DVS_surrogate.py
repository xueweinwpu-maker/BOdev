# -*- coding: utf-8 -*-
"""
This script trains and validates surrogate models for the 38 DV case 
(Ma=5, AoA=4) using the new flexible framework components.
"""
import pandas as pd
import torch
import os
import sys
from datetime import datetime

# Add project root to path to allow for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surrogates.model import GPyTorchSurrogateModel

def main():
    """
    Trains and validates surrogate models for the 38 DV case.
    """
    print("--- Starting Validation for 38 DV Case (Ma=5, AoA=4) ---")

    # 1. Define File Paths and Parameters
    train_path = "data/train_WDA_Ma5_AoA4.csv"
    valid_path = "data/valid_WDA_Ma5_AoA4.csv"
    
    # Define the 38 input design variables based on the CSV files
    input_cols = [f'dv_{i+1}' for i in range(38)]
    
    # Define the output columns to model from the CSV files
    output_cols_to_test = ['cl', 'cd', 'Volume']
    
    # Create a timestamped directory for the results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("validation_results", f"run_38dv_Ma5_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Validation results will be saved to: {output_dir}")

    # 2. Load and Clean Data
    try:
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        
        # Standardize column names
        train_df.columns = train_df.columns.str.strip()
        valid_df.columns = valid_df.columns.str.strip()
        
        # The new CSVs use 'ratio[x]' format for DVs, let's rename for clarity
        rename_dict = {f'ratio[{i}]': f'dv_{i+1}' for i in range(38)}
        train_df.rename(columns=rename_dict, inplace=True)
        valid_df.rename(columns=rename_dict, inplace=True)

        required_cols = input_cols + output_cols_to_test
        
        train_df.dropna(subset=required_cols, inplace=True)
        valid_df.dropna(subset=required_cols, inplace=True)
        
        print(f"Loaded and cleaned data. Training points: {len(train_df)}, Validation points: {len(valid_df)}")

    except FileNotFoundError as e:
        print(f"Error: Could not find data file. {e}")
        return
        
    if train_df.empty or valid_df.empty:
        print("Error: No valid data found after cleaning. Aborting.")
        return

    # 3. Prepare Data Tensors
    train_x_torch = torch.tensor(train_df[input_cols].values)
    valid_x_torch = torch.tensor(valid_df[input_cols].values)
    
    # 4. Loop Through Outputs, Train, and Validate
    for coeff_name in output_cols_to_test:
        train_y_torch = torch.tensor(train_df[coeff_name].values).unsqueeze(-1)
        valid_y_torch = torch.tensor(valid_df[coeff_name].values).unsqueeze(-1)
        
        model = GPyTorchSurrogateModel(train_x_torch, train_y_torch)
        model.train_model()
        
        model_name = f"{coeff_name} (Ma=5, 38 DVs)"
        
        # Validate the model and save plots/metrics
        GPyTorchSurrogateModel.validate_model(
            model, valid_x_torch, valid_y_torch, 
            name=model_name, output_dir=output_dir
        )
        
        # Plot ARD relevance using the MODIFIED function
        GPyTorchSurrogateModel.plot_ard_relevance(
            model, var_names=input_cols, 
            name=model_name, output_dir=output_dir
        )

    print(f"\n--- Validation testing complete. All results saved in {output_dir} ---")

if __name__ == '__main__':
    main()
