# -*- coding: utf-8 -*-
"""
This script loads all datasets, trains the surrogate models for each
design point (Ma 0.4, 6.0, and 10.0), and saves the final, trained
pipeline objects to disk for later use.

MODIFIED: This script now uses the new flexible pipeline and explicitly
configures it for the original 16-DV use case.
"""
import os
import pickle
import sys

# Add project root to path to allow for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surrogates.pipelines import SurrogateBackedPipeline

def main():
    """Main execution function."""
    print("--- Starting Surrogate Model Training and Saving Process (16-DV Cases) ---")
    
    output_dir = "trained_models"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Trained models will be saved in: '{output_dir}/'")
    
    # --- Define configurations for the models to train ---
    
    # Configuration for the original 16 DV case
    input_cols_16dv = [f'ratio[{i}]' for i in range(16)]
    output_cols_16dv = ['Cx', 'CY', 'CZ', 'CMx', 'CMY', 'CMZ']
    
    models_to_train = {
        "ma0.4": {
            "train_paths": ["data/train_dataM04.csv", "data/verify_dataM04.csv"],
            "input_cols": input_cols_16dv,
            "output_cols": output_cols_16dv,
            "calculate_derived_metrics": True,
            "alpha_deg": 4.0
        },
        "ma6.0": {
            "train_paths": ["data/train_dataM6.csv", "data/verify_dataM6.csv"],
            "input_cols": input_cols_16dv,
            "output_cols": output_cols_16dv,
            "calculate_derived_metrics": True,
            "alpha_deg": 4.0
        },
        "ma10.0": {
            "train_paths": ["data/train_dataM10.csv", "data/verify_dataM10.csv"],
            "input_cols": input_cols_16dv,
            "output_cols": output_cols_16dv,
            "calculate_derived_metrics": True,
            "alpha_deg": 4.0
        },
    }
    
    for name, config in models_to_train.items():
        print(f"\n--- Training model for {name} ---")
        try:
            # Use the new, flexible constructor with all parameters
            pipeline = SurrogateBackedPipeline(
                mach_str=name,
                train_paths=config["train_paths"],
                input_cols=config["input_cols"],
                output_cols=config["output_cols"],
                calculate_derived_metrics=config["calculate_derived_metrics"],
                alpha_deg=config["alpha_deg"]
            )
            
            save_path = os.path.join(output_dir, f"pipeline_{name}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            print(f"Successfully trained and saved model to '{save_path}'")
            
        except FileNotFoundError:
            print(f"Error: Could not find data files for Ma={name}. Please ensure data exists at {config['train_paths']}.")
        except Exception as e:
            print(f"An unexpected error occurred while training {name}: {e}")

    print("\n--- All models trained and saved successfully. ---")

if __name__ == '__main__':
    main()
