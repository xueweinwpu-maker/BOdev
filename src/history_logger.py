# -*- coding: utf-8 -*-
"""
Module for logging the full optimization history to structured CSV files.
This version saves both a main summary file and detailed logs for each
inner-loop optimization run.
"""
import pandas as pd
import numpy as np
import os
from src.bo_optimizer import BayesianOptimizer

def save_full_history(optimizer: BayesianOptimizer, output_dir: str):
    """
    Saves the complete optimization history, including a main summary CSV and
    detailed CSVs for each inner-loop run.
    
    Args:
        optimizer: The completed optimizer object containing the run history.
        output_dir: The main directory where results are being saved.
    """
    history_data = optimizer.full_run_history
    if not history_data:
        print("No history data to save.")
        return

    # --- Part 1: Save the main summary CSV (as before) ---
    summary_filename = os.path.join(output_dir, "full_optimization_history.csv")
    print(f"\n--- Saving main optimization history to '{summary_filename}' ---")
    
    summary_records = []
    for i, iteration_data in enumerate(history_data):
        record = {"evaluation": i + 1}
        for j, var in enumerate(iteration_data["offline_vars"]):
            record[f"offline_var_{j}"] = var
        for dp_name, online_vars in iteration_data["converged_online_vars"].items():
            if online_vars is not None:
                if isinstance(online_vars, (list, np.ndarray)):
                    for j, ovar in enumerate(online_vars):
                        record[f"{dp_name}_online_var_{j}"] = ovar
                else:
                     record[f"{dp_name}_online_var_0"] = online_vars
        for j, obj in enumerate(iteration_data["objectives"]):
            obj_name = optimizer.config.objectives[j].name.replace(" ", "_")
            record[f"obj_{obj_name}"] = obj
        if iteration_data["constraints"] is not None:
            for j, con in enumerate(iteration_data["constraints"]):
                con_name = optimizer.config.constraints[j].name.replace(" ", "_")
                record[f"con_{con_name}"] = con
        for dp_name, dp_results in iteration_data["design_point_results"].items():
            for key, value in dp_results.items():
                if isinstance(value, (int, float, np.number)):
                    record[f"{dp_name}_{key}"] = value
        record["convergence_metric"] = iteration_data["convergence_metric"]
        record["acqf_value"] = iteration_data["acqf_value"]
        summary_records.append(record)
    
    try:
        pd.DataFrame(summary_records).to_csv(summary_filename, index=False)
        print("Main history saved successfully.")
    except Exception as e:
        print(f"Error saving main history to CSV: {e}")

    # --- Part 2: Save detailed inner-loop histories ---
    details_dir = os.path.join(output_dir, "inner_loop_details")
    os.makedirs(details_dir, exist_ok=True)
    print(f"--- Saving detailed inner-loop histories to '{details_dir}' ---")

    for i, iteration_data in enumerate(history_data):
        eval_num = i + 1
        if "inner_loop_histories" not in iteration_data or not iteration_data["inner_loop_histories"]:
            continue
            
        for dp_name, inner_history in iteration_data["inner_loop_histories"].items():
            if not inner_history or not inner_history.online_vars_history:
                continue

            records = []
            num_steps = len(inner_history.online_vars_history)
            for step in range(num_steps):
                record = {"inner_step": step + 1}
                
                # Log online variables for this step
                online_vars = inner_history.online_vars_history[step]
                for j, var in enumerate(online_vars):
                    record[f"online_var_{j}"] = var
                    
                # Log objective for this step
                if step < len(inner_history.objective_history):
                    record["objective"] = inner_history.objective_history[step]
                    
                # Log constraints for this step
                for c_name, c_history in inner_history.constraint_history.items():
                    if step < len(c_history):
                        record[f"constraint_{c_name.replace(' ', '_')}"] = c_history[step]
                        
                records.append(record)
            
            if records:
                df_detail = pd.DataFrame(records)
                detail_filename = os.path.join(details_dir, f"eval_{eval_num:03d}_{dp_name}.csv")
                df_detail.to_csv(detail_filename, index=False)
    
    print("Detailed inner loop histories saved successfully.")
