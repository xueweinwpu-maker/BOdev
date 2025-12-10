# -*- coding: utf-8 -*-
"""
Module for post-processing and visualizing optimization results.
MODIFIED: Updated to use the new data attribute names from the refactored
BayesianOptimizer class (e.g., `raw_train_y_obj` instead of `train_y_obj`).
ADDED: A function to automatically find the best feasible design point and
save its design variable vector to a text file for subsequent CFD analysis.
MODIFIED: The parallel coordinates plot now shows the top N best feasible
designs instead of just the Pareto front, making it more insightful for
single-objective constrained problems.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
from botorch.utils.multi_objective.pareto import is_non_dominated
from src.bo_optimizer import BayesianOptimizer

def _save_best_feasible_design(optimizer: BayesianOptimizer, output_dir: str):
    """
    Finds the best feasible point and saves its design variables to a text file.
    'Best' is defined as the point with the highest value for the first objective.
    """
    print("\n--- Finding and Saving Best Feasible Design for Flow Analysis ---")
    
    final_y_obj = optimizer.raw_train_y_obj.cpu().numpy()
    final_y_con = optimizer.raw_train_y_con.cpu().numpy()
    final_x = optimizer.train_x.cpu().numpy()

    # Create a mask for feasible points
    feasible_mask = (final_y_con >= 0).all(axis=1) if final_y_con.shape[1] > 0 else np.ones(final_y_obj.shape[0], dtype=bool)

    if not np.any(feasible_mask):
        print("No feasible points were found during the optimization. Cannot save a best design.")
        return

    # Filter to only include feasible points
    feasible_objectives = final_y_obj[feasible_mask]
    feasible_dvs = final_x[feasible_mask]

    # Find the index of the best point based on the first objective
    # Assumes the first objective is the primary one to maximize
    best_point_index = np.argmax(feasible_objectives[:, 0])
    
    best_design_dvs = feasible_dvs[best_point_index]
    best_design_obj = feasible_objectives[best_point_index]

    # Save the design variables to a file
    save_path = os.path.join(output_dir, "best_feasible_design_dvs.txt")
    np.savetxt(save_path, best_design_dvs, fmt="%.8f", header="Design variables for the best feasible point found:")
    
    print(f"Best feasible point's objectives: {np.round(best_design_obj, 4)}")
    print(f"Design variables for this point saved to: '{save_path}'")


def _plot_parallel_coordinates(optimizer: BayesianOptimizer, output_dir: str):
    """
    Generates and saves a parallel coordinates plot for the top N best
    feasible designs, sorted by the primary objective.
    """
    print("\n--- Generating Parallel Coordinates Plot for Top Feasible Designs ---")
    
    final_y_obj = optimizer.raw_train_y_obj.cpu().numpy()
    final_y_con = optimizer.raw_train_y_con.cpu().numpy()
    final_x = optimizer.train_x.cpu().numpy()

    feasible_mask = (final_y_con >= 0).all(axis=1) if final_y_con.shape[1] > 0 else np.ones(final_y_obj.shape[0], dtype=bool)
    if not np.any(feasible_mask):
        print("No feasible points found to generate parallel coordinates plot.")
        return

    # --- NEW LOGIC START ---
    # Instead of plotting the Pareto front (which might be a single point),
    # we will plot the top N best feasible designs.

    # 1. Filter to get all feasible designs and their objectives.
    feasible_obj = final_y_obj[feasible_mask]
    feasible_x = final_x[feasible_mask]
    
    # 2. Sort the feasible designs by the first (primary) objective, in descending order.
    # We get the sorting indices from the first column of the objectives.
    sort_indices = np.argsort(feasible_obj[:, 0])[::-1]
    
    # 3. Define how many of the top designs we want to show.
    num_to_plot = min(20, len(feasible_obj)) # Show up to 20 designs
    
    # 4. Select the top N designs using the sorted indices.
    top_indices = sort_indices[:num_to_plot]
    top_x = feasible_x[top_indices]
    top_y = feasible_obj[top_indices]

    # --- NEW LOGIC END ---

    data_dict = {}
    obj_names = [obj.name for obj in optimizer.config.objectives]
    
    # The class column for coloring will be the primary objective.
    class_column_name = obj_names[0]
    data_dict[class_column_name] = top_y[:, 0]
    
    # Add other objectives if they exist
    for i in range(1, top_y.shape[1]):
        data_dict[obj_names[i]] = top_y[:, i]
        
    # Add design variables
    for i in range(top_x.shape[1]):
        data_dict[f'offline_var_{i}'] = top_x[:, i]
        
    df = pd.DataFrame(data_dict)
    
    if df.empty:
        print("No data to plot for parallel coordinates.")
        return

    fig = plt.figure(figsize=(20, 10))
    pd.plotting.parallel_coordinates(df, class_column=class_column_name, colormap='viridis', linewidth=2.5)
    plt.title(f'Parallel Coordinates of Top {num_to_plot} Feasible Solutions', fontsize=18)
    plt.xlabel('Design Variables and Objectives', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    fig.savefig(os.path.join(output_dir, "parallel_coordinates_plot.png"))
    plt.close(fig)


def _plot_single_objective_results(optimizer: BayesianOptimizer, output_dir: str):
    """
    Generates and saves the convergence plot for a single-objective problem.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    history_values = np.array(optimizer.convergence_history)
    plot_values = [val if not np.isnan(val) else None for val in history_values]
    
    iterations = np.arange(1, len(plot_values) + 1)
    
    ax.plot(iterations, plot_values, marker='o', linestyle='-', color='navy', label='Best Feasible Objective')
    
    ax.set_title("Convergence History", fontsize=16)
    ax.set_xlabel("Evaluation Number (Initial Samples + BO Iterations)", fontsize=12)
    ax.set_ylabel(f"Best Feasible: {optimizer.config.objectives[0].name}", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    valid_history = [v for v in plot_values if v is not None]
    if len(valid_history) > 1:
        y_min = np.min(valid_history)
        y_max = np.max(valid_history)
        y_range = max(y_max - y_min, 1e-6)
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence_history.png"))
    plt.close(fig)

def _plot_multi_objective_results(optimizer: BayesianOptimizer, output_dir: str):
    """Generates and saves the Pareto front and hypervolume convergence plots."""
    final_y_obj = optimizer.raw_train_y_obj.cpu().numpy()
    final_y_con = optimizer.raw_train_y_con.cpu().numpy()
    
    objective_names = [obj.name for obj in optimizer.config.objectives]
    constraint_names = [con.name for con in optimizer.config.constraints]
    
    # Combine objectives and constraints for plotting
    all_y = np.concatenate([final_y_obj, final_y_con], axis=1)
    all_names = objective_names + constraint_names

    feasible_mask = (final_y_con >= 0).all(axis=1) if final_y_con.shape[1] > 0 else np.ones(all_y.shape[0], dtype=bool)

    # Plot Objective Space
    fig1 = plt.figure(figsize=(10, 8))
    # We will plot the first objective vs the first constraint
    ax1 = fig1.add_subplot(111)
    ax1.scatter(final_y_obj[~feasible_mask, 0], final_y_con[~feasible_mask, 0], c='gray', s=30, alpha=0.5, label='Infeasible')
    ax1.scatter(final_y_obj[feasible_mask, 0], final_y_con[feasible_mask, 0], c='blue', s=50, alpha=0.7, label='Feasible')

    if np.any(feasible_mask):
        feasible_y_combined = torch.from_numpy(all_y[feasible_mask])
        pareto_mask = is_non_dominated(feasible_y_combined)
        pareto_points = feasible_y_combined[pareto_mask].cpu().numpy()
        ax1.scatter(pareto_points[:, 0], pareto_points[:, 1], c='red', s=120, marker='*', label='Pareto Front')

    ax1.set_xlabel(objective_names[0], fontsize=12)
    ax1.set_ylabel(constraint_names[0], fontsize=12)
    ax1.axhline(0, color='k', linestyle='--', label='Feasibility Boundary') # Add constraint boundary line
    ax1.set_title("Optimization Results: Objective Space", fontsize=16)
    ax1.legend()
    ax1.grid(True)
    fig1.savefig(os.path.join(output_dir, "objective_space.png"))
    plt.close(fig1)

    # Plot Hypervolume Convergence
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    iterations = np.arange(1, len(optimizer.convergence_history) + 1)
    ax2.plot(iterations, optimizer.convergence_history, marker='o', linestyle='-', color='purple')
    ax2.set_title("Hypervolume Convergence", fontsize=16)
    ax2.set_xlabel("Evaluation Number (Initial Samples + BO Iterations)", fontsize=12)
    ax2.set_ylabel("Hypervolume", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "hypervolume_convergence.png"))
    plt.close(fig2)

def plot_acquisition_convergence(optimizer: BayesianOptimizer, output_dir: str):
    """Generates and saves the convergence plot for the acquisition function value."""
    if not optimizer.acqf_value_history: return
    print("\n--- Generating Acquisition Function Convergence Plot ---")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    iterations = np.arange(1, len(optimizer.acqf_value_history) + 1)
    ax.plot(iterations, optimizer.acqf_value_history, marker='.', linestyle='--', color='darkorange')
    ax.set_title("Acquisition Function Value Convergence", fontsize=16)
    ax.set_xlabel("BO Iteration Number", fontsize=12)
    ax.set_ylabel("Max Acquisition Function Value", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "acquisition_convergence.png"))
    plt.close(fig)

def plot_results(optimizer: BayesianOptimizer, output_dir: str):
    """The main function to generate and save all relevant plots."""
    print("\n--- Generating and Saving Plots ---")
    if optimizer.is_multi_objective:
        _plot_multi_objective_results(optimizer, output_dir)
        # The parallel coordinates plot is now useful for constrained problems too
        _plot_parallel_coordinates(optimizer, output_dir)
    else:
        _plot_single_objective_results(optimizer, output_dir)
    
    plot_acquisition_convergence(optimizer, output_dir)
    
    if optimizer.config.constraints:
        _save_best_feasible_design(optimizer, output_dir)
    
    # Show the main convergence plot at the end of the run
    if optimizer.is_multi_objective:
        fig, ax = plt.subplots(figsize=(12, 7))
        iterations = np.arange(1, len(optimizer.convergence_history) + 1)
        ax.plot(iterations, optimizer.convergence_history, marker='o', linestyle='-', color='purple')
        ax.set_title("Hypervolume Convergence", fontsize=16)
        ax.set_xlabel("Evaluation Number", fontsize=12)
        ax.set_ylabel("Hypervolume", fontsize=12)
        ax.grid(True)
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        iterations = np.arange(1, len(optimizer.convergence_history) + 1)
        history_values = np.array(optimizer.convergence_history)
        plot_values = [val if not np.isnan(val) else None for val in history_values]
        ax.plot(iterations, plot_values, marker='o', linestyle='-', color='navy')
        ax.set_title("Convergence History", fontsize=16)
        ax.set_xlabel("Evaluation Number", fontsize=12)
        ax.set_ylabel(f"Best Feasible: {optimizer.config.objectives[0].name}", fontsize=12)
        ax.grid(True)
        plt.show()

