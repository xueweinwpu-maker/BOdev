# main.py
# -*- coding: utf-8 -*-
import argparse
import importlib
import os
from datetime import datetime
import torch

from src.bo_optimizer import BayesianOptimizer
from src.blackbox_wrapper import AeroEvaluation, AeroEvaluationAdaptive
from src.post_processing import plot_results
from src.history_logger import save_full_history
from src.problem_structures import OptimizationConfig

def main(args):
    print("--- Starting Optimization Framework ---")
    
    # 1. Load Problem Module
    try:
        problem_module = importlib.import_module(args.problem)
        define_problem_func = getattr(problem_module, 'define_problem')
    except Exception as e:
        print(f"FATAL: Could not load problem '{args.problem}'. Error: {e}")
        return

    # 2. Call define_problem (Compatibility wrapper)
    try:
        problem_config = define_problem_func(use_mtgp=args.use_mtgp)
    except TypeError:
        if args.use_mtgp:
            print("WARNING: --use_mtgp ignored for this problem definition.")
        problem_config = define_problem_func()
        
    # 3. Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode_str = "MTGP" if problem_config.model_type == "mtgp" else "STD"
    run_name = f"{args.problem.split('.')[-1]}_{mode_str}_{timestamp}"
    output_dir = os.path.join("results", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # 4. Select Black Box
    if args.adaptive:
        black_box_to_use = AeroEvaluationAdaptive
    else:
        black_box_to_use = AeroEvaluation

    # 5. Initialize & Run
    optimizer = BayesianOptimizer(
        config=problem_config,
        black_box_class=black_box_to_use
    )

    optimizer.run_optimization()

    save_full_history(optimizer, output_dir)
    plot_results(optimizer, output_dir)
    print("--- Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="src.problem_def_uncon", help="Problem module path")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive inner-loop")
    parser.add_argument("--use_mtgp", action="store_true", help="Enable Multi-Task GP")
    
    args = parser.parse_args()
    main(args)