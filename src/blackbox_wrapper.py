# -*- coding: utf-8 -*-
"""
The Black-Box wrapper.
This version now includes a new 'AeroEvaluationAdaptive' class that implements
the experimental bi-level approach using separate, live, 16D adaptive
surrogate models for the inner loops. This class is now fully flexible and
configures itself based on the problem definition.
This version fixes a bug where the adaptive model initialization failed when
using a derived metric (like 'K' or 'CL') as the inner-loop objective.
"""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import torch
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import warnings

from src.problem_structures import OptimizationConfig, DesignPoint, InnerLoopHistory
from src.adaptive_physics_model import AdaptivePhysicsModel


@dataclass
class _CacheEntry:
    results: Dict[str, Any]

class BlackBox(ABC):
    """Abstract base class for the black-box function."""
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.evaluation_counter = 0
        self.surrogate_call_counter = 0
        self.last_run_inner_histories: Dict[str, InnerLoopHistory] = {}
        self.warm_start_points: Dict[str, np.ndarray] = {}
        self.last_run_results_map: Dict[str, Dict[str, Any]] = {}
        self.dtype = config.offline_bounds.dtype
        self.device = config.offline_bounds.device

    @abstractmethod
    def evaluate(self, offline_vars_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

# --- STANDARD, NON-ADAPTIVE version for reference ---
class AeroEvaluation(BlackBox):
    """
    The concrete implementation of the black-box for aerodynamic evaluation.
    It orchestrates the inner-loop optimizations for each design point using
    a standard numerical optimizer (Scipy's SLSQP).
    """
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.last_run_converged_online: Dict[str, np.ndarray] = {}

    def _inner_loop_optimize(self, offline_vars: np.ndarray, design_point: DesignPoint) -> Dict[str, Any]:
        cfg = design_point.inner_opt_config
        cache: Dict[Tuple, _CacheEntry] = {}
        history = InnerLoopHistory()

        def get_from_pipeline_or_cache(online_vars_tuple: Tuple) -> Dict[str, Any]:
            if online_vars_tuple not in cache:
                self.surrogate_call_counter += 1
                cache[online_vars_tuple] = _CacheEntry(
                    results=design_point.pipeline.run(offline_vars, np.array(online_vars_tuple))
                )
            return cache[online_vars_tuple].results

        def objective_for_sqp(online_vars: np.ndarray) -> float:
            # Objective is negated because SLSQP minimizes
            obj_name = design_point.pipeline.inner_loop_objective
            raw_obj = get_from_pipeline_or_cache(tuple(online_vars)).get(obj_name, 0.0)
            return -raw_obj if not obj_name.startswith('-') else raw_obj.lstrip('-')

        inner_constraints_for_solver = []
        if cfg.constraints:
            for constraint in cfg.constraints:
                def constraint_func(online_vars: np.ndarray, c=constraint) -> float:
                    sim_results = get_from_pipeline_or_cache(tuple(online_vars))
                    return c.eval_function(sim_results)
                inner_constraints_for_solver.append({'type': 'ineq', 'fun': constraint_func})
                history.constraint_history[constraint.name] = []

        def callback(xk):
            current_results = get_from_pipeline_or_cache(tuple(xk))
            obj_name = design_point.pipeline.inner_loop_objective.lstrip('-')
            objective_value = current_results.get(obj_name, 0.0)
            
            log_msg = (f"    - Inner-loop step: sweep_angle = {xk[0]:.4f}, "
                       f"{obj_name} = {objective_value:.4f}")
            
            if cfg.constraints:
                con_msgs = []
                for constraint in cfg.constraints:
                    con_val = constraint.eval_function(current_results)
                    con_msgs.append(f"{constraint.name.split('_')[0]}={con_val:.4f}")
                    history.constraint_history[constraint.name].append(con_val)
                log_msg += ", Cons: (" + ", ".join(con_msgs) + ")"
            print(log_msg)
            history.online_vars_history.append(np.copy(xk))
            history.objective_history.append(objective_value)

        initial_guess = self.warm_start_points.get(design_point.name, cfg.x0)
        print(f"  - Starting standard inner loop for '{design_point.name}' with initial guess: {np.round(initial_guess, 4)}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                fun=objective_for_sqp, x0=initial_guess, method=cfg.method,
                bounds=cfg.bounds, constraints=inner_constraints_for_solver,
                options={"maxiter": cfg.maxiter}, tol=cfg.tol, callback=callback
            )
        
        optimal_online_vars = result.x
        final_results = get_from_pipeline_or_cache(tuple(optimal_online_vars))
        
        is_feasible = all(c.eval_function(final_results) >= 0 for c in cfg.constraints)
        feasibility_msg = "Feasible" if is_feasible else "Infeasible"
        print(f"  - Inner loop for '{design_point.name}' converged in {result.nit} iterations. Final state: {feasibility_msg}")
        
        if not is_feasible:
            print("    > WARNING: Inner loop failed to find a feasible solution. Returning penalty.")
        
        self.warm_start_points[design_point.name] = optimal_online_vars
        self.last_run_converged_online[design_point.name] = optimal_online_vars
        self.last_run_inner_histories[design_point.name] = history

        return final_results

    def evaluate(self, offline_vars_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.last_run_inner_histories.clear()
        self.last_run_converged_online.clear()
        self.last_run_results_map.clear()
        
        offline_vars_batch = offline_vars_tensor.cpu().numpy()
        all_objectives_batch, all_constraints_batch = [], []

        for offline_vars in offline_vars_batch:
            self.evaluation_counter += 1
            print(f"\n--- Black-Box Evaluation #{self.evaluation_counter} ---\nOffline Vars: {np.round(offline_vars, 4)}")
            
            results_map: Dict[str, Dict[str, Any]] = {}
            for dp in self.config.design_points:
                if dp.inner_opt_config:
                    final_results = self._inner_loop_optimize(offline_vars, dp)
                else:
                    final_results = dp.pipeline.run(offline_vars, None)
                    self.surrogate_call_counter += 1
                    self.last_run_converged_online[dp.name] = None
                
                results_map[dp.name] = final_results
            
            self.last_run_results_map = results_map.copy()

            objective_values = [obj.eval_function(results_map) for obj in self.config.objectives]
            all_objectives_batch.append(objective_values)
            
            constraint_values = [con.eval_function(results_map) for con in self.config.constraints] if self.config.constraints else []
            all_constraints_batch.append(constraint_values)

        print("--- Evaluation Complete ---")
        
        objectives_tensor = torch.tensor(all_objectives_batch, dtype=self.dtype, device=self.device)
        constraints_tensor = torch.tensor(all_constraints_batch, dtype=self.dtype, device=self.device) if self.config.constraints else torch.empty(len(all_objectives_batch), 0, dtype=self.dtype, device=self.device)
        
        return objectives_tensor, constraints_tensor


# --- FLEXIBLE ADAPTIVE version ---
class AeroEvaluationAdaptive(BlackBox):
    """
    An experimental version of the black-box that uses a live, adaptive 16D
    surrogate model for each design point to perform inner-loop optimizations.
    This class dynamically creates models based on the provided problem definition.
    """
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.last_run_converged_online: Dict[str, np.ndarray] = {}
        self.adaptive_models: Dict[str, AdaptivePhysicsModel] = {}
        self._initialize_adaptive_models()

    def _initialize_adaptive_models(self):
        """
        Dynamically loads data and creates an adaptive model for each
        DesignPoint defined in the optimization configuration.
        """
        print("\n--- Initializing Adaptive Physics Models based on Problem Definition ---")
        for dp in self.config.design_points:
            dp_name = dp.name
            print(f"  - Initializing model for Design Point: '{dp_name}'")

            paths = dp.pipeline.train_paths
            df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
            input_cols = dp.pipeline.input_cols
            
            inner_obj_name = dp.pipeline.inner_loop_objective
            if not inner_obj_name:
                raise ValueError(f"DesignPoint '{dp_name}' must have an inner_loop_objective defined in its pipeline for the adaptive strategy.")
            
            output_col = inner_obj_name.lstrip('-')

            # ** FIX: Calculate derived metrics if they are the objective **
            if dp.pipeline.calculate_derived_metrics:
                print(f"    - Objective '{output_col}' is a derived metric. Calculating from source data...")
                alpha = dp.pipeline.alpha_rad
                
                if 'Cx' not in df.columns or 'CY' not in df.columns:
                    raise KeyError(f"Source data CSV for '{dp_name}' must contain 'Cx' and 'CY' to calculate derived metrics.")

                cl_series = df['CY'] * np.cos(alpha) - df['Cx'] * np.sin(alpha)
                cd_series = df['CY'] * np.sin(alpha) + df['Cx'] * np.cos(alpha)
                
                train_y_series = None
                if output_col == 'CL':
                    train_y_series = cl_series
                elif output_col == 'CD':
                    train_y_series = cd_series
                elif output_col == 'K':
                    k_series = cl_series.divide(cd_series.where(cd_series.abs() > 1e-6, np.nan)).fillna(0)
                    train_y_series = k_series
            else:
                print(f"    - Objective '{output_col}' is a base metric. Loading directly from source data...")
                train_y_series = df[output_col]

            train_y_raw = torch.tensor(train_y_series.values, dtype=self.dtype, device=self.device).unsqueeze(-1)
            train_y = -train_y_raw if not inner_obj_name.startswith('-') else train_y_raw
            train_x = torch.tensor(df[input_cols].values, dtype=self.dtype, device=self.device)

            self.adaptive_models[dp_name] = AdaptivePhysicsModel(train_x, train_y)
        print("--- Adaptive models initialized and trained on baseline data. ---\n")

    def _inner_loop_optimize(self, offline_vars: np.ndarray, design_point: DesignPoint) -> Dict[str, Any]:
        """
        Performs inner-loop optimization using the corresponding adaptive model.
        """
        dp_name = design_point.name
        cfg = design_point.inner_opt_config
        adaptive_model = self.adaptive_models[dp_name]

        print(f"  - Starting adaptive inner loop for '{dp_name}'...")
        offline_vars_tensor = torch.tensor(offline_vars, dtype=self.dtype, device=self.device)
        online_bounds_tensor = torch.tensor(cfg.bounds, dtype=self.dtype, device=self.device).T

        optimal_online_tensor = adaptive_model.find_best_online_var(offline_vars_tensor, online_bounds_tensor)
        optimal_online_vars = optimal_online_tensor.cpu().numpy().flatten()
        print(f"    > Adaptive model suggests optimal online var: {np.round(optimal_online_vars, 4)}")

        final_results = design_point.pipeline.run(offline_vars, optimal_online_vars)
        self.surrogate_call_counter += 1

        new_x = torch.cat([offline_vars_tensor, optimal_online_tensor.squeeze(0)], dim=0).unsqueeze(0)
        
        inner_obj_name = design_point.pipeline.inner_loop_objective
        output_col = inner_obj_name.lstrip('-')
        new_y_val = final_results[output_col]
        if not inner_obj_name.startswith('-'):
            new_y_val = -new_y_val
        new_y = torch.tensor([[new_y_val]], dtype=self.dtype, device=self.device)

        adaptive_model.add_point_and_refit(new_x, new_y)

        is_feasible = all(c.eval_function(final_results) >= 0 for c in cfg.constraints)
        if not is_feasible:
            print(f"  - WARNING: Inner loop for '{dp_name}' found an infeasible solution.")
        
        self.last_run_converged_online[dp_name] = optimal_online_vars
        return final_results

    def evaluate(self, offline_vars_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.last_run_inner_histories.clear()
        self.last_run_converged_online.clear()
        self.last_run_results_map.clear()
        
        offline_vars_batch = offline_vars_tensor.cpu().numpy()
        all_objectives_batch, all_constraints_batch = [], []

        for offline_vars in offline_vars_batch:
            self.evaluation_counter += 1
            print(f"\n--- Black-Box Evaluation #{self.evaluation_counter} ---\nOffline Vars: {np.round(offline_vars, 4)}")
            
            results_map: Dict[str, Dict[str, Any]] = {}
            for dp in self.config.design_points:
                if dp.inner_opt_config:
                    final_results = self._inner_loop_optimize(offline_vars, dp)
                else: 
                    final_results = dp.pipeline.run(offline_vars, None)
                    self.surrogate_call_counter += 1
                    self.last_run_converged_online[dp.name] = None
                
                results_map[dp.name] = final_results
            
            self.last_run_results_map = results_map.copy()

            objective_values = [obj.eval_function(results_map) for obj in self.config.objectives]
            all_objectives_batch.append(objective_values)
            
            constraint_values = [con.eval_function(results_map) for con in self.config.constraints] if self.config.constraints else []
            all_constraints_batch.append(constraint_values)

        print("--- Evaluation Complete ---")
        
        objectives_tensor = torch.tensor(all_objectives_batch, dtype=self.dtype, device=self.device)
        constraints_tensor = torch.tensor(all_constraints_batch, dtype=self.dtype, device=self.device) if self.config.constraints else torch.empty(len(all_objectives_batch), 0, dtype=self.dtype, device=self.device)
        
        return objectives_tensor, constraints_tensor
