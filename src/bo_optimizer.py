# src/bo_optimizer.py
# -*- coding: utf-8 -*-
import torch
import warnings
import numpy as np
import gpytorch
import os
import pandas as pd

from botorch.models import MultiTaskGP, SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize

from src.problem_structures import OptimizationConfig
from src.blackbox_wrapper import BlackBox

class BayesianOptimizer:
    def __init__(self, config: OptimizationConfig, black_box_class: type[BlackBox]):
        self.config = config
        self.black_box = black_box_class(config)
        self.device = config.offline_bounds.device
        self.dtype = config.offline_bounds.dtype

        self.train_x = None
        self.train_y = None
        self.raw_train_y_obj = None
        self.raw_train_y_con = None
        
        self.convergence_history = []
        self.acqf_value_history = []
        self.full_run_history = []
        self.ref_point = None
        self.is_multi_objective = len(config.objectives) + len(config.constraints) > 1

    def _print_evaluation_results(self, y_obj_point, y_con_point):
        obj_values = y_obj_point.cpu().numpy().flatten()
        con_values = y_con_point.cpu().numpy().flatten()
        print("  > Results:")
        for i, obj in enumerate(self.config.objectives):
            print(f"    - Objective '{obj.name}': {obj_values[i]:.4f}")
        if self.config.constraints:
            for i, con in enumerate(self.config.constraints):
                print(f"    - Constraint '{con.name}': {con_values[i]:.4f}")

    def _initialize_data(self):
        num_user_points = 0
        initial_x_list = []
        if self.config.user_initial_points is not None:
            initial_x_list.append(self.config.user_initial_points)
            num_user_points = self.config.user_initial_points.shape[0]

        num_random_points = self.config.n_initial_samples - num_user_points
        if num_random_points > 0:
            random_points = draw_sobol_samples(bounds=self.config.offline_bounds, n=num_random_points, q=1).squeeze(-2)
            initial_x_list.append(random_points)
        
        initial_x = torch.cat(initial_x_list, dim=0).to(self.device, self.dtype)

        for i in range(self.config.n_initial_samples):
            x_point = initial_x[i].unsqueeze(0)
            print(f"\n--- Initial Point {i+1} ---")
            y_obj, y_con = self.black_box.evaluate(x_point)
            self._print_evaluation_results(y_obj, y_con)

            y_comb = torch.cat([y_obj, y_con], dim=-1)
            if i == 0:
                self.train_x, self.train_y = x_point, y_comb
                self.raw_train_y_obj, self.raw_train_y_con = y_obj, y_con
            else:
                self.train_x = torch.cat([self.train_x, x_point], dim=0)
                self.train_y = torch.cat([self.train_y, y_comb], dim=0)
                self.raw_train_y_obj = torch.cat([self.raw_train_y_obj, y_obj], dim=0)
                self.raw_train_y_con = torch.cat([self.raw_train_y_con, y_con], dim=0)
            
            if self.is_multi_objective and self.ref_point is None:
                 obj_ref = self.raw_train_y_obj.min(0).values - 0.2 * abs(self.raw_train_y_obj.min(0).values)
                 con_ref = torch.zeros(len(self.config.constraints), device=self.device, dtype=self.dtype)
                 self.ref_point = torch.cat([obj_ref, con_ref])
            
            self._update_convergence_metric()
            self._log_iteration_data(is_initial_point=True)

    def _log_iteration_data(self, is_initial_point=False):
        iteration_data = {
            "offline_vars": self.train_x[-1].cpu().numpy(),
            "objectives": self.raw_train_y_obj[-1].cpu().numpy(),
            "constraints": self.raw_train_y_con[-1].cpu().numpy() if self.raw_train_y_con.numel() > 0 else None,
            "converged_online_vars": self.black_box.last_run_converged_online.copy(),
            "design_point_results": self.black_box.last_run_results_map.copy(),
            "convergence_metric": self.convergence_history[-1] if self.convergence_history else None,
            "acqf_value": self.acqf_value_history[-1] if not is_initial_point and self.acqf_value_history else None,
            "inner_loop_histories": self.black_box.last_run_inner_histories.copy()
        }
        self.full_run_history.append(iteration_data)

    def _update_convergence_metric(self):
        if not self.is_multi_objective:
            self.convergence_history.append(self.train_y.max().item())
        else:
             self.convergence_history.append(0.0) # Placeholder for MOO metric

    def _get_and_fit_model(self):
        if self.config.model_type == "mtgp":
            return self._fit_mtgp_model()
        else:
            return self._fit_standard_model()

    def _fit_standard_model(self):
        """Fits Independent SingleTaskGPs."""
        normalized_train_x = normalize(self.train_x, self.config.offline_bounds)
        full_train_y = self.train_y
        
        models = []
        for i in range(full_train_y.shape[-1]):
            train_y_i = full_train_y[:, [i]]
            model = SingleTaskGP(
                train_X=normalized_train_x,
                train_Y=train_y_i,
                outcome_transform=Standardize(m=1),
                covar_module=gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=normalized_train_x.shape[-1])
                )
            )
            models.append(model)
        
        model_list = ModelListGP(*models)
        mll = gpytorch.mlls.SumMarginalLogLikelihood(model_list.likelihood, model_list)
        with gpytorch.settings.cholesky_jitter(1e-5):
            fit_gpytorch_mll(mll)
        return model_list

    def _fit_mtgp_model(self):
        """Fits a Correlated Multi-Task GP."""
        normalized_train_x = normalize(self.train_x, self.config.offline_bounds)
        num_tasks = self.train_y.shape[1]
        n_samples = self.train_x.shape[0]

        X_list, Y_list = [], []
        for i in range(num_tasks):
            task_idx = torch.full((n_samples, 1), i, device=self.device, dtype=self.dtype)
            X_with_task = torch.cat([normalized_train_x, task_idx], dim=1)
            X_list.append(X_with_task)
            Y_list.append(self.train_y[:, i].unsqueeze(1))
        
        full_X = torch.cat(X_list, dim=0)
        full_Y = torch.cat(Y_list, dim=0)
        
        model = MultiTaskGP(
            full_X, full_Y, task_feature=-1, rank=self.config.mtgp_rank,
            outcome_transform=Standardize(m=1)
        )
        
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with gpytorch.settings.cholesky_jitter(1e-5):
            fit_gpytorch_mll(mll)
        return model

    def _optimize_acquisition_function(self):
        model = self._get_and_fit_model()
        normalized_bounds = torch.tensor([[0.0]*self.config.offline_dim, [1.0]*self.config.offline_dim], 
                                         device=self.device, dtype=self.dtype)

        if self.config.model_type == "mtgp":
            # MTGP Strategy: Weighted Sum
            num_tasks = self.train_y.shape[1]
            weights = torch.full((num_tasks,), 1.0/num_tasks, device=self.device, dtype=self.dtype)
            
            def weighted_obj(samples):
                return torch.matmul(samples, weights)
            
            objective = GenericMCObjective(objective=weighted_obj)
            acq_func = qExpectedImprovement(
                model=model,
                best_f=objective(self.train_y.unsqueeze(0)).max(),
                objective=objective
            )
        else:
            # Standard Strategy
            if self.is_multi_objective:
                if len(self.config.objectives) == 2 and not self.config.constraints:
                     # Weighted sum fallback for benchmark compatibility
                     weights = torch.tensor([0.5, 0.5], device=self.device, dtype=self.dtype)
                     obj = GenericMCObjective(objective=lambda s: torch.matmul(s, weights))
                     acq_func = qExpectedImprovement(model=model, best_f=obj(self.train_y.unsqueeze(0)).max(), objective=obj)
                else:
                    acq_func = qExpectedHypervolumeImprovement(
                        model=model, ref_point=self.ref_point,
                        partitioning=FastNondominatedPartitioning(self.ref_point, self.train_y),
                        objective=IdentityMCMultiOutputObjective(outcomes=list(range(self.train_y.shape[-1])))
                    )
            else:
                acq_func = qExpectedImprovement(model=model, best_f=self.train_y.max())

        candidates, acqf_value = optimize_acqf(
            acq_function=acq_func,
            bounds=normalized_bounds,
            q=self.config.batch_size,
            num_restarts=self.config.num_restarts,
            raw_samples=self.config.raw_samples,
        )
        return unnormalize(candidates, self.config.offline_bounds), acqf_value

    def _summarize_problem(self):
        print("\n" + "="*70)
        print(" Optimization Summary")
        print(f" Strategy: {self.config.model_type.upper()}")
        print("="*70 + "\n")

    def run_optimization(self):
        self._initialize_data()
        self._summarize_problem()
        for i in range(self.config.n_bo_iterations):
            print(f"\n--- BO Iteration {i+1}/{self.config.n_bo_iterations} ---")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_x, new_val = self._optimize_acquisition_function()
            
            self.acqf_value_history.append(new_val.item())
            print("Evaluating new candidate...")
            new_y_obj, new_y_con = self.black_box.evaluate(new_x)
            self._print_evaluation_results(new_y_obj, new_y_con)

            self.train_x = torch.cat([self.train_x, new_x])
            y_comb = torch.cat([new_y_obj, new_y_con], dim=-1)
            self.train_y = torch.cat([self.train_y, y_comb], dim=0)
            self.raw_train_y_obj = torch.cat([self.raw_train_y_obj, new_y_obj], dim=0)
            self.raw_train_y_con = torch.cat([self.raw_train_y_con, new_y_con], dim=0)
            
            self._update_convergence_metric()
            self._log_iteration_data()
        print("\n--- Optimization Finished ---")