# -*- coding: utf-8 -*-
"""
This module defines the AdaptivePhysicsModel, a wrapper for a live, 
surrogate model that gets updated during the optimization run.
"""
import torch
import warnings
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import PosteriorMean

class AdaptivePhysicsModel:
    """
    Manages a single, adaptive surrogate model for a specific Mach number.
    This model is updated with new data points as the optimization progresses.
    """
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.device = train_x.device
        self.dtype = train_x.dtype
        self.train_x = train_x
        self.train_y = train_y
        self.model = None
        self._fit_model()

    def _fit_model(self):
        """Fits or re-fits the internal Gaussian Process model."""
        print(f"Fitting adaptive model with {self.train_x.shape[0]} points...")
        # Use a standardized model for robustness
        self.model = SingleTaskGP(
            self.train_x, self.train_y, outcome_transform=Standardize(m=1)
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)
        print("Fitting complete.")

    def add_point_and_refit(self, new_x: torch.Tensor, new_y: torch.Tensor):
        """Adds a new data point to the training set and re-fits the model."""
        self.train_x = torch.cat([self.train_x, new_x], dim=0)
        self.train_y = torch.cat([self.train_y, new_y], dim=0)
        self._fit_model()

    def find_best_online_var(self, offline_vars: torch.Tensor, online_bounds: torch.Tensor) -> torch.Tensor:
        """
        Optimizes the online variable by finding the maximum of the model's
        posterior mean, given a fixed set of offline variables.
        """
        if offline_vars.dim() == 1:
            offline_vars = offline_vars.unsqueeze(0)
            
        num_offline_dims = offline_vars.shape[-1]

        def objective_function(online_vars):
            # online_vars shape: (num_candidates, 1, 1) -> (num_candidates, 1)
            online_vars = online_vars.squeeze(-1)
            # Repeat offline_vars to match the batch size of online_vars
            batch_size = online_vars.shape[0]
            repeated_offline = offline_vars.repeat(batch_size, 1)
            # Combine to form the full input for the model
            full_vars = torch.cat([repeated_offline, online_vars], dim=1)
            
            posterior = self.model.posterior(full_vars)
            return posterior.mean

        # Use PosteriorMean as a simple acquisition function to find the peak
        acq_func = PosteriorMean(model=self.model, objective=objective_function)
        
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=online_bounds,
            q=1,
            num_restarts=7,
            raw_samples=64,
        )
        return candidates.detach()
