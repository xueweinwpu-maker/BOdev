# -*- coding: utf-8 -*-
"""
This module defines the Gaussian Process surrogate model using GPyTorch and BoTorch.
MODIFIED: Added logic to only show y-axis label on the first subplot of a figure.
"""
import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from typing import List, Tuple, Dict
import re

class GPyTorchSurrogateModel:
    """
    A wrapper for a single-output Gaussian Process model using GPyTorch/BoTorch.
    """
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        self.x_scaler = MinMaxScaler()
        self.y_scaler = StandardScaler()

        self.train_x_scaled = torch.tensor(self.x_scaler.fit_transform(train_x), device=self.device, dtype=self.dtype)
        if train_y.ndim == 1:
            train_y = train_y.reshape(-1, 1)
        self.train_y_scaled = torch.tensor(self.y_scaler.fit_transform(train_y), device=self.device, dtype=self.dtype)

        self.model = SingleTaskGP(self.train_x_scaled, self.train_y_scaled)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def train_model(self):
        """Trains the GP model by maximizing the marginal log likelihood."""
        print("Training the GP model...")
        self.model.train()
        self.model.likelihood.train()
        fit_gpytorch_mll(self.mll)
        print("Training complete.")

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Makes predictions with the trained GP model."""
        self.model.eval()
        self.model.likelihood.eval()
        x_scaled = torch.tensor(self.x_scaler.transform(x), device=self.device, dtype=self.dtype)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model.posterior(x_scaled)
            y_pred_scaled = posterior.mean
            y_std_scaled = torch.sqrt(posterior.variance)
        
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled.cpu().numpy().reshape(-1, 1))
        y_std_val = self.y_scaler.var_[0] if hasattr(self.y_scaler, 'var_') and self.y_scaler.var_ is not None else 1.0
        y_std = y_std_scaled.cpu().numpy() * np.sqrt(y_std_val)

        return y_pred.flatten(), y_std.flatten()

    def validate(self, valid_x: torch.Tensor, valid_y: np.ndarray) -> Dict[str, float]:
        """Validates the model and returns key performance metrics."""
        y_pred, _ = self.predict(valid_x)
        if valid_y.ndim > 1:
            valid_y = valid_y.flatten()
        metrics = {
            "R2": r2_score(valid_y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(valid_y, y_pred)),
            "MAE": mean_absolute_error(valid_y, y_pred),
            "MaxAE": np.max(np.abs(valid_y - y_pred))
        }
        return metrics

    def plot_validation(self, name: str, valid_x: torch.Tensor, valid_y: np.ndarray, output_dir: str = '.'):
        """Creates and saves detailed validation plots."""
        y_pred, y_std = self.predict(valid_x)
        valid_y = valid_y.flatten()
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'

        # Prediction vs. Actual Plot
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.errorbar(valid_y, y_pred, yerr=2*y_std, fmt='o', color='royalblue', ecolor='lightsteelblue', capsize=0, alpha=0.7, label='Prediction with 95% CI')
        ax1.plot([min(valid_y), max(valid_y)], [min(valid_y), max(valid_y)], 'r--', lw=2, label='Perfect Fit')
        ax1.set_xlabel("Actual Values", fontsize=14); ax1.set_ylabel("Predicted Values", fontsize=14)
        ax1.set_title(f"Prediction vs. Actual for {name}", fontsize=16, weight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6); ax1.legend()
        fig1.tight_layout()
        fig1.savefig(os.path.join(output_dir, f"pred_vs_actual_{name}.png"), dpi=300)
        plt.close(fig1)

        # Residual Plot
        residuals = valid_y - y_pred
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(y_pred, residuals, c=residuals, cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel("Predicted Values", fontsize=14); ax2.set_ylabel("Residuals (Actual - Predicted)", fontsize=14)
        ax2.set_title(f"Residuals for {name}", fontsize=16, weight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, f"residuals_{name}.png"), dpi=300)
        plt.close(fig2)
        print(f"Saved validation plots for {name} to {output_dir}")

    def plot_ard_relevance(self, name: str, var_names: List[str], output_dir: str = '.', save_plot: bool = True, ax=None):
        """Calculates and plots the ARD relevance with the preferred low-saturation colormap."""
        if not hasattr(self.model.covar_module, 'base_kernel') or not hasattr(self.model.covar_module.base_kernel, 'lengthscale'):
            print("Warning: ARD lengthscales not found. Skipping plot.")
            return

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['text.usetex'] = False 
        plt.rcParams['mathtext.fontset'] = 'stix'

        lengthscales = self.model.covar_module.base_kernel.lengthscale.squeeze().detach().cpu().numpy()
        relevance = 1.0 / lengthscales
        sorted_indices = np.argsort(relevance)
        sorted_relevance = relevance[sorted_indices]
        sorted_names = [var_names[i] for i in sorted_indices]

        if ax is None:
            fig, own_ax = plt.subplots(figsize=(10, 12))
        else:
            own_ax = ax; fig = own_ax.get_figure()

        # Revert to the low-saturation, diverging colormap (Blue-White-Red)
        norm = plt.Normalize(vmin=sorted_relevance.min(), vmax=sorted_relevance.max())
        colors = plt.cm.seismic(norm(sorted_relevance))

        own_ax.barh(np.arange(len(sorted_names)), sorted_relevance, color=colors, edgecolor='black', linewidth=0.7)
        
        dv_labels = []
        for s_name in sorted_names:
            match = re.search(r'\[(\d+)\]', s_name)
            if match:
                idx = int(match.group(1))
                dv_labels.append(f'DV{idx + 1}')
            else:
                dv_labels.append(s_name)
        
        own_ax.set_yticks(np.arange(len(sorted_names))); own_ax.set_yticklabels(dv_labels, fontsize=12)
        own_ax.set_xlabel('ARD Relevance (1 / Length-scale)', fontsize=14)
        
        # CRITICAL CHANGE: Only add the y-axis label to the first subplot in a row
        if own_ax.get_subplotspec().is_first_col():
            own_ax.set_ylabel('Design Variable', fontsize=14)

        own_ax.set_title(name, fontsize=16, weight='normal')
        own_ax.grid(axis='x', linestyle='--', alpha=0.6); own_ax.spines['top'].set_visible(False); own_ax.spines['right'].set_visible(False)
        own_ax.invert_yaxis()

        if ax is None and save_plot:
            clean_name = re.sub(r'[^a-zA-Z0_]', '', name.split('(')[0].strip()).replace(' ', '_')
            plot_filename = os.path.join(output_dir, f"ard_{clean_name}.pdf")
            fig.tight_layout(); fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved ARD plot to {plot_filename}")

