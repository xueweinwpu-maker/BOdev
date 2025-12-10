# surrogates/pipelines.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import sys
import os
import pickle
from typing import Dict, Any, Optional, List, Tuple

# Ensure we can import from src if running locally
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surrogates.model import GPyTorchSurrogateModel
from src.base_pipeline import AeroPipeline

class SurrogateBackedPipeline(AeroPipeline):
    """
    Standard pipeline for training/prediction on raw data.
    """
    def __init__(self,
                 train_paths: List[str],
                 input_cols: List[str],
                 output_cols: List[str],
                 calculate_derived_metrics: bool = False,
                 alpha_deg: float = 0.0,
                 inner_loop_objective: Optional[str] = None,
                 mach_str: str = "custom",
                 train_on_init: bool = True):

        self.mach_str = mach_str
        self.train_paths = train_paths
        self.inner_loop_objective = inner_loop_objective
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.calculate_derived_metrics = calculate_derived_metrics
        self.alpha_rad = np.deg2rad(alpha_deg) if calculate_derived_metrics else 0.0

        self.models: Dict[str, GPyTorchSurrogateModel] = {}
        if train_on_init:
            self._train_all_models()

    def _transform_coeffs(self, results_dict: Dict[str, float]) -> Tuple[float, float]:
        cl = results_dict['CY'] * np.cos(self.alpha_rad) - results_dict['Cx'] * np.sin(self.alpha_rad)
        cd = results_dict['CY'] * np.sin(self.alpha_rad) + results_dict['Cx'] * np.cos(self.alpha_rad)
        return cl, cd

    def _train_all_models(self):
        print(f"--- Initializing Surrogate Pipeline for Ma = {self.mach_str} ---")
        all_dfs = []
        for path in self.train_paths:
            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                required_cols = self.input_cols + self.output_cols
                df.dropna(subset=required_cols, inplace=True)
                all_dfs.append(df)
            except FileNotFoundError:
                print(f"WARNING: Could not find training data at {path}.")

        if not all_dfs:
            print(f"FATAL: No training data could be loaded for Ma = {self.mach_str}. Exiting.")
            sys.exit(1)

        full_train_df = pd.concat(all_dfs, ignore_index=True)
        train_x_torch = torch.tensor(full_train_df[self.input_cols].values)

        for coeff_name in self.output_cols:
            print(f"Training model for {coeff_name}...")
            train_y_torch = torch.tensor(full_train_df[coeff_name].values).unsqueeze(-1)
            model = GPyTorchSurrogateModel(train_x_torch, train_y_torch)
            model.train_model()
            self.models[coeff_name] = model
        print(f"--- Pipeline for Ma = {self.mach_str} is ready. ---")

    def run(self, offline_vars: np.ndarray, online_vars: Optional[np.ndarray]) -> Dict[str, Any]:
        if online_vars is not None:
            full_vars = np.concatenate([offline_vars, online_vars])
        else:
            full_vars = offline_vars

        input_tensor = torch.tensor(full_vars, dtype=torch.double).reshape(1, -1)
        results = {}
        for coeff_name in self.output_cols:
            pred, _ = self.models[coeff_name].predict(input_tensor)
            results[coeff_name] = pred.item()

        if self.calculate_derived_metrics:
            cl, cd = self._transform_coeffs(results)
            results["CL"] = cl
            results["CD"] = cd
            if 'CMZ' in results and abs(results.get('CY', 0)) > 1e-6:
                results["Xcp"] = results['CMZ'] / results['CY']
            else:
                results["Xcp"] = 0.0
        else:
            if 'cl' in results: results['CL'] = results['cl']
            if 'cd' in results: results['CD'] = results['cd']

        if 'CL' in results and 'CD' in results:
            results["K"] = results["CL"] / results["CD"] if results["CD"] > 1e-6 else 0.0
            results["-CL"] = -results["CL"]
            results["-CD"] = -results["CD"]
            results["-K"] = -results["K"]

        return results

class TruthPipeline(AeroPipeline):
    """
    Acts as the 'Simulator' for benchmarking.
    Loads a pre-trained .pkl and enforces fixed ratio[15].
    """
    def __init__(self, pipeline_name: str, fixed_ratio_15: float = 1.0):
        self.fixed_ratio_15 = fixed_ratio_15
        self.pipeline_name = pipeline_name
        model_path = os.path.join("trained_models", f"pipeline_{pipeline_name}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FATAL: Truth model not found at '{model_path}'.\n"
                f"Please run 'tools/run_validation.py' with run_name='{pipeline_name}' first."
            )
        print(f"--- [TruthMachine] Loading Oracle Model: {pipeline_name} ---")
        with open(model_path, 'rb') as f:
            self.internal_pipeline = pickle.load(f)
            
    def run(self, offline_vars: np.ndarray, online_vars: Optional[np.ndarray]) -> Dict[str, Any]:
        # Validate input dimension (Optimizer gives 15 vars)
        if offline_vars.shape[0] != 15:
             raise ValueError(f"TruthPipeline expects 15 input variables, got {offline_vars.shape[0]}.")
        
        # Append fixed sweep
        full_input = np.append(offline_vars, self.fixed_ratio_15)
        
        # Query
        raw_results = self.internal_pipeline.run(full_input, None)
        results = raw_results.copy()
        
        # Standardize Physics (L/D, Xcp)
        alpha_rad = 0.0 
        if 'Cx' in results and 'CY' in results:
            cl = results['CY'] * np.cos(alpha_rad) - results['Cx'] * np.sin(alpha_rad)
            cd = results['CY'] * np.sin(alpha_rad) + results['Cx'] * np.cos(alpha_rad)
            k = cl / cd if abs(cd) > 1e-8 else 0.0
            results['CL'] = cl
            results['CD'] = cd
            results['K'] = k
            
        if 'CMz' in results and 'CY' in results:
            xcp = results['CMz'] / results['CY'] if abs(results['CY']) > 1e-6 else 0.0
            results['Xcp'] = xcp

        return results