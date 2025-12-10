# src/problem_structures.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Dict, Any, Optional
import torch
import numpy as np

# Updated import for parallel structure
from src.base_pipeline import AeroPipeline

@dataclass
class InnerLoopHistory:
    online_vars_history: List[np.ndarray] = field(default_factory=list)
    objective_history: List[float] = field(default_factory=list)
    constraint_history: Dict[str, List[float]] = field(default_factory=dict)

@dataclass
class Objective:
    name: str
    eval_function: Callable[[Dict[str, Dict[str, Any]]], float]

@dataclass
class InnerConstraint:
    name: str
    description: str
    eval_function: Callable[[Dict[str, Any]], float]

@dataclass
class InnerOptConfig:
    bounds: List[Tuple[float, float]]
    x0: List[float]
    constraints: List[InnerConstraint] = field(default_factory=list)
    method: str = "SLSQP"
    maxiter: int = 100
    tol: float = 1e-6

@dataclass
class DesignPoint:
    name: str
    pipeline: AeroPipeline
    inner_opt_config: Optional[InnerOptConfig] = None

@dataclass
class GlobalConstraint:
    name: str
    description: str
    eval_function: Callable[[Dict[str, Dict[str, Any]]], float]

@dataclass
class OptimizationConfig:
    offline_dim: int
    online_dim: int
    offline_bounds: torch.Tensor
    design_points: List[DesignPoint]
    objectives: List[Objective]
    constraints: List[GlobalConstraint] = field(default_factory=list)
    user_initial_points: Optional[torch.Tensor] = None
    n_initial_samples: int = 10
    n_bo_iterations: int = 50
    batch_size: int = 1
    num_restarts: int = 10
    raw_samples: int = 512
    
    # --- NEW FIELDS FOR MTGP SUPPORT ---
    model_type: str = "standard"  # "standard" or "mtgp"
    mtgp_rank: int = 1