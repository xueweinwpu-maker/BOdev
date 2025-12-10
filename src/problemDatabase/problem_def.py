# -*- coding: utf-8 -*-
"""
Module for defining the optimization problem.
This file serves as the master configuration for a specific optimization run.
**This version is updated to use the data-driven SurrogateBackedPipelines.**
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Dict, Any, Optional
import torch
from collections import defaultdict

# Use the correct import path for our new base class and surrogate pipeline
from src.base_pipeline import AeroPipeline
from surrogates.pipelines import SurrogateBackedPipeline

@dataclass
class Objective:
    """Defines a single objective for the outer optimization problem."""
    name: str  # The name to be used for plotting and logging
    eval_function: Callable[[Dict[str, Dict[str, Any]]], float]

@dataclass
class InnerConstraint:
    """Defines an inner-loop constraint by referencing an index from the pipeline's output."""
    name: str
    index: int # Index in the 'constraints' list returned by the pipeline

@dataclass
class InnerOptConfig:
    """Configuration for the inner-loop optimization of online variables."""
    bounds: List[Tuple[float, float]]
    x0: List[float]
    constraints: List[InnerConstraint] = field(default_factory=list)
    method: str = "SLSQP"
    maxiter: int = 100
    tol: float = 1e-6

@dataclass
class DesignPoint:
    """Defines a single flight condition and its associated settings."""
    name: str
    pipeline: AeroPipeline
    inner_opt_config: Optional[InnerOptConfig] = None

@dataclass
class GlobalConstraint:
    """
    Defines a single GLOBAL constraint for the outer optimization problem.
    The eval_function must be formulated such that a value >= 0 indicates a
    satisfied constraint.
    """
    name: str
    eval_function: Callable[[Dict[str, Dict[str, Any]]], float]

@dataclass
class OptimizationConfig:
    """The main container for the entire problem definition."""
    offline_dim: int
    online_dim: int
    offline_bounds: torch.Tensor # A 2 x d tensor: row 0 = lower, row 1 = upper
    design_points: List[DesignPoint]
    objectives: List[Objective]
    constraints: List[GlobalConstraint] = field(default_factory=list)
    n_initial_samples: int = 10
    n_bo_iterations: int = 25
    batch_size: int = 1

def define_problem() -> OptimizationConfig:
    """
    Defines and assembles the complete optimization problem configuration
    using the surrogate-backed pipelines.
    """
    # Instantiate the data-driven pipelines
    low_speed_pipeline = SurrogateBackedPipeline(
        mach_str="0.4",
        train_path="data/train_dataM04.csv"
    )
    
    # TODO: Add the Ma=3.0 pipeline once data is available
    # super_sonic_pipeline = SurrogateBackedPipeline(...)

    high_speed_pipeline = SurrogateBackedPipeline(
        mach_str="10.0",
        train_path="data/train_dataM10.csv"
    )

    design_points = [
        DesignPoint(
            name="Subsonic_Ma0.4",
            pipeline=low_speed_pipeline,
            inner_opt_config=InnerOptConfig(
                bounds=[(0.0, 30.0)], x0=[15.0],
                # K_min constraint will be handled as a global constraint for simplicity
            )
        ),
        # TODO: Uncomment and configure when Ma=3.0 data is ready
        # DesignPoint(
        #     name="Supersonic_Ma3.0",
        #     pipeline=super_sonic_pipeline,
        #     inner_opt_config=InnerOptConfig(bounds=[(30.0, 60.0)], x0=[45.0])
        # ),
        DesignPoint(
            name="Hypersonic_Ma10.0",
            pipeline=high_speed_pipeline,
            inner_opt_config=InnerOptConfig(bounds=[(50.0, 75.0)], x0=[65.0])
        ),
    ]

    # Define objectives based on the output of the surrogate pipelines
    objectives = [
        Objective(name="Max CL (Ma 0.4)", eval_function=lambda res_map: res_map["Subsonic_Ma0.4"]["CL"]),
        # Objective(name="Max L/D (Ma 3.0)", eval_function=lambda res_map: res_map["Supersonic_Ma3.0"]["K"]),
        Objective(name="Max L/D (Ma 10.0)", eval_function=lambda res_map: res_map["Hypersonic_Ma10.0"]["K"]),
    ]

    # Define global constraints
    # NOTE: The original K_min_subsonic constraint is better handled here
    # to avoid needing a constrained inner-loop solver for this setup.
    constraints = [
        GlobalConstraint(
            name="K_min_subsonic",
            # Constraint is K >= 4.0, which means K - 4.0 >= 0
            eval_function=lambda res_map: res_map["Subsonic_Ma0.4"]["K"] - 4.0
        ),
        # TODO: Uncomment when Ma=3.0 data is ready
        # GlobalConstraint(
        #     name="Xcp_Stability",
        #     # Constraint is |Xcp_3 - Xcp_10| <= 0.05 => 0.05 - |Xcp_3 - Xcp_10| >= 0
        #     eval_function=lambda res_map: 0.05 - abs(res_map["Supersonic_Ma3.0"]["Xcp"] - res_map["Hypersonic_Ma10.0"]["Xcp"])
        # )
    ]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.double

    # The surrogate model was trained on all 16 vars.
    # The optimizer's design space is the 15 offline vars. The 1 online var is handled by the inner loop.
    problem_config = OptimizationConfig(
        offline_dim=15,
        online_dim=1,
        # Bounds for the 15 FFD control points
        offline_bounds=torch.tensor([[-2.0] * 15, [2.0] * 15], dtype=DTYPE, device=DEVICE),
        design_points=design_points,
        objectives=objectives,
        constraints=constraints,
        n_initial_samples=10,
        n_bo_iterations=50,
        batch_size=1
    )

    _validate_configuration(problem_config)
    return problem_config

def _validate_configuration(config: OptimizationConfig):
    """Performs critical checks on the problem configuration."""
    print("Validating problem configuration...")

    if config.offline_bounds.shape != (2, config.offline_dim):
        raise ValueError(f"Shape of offline_bounds is {config.offline_bounds.shape}, but expected (2, {config.offline_dim}).")

    dp_names = [dp.name for dp in config.design_points]
    if len(dp_names) != len(set(dp_names)):
        raise ValueError("DesignPoint names must be unique.")

    for dp in config.design_points:
        if dp.inner_opt_config:
            cfg = dp.inner_opt_config
            if len(cfg.x0) != len(cfg.bounds) or len(cfg.x0) != config.online_dim:
                raise ValueError(f"In '{dp.name}', dimension of x0/bounds does not match online_dim.")

    # Test that objectives and constraints can be evaluated with dummy data
    dp_name_set = set(dp_names)
    dummy_results = {name: defaultdict(float) for name in dp_name_set}
    
    for obj in config.objectives:
        try:
            obj.eval_function(dummy_results)
        except KeyError as e:
            raise KeyError(f"Objective '{obj.name}' uses a name ('{e.args[0]}') that is not a defined DesignPoint name.")

    for con in config.constraints:
        try:
            con.eval_function(dummy_results)
        except KeyError as e:
            raise KeyError(f"Global Constraint '{con.name}' uses a name ('{e.args[0]}') that is not a defined DesignPoint name.")

    print("Configuration is valid.")

if __name__ == '__main__':
    print("--- Testing problem_def.py ---")
    config = define_problem()
    print(f"\nLoaded configuration for a problem with {config.offline_dim} offline and {config.online_dim} online variables.")
    print(f"Found {len(config.design_points)} design points.")
    print(f"Found {len(config.objectives)} objectives:")
    for obj in config.objectives:
        print(f"- {obj.name}")
    print(f"Found {len(config.constraints)} global constraints:")
    for con in config.constraints:
        print(f"- {con.name}")
    print("\n--- problem_def.py test PASSED ---")

