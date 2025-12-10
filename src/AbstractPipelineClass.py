# -*- coding: utf-8 -*-
"""
This module defines the abstract base class for all aerodynamic evaluation pipelines.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class AeroPipeline(ABC):
    """
    Abstract base class (ABC) for an evaluation pipeline.
    All pipeline classes, whether based on CFD, analytical functions, or
    surrogate models, must inherit from this class and implement the run method.
    """
    @abstractmethod
    def run(self, offline_vars: np.ndarray, online_vars: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Executes the evaluation for a given set of design variables.

        Args:
            offline_vars: A NumPy array of the offline (fixed) design variables.
            online_vars: A NumPy array of the online (in-flight) design variables.

        Returns:
            A dictionary containing the aerodynamic results (e.g., CL, CD, Xcp)
            and any other relevant outputs like objective or constraint values.
        """
        pass
