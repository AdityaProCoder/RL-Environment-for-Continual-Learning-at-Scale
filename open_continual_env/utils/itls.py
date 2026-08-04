"""
Inference-Time Learning Signal (ITLS) - Novel Mathematical Contribution D3.
Measures parameter sensitivity during forward passes: ITLS = -∇log p(y|x,θ) · Δθ.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class ITLSSignal:
    """
    Computes Inference-Time Learning Signal (ITLS) to measure model uncertainty
    and parameter sensitivity directly during inference forward passes.
    """

    @staticmethod
    def compute_itls(log_probs: List[float], grad_norms: List[float]) -> float:
        """
        Calculates ITLS signal scalar: -mean(log_probs) * mean(grad_norms).
        Higher values indicate high learning potential / surprise.
        """
        if not log_probs or not grad_norms:
            return 1.0
        nll = -float(np.mean(log_probs))
        g_norm = float(np.mean(grad_norms))
        signal = nll * g_norm
        return float(1.0 / (1.0 + np.exp(-signal)))  # Sigmoid normalized [0, 1]
