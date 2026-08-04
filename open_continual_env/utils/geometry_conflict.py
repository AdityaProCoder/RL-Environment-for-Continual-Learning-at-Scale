"""
Geometry Conflict (GCWM) - Task Compatibility via Wasserstein Distance (May 2026).
Reference: Wang et al. (arXiv:2605.09608).
"""

from typing import List, Dict, Any, Optional
import numpy as np


class GeometryConflict:
    """
    Estimates covariance geometry conflict between task update matrices using Gaussian Wasserstein barycenters.
    """

    @staticmethod
    def compute_wasserstein_distance(cov1: np.ndarray, cov2: np.ndarray) -> float:
        """
        Calculates 2-Wasserstein distance W_2(N(0, cov1), N(0, cov2)) between Gaussian distributions.
        W_2^2 = Tr(cov1) + Tr(cov2) - 2 * Tr((cov1^{1/2} cov2 cov1^{1/2})^{1/2}).
        """
        if cov1.shape != cov2.shape:
            return 1.0

        tr1 = float(np.trace(cov1))
        tr2 = float(np.trace(cov2))

        try:
            from scipy.linalg import sqrtm
            cov1_sqrt = sqrtm(cov1)
            middle = np.matmul(np.matmul(cov1_sqrt, cov2), cov1_sqrt)
            middle_sqrt = sqrtm(middle)
            cross_tr = float(np.real(np.trace(middle_sqrt)))
        except Exception:
            evals1 = np.maximum(0.0, np.linalg.eigvalsh(cov1))
            evals2 = np.maximum(0.0, np.linalg.eigvalsh(cov2))
            cross_tr = float(np.sum(np.sqrt(evals1 * evals2)))

        dist_sq = max(0.0, tr1 + tr2 - 2.0 * cross_tr)
        return float(np.sqrt(dist_sq))

    @classmethod
    def is_compatible(cls, cov_task_a: np.ndarray, cov_task_b: np.ndarray, threshold: float = 0.5) -> bool:
        dist = cls.compute_wasserstein_distance(cov_task_a, cov_task_b)
        return dist <= threshold
