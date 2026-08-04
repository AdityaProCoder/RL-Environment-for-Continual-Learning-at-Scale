"""
Metrics suite for evaluation of continual learning algorithms in OpenContinualEnv.
"""

import math
import statistics
from typing import Any, Dict, List, Optional, Union


class ContinualMetrics:
    """Computes standardized continual learning performance & forgetting metrics."""

    @staticmethod
    def task_success_rate(pass_results: Any) -> float:
        """Calculate pass rate across evaluated tasks."""
        if not pass_results:
            return 0.0
        if isinstance(pass_results, dict):
            pass_results = list(pass_results.values())
        
        total = len(pass_results)
        if total == 0:
            return 0.0

        successes = 0
        for item in pass_results:
            if isinstance(item, bool):
                if item:
                    successes += 1
            elif isinstance(item, (int, float)):
                if float(item) >= 0.8 or float(item) == 1.0:
                    successes += 1
            elif isinstance(item, dict):
                if item.get("success") is True or item.get("pass_rate") == 1.0 or item.get("reward", 0.0) >= 0.8:
                    successes += 1
        return float(successes / total)

    @staticmethod
    def sample_efficiency(steps_to_success: List[Any], max_steps: int = 10) -> float:
        """Calculate average sample efficiency or mean steps."""
        if not steps_to_success:
            return 0.0
        numeric_steps = [float(s) for s in steps_to_success if isinstance(s, (int, float))]
        if not numeric_steps:
            return 0.0
        return float(sum(numeric_steps) / len(numeric_steps))

    @staticmethod
    def learning_speed(history: List[float], max_steps: int = 10) -> float:
        """Calculate learning speed as delta/improvement across history."""
        if not history or len(history) < 2:
            return 0.0
        if history[0] == history[-1]:
            return 0.0
        return float(history[-1] - history[0])

    @staticmethod
    def catastrophic_forgetting(
        initial_performances: Any,
        final_performances: Optional[List[float]] = None,
    ) -> float:
        """
        Calculate Catastrophic Forgetting.
        Supports 2D performance matrix R or two 1D performance lists.
        Clips negative drops (improvements) to 0.0.
        """
        if not initial_performances:
            return 0.0

        if isinstance(initial_performances, list) and initial_performances:
            first_elem = initial_performances[0]
            if isinstance(first_elem, list):
                R = initial_performances
                N = len(R)
                if N <= 1 or len(first_elem) == 0:
                    return 0.0
                forgetting_sum = 0.0
                valid_count = 0
                for i in range(N - 1):
                    if i < len(R[i]) and i < len(R[N - 1]):
                        drop = R[i][i] - R[N - 1][i]
                        forgetting_sum += max(0.0, float(drop))
                        valid_count += 1
                return float(forgetting_sum / valid_count) if valid_count > 0 else 0.0

        if not initial_performances or final_performances is None or len(initial_performances) != len(final_performances):
            return 0.0
        diffs = [max(0.0, float(i - f)) for i, f in zip(initial_performances, final_performances)]
        return float(sum(diffs) / len(diffs))

    @staticmethod
    def backward_transfer(
        initial_performances: Any,
        final_performances: Optional[List[float]] = None,
    ) -> float:
        """
        Calculate Backward Transfer (BWT).
        Signed value measuring improvement or drop relative to initial accuracy.
        """
        if not initial_performances:
            return 0.0

        if isinstance(initial_performances, list) and initial_performances:
            first_elem = initial_performances[0]
            if isinstance(first_elem, list):
                R = initial_performances
                N = len(R)
                if N <= 1 or len(first_elem) == 0:
                    return 0.0
                bwt_sum = 0.0
                valid_count = 0
                for i in range(N - 1):
                    if i < len(R[i]) and i < len(R[N - 1]):
                        bwt_sum += (R[N - 1][i] - R[i][i])
                        valid_count += 1
                return float(bwt_sum / valid_count) if valid_count > 0 else 0.0

        if not initial_performances or final_performances is None or len(initial_performances) != len(final_performances):
            return 0.0
        diffs = [float(f - i) for i, f in zip(initial_performances, final_performances)]
        return float(sum(diffs) / len(diffs))

    @staticmethod
    def forward_transfer(
        baseline_performances: Any,
        adapted_performances: Optional[List[float]] = None,
        baseline_accuracies: Optional[List[float]] = None,
    ) -> float:
        """Calculate Forward Transfer (FWT)."""
        if not baseline_performances:
            return 0.0

        if isinstance(baseline_performances, list) and baseline_performances:
            first_elem = baseline_performances[0]
            if isinstance(first_elem, list):
                R = baseline_performances
                N = len(R)
                if N <= 1 or len(first_elem) == 0:
                    return 0.0
                fwt_sum = 0.0
                valid_count = 0
                for i in range(1, N):
                    if i < len(R[i - 1]):
                        b_acc = baseline_accuracies[i] if (baseline_accuracies and i < len(baseline_accuracies)) else 0.0
                        fwt_sum += (R[i - 1][i] - b_acc)
                        valid_count += 1
                return float(fwt_sum / valid_count) if valid_count > 0 else 0.0

        if not baseline_performances or adapted_performances is None or len(baseline_performances) != len(adapted_performances):
            return 0.0
        diffs = [float(a - b) for b, a in zip(baseline_performances, adapted_performances)]
        return float(sum(diffs) / len(diffs))

    @staticmethod
    def weight_stability(rewards_over_time: List[float]) -> float:
        """
        Calculate stability using coefficient of variation: 1.0 - (std(rewards) / max(mean(rewards), 1e-8)).
        Higher value (closer to 1.0) means higher performance stability. Returns 1.0 if empty or all zero.
        """
        if not rewards_over_time or all(r == 0.0 for r in rewards_over_time):
            return 1.0
        n = len(rewards_over_time)
        mean_val = float(sum(rewards_over_time) / n)
        if mean_val <= 1e-8:
            return 1.0
        variance = sum((r - mean_val) ** 2 for r in rewards_over_time) / n
        std_val = float(variance ** 0.5)
        cov = std_val / mean_val
        return float(max(0.0, min(1.0, 1.0 - cov)))


    @staticmethod
    def stability_index(rewards_over_time: List[float]) -> float:
        """Alias for weight_stability."""
        return ContinualMetrics.weight_stability(rewards_over_time)

    @staticmethod
    def compute_learning_efficiency_frontier(sample_efficiencies: List[float], retentions: List[float]) -> float:
        """
        Novel Contribution D2: Learning Efficiency Frontier (LEF).
        Calculates Pareto AUC metric over (Sample_Efficiency, Retention) space.
        """
        if not sample_efficiencies or not retentions or len(sample_efficiencies) != len(retentions):
            return 0.0
        pairs = sorted(zip(sample_efficiencies, retentions), key=lambda x: x[0])
        auc = 0.0
        for i in range(1, len(pairs)):
            dx = pairs[i][0] - pairs[i - 1][0]
            y_avg = (pairs[i][1] + pairs[i - 1][1]) / 2.0
            auc += dx * y_avg
        return float(max(0.0, auc))


    @staticmethod
    def compute_all_metrics(
        task_results: Optional[List[Any]] = None,
        steps_list: Optional[List[Any]] = None,
        history: Optional[List[float]] = None,
        matrix: Optional[List[List[float]]] = None,
        rewards: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Computes all standard metrics and returns a summary dictionary."""
        return {
            "task_success_rate": ContinualMetrics.task_success_rate(task_results or []),
            "sample_efficiency": ContinualMetrics.sample_efficiency(steps_list or []),
            "learning_speed": ContinualMetrics.learning_speed(history or []),
            "catastrophic_forgetting": ContinualMetrics.catastrophic_forgetting(matrix or []),
            "backward_transfer": ContinualMetrics.backward_transfer(matrix or []),
            "forward_transfer": ContinualMetrics.forward_transfer(matrix or []),
            "weight_stability": ContinualMetrics.weight_stability(rewards or []),
        }


compute_all_metrics = ContinualMetrics.compute_all_metrics
task_success_rate = ContinualMetrics.task_success_rate
sample_efficiency = ContinualMetrics.sample_efficiency
learning_speed = ContinualMetrics.learning_speed
catastrophic_forgetting = ContinualMetrics.catastrophic_forgetting
backward_transfer = ContinualMetrics.backward_transfer
forward_transfer = ContinualMetrics.forward_transfer
weight_stability = ContinualMetrics.weight_stability
stability_index = ContinualMetrics.stability_index
ContinualMetricsCalculator = ContinualMetrics
