"""
NoveltyGate for compute allocation based on trajectory novelty, NLL surprise scoring (SuRe 2025), and reward gating.
"""

from typing import List, Optional, Any
import numpy as np


class NoveltyGate:
    """
    Computes novelty relative to stored memory and adaptive thresholding for update gating.
    Includes SuRe (2025) NLL-based surprise scoring.
    """

    def __init__(
        self,
        memory: Any = None,
        base_threshold: float = 0.3,
        window_size: int = 20
    ):
        self.memory = memory
        self.base_threshold = base_threshold
        self.window_size = window_size
        self.recent_novelty_scores: List[float] = []

    def compute_novelty(self, prompt: str) -> float:
        if self.memory is None:
            return 1.0

        if hasattr(self.memory, "get_nearest_distance"):
            score = self.memory.get_nearest_distance(prompt)
        elif hasattr(self.memory, "query"):
            results = self.memory.query(prompt, top_k=1)
            if not results:
                # No memories yet — unknown, not novel.  Returning 1.0 here
                # caused every prompt to look maximally novel when memory was
                # empty, which broke downstream threshold logic.
                score = 0.0
            else:
                score = 0.5
        else:
            score = 0.0

        self.recent_novelty_scores.append(score)
        if len(self.recent_novelty_scores) > self.window_size:
            self.recent_novelty_scores.pop(0)

        return float(score)

    def compute_nll_surprise(self, log_probs: List[float]) -> float:
        """
        SuRe (2025): Surprise-driven score based on Negative Log Likelihood (NLL).
        Higher NLL indicates model uncertainty/surprise.
        """
        if not log_probs:
            return 0.5
        nll = -float(np.mean(log_probs))
        return float(1.0 / (1.0 + np.exp(-nll)))  # Sigmoid scaled to [0, 1]

    def get_adaptive_threshold(self) -> float:
        if not self.recent_novelty_scores:
            return self.base_threshold
        arr = np.array(self.recent_novelty_scores)
        thresh = float(np.mean(arr) + 0.5 * np.std(arr))
        return max(0.1, min(0.9, thresh))

    def should_update(self, prompt: str, reward: float, min_reward: float = 0.8) -> bool:
        """
        Fix B2: Returns True if trajectory is high-reward OR sufficiently novel (>= 0.8),
        allowing learning discovery on novel hard tasks even if initial reward is low.
        """
        novelty = self.compute_novelty(prompt)
        threshold = self.get_adaptive_threshold()

        # High novelty override to allow learning discovery on hard novel tasks
        if novelty >= 0.8:
            return True
        if reward < min_reward:
            return False
        return novelty >= threshold
