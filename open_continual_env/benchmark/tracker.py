"""
Hugging Face TrackIO & Visualization Tracker for OpenContinualEnv.

Uses Hugging Face TrackIO (pip install trackio) to log ML metrics, episode rewards,
catastrophic forgetting curves, and launch local interactive Gradio dashboards.
"""

import os
from typing import Any, Dict, List, Optional

try:
    import trackio
    HAS_TRACKIO = True
except ImportError:
    HAS_TRACKIO = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ExperimentTracker:
    """
    Hugging Face TrackIO & visualization tracker for OpenContinualEnv experiments.
    """

    def __init__(
        self,
        project_name: str = "open_continual_env",
        experiment_name: str = "continual_benchmark",
        config: Optional[Dict[str, Any]] = None,
        use_trackio: bool = True,
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config or {}
        self.use_trackio = use_trackio and HAS_TRACKIO
        self.step_count = 0
        self._history: List[Dict[str, Any]] = []

        if self.use_trackio:
            try:
                trackio.init(project=self.project_name, name=self.experiment_name, config=self.config)
            except Exception:
                self.use_trackio = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics for the current step."""
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1

        record = {"step": self.step_count, **metrics}
        self._history.append(record)

        if self.use_trackio:
            try:
                trackio.log(metrics, step=self.step_count)
            except Exception:
                pass

    def finish() -> None:
        """Close active TrackIO run."""
        if self.use_trackio:
            try:
                trackio.finish()
            except Exception:
                pass

    def generate_plots(self, output_dir: str = "benchmark_results/plots") -> Dict[str, str]:
        """Generate Matplotlib performance & forgetting plots."""
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        if not HAS_MATPLOTLIB or not self._history:
            return paths

        steps = [h["step"] for h in self._history if "reward" in h]
        rewards = [h["reward"] for h in self._history if "reward" in h]

        if steps and rewards:
            plt.figure(figsize=(8, 4))
            plt.plot(steps, rewards, marker="o", color="#2b5c8f", linewidth=2, label="Reward")
            plt.title("OpenContinualEnv — Reward Trajectory over Steps")
            plt.xlabel("Step")
            plt.ylabel("Reward")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            reward_plot_path = os.path.join(output_dir, "reward_trajectory.png")
            plt.savefig(reward_plot_path, dpi=150)
            plt.close()
            paths["reward_trajectory"] = reward_plot_path

        return paths
