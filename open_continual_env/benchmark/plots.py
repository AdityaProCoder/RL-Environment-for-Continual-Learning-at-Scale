"""
Plotting and visualization tools for OpenContinualEnv benchmark results.
"""

from typing import Any, Dict, List, Optional
import os


class BenchmarkPlotter:
    """Generates ASCII/Markdown metric tables and plots for benchmark reporting."""

    @staticmethod
    def generate_markdown_table(benchmark_results: Dict[str, Any]) -> str:
        """Generate a formatted Markdown summary table of benchmark results."""
        results = benchmark_results.get("results", [])
        table = "# OpenContinualEnv Benchmark Summary\n\n"
        table += f"- **Workers**: {benchmark_results.get('num_workers', 1)}\n"
        table += f"- **Total Tasks**: {benchmark_results.get('total_tasks', 0)}\n"
        table += f"- **Pass Rate**: {benchmark_results.get('pass_rate', 0.0) * 100:.1f}%\n"
        table += f"- **Mean Reward**: {benchmark_results.get('mean_reward', 0.0):.3f}\n\n"

        table += "| Task ID | Status | Pass Rate | Reward | Exec Time (s) |\n"
        table += "|---|---|---|---|---|\n"

        for r in results:
            status = "PASSED" if r.get("success") else "FAILED"
            table += f"| {r.get('task_id')} | {status} | {r.get('pass_rate', 0.0)*100:.0f}% | {r.get('reward', 0.0):.2f} | {r.get('execution_time', 0.0):.3f} |\n"

        return table

    @staticmethod
    def save_report(benchmark_results: Dict[str, Any], output_path: str) -> str:
        """Save benchmark summary markdown to disk."""
        report = BenchmarkPlotter.generate_markdown_table(benchmark_results)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        return report


def plot_learning_curve(rewards: List[float], output_path: Optional[str] = None) -> str:
    """Generate ASCII/text learning curve plot."""
    text = "Learning Curve:\n"
    for i, r in enumerate(rewards, 1):
        bar = "#" * int(r * 20)
        text += f"Step {i:2d}: [{bar:<20}] {r:.2f}\n"

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def plot_forgetting_matrix(matrix: List[List[float]], output_path: Optional[str] = None) -> str:
    """Generate ASCII/text forgetting matrix plot."""
    text = "Forgetting Matrix:\n"
    for row in matrix:
        text += " ".join(f"{val:.2f}" for val in row) + "\n"

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
