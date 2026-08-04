"""Real figures from measured runs. One figure per paper block; always reads
metrics.json produced by an actual run of gcl.experiment (nothing faked)."""
from __future__ import annotations

import json, os
from typing import Any, Dict, List


def load_metrics(out_dir: str) -> Dict[str, Any]:
    with open(os.path.join(out_dir, "metrics.json")) as f:
        return json.load(f)


def plot_family_curves(reports: Dict[str, Any], out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=150)
    for name, d in reports["learners"].items():
        fc = d.get("family_curve", [])
        if not fc:
            continue
        ax.plot(range(1, len(fc)+1), fc, marker="o", label=name)
    ax.set_xlabel("Task family (stream position)")
    ax.set_ylabel("Mean verified reward during family")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    path = os.path.join(out_dir, "fig_family_curves.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_final_vs_zero(reports: Dict[str, Any], out_dir: str) -> str:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    first = next(iter(reports["learners"].values()), {})
    fams = None
    # infer family count from any learner R_pairs
    nF = max((len(d["R_pairs"]["final"]) for d in reports["learners"].values()), default=0)
    inds = np.arange(nF)
    width = 0.8 / max(1, len(reports["learners"]))
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=150)
    for k, (name, d) in enumerate(reports["learners"].items()):
        final = d["R_pairs"]["final"]
        zero = d["R_pairs"]["zero_shot"]
        x = inds + k * width
        ax.bar(x - width/4, zero, width/2, alpha=0.5, label=f"{name} zero" if k == 0 else None, color=f"C{k*2}")
        ax.bar(x + width/4, final, width/2, alpha=0.9, label=f"{name} final", color=f"C{k*2+1}")
    ax.set_xticks(inds + width*(len(reports["learners"])-1)/2)
    ax.set_xticklabels([f"F{i+1}" for i in inds])
    ax.set_ylim(0, 1.0); ax.set_ylabel("Pass@1 (verified)")
    ax.set_title("Family accuracy before (zero-shot) vs after stream (final)")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=6, ncol=2)
    path = os.path.join(out_dir, "fig_final_vs_zero.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_frontier(reports: Dict[str, Any], out_dir: str) -> str:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=150)
    for name, d in reports["learners"].items():
        r = d["report"]
        ax.scatter(d["updates"], r["acc"], s=60, label=name)
        ax.annotate(f"{r['forgetting']:.2f}", (d["updates"], r["acc"]), fontsize=7)
    ax.set_xlabel("# real updates (cost)"); ax.set_ylabel("ACC (final avg)")
    ax.set_title("Stability–plasticity frontier")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    path = os.path.join(out_dir, "fig_frontier.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def write_tables(reports: Dict[str, Any], out_dir: str) -> str:
    lines = ["# Results (auto-generated from metrics.json)", ""]
    cfg = reports.get("config", {})
    lines.append(f"Model: `{cfg.get('model_name')}`  | LoRA r={cfg.get('lora_r')}  | seed {cfg.get('seed')}")
    canary = reports.get("canary", {})
    lines.append(f"Anti-contamination overlap: {canary.get('overlap')} | families: {len([d for d in reports['learners'].values()])}")
    lines.append("")
    lines.append("| Learner | ACC | BWT | FWT | Forgetting | AUC | Updates | Rollbacks | Frontier |")
    lines.append("|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for name, d in reports["learners"].items():
        r = d["report"]
        lines.append(f"| {name} | {r['acc']:.3f} | {r['bwt']:+.3f} | {r['fwt']:+.3f} | {r['forgetting']:.3f} | {r['auc']:.3f} | {d['updates']} | {d['rollbacks']} | {d['frontier_score']:+.3f} |")
    with open(os.path.join(out_dir, "TABLES.md"), "w") as f:
        f.write("\n".join(lines))
    return os.path.join(out_dir, "TABLES.md")


def render_all(out_dir: str) -> List[str]:
    reports = load_metrics(out_dir)
    made = []
    for fn in (plot_family_curves, plot_final_vs_zero, plot_frontier):
        try:
            made.append(fn(reports, out_dir))
        except Exception as e:
            made.append(f"ERR {fn.__name__}: {e}")
    made.append(write_tables(reports, out_dir))
    return made
