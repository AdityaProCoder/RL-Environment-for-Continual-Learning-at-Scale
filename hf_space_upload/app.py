import os
import sys
import json
import time
import subprocess
import threading
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

# 1. Launch background experiment execution
def run_notebook():
    print("STARTING BACKGROUND NOTEBOOK EXECUTION...", flush=True)
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=-1",
        "--output", "executed_notebook.ipynb",
        "kaggle_continual_learning.ipynb"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("NOTEBOOK FINISHED WITH EXIT CODE:", res.returncode, flush=True)
    if res.returncode != 0:
        print("NOTEBOOK ERROR LOGS:\n", res.stderr[-2000:], flush=True)

thread = threading.Thread(target=run_notebook, daemon=True)
thread.start()

# 2. Live Data Extraction
def get_live_data():
    paths = [
        "runs/main/hf_main/metrics_partial.json",
        "runs/main/hf_main/metrics.json",
        "/app/runs/main/hf_main/metrics_partial.json",
        "/app/runs/main/hf_main/metrics.json",
    ]
    data = None
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                break
            except Exception:
                pass

    if not data or "learners" not in data:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#1f2937")
        ax.text(0.5, 0.5, "⏳ Qwen3.5-2B Model Executing...\nEvaluating Baseline & Online LoRA", 
                ha="center", va="center", fontsize=13, color="#60a5fa", weight="bold")
        ax.axis("off")
        fig.tight_layout()
        return "🚀 Experiment Active: Initializing Qwen3.5-2B Baseline...", pd.DataFrame(), fig

    learners_dict = data.get("learners", {})
    rows = []
    learner_names = []
    accs = []
    bwts = []
    for name, info in learners_dict.items():
        rep = info.get("report", {})
        acc = rep.get("acc", 0.0)
        bwt = rep.get("bwt", 0.0)
        forget = rep.get("forgetting", 0.0)
        upd = info.get("updates", 0)
        rb = info.get("rollbacks", 0)
        wall = info.get("wallclock_s", 0)
        rows.append({
            "Learner": name,
            "Accuracy (ACC)": f"{acc:.3f}",
            "BWT": f"{bwt:+.3f}",
            "Forgetting": f"{forget:.3f}",
            "Updates": upd,
            "Rollbacks": rb,
            "Time (s)": round(wall, 1)
        })
        learner_names.append(name)
        accs.append(acc)
        bwts.append(bwt)

    df = pd.DataFrame(rows)

    # Matplotlib Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#111827")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1f2937")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#374151")

    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ef4444"][:len(learner_names)]
    ax1.bar(learner_names, accs, color=colors)
    ax1.set_title("Final Accuracy (ACC)")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Accuracy")

    ax2.bar(learner_names, bwts, color=colors)
    ax2.set_title("Backward Transfer (BWT)")
    ax2.set_ylabel("BWT Score")

    fig.tight_layout()
    status_text = f"✅ Progress: {len(learners_dict)}/5 Learners Finished | Active Hardware: 1x Nvidia L4 GPU"
    return status_text, df, fig

# 3. Gradio Interface Layout
with gr.Blocks(title="GCL Qwen3.5 Continual Learning Benchmark") as demo:
    gr.Markdown("# 🧠 Grounded Continual Learning (GCL) — Qwen3.5-2B Benchmark\n### Live Online PEFT LoRA Continual Learning & Verified Safety Controller Dashboard")
    status_box = gr.Textbox(label="Live Container Status", value="🚀 Booting Experiment Environment...", interactive=False)
    
    with gr.Row():
        with gr.Column(scale=1):
            table_output = gr.DataFrame(label="Learner Performance Summary")
        with gr.Column(scale=1):
            plot_output = gr.Plot(label="Live Performance Metrics (ACC & BWT)")

    timer = gr.Timer(value=5)
    timer.tick(fn=get_live_data, outputs=[status_box, table_output, plot_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
