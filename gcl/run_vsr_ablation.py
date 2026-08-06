#!/usr/bin/env python3
import os, sys, json, time, copy
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/root/gclwork")
import torch

from gcl.config import ExperimentConfig
from gcl.experiment import run_experiment
from gcl.curriculum import StreamAssembler, spec_perturb

cfg = ExperimentConfig(
    model_name="Qwen/Qwen3.5-2B", device="cuda", dtype="bfloat16",
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    learning_rate=3.5e-4, train_steps_per_update=3, max_updates=40,
    temperature=0.3, top_p=0.9, max_new_tokens=256, max_seq_len=512,
    gate_epsilon=0.05, holdout_size=12,
    use_reference_injection=True, use_vsr_gate=True,
    vsr_learners=["vsr", "vsr_nogold", "vsr_bounded"],
    refinject_learners=["vsr", "vsr_bounded"],
    vault_commit_min=0.9, vault_retrieve_k=3, vault_gate_check=3,
    anchor_lambda=0.5, anchor_learners=["vsr_nogold", "vsr_bounded"],
    replay_frac=0.5, replay_frac_learners=["vsr", "vsr_nogold", "vsr_bounded"],
    lr_decay=True, vault_dedup_sim=0.995, vault_dedup_learners=["vsr", "vsr_nogold", "vsr_bounded"],
    self_taught_k=6, self_taught_temp=0.7, self_taught_repair_rounds=1, self_taught_repair_k=2,
    out_dir="runs/vsr_ablation3", seed=42,
)

asm = StreamAssembler(seed=42)
fams = asm.assemble([
    dict(corpus="mbpp", name="A_basic",   n_train=8, n_holdout=4, offset=0),
    dict(corpus="mbpp", name="B_strings", n_train=8, n_holdout=4, offset=12),
    dict(corpus="math", name="C_math",    n_train=8, n_holdout=4, offset=0),
    dict(corpus="mbpp", name="D_drift",   n_train=8, n_holdout=4, offset=24, drift=spec_perturb),
])

learners = ["frozen", "vsr", "vsr_nogold", "vsr_bounded"]
print("VSR ablation launch:", learners)

reports = run_experiment(cfg, learners, fams, cfg.out_dir)

out = {}
for k, v in reports["learners"].items():
    out[k] = {
        "zero_shot": v["R_pairs"]["zero_shot"], "final": v["R_pairs"]["final"],
        "trained": v["R_pairs"].get("final_trained", []), "acc": v["report"]["acc"],
        "bwt": v["report"]["bwt"], "forgetting": v["report"]["forgetting"],
        "auc": v["report"]["auc"], "updates": v["updates"], "rollbacks": v["rollbacks"],
        "vsr": v.get("vsr", {}),
    }
os.makedirs(cfg.out_dir, exist_ok=True)
with open(cfg.out_dir + "/results.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
