"""Standalone VSR-only run on the real Qwen3.5-2B (3 drifted families).
Writes per-episode progress to runs/vsr_real2/progress_vsr.jsonl.
Controls (frozen / always_lora) are reused from the prior measured run — no re-run."""
import os, json, time, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/root/gclwork")
import torch  # noqa

from gcl.config import ExperimentConfig
from gcl.experiment import run_experiment
from gcl.curriculum import StreamAssembler, spec_perturb

OUT = "/root/gclwork/runs/vsr_real2"
os.makedirs(OUT, exist_ok=True)
PROG = os.path.join(OUT, "progress_vsr.jsonl")

def plog(rec):
    with open(PROG, "a") as f:
        f.write(json.dumps(rec) + "\n")

cfg = ExperimentConfig(
    model_name="Qwen/Qwen3.5-2B", device="cuda", dtype="bfloat16",
    lora_r=16, lora_alpha=32, learning_rate=3e-4, train_steps_per_update=3,
    max_updates=40, max_new_tokens=220, temperature=0.1, max_seq_len=512,
    gate_epsilon=0.05, holdout_size=6,
    use_reference_injection=True, use_vsr_gate=True,
    vsr_learners=["vsr"], refinject_learners=["vsr"],
    vault_commit_min=0.9, vault_retrieve_k=2, vault_gate_check=2,
    seed=42, out_dir=OUT,
)
asm = StreamAssembler(seed=42)
fams = asm.assemble([
    dict(corpus="mbpp", name="A_arith",  n_train=8, n_holdout=4, offset=0),
    dict(corpus="mbpp", name="B_string", n_train=8, n_holdout=4, offset=12),
    dict(corpus="mbpp", name="C_drift",  n_train=8, n_holdout=4, offset=24, drift=spec_perturb),
])
plog({"event": "start_vsr_only", "families": [f.name for f in fams]})
reports = run_experiment(cfg, ["vsr"], fams, OUT)
d = reports["learners"]["vsr"]
plog({"event": "done_vsr", "acc": d["report"]["acc"], "bwt": d["report"]["bwt"],
      "forgetting": d["report"]["forgetting"], "updates": d["updates"], "rollbacks": d["rollbacks"],
      "vsr": d.get("vsr", {})})
print(json.dumps(d["report"].__dict__, indent=2))
