"""VSR real-model benchmark: VSR vs frozen vs always_lora on Qwen3.5-2B.

Curriculum: A_arith (mbpp) -> B_string (mbpp) -> C_math (math: cross-domain drift).
VSR gets the SkillVault (reference injection + vault-test gate). Frozen = negative
control. always_lora = positive control (self-distillation floor).

The hard gates this run tries to meet:
  - frozen ACC << 0.45 (baseline floor)
  - always_lora forgets or, at best, no gain (BWT <= 0*on C* + no accumulation)
  - vsr has ACC > frozen, forgetting <=0.05, updates >0, rollbacks < 20%
"""
import os, json, time, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
sys.path.insert(0, "/root/gclwork")
from gcl.config import ExperimentConfig
from gcl.experiment import run_experiment
from gcl.curriculum import StreamAssembler

OUT = "/root/gclwork/runs/vsr_real2"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "progress.jsonl")

def log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

cfg = ExperimentConfig(
    model_name="Qwen/Qwen3.5-2B", device="cuda", dtype="bfloat16",
    lora_r=16, lora_alpha=32, learning_rate=3e-4,
    train_steps_per_update=3, max_updates=40, max_new_tokens=220,
    temperature=0.1, max_seq_len=512, gate_epsilon=0.05, holdout_size=6,
    use_reference_injection=True, use_vsr_gate=True,
    vsr_learners=["vsr"], refinject_learners=["vsr"],
    vault_commit_min=0.85, vault_retrieve_k=2, vault_gate_check=2,
    seed=42, out_dir=OUT,
)

asm = StreamAssembler(seed=42)
fams = asm.assemble([
    dict(corpus="mbpp",  name="A_arith",  n_train=6, n_holdout=4, offset=0),
    dict(corpus="mbpp",  name="B_string", n_train=6, n_holdout=4, offset=10),
    dict(corpus="math",  name="C_math",   n_train=6, n_holdout=4, offset=0),
])

learners = ["frozen", "always_lora", "vsr"]
log({"event": "start", "learners": learners,
     "families": [f.name for f in fams]})

t0 = time.time()
reports = run_experiment(cfg, learners, fams, OUT)

summary = {}
for name, d in reports["learners"].items():
    r = d["report"]
    summary[name] = {
        "acc": r["acc"], "bwt": r["bwt"], "forgetting": r["forgetting"],
        "updates": d["updates"], "rollbacks": d["rollbacks"], "vsr": d.get("vsr", {}),
    }
    log({"event": "learner_done", "learner": name, **summary[name]})

log({"event": "done_all", "wallclock_s": round(time.time() - t0, 1)})
print(json.dumps(summary, indent=2))
print("RESULTS:", os.path.join(OUT, "metrics.json"))
