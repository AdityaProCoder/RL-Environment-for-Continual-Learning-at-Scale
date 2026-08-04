"""End-to-end smoke: run a real continual-learning loop on GPU.

Uses two synthetic task families with measured drift (api_rename) so the loop is
deterministic/fast, exercising: real verified reward + real gated LoRA updates.
"""
import os, sys, json, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcl import (ExperimentConfig, Task, Family, GroundedContinualEnv, Action, LearnOp, LEARNERS)
from gcl.engine import TrainingEngine
from gcl.verify import Verifier
from gcl.sandbox import PythonSandbox
from gcl.experiment import run_experiment
from gcl.curriculum import StreamAssembler, api_rename, spec_paraphrase


def synthetic_tasks():
    fam_a = [Task(task_id="A_add", family="arith", domain="code",
                  prompt="Write a function add(a,b) returning a+b.",
                  test_code="assert add(2,3)==5\nassert add(-1,1)==0\nassert add(0,0)==0",
                  reference_answer="def add(a,b):\n    return a+b", entry_point="add"),
             Task(task_id="A_sub", family="arith", domain="code",
                  prompt="Write a function sub(a,b) returning a-b.",
                  test_code="assert sub(5,3)==2\nassert sub(0,0)==0", entry_point="sub"),
             Task(task_id="A_mul", family="arith", domain="code",
                  prompt="Write a function mul(a,b) returning a*b.",
                  test_code="assert mul(2,3)==6\nassert mul(0,5)==0", entry_point="mul")]
    # family B is a *drift* of A: operation name changed (api_rename analog)
    fam_b = [api_rename(Task(task_id="B_sumify", family="arith_drift", domain="code",
                             prompt="Write a function sumify(a,b) returning a+b.",
                             test_code="assert sumify(2,3)==5\nassert sumify(-1,1)==0",
                             reference_answer="def sumify(a,b):\n    return a+b", entry_point="sumify"),
                        "sumify", "aggregate"),
             Task(task_id="B_negate", family="arith_drift", domain="code",
                  prompt="Write a function negate(x) returning -x.",
                  test_code="assert negate(5)==-5\nassert negate(-1)==1", entry_point="negate"),
             Task(task_id="B_square", family="arith_drift", domain="code",
                  prompt="Write a function square(x) returning x*x.",
                  test_code="assert square(4)==16\nassert square(-2)==4", entry_point="square")]
    return [Family("arith", fam_a, holdout=[]),
            Family("arith_drift", fam_b, holdout=[])]


def main():
    cfg = ExperimentConfig(model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                           lora_r=8, lora_alpha=16, learning_rate=1e-3,
                           train_steps_per_update=2, max_updates=20, holdout_size=0,
                           temperature=0.0, max_new_tokens=200)
    fams = synthetic_tasks()
    out_dir = "/root/gcl/runs/smoke"
    reports = run_experiment(cfg, ["frozen", "always_lora"], fams, out_dir)
    for name, d in reports["learners"].items():
        r = d["report"]
        print(f"[SMOKE][{name:12s}] ACC={r['acc']:.3f} BWT={r['bwt']:+.3f} forget={r['forgetting']:.3f} "
              f"updates={d['updates']} rollbacks={d['rollbacks']} auc={r['auc']:.3f} wall={d['wallclock_s']}s")
    # sanity: real updates happened for always_lora
    al = reports["learners"]["always_lora"]
    print("[SMOKE] always_lora adapter_versions:", len(al.get("adapter_history", [])))
    with open(os.path.join(out_dir, "smoke_done.json"), "w") as f:
        json.dump({"ok": True, "learners": list(reports["learners"].keys())}, f)
    print("SMOKE_DONE")


if __name__ == "__main__":
    main()
