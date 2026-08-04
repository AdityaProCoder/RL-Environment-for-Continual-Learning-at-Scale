"""CLI runner: `python -m gcl.runner --config configs/bench_quick.json`."""
from __future__ import annotations

import argparse, json, os
from .config import ExperimentConfig
from .curriculum import StreamAssembler, api_rename, spec_paraphrase, spec_perturb
from .experiment import run_experiment
from . import plots


def load_families(stream_spec):
    import importlib
    asm = StreamAssembler(seed=stream_spec.get("seed", 42))
    spec = []
    for s in stream_spec["families"]:
        drift = None
        d = s.get("drift")
        if d and d.get("kind") == "api_rename":
            drift = lambda t, o=d["old"], n=d["new"]: api_rename(t, o, n)
        elif d and d.get("kind") == "spec_paraphrase":
            drift = lambda t, p=d["prefix"]: spec_paraphrase(t, p)
        elif d and d.get("kind") == "spec_perturb":
            drift = spec_perturb
        spec.append({"corpus": s["corpus"], "name": s["name"], "n_train": s["n_train"],
                     "n_holdout": s["n_holdout"], "drift": drift, "offset": s.get("offset", 0)})
    return asm.assemble(spec)


def main():
    ap = argparse.ArgumentParser(description="Grounded Continual Learning runner")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    exp = cfg["experiment"]
    e = ExperimentConfig(**exp)
    fams = load_families(cfg["stream"])
    out_dir = cfg.get("out_dir", "runs/run")
    reports = run_experiment(e, cfg["learners"], fams, out_dir)
    made = plots.render_all(out_dir)
    print("MADE:", " | ".join(made))
    for name, d in reports["learners"].items():
        r = d["report"]
        print(f"[{name:12s}] ACC={r['acc']:.3f} BWT={r['bwt']:+.3f} FWT={r['fwt']:+.3f} "
              f"forget={r['forgetting']:.3f} updates={d['updates']} rollbacks={d['rollbacks']} frontier={d['frontier_score']:+.3f}")


if __name__ == "__main__":
    main()
