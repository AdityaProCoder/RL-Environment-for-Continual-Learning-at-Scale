# Grounded Continual Learning — Deployment & Verification (handoff to deployer agent)

This file is the *operational contract*. The research/architecture is implemented in `gcl/`. The deployer agent executes these steps and reports the artifacts back to the research lead.

Environment contract (verified to work):

| Component | Version | Note |
|---|---|---|
| Host | Windows 11 + WSL2 (Ubuntu-22.04) | GPU passthrough OK |
| GPU | NVIDIA RTX 4060 Ti (16 GB, sm_89) | bf16 OK; 17.18 GB visible in WSL |
| venv | `/root/.oce_kit/.venv` (uv, py3.11) | **the only trustworthy env** (system python3 torch is broken) |
| torch | 2.7.1+cu128 | CUDA confirmed |
| transformers | 4.56.2, peft 0.13.2, trl 0.18.2, accelerate 1.7.0, datasets 3.6.0 | pinned-compat stack |

---

## Step 0 — Sanity (2 min)

```bash
wsl -d Ubuntu-22.04 -e bash -lc 'cd /root/gcl && /root/.oce_kit/.venv/bin/python -c "import gcl; print(gcl.__version__, sorted(gcl.LEARNERS))"'
```
Expected: prints `0.2.0 ...` and the learner list including `controller, ewc, replay, frozen, always_lora, grpo`.

## Step 1 — Real-learning smoke (GPU, ~30 s)

```bash
wsl -d Ubuntu-22.04 -e bash -lc 'cd /root/gcl && /root/.oce_kit/.venv/bin/python gcl_smoke.py'
```
Expected: two blocks `[SMOKE][frozen] ...` and `[SMOKE][always_lora] ... `with nonzero `updates` for always_lora and `REAL` behaviour. Artifacts in `runs/smoke/`: `metrics.json`, `trajectories_*.jsonl`.

> If this fails with CUDA/timeout, do not proceed — re-check WSL state and disk space on C: (must be >20 GB free for swap).

## Step 2 — Real-corpus continual drift experiment (GPU, ~15–45 min)

```bash
wsl -d Ubuntu-22.04 -e bash -lc 'cd /root/gcl && HF_TOKEN=<hf_read_token_optional> /root/.oce_kit/.venv/bin/python -m gcl.runner --config configs/real_mbpp_drift.json 2>&1 | tee runs/real_mbpp_drift_console.log'
```
Expected console lines per learner:
```
[frozen     ] ACC=... BWT=+0.000 forget=0.000 updates=0
[always_lora] ACC=... BWT=... forget=... updates>0
[replay     ] ...
[ewc        ] ...
[controller ] ...
```
Artifacts `runs/real_mbpp_drift/`:
- `config.json`, `metrics.json`, `summary.md`, `trajectories_<learner>.jsonl`, `fig_family_curves.png`, `fig_final_vs_zero.png`, `fig_frontier.png`, `TABLES.md`.

## Step 3 (optional transfer to HF)

Same code path; model override and credits:
```bash
wsl -d Ubuntu-22.04 -e bash -lc 'cd /root/gcl && HF_TOKEN=<token> /root/.oce_kit/.venv/bin/python -m gcl.runner --config configs/real_hf_scale.json'
```

---

## Failure triage (most common)
- `Bus error`/torch crash: global env pollution — never use system python3; only the venv above.
- `Wsl/CreateInstance/E_FAIL`: C: disk full; free space, then `wsl --shutdown`.
- `ModuleNotFoundError: trl`/enable_grpo path: set `"enable_grpo": true` in config or accept Replay fallback (honest).
- vLLM Track-A (server at scale): not required for the paper claims; Track-B HF generate is the source of truth.

---

## What to send back to the research lead
1. console log tail (20 lines) from Step 2
2. `runs/real_mbpp_drift/summary.md` and `TABLES.md`
3. the two figures `fig_final_vs_zero.png` and `fig_frontier.png`
4. whether the expected ordering holds: `forgetting(always_lora) > forgetting(ewc) ≥ forgetting(replay) > forgetting(frozen) = 0`, and whether `controller` achieves Pareto-best frontier (high ACC, low forgetting per update).
