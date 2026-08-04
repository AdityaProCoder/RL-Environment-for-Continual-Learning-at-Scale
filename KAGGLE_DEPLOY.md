# Kaggle Deployment — Grounded Continual Learning on Qwen3.5-2B (BF16)

**Everything runs on Kaggle. Nothing local.** The notebook `kaggle_continual_learning.ipynb` is fully self-contained: it embeds the entire `gcl` research package, downloads the BF16 model, builds the drift curriculum, runs the 5-learner benchmark, streams checkpoints + logs to the kernels working folder, and writes all artifacts (`metrics.json`, `results.tex`, figures, trajectories).

---

## Files to upload

| File | Where |
|---|---|
| `kaggle_continual_learning.ipynb` | root directory of this project folder (already built) |

No other file needed — the notebook is self-packing.

## Pre-flight checklist

- You have a Kaggle account with **Phone-verified → GPU enabled** (free tier gives P100/T4 x2).
- Internet is ON in the notebook settings (model downloads ~5GB) via HF hub.
- The notebook can use ~16GB GPU RAM (fp16). Your local RTX 4060 Ti is only for reference; Kaggle does the real work.

## One-liner to push

```bash
kaggle kernels push --kernel qwen35-continual-gcl --title "continual-learning" --path kaggle_continual_learning.ipynb --kernel-type notebook --enable-gpu true --enable-internet true
```

*(If you hit quota errors, rerun; the kernel consumes the same name. First run is slowest because model caching warms up.)*

## Within the notebook (what you will see)

1. ✅ The notebook first installs deps (`pip`).
2. ✅ It unpacks the embedded `gcl` package and prints version.
3. ✅ It downloads the BF16 model `Qwen/Qwen3.5-2B` into the HF cache (what you see: "model cached at: /kaggle/input/-torch-transformers...").
4. ✅ It builds a CANARY-clean 4-family drift curriculum (`canary: {'Clean': True}` is assert — it must be True or it will stop).
5. ✅ It then starts the Lifelong Learning experiment for 5 learners (frozen, always_lora, replay, ewc, controller) and writes a log that you can print with the variant cell.

Mid-run monitoring: tail `runs/main/logs/RUN.log` and `runs/main/logs/CHECKPOINT.json` from the kernels output — that prints recent reward, updates, rollbacks.

## After the run finishes

The last cells produce:
- `output/metrics.json` (raw scores per learner: ACC, BWT, FWT, forgetting, AUC, updates, rollback counts)
- `output/results.tex` (LaTeX fragment that the paper main.tex will load)
- `fig_family_curves.png`, `fig_frontier.png`, etc.
- `TABLES.md` (Markdown tables for the report)

The paper's `results.tex` can be consumed by the provided LaTeX file (paper/main.tex) to fill in actual values for the tables after you run `gcl.report` on it on your machine or just download the notebook's output artifacts and read them directly.

## Failure fallbacks

- **`OutOfMemoryError` CUDA**: drop `lora_r=16` to `8`, reduce `n_train=25`→`12`, or increase `max_new_tokens=256`. The notebook cell has the `ExperimentConfig` parameters to change directly.
- **`kaggle: command not found`**: run `pip install kaggle` first: `pip install kaggle`.
- **Gateway/timeout**: switch to a P100 VM.

## Quick run-status sanity

While it's running, execute this in a separate Kaggle cell to peek progress live:

```python
import json, os
for line in open('/kaggle/working/runs/main/logs/RUN.log'):
    print(line.strip())
```
