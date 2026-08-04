# Continual Learning Empirical Benchmark Summary

**Execution Timestamp**: `2026-07-25T19:49:00.553592+00:00`  
**Inference Endpoint**: `http://127.0.0.1:8000/v1`  
**Model Name**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`  
**Total Benchmark Tasks**: `4`  
**Episodes per Task**: `3`  

---

## Baseline Comparison Table

| Metric | Memory Replay | LoRA Online | Hybrid (Replay + LoRA) |
|---|:---:|:---:|:---:|
| **Task Success Rate (Pass@1)** | 75.0% | 75.0% | 75.0% |
| **Sample Efficiency (avg steps)** | 1.00 | 1.00 | 1.00 |
| **Learning Speed ($\Delta$ Reward)** | -0.200 | -0.200 | -0.200 |
| **Catastrophic Forgetting** | 0.000 | 0.000 | 0.000 |
| **Backward Transfer (BWT)** | 0.000 | -0.000 | -0.000 |
| **Forward Transfer (FWT)** | 0.000 | -0.000 | -0.000 |
| **Performance Stability Index** | 0.513 | 0.513 | 0.513 |
| **Mean Reward** | 0.949 | 0.949 | 0.949 |

---

## Detailed Performance Matrices ($R_{i,j}$)

### 1. Memory Replay Matrix
```json
[
  [
    0.9989780848041445,
    0.9989799990014335,
    0.9989801089576168,
    0.7989466135214627
  ],
  [
    0.9989807237167367,
    0.9989816983503181,
    0.9989803738528678,
    0.7989380933560184
  ],
  [
    0.9989791243592172,
    0.9989813134927469,
    0.9989821981698336,
    0.7989458393549633
  ],
  [
    0.9989806987263272,
    0.9989806637398567,
    0.998981393462885,
    0.7989445208022536
  ]
]
```

### 2. LoRA Online Matrix
```json
[
  [
    0.9989807686994326,
    0.9989783946692267,
    0.9989810885772661,
    0.798947712359056
  ],
  [
    0.998977999841462,
    0.9989803238725474,
    0.9989765455028441,
    0.7989467134147326
  ],
  [
    0.9989759607806844,
    0.9989791843341136,
    0.9989788594705431,
    0.7989490160113846
  ],
  [
    0.9989788794620613,
    0.9989810086076157,
    0.9989787595127994,
    0.7989517832635156
  ]
]
```

### 3. Hybrid Baseline Matrix
```json
[
  [
    0.9989791343550283,
    0.9989763056159929,
    0.9989806087611928,
    0.7989516433987436
  ],
  [
    0.9989796191549469,
    0.9989796041610401,
    0.9989780448217519,
    0.7989433321367465
  ],
  [
    0.9989777199654972,
    0.9989796841284433,
    0.9989759158023611,
    0.7989468432763458
  ],
  [
    0.998975001251415,
    0.9989761057111936,
    0.9989799440235463,
    0.7989405853289955
  ]
]
```

---

## Artifact Traceability

All raw execution steps, model generation outputs, sandbox logs, and continual update signals have been recorded into:
- **Trajectories Log**: `benchmark_results/trajectories.jsonl`
- **Execution Log**: `benchmark_results/benchmark_execution.log`
- **JSON Metrics Summary**: `benchmark_results/metrics_summary.json`
- **Markdown Report**: `benchmark_results/summary.md`
