# Continual Learning Empirical Benchmark Summary

**Execution Timestamp**: `2026-07-25T18:51:10.870553+00:00`  
**Inference Endpoint**: `http://127.0.0.1:1234/v1`  
**Model Name**: `google/gemma-4-e4b`  
**Total Benchmark Tasks**: `4`  
**Episodes per Task**: `3`  

---

## Baseline Comparison Table

| Metric | Memory Replay | LoRA Online | Hybrid (Replay + LoRA) |
|---|:---:|:---:|:---:|
| **Task Success Rate (Pass@1)** | 75.0% | 75.0% | 75.0% |
| **Sample Efficiency (avg steps)** | 1.00 | 1.00 | 1.00 |
| **Learning Speed ($\Delta$ Reward)** | -0.205 | -0.200 | -0.200 |
| **Catastrophic Forgetting** | 0.003 | 0.000 | 0.000 |
| **Backward Transfer (BWT)** | -0.003 | 0.002 | -0.000 |
| **Forward Transfer (FWT)** | -0.002 | -0.001 | -0.000 |
| **Performance Stability Index** | 0.514 | 0.513 | 0.513 |
| **Mean Reward** | 0.947 | 0.949 | 0.949 |

---

## Detailed Performance Matrices ($R_{i,j}$)

### 1. Memory Replay Matrix
```json
[
  [
    0.9989800689735139,
    0.9989789894156906,
    0.998980423833238,
    0.7989399111356885
  ],
  [
    0.9989815134183775,
    0.9989817783211176,
    0.9944807437090006,
    0.7989492857367633
  ],
  [
    0.993980378850898,
    0.9939760957159427,
    0.9944777649454444,
    0.7944348275114825
  ],
  [
    0.9939786945405049,
    0.9939797640960198,
    0.9944794042430466,
    0.7944451151455252
  ]
]
```

### 2. LoRA Online Matrix
```json
[
  [
    0.9984802638961472,
    0.9989814334480019,
    0.9944818382992896,
    0.7989371745035065
  ],
  [
    0.9989814684350568,
    0.9939780698107435,
    0.994473636952569,
    0.7974360958722393
  ],
  [
    0.9989801139556206,
    0.998981203533959,
    0.9944817233411366,
    0.7974509590658849
  ],
  [
    0.9989810086076157,
    0.9989804838098302,
    0.9944800839675146,
    0.7974531619517173
  ]
]
```

### 3. Hybrid Baseline Matrix
```json
[
  [
    0.9989797840878999,
    0.9989822481520282,
    0.9989815384091488,
    0.7989484665751805
  ],
  [
    0.9989785845874789,
    0.9989777699432137,
    0.9944802838882726,
    0.7989466734574371
  ],
  [
    0.9989790993696074,
    0.9989796441447257,
    0.9989823631113428,
    0.7989450202499009
  ],
  [
    0.9989794792121891,
    0.9989784746347402,
    0.9989819032757163,
    0.798951123904305
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
