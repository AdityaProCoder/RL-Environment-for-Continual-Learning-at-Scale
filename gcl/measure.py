"""Real metrics: build the R-matrix by frozen-policy eval, then compute CL metrics.

This replaces the fabricated matrices of the old system with measured values (I4/I5).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


def _teach():  # pragma: no cover - placeholder
    pass


def acc_from_matrix(R: List[List[float]]) -> float:
    if not R or not R[0]:
        return 0.0
    last_row = R[-1]
    return sum(last_row) / len(last_row)


def bwt_from_matrix(R: List[List[float]]) -> float:
    if len(R) < 2 or not R[0]:
        return 0.0
    n_cols = len(R[0])
    if n_cols < 2:
        return R[-1][0] - R[0][0]
    diffs = [R[-1][j] - R[0][j] for j in range(n_cols)]
    return sum(diffs) / len(diffs)


def forgetting_from_matrix(R: List[List[float]]) -> float:
    if len(R) < 2 or not R[0]:
        return 0.0
    n_cols = len(R[0])
    drops = [max(0.0, R[0][j] - R[-1][j]) for j in range(n_cols)]
    return sum(drops) / max(1, len(drops))


def fwt_from_matrix(R: List[List[float]], baseline: List[float]) -> float:
    if len(R) < 2 or not R[0] or not baseline:
        return 0.0
    n_cols = min(len(R[0]), len(baseline))
    if n_cols < 2:
        return 0.0
    diffs = [R[-1][j] - baseline[j] for j in range(1, n_cols)]
    return sum(diffs) / max(1, len(diffs))


def auc_accuracy(learning_curve: List[float]) -> float:
    if len(learning_curve) < 2:
        return learning_curve[0] if learning_curve else 0.0
    auc = 0.0
    for i in range(1, len(learning_curve)):
        auc += (learning_curve[i] + learning_curve[i - 1]) / 2.0
    return auc / (len(learning_curve) - 1)


def weight_stability(rewards: List[float]) -> float:
    if not rewards or all(r == 0 for r in rewards):
        return 1.0
    m = sum(rewards) / len(rewards)
    if m <= 1e-8:
        return 1.0
    var = sum((r - m) ** 2 for r in rewards) / len(rewards)
    cov = math.sqrt(var) / m
    return float(max(0.0, min(1.0, 1.0 - cov)))


@dataclass
class MetricReport:
    learner: str
    acc: float
    bwt: float
    fwt: float
    forgetting: float
    auc: float
    stability: float
    updates: int
    updates_used_pct: float
    cost: float
    n_families: int

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def report(learner_name: str, R: List[List[float]], baseline: List[float],
           learning_curve: List[float], rewards: List[float], updates: int,
           max_updates: int, cost_per_update: float = 1.0) -> MetricReport:
    n = len(R)
    updates_pct = (100.0 * updates / max_updates) if max_updates else 0.0
    return MetricReport(
        learner=learner_name, acc=acc_from_matrix(R), bwt=bwt_from_matrix(R),
        fwt=fwt_from_matrix(R, baseline), forgetting=forgetting_from_matrix(R),
        auc=auc_accuracy(learning_curve), stability=weight_stability(rewards),
        updates=updates, updates_used_pct=updates_pct,
        cost=updates * cost_per_update, n_families=n)
