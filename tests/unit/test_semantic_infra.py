"""
Unit tests for FAISSMemory, EmbeddingRouter, and AdapterRegistry.
"""

import os
import tempfile
import pytest
from open_continual_env.memory.faiss_memory import FAISSMemory
from open_continual_env.routing.embedding_router import EmbeddingRouter
from open_continual_env.routing.adapter_registry import AdapterRegistry
from open_continual_env.trajectory.schema import Trajectory


def test_faiss_memory_add_and_query():
    memory = FAISSMemory()
    t1 = Trajectory(trajectory_id="t1", prompt="Sort an array of integers in Python", generated_code="def sort_arr(arr): return sorted(arr)")
    t2 = Trajectory(trajectory_id="t2", prompt="Calculate matrix multiplication in Python", generated_code="def matmul(a, b): pass")

    memory.add(t1)
    memory.add(t2)

    assert len(memory) == 2

    results = memory.query("how to sort list", top_k=1)
    assert len(results) == 1
    assert results[0].trajectory_id == "t1"


def test_faiss_memory_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = FAISSMemory()
        t1 = Trajectory(trajectory_id="t1", prompt="Write hello world", generated_code="print('hello')")
        memory.add(t1)
        memory.save(tmpdir)

        memory2 = FAISSMemory()
        memory2.load(tmpdir)
        assert len(memory2) == 1
        assert memory2.trajectories[0].trajectory_id == "t1"


def test_embedding_router_cluster_assignment():
    router = EmbeddingRouter()
    c1 = router.get_cluster_id("Write a function to sort an array using quicksort")
    c2 = router.get_cluster_id("Calculate the prime factors of a number")
    c3 = router.get_cluster_id("Parse a string regex pattern")

    assert c1 == "cluster_algorithms"
    assert c2 == "cluster_math"
    assert c3 == "cluster_text"

    clusters = router.list_clusters()
    assert clusters["cluster_algorithms"] >= 1
    assert clusters["cluster_math"] >= 1


def test_adapter_registry_lru_eviction():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = AdapterRegistry(adapter_dir=tmpdir, max_active_adapters=2)

        os.makedirs(os.path.join(tmpdir, "c1"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "c2"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "c3"), exist_ok=True)
        registry._scan_adapters()

        registry.touch_active("c1")
        registry.touch_active("c2")
        evicted = registry.touch_active("c3")

        assert evicted == "c1"
