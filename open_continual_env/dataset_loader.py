"""
Unified Multi-Dataset Loader for OpenContinualEnv.
Loads and normalizes coding (MBPP, HumanEval) and math (GSM8K) evaluation datasets.
"""

import os
import sys
import json
import gzip
import ssl
import re
import urllib.request
from typing import List, Dict, Any, Optional

ssl._create_default_https_context = ssl._create_unverified_context

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def extract_gsm8k_answer(answer_str: str) -> str:
    """Extracts numeric or text answer following '####' in GSM8K solutions."""
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    # Fallback to finding trailing numbers
    numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_str)
    return numbers[-1] if numbers else answer_str.strip()


class UnifiedDatasetLoader:
    """
    Streams and normalizes tasks from MBPP, HumanEval, and GSM8K datasets.
    """

    @staticmethod
    def load_mbpp(max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """Loads MBPP coding tasks."""
        print("Fetching MBPP dataset (974 tasks)...")
        tasks = []
        try:
            with urllib.request.urlopen(MBPP_URL) as resp:
                lines = resp.read().decode("utf-8").splitlines()
                for i, line in enumerate(lines):
                    if max_tasks and len(tasks) >= max_tasks:
                        break
                    row = json.loads(line)
                    test_list = row.get("test_list", [])
                    test_code = "\n".join(test_list)
                    prompt = row.get("text", "")
                    entry_point = ""
                    if test_list:
                        match = re.search(r'assert\s+([a-zA-Z_0-9]+)\s*\(', test_list[0])
                        if match:
                            entry_point = match.group(1)
                            prompt += f"\nYour function should be named `{entry_point}`."
                    tasks.append({
                        "task_id": f"mbpp_{row.get('task_id', i+1)}",
                        "domain": "code",
                        "prompt": prompt,
                        "test_code": test_code,
                        "reference_answer": row.get("code", ""),
                        "entry_point": entry_point,
                    })
            print(f"Loaded {len(tasks)} MBPP tasks.")
        except Exception as e:
            print(f"Error loading MBPP dataset: {e}")
        return tasks

    @staticmethod
    def load_humaneval(max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """Loads HumanEval coding tasks."""
        print("Fetching HumanEval dataset (164 tasks)...")
        tasks = []
        try:
            req = urllib.request.Request(HUMANEVAL_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                content = gzip.decompress(resp.read()).decode("utf-8")
                for line in content.splitlines():
                    if max_tasks and len(tasks) >= max_tasks:
                        break
                    row = json.loads(line)
                    task_id = str(row.get("task_id", "")).replace("/", "_")
                    entry_point = row.get("entry_point", "")
                    test_code = f"{row.get('test', '')}\ncheck({entry_point})"
                    tasks.append({
                        "task_id": f"humaneval_{task_id}",
                        "domain": "code",
                        "prompt": row.get("prompt", ""),
                        "test_code": test_code,
                        "reference_answer": row.get("canonical_solution", ""),
                        "entry_point": entry_point,
                    })
            print(f"Loaded {len(tasks)} HumanEval tasks.")
        except Exception as e:
            print(f"Error loading HumanEval dataset: {e}")
        return tasks

    @staticmethod
    def load_gsm8k(max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """Loads GSM8K grade-school math reasoning tasks."""
        print("Fetching GSM8K dataset (1,319 tasks)...")
        tasks = []
        try:
            with urllib.request.urlopen(GSM8K_URL) as resp:
                lines = resp.read().decode("utf-8").splitlines()
                for i, line in enumerate(lines):
                    if max_tasks and len(tasks) >= max_tasks:
                        break
                    row = json.loads(line)
                    raw_answer = row.get("answer", "")
                    target_answer = extract_gsm8k_answer(raw_answer)
                    prompt = (
                        f"Solve the following math problem step-by-step. "
                        f"End your final answer clearly with '#### <number>'.\n\n"
                        f"Problem: {row.get('question', '')}"
                    )
                    tasks.append({
                        "task_id": f"gsm8k_{i+1}",
                        "domain": "math",
                        "prompt": prompt,
                        "test_code": "",
                        "reference_answer": target_answer,
                        "full_solution": raw_answer,
                        "entry_point": "",
                    })
            print(f"Loaded {len(tasks)} GSM8K tasks.")
        except Exception as e:
            print(f"Error loading GSM8K dataset: {e}")
        return tasks

    @classmethod
    def load_dataset(
        cls,
        name: str = "all",
        max_tasks: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Loads specified dataset ('mbpp', 'humaneval', 'gsm8k', or 'all').
        """
        name_clean = str(name).lower().strip()
        if name_clean == "mbpp":
            return cls.load_mbpp(max_tasks=max_tasks)
        elif name_clean == "humaneval":
            return cls.load_humaneval(max_tasks=max_tasks)
        elif name_clean == "gsm8k":
            return cls.load_gsm8k(max_tasks=max_tasks)
        elif name_clean in ("all", "combined"):
            all_tasks = []
            mbpp_tasks = cls.load_mbpp()
            he_tasks = cls.load_humaneval()
            gsm_tasks = cls.load_gsm8k()
            all_tasks.extend(mbpp_tasks)
            all_tasks.extend(he_tasks)
            all_tasks.extend(gsm_tasks)
            if max_tasks:
                all_tasks = all_tasks[:max_tasks]
            print(f"Combined total tasks loaded across domains: {len(all_tasks)}")
            return all_tasks
        else:
            raise ValueError(f"Unknown dataset name '{name}'. Supported: mbpp, humaneval, gsm8k, all")
