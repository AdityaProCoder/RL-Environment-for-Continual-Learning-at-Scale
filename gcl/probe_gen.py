import os; os.environ.setdefault("HF_HUB_OFFLINE","1")
import torch
from gcl.config import ExperimentConfig
from gcl.engine import TrainingEngine, extract_code, _build_prompt
from gcl.verify import Verifier
from gcl.curriculum import StreamAssembler

cfg = ExperimentConfig(
    model_name="Qwen/Qwen3.5-2B", device="cuda", dtype="bfloat16",
    max_new_tokens=220, temperature=0.0, lora_r=8, lora_alpha=16,
    holdout_size=0, max_seq_len=512)

eng = TrainingEngine(cfg, adapter_root="/root/gclwork/runs/probe/adapters")
v = Verifier()
asm = StreamAssembler(seed=42)
fam = asm.assemble([dict(corpus="mbpp", name="A", n_train=2, n_holdout=0, offset=0)])[0]

for t in fam.tasks:
    prompt = _build_prompt(t)        # env/experiment style prompt
    raw = eng.generate(prompt, adapter_on=True)
    code = extract_code(raw)
    print("=" * 58)
    print("TASK:", t.task_id)
    print("PROMPT_TAIL:", repr(prompt[-90:]))
    print("RAW:")
    print(raw[:500])
    print("EXTRACTED:")
    print(code)
    r, i, _ = v.reward(domain="code", code=code, test_code=t.test_code,
                       reference_answer=t.reference_answer)
    print("SCORE:", round(r, 3), "| err:", i.get("error_type"), "| quality:", i.get("quality"))
