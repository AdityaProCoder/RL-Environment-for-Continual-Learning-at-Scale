import os
import json
import torch
from datasets import Dataset
import logging
from typing import Optional, List, Dict, Any

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_ella_subspace_loss(model: Any, alpha: float = 0.01) -> torch.Tensor:
    """
    ELLA (EACL 2026): Selective Subspace De-correlation Loss.
    Penalizes alignment along high-energy task-specific directions in LoRA weights.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            # Subspace de-correlation: Frobenious norm of off-diagonal covariance
            cov = torch.matmul(param, param.T)
            diag = torch.diag(torch.diag(cov))
            off_diag = cov - diag
            loss = loss + torch.norm(off_diag, p="fro")
    return alpha * loss


def train_adapter(
    cluster_id: str,
    data: list,
    base_model_id: str = "google/gemma-4-e4b",
    output_dir: str = "./adapters",
    use_ella: bool = True,
):
    """
    Train a new LoRA adapter for a specific cluster of trajectories.
    Uses Unsloth (if available) for 2x-5x faster training & 70% lower VRAM,
    otherwise falls back to Hugging Face PEFT.
    Includes ELLA selective subspace de-correlation (EACL 2026).
    """
    logger.info(f"Starting async training for cluster {cluster_id} with {len(data)} examples")
    adapter_path = os.path.join(output_dir, cluster_id)
    os.makedirs(adapter_path, exist_ok=True)

    if UNSLOTH_AVAILABLE:
        logger.info("[UNSLOTH] Engine detected! Using Unsloth FastLanguageModel for ultra-fast training.")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    else:
        logger.info("Using standard Hugging Face PEFT + Trainer.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        target_mods = ["q_proj", "v_proj"]
        named_mods = [name for name, _ in model.named_modules()]
        if not any("q_proj" in m for m in named_mods):
            if any("c_attn" in m for m in named_mods):
                target_mods = ["c_attn", "c_proj"]
            else:
                target_mods = "all-linear"

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=target_mods
        )
        model = get_peft_model(model, lora_config)

    def format_ds(examples):
        # Fix Bug 2: check for 'generated_code' as well as 'response'
        codes = examples.get('generated_code') or examples.get('response') or [''] * len(examples['prompt'])
        texts = [f"Task: {p}\n```python\n{c}\n```" for p, c in zip(examples['prompt'], codes)]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

    ds = Dataset.from_list(data)
    tokenized_ds = ds.map(format_ds, batched=True, remove_columns=ds.column_names)

    training_args = TrainingArguments(
        output_dir=adapter_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=1,
        save_strategy="no",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    class ELLA_Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            if use_ella:
                ella_loss = compute_ella_subspace_loss(model)
                loss = loss + ella_loss
            return (loss, outputs) if return_outputs else loss

    trainer = ELLA_Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=lambda data: {
            'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
            'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
            'labels': torch.stack([torch.tensor(f['input_ids']) for f in data])
        }
    )

    trainer.train()

    if UNSLOTH_AVAILABLE:
        model.save_pretrained_merged(adapter_path, tokenizer, save_method="lora")
    else:
        model.save_pretrained(adapter_path)

    logger.info(f"Successfully trained and saved adapter {cluster_id} to {adapter_path}")
    return {"status": "success", "cluster_id": cluster_id, "adapter_path": adapter_path}
