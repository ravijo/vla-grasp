"""
Source:
    vla-scripts/finetune.py

Run:
    python finetune_bottle.py --max_steps 2000 --batch_size 1 --grad_accumulation_steps 8

Smoke test:
    python finetune_bottle.py --max_steps 3 --save_steps 100000 --no_save True
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
import tqdm
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# Make sure OpenVLA files are accessible
# Set the REPO_ROOT to the OpenVLA repo root directory, and add it to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from bottle_dataset import DATASET_NAME, BottleDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Config:
    vla_path: str = "openvla/openvla-7b"
    demo_dir: str = os.path.join(os.path.dirname(__file__), "demos")
    run_root_dir: Path = Path(os.path.join(os.path.dirname(__file__), "runs"))
    adapter_tmp_dir: Path = Path(os.path.join(
        os.path.dirname(__file__), "adapter-tmp")
    )

    batch_size: int = 1
    grad_accumulation_steps: int = 8
    max_steps: int = 2000
    save_steps: int = 500
    learning_rate: float = 5e-4
    num_workers: int = 0

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = True  # 4-bit QLoRA
    no_save: bool = False  # skip checkpoint saving (for smoke tests)


@draccus.wrap()
def finetune(cfg: Config) -> None:
    assert torch.cuda.is_available(), "Need a GPU!"
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    exp_id = f"openvla-7b+{DATASET_NAME}+b{cfg.batch_size * cfg.grad_accumulation_steps}+lr{cfg.learning_rate}"
    exp_id += f"+lora-r{cfg.lora_rank}" + \
        ("+q4bit" if cfg.use_quantization else "")
    run_dir = cfg.run_root_dir / exp_id
    adapter_dir = cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    print(f"[finetune] run_dir = {run_dir}")

    # Register OpenVLA with HF AutoClasses
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(
        cfg.vla_path,
        trust_remote_code=True,
    )

    quantization_config = None
    if cfg.use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        # 4-bit models must be placed via device_map, not .to() afterwards
        device_map={"": 0} if cfg.use_quantization else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    optimizer = AdamW(
        [p for p in vla.parameters()if p.requires_grad],
        lr=cfg.learning_rate
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    dataset = BottleDataset(
        cfg.demo_dir,
        action_tokenizer,
        processor.tokenizer,
        processor.image_processor.apply_transform,
        PurePromptBuilder,
    )
    print(
        f"[finetune] dataset: {len(dataset)} transitions from {DATASET_NAME}"
    )
    save_dataset_statistics(dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.num_workers,
    )

    def save_checkpoint(tag):
        if cfg.no_save:
            return

        print(f"[finetune] saving checkpoint ({tag}) ...")
        processor.save_pretrained(run_dir)
        vla.save_pretrained(adapter_dir)
        base = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        merged = PeftModel.from_pretrained(
            base, adapter_dir
        ).merge_and_unload()
        merged.save_pretrained(run_dir)
        # keep stats next to weights
        save_dataset_statistics(dataset.dataset_statistics, run_dir)
        del base, merged
        torch.cuda.empty_cache()
        print(f"[finetune] saved merged checkpoint to {run_dir}")

    vla.train()
    optimizer.zero_grad()
    step, micro = 0, 0
    pbar = tqdm.tqdm(total=cfg.max_steps)
    while step < cfg.max_steps:
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(
                        torch.bfloat16).to(device),
                    labels=batch["labels"].to(device),
                )
                loss = out.loss
            (loss / cfg.grad_accumulation_steps).backward()

            # action-token accuracy (monitoring)
            with torch.no_grad():
                n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches
                logits = out.logits[:, n_patch:-1]
                preds = logits.argmax(dim=2)
                gt = batch["labels"][:, 1:].to(device)
                m = gt > action_tokenizer.action_token_begin_idx
                acc = ((preds == gt) & m).sum().float() / \
                    m.sum().clamp(min=1).float()

            micro += 1
            if micro % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                pbar.update(1)
                pbar.set_description(
                    f"loss={loss.item():.3f} acc={acc.item():.3f}"
                )
                if step % cfg.save_steps == 0:
                    save_checkpoint(f"step{step}")
                if step >= cfg.max_steps:
                    break
    pbar.close()
    save_checkpoint("final")
    print("[finetune] done.")


if __name__ == "__main__":
    finetune()
