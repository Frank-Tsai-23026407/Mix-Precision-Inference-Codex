#!/usr/bin/env python
"""Post-training quantization script for Llama 3.2 1B.

This script downloads the Llama 3.2 1B model from Hugging Face and

performs post-training quantization. If CUDA is available it uses
``AutoGPTQ`` for 8-bit or 2-bit quantization. On CPU-only machines it
falls back to PyTorch dynamic quantization (8-bit only).

Example usage:
    python quantize_llama_3.2_1b.py --bits 8 --output_dir ../model/llama3_2_1b_8bit
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
try:  # AutoGPTQ is optional and requires CUDA
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
except Exception:  # pragma: no cover - auto_gptq may be absent
    AutoGPTQForCausalLM = BaseQuantizeConfig = None



def build_calibration_dataset(tokenizer, nsamples: int, max_length: int):
    """Yield tokenized samples for calibration."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    collected = 0
    for text in dataset["text"]:
        if not text.strip():
            continue
        yield tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
        collected += 1
        if collected >= nsamples:
            break


def quantize(model_id: str, bits: int, output_dir: str, nsamples: int, max_length: int):
    """Quantize the specified model and save it to ``output_dir``."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)


    if torch.cuda.is_available() and AutoGPTQForCausalLM is not None:
        quant_cfg = BaseQuantizeConfig(bits=bits, group_size=128, desc_act=False)
        model = AutoGPTQForCausalLM.from_pretrained(
            model_id,
            quantize_config=quant_cfg,
            use_safetensors=True,
            device_map="auto",
        )
        cal_dataset = list(build_calibration_dataset(tokenizer, nsamples, max_length))
        model.quantize(tokenizer=tokenizer, calibration_dataset=cal_dataset)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_quantized(output_dir, use_safetensors=True)
        tokenizer.save_pretrained(output_dir)
    else:
        if bits != 8:
            raise ValueError("CPU dynamic quantization only supports 8 bits.")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()
        model_quant = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_quant.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-training quantization for Llama 3.2 1B")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.2-1B", help="HuggingFace model id")
    parser.add_argument("--bits", type=int, default=8, choices=[2, 8], help="Quantization bits")
    parser.add_argument("--output_dir", default="model/llama3_2_1b_quantized", help="Directory to save the quantized model")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length of calibration samples")
    args = parser.parse_args()

    quantize(args.model_id, args.bits, args.output_dir, args.nsamples, args.max_length)
