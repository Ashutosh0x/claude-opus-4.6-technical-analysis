"""
Export Model to GGUF Format (for llama.cpp inference).

GGUF (GGML Universal File Format) is self-contained — includes
tokenizer, config, and weights in ONE file.

Conversion pipeline:
    HF SafeTensors → GGUF F16 → Quantized GGUF

Available quantization formats:
    Format    | BPW  | Size (2T) | Quality
    ──────────────────────────────────────────
    Q8_0      | 8.5  | ~2.1 TB   | Near-lossless
    Q6_K      | 6.6  | ~1.6 TB   | Excellent
    Q5_K_M    | 5.7  | ~1.4 TB   | Very good
    Q4_K_M    | 4.8  | ~1.2 TB   | Good (most popular)
    Q3_K_M    | 3.9  | ~980 GB   | Acceptable
    Q2_K      | 3.2  | ~800 GB   | Degraded
    IQ2_XXS   | 2.1  | ~520 GB   | Research-grade

K-Quant layer-differentiated quantization:
    K_M promotes attention Q/K/V and FFN down to 6-bit,
    keeping the most sensitive weights at higher precision.

GGUF binary structure:
    [4 bytes: magic "GGUF"]
    [4 bytes: version]
    [8 bytes: tensor_count]
    [8 bytes: metadata_kv_count]
    [Metadata Key-Value Store]
    [Tensor Info Block]
    [Tensor Data Block]  (32-byte aligned)

References:
    - Gerganov, GGUF specification (llama.cpp), 2023
    - Frantar et al., "GPTQ: Accurate Post-Training Quantization
      for Generative Pre-trained Transformers", 2023
"""

import os
import json
import struct
import logging
import argparse
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# GGUF metadata keys for Claude Opus 4.6
GGUF_METADATA = {
    "general.architecture": "llama",         # Compatible with llama.cpp
    "general.name": "Claude Opus 4.6",
    "general.organization": "Anthropic",
    "general.description": "Hypothetical open-weight release of Claude Opus 4.6",
    "general.file_type": 1,                  # F16

    # Architecture
    "llama.context_length": 1048576,          # 1M tokens
    "llama.embedding_length": 16384,          # d_model
    "llama.block_count": 160,                 # L (layers)
    "llama.feed_forward_length": 65536,       # d_ff
    "llama.attention.head_count": 128,        # n_h
    "llama.attention.head_count_kv": 16,      # n_kv (GQA)
    "llama.attention.layer_norm_rms_epsilon": 1e-5,

    # RoPE
    "llama.rope.freq_base": 500000.0,
    "llama.rope.dimension_count": 128,        # d_h

    # MoE
    "llama.expert_count": 128,                # E
    "llama.expert_used_count": 2,             # top-k

    # Tokenizer
    "tokenizer.ggml.model": "gpt2",           # BPE type
    "tokenizer.ggml.bos_token_id": 1,
    "tokenizer.ggml.eos_token_id": 2,
    "tokenizer.ggml.padding_token_id": 0,
}


def generate_gguf_conversion_script(
    model_dir: str,
    output_path: str,
    quantization: str = "Q4_K_M",
    compute_imatrix: bool = True,
    calibration_data: Optional[str] = None,
) -> str:
    """
    Generate a shell script for GGUF conversion.

    The actual conversion uses llama.cpp tools:
        1. convert_hf_to_gguf.py  → GGUF F16 (unquantized)
        2. llama-imatrix          → importance matrix (optional)
        3. llama-quantize         → quantized GGUF

    Args:
        model_dir: Path to HuggingFace model directory.
        output_path: Path for output GGUF file.
        quantization: Target quantization (Q4_K_M, Q8_0, etc.).
        compute_imatrix: Whether to compute importance matrix first.
        calibration_data: Path to calibration text for imatrix.

    Returns:
        Shell script content as string.
    """
    model_name = Path(model_dir).stem
    f16_path = f"{model_name}-F16.gguf"
    imatrix_path = "imatrix.dat"

    lines = [
        "#!/bin/bash",
        "# GGUF Conversion Pipeline for Claude Opus 4.6",
        f"# Target: {quantization}",
        "",
        "set -euo pipefail",
        "",
        "LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-\"../llama.cpp\"}",
        "",
        "# Step 1: Convert HF SafeTensors → GGUF F16",
        f"echo 'Converting {model_dir} → GGUF F16...'",
        f"python ${{LLAMA_CPP_DIR}}/convert_hf_to_gguf.py \\",
        f"    --model {model_dir} \\",
        f"    --outfile {f16_path} \\",
        "    --outtype f16",
        "",
    ]

    if compute_imatrix and calibration_data:
        lines.extend([
            "# Step 2: Compute importance matrix (for IQ/K quants)",
            "echo 'Computing importance matrix...'",
            f"${{LLAMA_CPP_DIR}}/llama-imatrix \\",
            f"    --model {f16_path} \\",
            f"    --cal-data {calibration_data} \\",
            f"    --output {imatrix_path} \\",
            "    --n-gpu-layers 999",
            "",
            f"# Step 3: Quantize to {quantization}",
            f"echo 'Quantizing to {quantization}...'",
            f"${{LLAMA_CPP_DIR}}/llama-quantize \\",
            f"    --imatrix {imatrix_path} \\",
            f"    {f16_path} \\",
            f"    {output_path} \\",
            f"    {quantization}",
        ])
    else:
        lines.extend([
            f"# Step 2: Quantize to {quantization}",
            f"echo 'Quantizing to {quantization}...'",
            f"${{LLAMA_CPP_DIR}}/llama-quantize \\",
            f"    {f16_path} \\",
            f"    {output_path} \\",
            f"    {quantization}",
        ])

    lines.extend([
        "",
        f"echo 'Done: {output_path}'",
        f"ls -lh {output_path}",
    ])

    return "\n".join(lines)


def write_gguf_metadata_json(output_path: str):
    """
    Write GGUF metadata as JSON for documentation.

    This shows what metadata keys would be in the GGUF file.
    The actual GGUF writing is done by llama.cpp's convert tool.
    """
    with open(output_path, "w") as f:
        json.dump(GGUF_METADATA, f, indent=2)

    logger.info(f"GGUF metadata reference written: {output_path}")


def generate_all_quants_script(
    model_dir: str,
    output_dir: str = "gguf_quants",
) -> str:
    """
    Generate script to create all standard quantization variants.

    Community derivative files (hypothetical):
        claude-opus-4-6-Q8_0.gguf     ~2.1 TB  (near-lossless)
        claude-opus-4-6-Q6_K.gguf     ~1.6 TB  (excellent)
        claude-opus-4-6-Q5_K_M.gguf   ~1.4 TB  (very good)
        claude-opus-4-6-Q4_K_M.gguf   ~1.2 TB  (most popular)
        claude-opus-4-6-IQ4_XS.gguf   ~1.05 TB (best 4-bit)
        claude-opus-4-6-Q3_K_M.gguf   ~980 GB
        claude-opus-4-6-Q2_K.gguf     ~800 GB  (degraded)
        claude-opus-4-6-IQ2_XXS.gguf  ~520 GB  (research)
    """
    model_name = Path(model_dir).stem
    f16_gguf = f"{model_name}-F16.gguf"

    quants = [
        "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M",
        "IQ4_XS", "Q3_K_M", "Q2_K", "IQ2_XXS",
    ]

    lines = [
        "#!/bin/bash",
        "# Generate all GGUF quantization variants",
        f"# Model: {model_name}",
        "",
        "set -euo pipefail",
        "",
        f"mkdir -p {output_dir}",
        "",
        f"# First: convert to F16 GGUF",
        f"python convert_hf_to_gguf.py --model {model_dir} "
        f"--outfile {f16_gguf} --outtype f16",
        "",
        "# Compute importance matrix",
        f"./llama-imatrix --model {f16_gguf} "
        f"--cal-data calibration.txt --output imatrix.dat",
        "",
    ]

    for quant in quants:
        out_file = f"{output_dir}/{model_name}-{quant}.gguf"
        imatrix_flag = "--imatrix imatrix.dat " if "IQ" in quant else ""
        lines.append(
            f"echo 'Quantizing {quant}...' && "
            f"./llama-quantize {imatrix_flag}"
            f"{f16_gguf} {out_file} {quant}"
        )

    lines.extend([
        "",
        f"echo 'All quantizations complete!'",
        f"ls -lh {output_dir}/",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate GGUF export scripts for Claude Opus 4.6"
    )
    parser.add_argument(
        "--model-dir", default="claude-opus-4-6/",
        help="HuggingFace model directory"
    )
    parser.add_argument(
        "--output", default="claude-opus-4-6-Q4_K_M.gguf",
        help="Output GGUF file"
    )
    parser.add_argument(
        "--quant", default="Q4_K_M",
        help="Quantization format"
    )
    parser.add_argument(
        "--calibration-data", default=None,
        help="Calibration data for importance matrix"
    )
    parser.add_argument(
        "--all-quants", action="store_true",
        help="Generate script for all quantization variants"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.all_quants:
        script = generate_all_quants_script(args.model_dir)
        script_path = "quantize_all.sh"
    else:
        script = generate_gguf_conversion_script(
            model_dir=args.model_dir,
            output_path=args.output,
            quantization=args.quant,
            calibration_data=args.calibration_data,
        )
        script_path = "convert_gguf.sh"

    with open(script_path, "w") as f:
        f.write(script)

    print(f"Script written: {script_path}")

    # Also write metadata reference
    write_gguf_metadata_json("gguf_metadata_reference.json")
