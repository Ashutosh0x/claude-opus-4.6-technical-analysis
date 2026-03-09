"""
BPE Tokenizer Training Script.

Claude uses a variant of BPE (Byte-Pair Encoding), likely
SentencePiece or a custom implementation:

    1. Start with individual bytes/characters as tokens
    2. Iteratively merge the most frequent adjacent pair
    3. Repeat until vocabulary size |V| is reached

Estimated vocabulary:
    |V| ≈ 100,000–150,000 tokens  (likely 131,072 = 2^17)
    Encoding: Byte-level BPE (UTF-8 fallback)
    Avg tokens/English word: ~1.3
    Embedding dimension: d_model (~16,384)
    Embedding parameters: |V| × d_model ≈ 2.15B

Tokenizer fertility by language:
    English:  ~1.3 tokens/word   → ~750K words in 1M context
    Spanish:  ~1.5 tokens/word   → ~667K words
    Chinese:  ~1.5–2.0/char      → ~500K–667K chars
    Japanese: ~2.0–3.0/char      → ~333K–500K chars
    Hindi:    ~3.0–4.0/word      → ~250K–333K words

References:
    - Sennrich et al., "Neural Machine Translation of Rare Words
      with Subword Units" (BPE), 2016
    - Kudo & Richardson, "SentencePiece" (unigram/BPE toolkit), 2018
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def train_sentencepiece_tokenizer(
    input_files: List[str],
    output_dir: str,
    vocab_size: int = 131072,
    model_type: str = "bpe",
    character_coverage: float = 0.9999,
    num_threads: int = 16,
    special_tokens: Optional[List[str]] = None,
):
    """
    Train a SentencePiece BPE tokenizer.

    This produces:
        - tokenizer.model (~4.5 MB SentencePiece binary)
        - tokenizer.json  (~10 MB BPE vocab with merge rules)

    Target vocabulary: 131,072 tokens (2^17)
    This is a power-of-2 for efficient GPU embedding lookups.

    Args:
        input_files: List of text files for training corpus.
        output_dir: Directory to save tokenizer files.
        vocab_size: Target vocabulary size (default: 131072).
        model_type: Tokenizer type ("bpe" or "unigram").
        character_coverage: Coverage of characters in training data.
        num_threads: Number of training threads.
        special_tokens: Additional special tokens to add.
    """
    try:
        import sentencepiece as spm
    except ImportError:
        logger.error(
            "sentencepiece not installed. "
            "Run: pip install sentencepiece"
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, "tokenizer")

    # Default special tokens for Claude-style models
    if special_tokens is None:
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|pad|>",
            "<|start_header|>",
            "<|end_header|>",
            "<|eot_id|>",
            "<tool_call>",
            "</tool_call>",
            "<tool_result>",
            "</tool_result>",
            "<thinking>",
            "</thinking>",
            "<|image|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
        ]

    logger.info(
        f"Training {model_type.upper()} tokenizer with "
        f"vocab_size={vocab_size:,}"
    )

    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=",".join(input_files),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
        # Byte fallback for unknown characters
        byte_fallback=True,
        # Special tokens
        user_defined_symbols=special_tokens,
        # BOS/EOS handling
        bos_id=1,   # <|begin_of_text|>
        eos_id=2,   # <|end_of_text|>
        pad_id=0,   # <|pad|>
        unk_id=-1,  # No UNK (byte fallback covers all)
        # Training parameters
        max_sentence_length=65536,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )

    logger.info(f"Tokenizer model saved: {model_prefix}.model")

    # Also export vocab list for reference
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    vocab_path = os.path.join(output_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{i}\t{sp.id_to_piece(i)}\n")

    logger.info(
        f"Vocab exported: {sp.get_piece_size():,} tokens → {vocab_path}"
    )


def export_hf_tokenizer(
    sp_model_path: str,
    output_dir: str,
    vocab_size: int = 131072,
):
    """
    Convert SentencePiece model to HuggingFace tokenizer format.

    Produces:
        - tokenizer.json        (~10 MB, BPE vocab + merge rules)
        - tokenizer_config.json (~5 KB, tokenizer class and settings)
        - special_tokens_map.json (~2 KB, special token mapping)

    These files are what HuggingFace Transformers reads to
    instantiate the tokenizer.
    """
    try:
        from tokenizers import SentencePieceBPETokenizer
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        logger.error(
            "transformers/tokenizers not installed. "
            "Run: pip install transformers tokenizers"
        )
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load SentencePiece model
    tokenizer = SentencePieceBPETokenizer.from_file(sp_model_path)

    # Wrap as HuggingFace tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        pad_token="<|pad|>",
        model_max_length=1048576,  # 1M token context
    )

    # Add chat template
    hf_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|start_header|>system<|end_header|>\n"
        "{{ message['content'] }}<|eot_id|>"
        "{% elif message['role'] == 'user' %}"
        "<|start_header|>user<|end_header|>\n"
        "{{ message['content'] }}<|eot_id|>"
        "{% elif message['role'] == 'assistant' %}"
        "<|start_header|>assistant<|end_header|>\n"
        "{{ message['content'] }}<|eot_id|>"
        "{% endif %}{% endfor %}"
        "<|start_header|>assistant<|end_header|>\n"
    )

    # Save
    hf_tokenizer.save_pretrained(output_dir)
    logger.info(f"HuggingFace tokenizer saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer for Claude Opus 4.6"
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Input text files for training"
    )
    parser.add_argument(
        "--output", default="tokenizer_output",
        help="Output directory"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=131072,
        help="Vocabulary size (default: 131072)"
    )
    parser.add_argument(
        "--threads", type=int, default=16,
        help="Number of training threads"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train_sentencepiece_tokenizer(
        input_files=args.input,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        num_threads=args.threads,
    )
