"""Download a SentencePiece BPE model for testing."""

import os
import shutil

import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as sp_pb2
from transformers import AutoTokenizer

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(MODEL_DIR, "bpe.model")

# SentencePiece model type constants
MODEL_TYPE_NAMES = {1: "UNIGRAM", 2: "BPE", 3: "WORD", 4: "CHAR"}
BPE_TYPE = 2


def get_model_type(path):
    """Get model type from a sentencepiece model file via protobuf."""
    m = sp_pb2.ModelProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    return m.trainer_spec.model_type


def try_download(model_name):
    """Try to download a tokenizer and return (vocab_file, is_bpe) or (None, False)."""
    print(f"Trying '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        src = tokenizer.vocab_file
        model_type = get_model_type(src)
        type_name = MODEL_TYPE_NAMES.get(model_type, f"UNKNOWN({model_type})")
        print(f"  Found model at: {src}")
        print(f"  Model type: {type_name}")
        if model_type == BPE_TYPE:
            return src, True
        print(f"  Skipping: not BPE")
        return None, False
    except Exception as e:
        print(f"  Failed: {e}")
        return None, False


def main():
    candidates = [
        # NLLB uses actual BPE sentencepiece (confirmed type=2), ~4.6 MB model file
        "facebook/nllb-200-distilled-600M",
        # These claim BPE in filename but are actually UNIGRAM; kept as fallback
        "camembert-base",
        "facebook/mbart-large-50-many-to-many-mmt",
    ]

    src = None
    chosen = None
    for name in candidates:
        src, is_bpe = try_download(name)
        if src is not None and is_bpe:
            chosen = name
            break

    if src is None:
        raise RuntimeError(
            "Could not download any BPE SentencePiece model from the candidates."
        )

    shutil.copy2(src, OUTPUT)
    print(f"\nSaved to: {OUTPUT}")

    # Final verification
    sp = spm.SentencePieceProcessor()
    sp.Load(OUTPUT)
    vocab_size = sp.GetPieceSize()
    model_type = get_model_type(OUTPUT)
    type_name = MODEL_TYPE_NAMES.get(model_type, f"UNKNOWN({model_type})")

    print(f"Model:      {chosen}")
    print(f"Vocab size: {vocab_size}")
    print(f"Model type: {type_name}")
    print(f"Is BPE:     {model_type == BPE_TYPE}")

    if model_type != BPE_TYPE:
        raise RuntimeError("Downloaded model is NOT BPE!")
    print("\nConfirmed: model is BPE.")


if __name__ == "__main__":
    main()
