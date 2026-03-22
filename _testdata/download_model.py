"""Download the DeBERTa v3 small SentencePiece model file."""

import os
import shutil
from transformers import AutoTokenizer

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(MODEL_DIR, "spm.model")


def main():
    print("Downloading microsoft/deberta-v3-small tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

    # The vocab file is the sentencepiece model
    src = tokenizer.vocab_file
    print(f"Found model at: {src}")

    shutil.copy2(src, OUTPUT)
    print(f"Saved to: {OUTPUT}")

    # Quick sanity check
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(OUTPUT)
    print(f"Vocab size: {sp.GetPieceSize()}")
    print(f"Vocab size check passed.")
    print("Done.")


if __name__ == "__main__":
    main()
