"""Download HuggingFace tokenizer.json for testing.

Downloads from onnx-community/mmBERT-small-ONNX.
"""

import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "tokenizer.json")
URL = "https://huggingface.co/onnx-community/mmBERT-small-ONNX/resolve/main/tokenizer.json"


def main():
    if os.path.exists(OUTPUT):
        size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
        print(f"Already exists: {OUTPUT} ({size_mb:.1f} MB)")
        return

    print(f"Downloading {URL}...")
    urllib.request.urlretrieve(URL, OUTPUT)
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"Saved to: {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
