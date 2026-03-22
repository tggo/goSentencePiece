"""Generate golden test cases for HuggingFace tokenizer.json format.

Produces: _testdata/golden/hf_test_cases.jsonl
Requires: pip install tokenizers
"""

import json
import os
import random

from tokenizers import Tokenizer

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(HERE, "tokenizer.json")
OUTPUT = os.path.join(HERE, "golden", "hf_test_cases.jsonl")


def make_case(tok, text, desc):
    enc = tok.encode(text, add_special_tokens=False)
    ids = enc.ids
    pieces = enc.tokens
    decoded = tok.decode(ids)
    return {"input": text, "pieces": pieces, "ids": ids, "decoded": decoded, "description": desc}


def generate(tok):
    cases = []
    seen = set()

    def add(text, desc):
        if text in seen:
            return
        seen.add(text)
        cases.append(make_case(tok, text, desc))

    # Basic
    add("", "empty")
    add("a", "single_char")
    add("hello", "single_word")
    add("Hello world", "two_words")
    add("Hello World", "two_words_cap")
    add("the quick brown fox jumps over the lazy dog", "pangram")

    # Numbers & punctuation
    add("42", "number")
    add("3.14", "float")
    add("-1", "negative")
    add("!?!?", "punctuation")
    add("...", "ellipsis")

    # Unicode
    add("café", "latin_accent")
    add("Привіт світ", "ukrainian")
    add("Привет мир", "russian")
    add("你好世界", "chinese")
    add("こんにちは", "japanese")
    add("한국어", "korean")
    add("مرحبا بالعالم", "arabic")
    add("สวัสดีครับ", "thai")
    add("Hello Привіт 你好 🚀", "mixed_scripts")

    # Emoji
    add("🚀", "rocket")
    add("👨\u200d👩\u200d👧\u200d👦", "family_zwj")
    add("🇺🇦", "flag_ukraine")
    add("😀😁😂🤣", "emoji_sequence")

    # Whitespace
    add(" ", "single_space")
    add("  ", "double_space")
    add("\t", "tab")
    add("\n", "newline")
    add(" hello ", "padded")
    add("hello  world", "double_space_words")

    # Special chars
    add("\u200b", "zwsp")
    add("\ufeff", "bom")
    add("\u00ad", "soft_hyphen")
    add("a\u0300", "combining_grave")

    # Code
    add('func main() { fmt.Println("hello") }', "go_code")
    add("SELECT * FROM users WHERE id = 1;", "sql")
    add('{"key": "value"}', "json")
    add("https://example.com/path?q=hello", "url")
    add("user@example.com", "email")

    # Long text
    add("a" * 100, "repeated_a_100")
    add("hello " * 50, "hello_repeated_50")
    add("The quick brown fox. " * 20, "pangram_repeated")

    # Real sentences
    sentences = [
        "The transformer architecture was introduced in 2017.",
        "Machine learning models require large datasets for training.",
        "Natural language processing has made significant advances.",
        "Go is a statically typed, compiled programming language.",
        "SentencePiece provides a language-independent tokenizer.",
        "Київ — столиця і найбільше місто України.",
        "Програмування — це мистецтво.",
        "Die Würde des Menschen ist unantastbar.",
        "Liberté, égalité, fraternité.",
        "吾輩は猫である。名前はまだ無い。",
    ]
    for i, s in enumerate(sentences):
        add(s, f"sentence_{i}")

    # ASCII range
    for i in range(32, 127):
        add(chr(i), f"ascii_{i:03d}")

    # Various lengths
    for n in [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]:
        add("a" * n, f"len_a_{n}")
        add("x" * n, f"len_x_{n}")

    # Subword patterns
    words = ["unhappiness", "internationalization", "tokenization",
             "antidisestablishmentarianism", "preprocessing",
             "microservices", "multithreading"]
    for w in words:
        add(w, f"subword_{w}")

    # Random strings
    for i in range(100):
        n = random.randint(5, 100)
        chars = [chr(random.randint(32, 126)) for _ in range(n)]
        add("".join(chars), f"random_{i}")

    # Random Unicode
    ranges = [(0x0041, 0x007A), (0x0400, 0x04FF), (0x4E00, 0x4EFF)]
    for i in range(50):
        n = random.randint(5, 50)
        chars = [chr(random.randint(*random.choice(ranges))) for _ in range(n)]
        add("".join(chars), f"random_uni_{i}")

    # BPE-specific: pairs that test merge ordering
    add("aaaa", "repeat_a4")
    add("aabb", "aabb")
    add("abab", "abab")
    add("abcabc", "abcabc")
    add("ababab", "ababab")
    add("aabbcc", "aabbcc")

    # mmBERT-specific: multilingual text that exercises the model
    add("мене це реально дратує але я мовчу", "mmbert_uk")
    add("I love programming in Go", "mmbert_en")
    add("Je suis un programmeur", "mmbert_fr")
    add("Ich bin ein Programmierer", "mmbert_de")

    return cases


def main():
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"Loaded tokenizer: vocab_size={tok.get_vocab_size()}")

    cases = generate(tok)
    print(f"Generated {len(cases)} test cases")

    # Validate
    errors = 0
    for tc in cases:
        enc = tok.encode(tc["input"], add_special_tokens=False)
        if enc.ids != tc["ids"]:
            print(f"MISMATCH: {tc['description']}")
            errors += 1
    if errors:
        raise SystemExit(f"{errors} validation errors")
    print("All validated ✓")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for tc in cases:
            f.write(json.dumps(tc, ensure_ascii=False) + "\n")
    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
