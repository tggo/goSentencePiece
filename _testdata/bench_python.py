#!/usr/bin/env python3
"""Benchmark the Python sentencepiece library on the same inputs as Go benchmarks.

Usage:
    .venv/bin/python _testdata/bench_python.py
"""

import time
import statistics
import sentencepiece as spm

MODEL_PATH = "_testdata/spm.model"
ITERATIONS = 10000

SHORT = "Hello world"
MEDIUM = (
    "The quick brown fox jumps over the lazy dog. "
    "This is a medium length string for benchmarking the tokenizer performance."
)
LONG = "The quick brown fox jumps over the lazy dog. " * 100


def bench(name: str, func, iterations: int = ITERATIONS) -> None:
    """Run a benchmark and print results."""
    # Warmup
    for _ in range(min(100, iterations)):
        func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)

    avg = statistics.mean(times)
    med = statistics.median(times)
    p99 = sorted(times)[int(len(times) * 0.99)]

    if avg >= 1_000_000:
        unit, divisor = "ms", 1_000_000
    elif avg >= 1_000:
        unit, divisor = "us", 1_000
    else:
        unit, divisor = "ns", 1

    print(
        f"{name:30s}  avg={avg / divisor:8.1f} {unit}  "
        f"med={med / divisor:8.1f} {unit}  "
        f"p99={p99 / divisor:8.1f} {unit}  "
        f"(n={iterations})"
    )


def main() -> None:
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)

    print(f"Model: {MODEL_PATH}")
    print(f"Vocab size: {sp.get_piece_size()}")
    print(f"Iterations: {ITERATIONS}")
    print("-" * 90)

    # Encode benchmarks
    bench("Encode/short", lambda: sp.encode(SHORT))
    bench("Encode/medium", lambda: sp.encode(MEDIUM))
    bench("Encode/long", lambda: sp.encode(LONG))

    # EncodeAsPieces benchmarks
    bench("EncodeAsPieces/short", lambda: sp.encode(SHORT, out_type=str))
    bench("EncodeAsPieces/medium", lambda: sp.encode(MEDIUM, out_type=str))
    bench("EncodeAsPieces/long", lambda: sp.encode(LONG, out_type=str))

    # Decode benchmarks
    short_ids = sp.encode(SHORT)
    medium_ids = sp.encode(MEDIUM)
    long_ids = sp.encode(LONG)

    bench("Decode/short", lambda: sp.decode(short_ids))
    bench("Decode/medium", lambda: sp.decode(medium_ids))
    bench("Decode/long", lambda: sp.decode(long_ids))


if __name__ == "__main__":
    main()
