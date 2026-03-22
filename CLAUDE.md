# Project: goSentencePiece

Pure Go SentencePiece tokenizer (Unigram + BPE). Loads both SentencePiece `.model` (protobuf) and HuggingFace `tokenizer.json` formats.

## Python

- Use Python via venv: `.venv/` in project root
- Activate: `source .venv/bin/activate`
- All Python commands must run inside the venv
- Create venv via `make venv`

## Go

- Module: `github.com/tggo/goSentencePiece`
- Min version: Go 1.23
- Dependencies: stdlib + `google.golang.org/protobuf`

## Commands

- `make venv` — create Python venv and install dependencies
- `make golden` — generate golden test datasets (SentencePiece + HuggingFace)
- `make proto` — generate Go protobuf code
- `make test` — run Go tests
- `make bench` — run benchmarks
- `make fuzz` — run fuzz tests
- `make all` — full pipeline

## Structure

- `_testdata/` — test data (spm.model, bpe.model, tokenizer.json, golden/)
- `proto/` — generated protobuf Go code
