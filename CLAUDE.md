# Project: goSentencePiece

Pure Go port of the SentencePiece Unigram tokenizer for DeBERTa v3 models.

## Python

- Use Python via venv: `.venv/` in project root
- Activate: `source .venv/bin/activate`
- All Python commands must run inside the venv
- Create venv via `make venv`

## Go

- Module: `github.com/tggo/goSentencePiece`
- Min version: Go 1.22
- Dependencies: stdlib + `google.golang.org/protobuf`

## Commands

- `make venv` — create Python venv and install dependencies
- `make golden` — generate golden test dataset
- `make proto` — generate Go protobuf code
- `make test` — run Go tests
- `make bench` — run benchmarks
- `make fuzz` — run fuzz tests
- `make all` — full pipeline

## Structure

- `_testdata/` — test data (spm.model, golden/)
- `proto/` — generated protobuf Go code
- `status.log` — work progress log
