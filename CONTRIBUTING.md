# Contributing to goSentencePiece

Thank you for your interest in contributing!

## Development Setup

1. **Go 1.22+** is required.
2. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/tggo/goSentencePiece.git
   cd goSentencePiece
   ```
3. Set up the Python venv (for generating golden test data):
   ```bash
   make venv
   ```
4. Download the test model and generate golden data:
   ```bash
   make golden
   ```

## Code Style

- Format code with `gofmt` (enforced by CI).
- Run `go vet` and `staticcheck` before submitting.
- If you have `golangci-lint` installed, run `make lint` for a full check.

## Running Tests

```bash
# Unit + golden tests
make test

# Test with coverage
make cover

# Fuzz testing (60s default)
make fuzz
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Keep changes focused -- one feature or fix per PR.
3. Add or update tests for any changed behavior.
4. Run `make test` and `make lint` and ensure everything passes.
5. Open a pull request against `main` with a clear description of the change.

## Reporting Issues

Please open a GitHub issue with steps to reproduce. If the issue involves
incorrect tokenization, include the input text, expected token IDs (from
the Python `sentencepiece` library), and actual token IDs from this library.
