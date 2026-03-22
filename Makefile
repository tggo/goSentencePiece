SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
MODEL := _testdata/spm.model
GOLDEN := _testdata/golden/test_cases.jsonl
PROTO_SRC := proto/sentencepiece_model.proto
PROTO_GO := proto/sentencepiece_model.pb.go

.PHONY: all venv deps golden proto test bench fuzz lint cover ci clean generate

all: venv deps golden proto test

# ── Python venv ──────────────────────────────────────────────
venv: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install sentencepiece transformers protobuf torch tokenizers --extra-index-url https://download.pytorch.org/whl/cpu
	@echo "✓ venv ready"

deps: venv

# ── Golden test data ─────────────────────────────────────────
HF_GOLDEN := _testdata/golden/hf_test_cases.jsonl

golden: $(GOLDEN) $(HF_GOLDEN)

$(GOLDEN): $(MODEL) _testdata/generate_golden.py | venv
	$(PYTHON) _testdata/generate_golden.py
	@echo "✓ golden data generated"

_testdata/tokenizer.json: _testdata/download_hf_tokenizer.py | venv
	$(PYTHON) _testdata/download_hf_tokenizer.py
	@echo "✓ HF tokenizer downloaded"

$(HF_GOLDEN): _testdata/tokenizer.json _testdata/generate_hf_golden.py | venv
	$(PYTHON) _testdata/generate_hf_golden.py
	@echo "✓ HF golden data generated"

$(MODEL): _testdata/download_model.py | venv
	$(PYTHON) _testdata/download_model.py
	@echo "✓ model downloaded"

# ── Protobuf ─────────────────────────────────────────────────
proto: $(PROTO_GO)

$(PROTO_GO): $(PROTO_SRC)
	protoc --go_out=. --go_opt=paths=source_relative $(PROTO_SRC)
	@echo "✓ protobuf generated"

# ── Go ───────────────────────────────────────────────────────
test:
	go test -v -count=1 ./...

bench:
	go test -bench=. -benchmem -count=3 ./...

fuzz:
	go test -fuzz=FuzzEncode -fuzztime=60s ./...

# ── Lint / Coverage / CI ────────────────────────────────────
lint:
	@which golangci-lint > /dev/null 2>&1 && golangci-lint run ./... || (echo "golangci-lint not found, falling back to go vet + staticcheck"; go vet ./...; go install honnef.co/go/tools/cmd/staticcheck@latest; staticcheck ./...)

cover:
	go test -coverprofile=coverage.out ./...
	go tool cover -func=coverage.out | tail -1

ci: test lint
	go build ./...

# ── Generate ─────────────────────────────────────────────────
generate:
	go generate ./...

# ── Cleanup ──────────────────────────────────────────────────
clean:
	rm -rf $(VENV)
	rm -f $(GOLDEN) _testdata/golden/model_info.json
	rm -f $(PROTO_GO)
