# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly by emailing the maintainers directly rather than opening a public issue.

## Scope

This library processes untrusted input (arbitrary text) and model files (protobuf). Security considerations:

- **Input text**: The tokenizer handles arbitrary UTF-8 and invalid byte sequences without panics (fuzz-tested).
- **Model files**: Only load model files from trusted sources. Malformed protobuf data may cause errors but should not cause panics.
