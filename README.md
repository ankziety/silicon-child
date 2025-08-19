# AI-Infant

Minimal, long-running research agent that ingests data, records provenance, learns incrementally, and reports progress.

## Quick Start

```bash
# Install development dependencies
make install

# Run all checks
make verify

# Run individual checks
make fmt        # Format code
make typecheck  # Type checking
make test       # Run tests
```

## Architecture

This project implements a two-layer architecture:

1. **Immutable Contracts (Schemas)**: DocV1, TraceV1, JobV1
2. **Processing Pipeline**: Components that operate on validated data

## Schema Rules

- Never modify existing schema versions
- Always add vN+1 for breaking changes
- All data must validate against published schemas
- Maintain backward compatibility

## Development

See `docs/ADR-0000.md` for detailed architecture decisions.
