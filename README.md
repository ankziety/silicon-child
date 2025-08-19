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

## Features

### Core Research Pipeline
- **Data Ingestion**: Web crawling with robots.txt compliance
- **Document Processing**: HTML/PDF parsing with anchored quotes
- **Storage**: DuckDB + JSONL with deduplication
- **Job Logging**: Complete audit trail of all operations

### LLM Jury Evaluation System
- **Multi-Model Judges**: GPT-4o-mini, GPT-5, Claude Haiku, Claude Sonnet, Command R+
- **Reference-Free Evaluation**: No ground truth data required
- **Cost Optimization**: 70% cost savings with affordable jury configurations
- **Ring Buffer Management**: Automatic adapter versioning with rollback
- **Comprehensive Logging**: All evaluations logged to JobV1 schema

### Jury Configurations
- **Frontier Jury**: GPT-5 + Claude Sonnet (highest performance)
- **Affordable Jury**: GPT-4o-mini + Claude Haiku (cost-effective)
- **Diverse Jury**: Mixed model families (bias reduction)
- **Specialized Jury**: Single model family (focused evaluation)
- **Mixed Jury**: High-performance + affordable (balanced)

## Architecture

This project implements a two-layer architecture:

1. **Immutable Contracts (Schemas)**: DocV1, TraceV1, JobV1
2. **Processing Pipeline**: Components that operate on validated data

## Environment Setup

For LLM Jury evaluation, set these environment variables:

```bash
# Required for LLM judges
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export COHERE_API_KEY="your_cohere_api_key"
```

## Schema Rules

- Never modify existing schema versions
- Always add vN+1 for breaking changes
- All data must validate against published schemas
- Maintain backward compatibility

## Development

See `docs/ADR-0000.md` for detailed architecture decisions.

## Model Pricing (per 1K tokens)

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| GPT-4o-mini | $0.00015 | $0.0006 | Cost-effective evaluation |
| GPT-5 | $0.005 | $0.015 | High-performance evaluation |
| Claude Haiku | $0.00025 | $0.00125 | Affordable Claude evaluation |
| Claude Sonnet | $0.003 | $0.015 | High-performance Claude evaluation |
| Command R+ | $0.0005 | $0.0015 | Cohere evaluation |

## Testing

```bash
# Run all tests
make test

# Run evaluation tests only
pytest tests/test_eval_promote.py -v

# Run with coverage
pytest --cov=ai_infant tests/
```
