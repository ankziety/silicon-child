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

# Start the self-curious research agent (default behavior)
python -m ai_infant

# Run self-curious research for specific duration
python -m ai_infant --duration 30

# Run research on a specific question
python -m ai_infant research "What are the latest developments in quantum computing?"

# Run a timed session with predefined questions
python -m ai_infant session --duration 15

# See all available commands
python -m ai_infant --help
```

## Features

### Core Research Pipeline
- **Data Ingestion**: Web crawling with robots.txt compliance
- **Document Processing**: HTML/PDF parsing with anchored quotes
- **Storage**: DuckDB + JSONL with deduplication
- **Job Logging**: Complete audit trail of all operations

## Running the System

The AI-Infant research agent is self-curious by default and will autonomously generate and research questions.

### Default Self-Curious Mode
Start the agent and let it research autonomously:
```bash
python -m ai_infant
```

The system is autonomously curious - it explores topics based on its own interests, follows its curiosity wherever it leads, and decides how much to research without artificial constraints. It generates its own questions and explores them naturally, making its own decisions about what interests it and how deeply to investigate.

Options:
- `--duration N`: Session duration in minutes (default: 60)
- `--db-path PATH`: Database path (default: data/ai_infant.db)

### Research Mode
Run research on a specific question:
```bash
python -m ai_infant research "What are the latest developments in quantum computing?"
```

Options:
- `--max-iterations N`: Maximum research iterations (default: 20)
- `--min-quotes N`: Minimum quotes required (default: 3)
- `--db-path PATH`: Database path (default: data/ai_infant.db)

### Session Mode
Run a timed research session with predefined questions:
```bash
python -m ai_infant session --duration 15
```

Options:
- `--duration N`: Session duration in minutes (default: 15)
- `--questions Q1 Q2 Q3`: Custom list of questions to research
- `--db-path PATH`: Database path (default: data/ai_infant.db)

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
