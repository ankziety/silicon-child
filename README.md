# AI-Infant 🤖🧠

A sophisticated, long-running research agent that combines **adaptive reasoning**, **continuous learning**, and **vision-based web automation** to conduct intelligent research. The system ingests data from the web, records complete provenance, learns incrementally from its experiences, and provides comprehensive research reports.

## Core Philosophy

AI-Infant implements the concept of **"emergent curiosity"** - like how the human brain develops specialized neurons (e.g., the "Jennifer Aniston neuron") through natural learning experiences. The system doesn't have pre-programmed research topics but develops its own interests organically through research experiences.

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

## Key Features

### 🤔 Adaptive Reasoning Engine
- **Human-like Research Process**: Shows AI's thinking process in real-time with detailed reasoning traces
- **Knowledge Gap Identification**: Automatically identifies what it still needs to know
- **Iterative Research**: Searches based on identified knowledge gaps, not just initial queries
- **Transparent Decision Making**: Each thought includes confidence scores and evidence sources
- **Working Memory**: Maintains context and builds upon previous findings

### 🧠 Continuous Learning System
- **Online Model Updates**: Model learns and improves during research sessions
- **LLM Jury Evaluation**: Multi-model evaluation system for quality assessment
- **Dataset Selection**: Intelligent filtering of high-quality learning examples
- **LoRA Training**: Parameter-efficient fine-tuning with resume-safe checkpoints
- **A/B Model Promotion**: Automated evaluation and promotion of improved model versions

### 👁️ Vision-Based Browser Automation
- **Visual Web Interaction**: Uses vision models (GPT-4V, Claude) to understand web pages visually
- **Intelligent Navigation**: Makes decisions based on visual analysis of page layouts
- **Goal-Oriented Automation**: Executes actions to achieve user-specified research goals
- **Multi-Provider Support**: Compatible with OpenAI, Anthropic, and local vision models
- **Action Confidence Scoring**: Only executes high-confidence automation actions

### 🏗️ Modular Architecture
- **Schema-Based Contracts**: Immutable data contracts (DocV1, TraceV1, JobV1)
- **Component Separation**: Clean separation between data ingestion, processing, and learning
- **Extensible Design**: Easy to add new vision providers, evaluation models, or learning algorithms
- **Robust Error Handling**: Graceful degradation with comprehensive logging

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

## Architecture Overview

AI-Infant implements a sophisticated layered architecture designed for reliability, extensibility, and research-grade operation:

### 🏗️ System Layers

1. **Schema Contracts (Foundation)**
   - **DocV1**: Document ingestion and storage contract
   - **TraceV1**: Execution trace logging for complete audit trails
   - **JobV1**: Job orchestration and metadata tracking
   - **Immutable Design**: Schema versions never change, only add new versions

2. **Data Processing Pipeline**
   - **Crawl Layer**: Browser automation and web data ingestion
   - **Text Layer**: Content parsing, cleaning, and analysis
   - **Learn Layer**: Continuous learning and model improvement
   - **Plan Layer**: Research planning and policy execution

3. **Core Orchestration**
   - **Adaptive Loop**: Main research orchestration engine
   - **Reasoning Engine**: Thought process management and knowledge gap identification
   - **Store**: Centralized data persistence with DuckDB + JSONL

### 🔄 Research Workflow

1. **Question Analysis**: AI analyzes research questions and generates initial search queries
2. **Web Exploration**: Vision-based browser automation explores relevant web content
3. **Content Analysis**: Extracts facts, identifies knowledge gaps, forms hypotheses
4. **Iterative Research**: Searches based on knowledge gaps in human-like research process
5. **Learning & Adaptation**: Updates internal models with new knowledge
6. **Conclusion Formation**: Synthesizes findings into coherent conclusions
7. **Final Answer**: Generates comprehensive research reports

### 🛡️ Quality Assurance

- **LLM Jury Evaluation**: Multi-model assessment of research quality
- **Confidence Scoring**: All decisions include uncertainty estimates
- **Audit Trails**: Complete provenance tracking for all operations
- **Error Recovery**: Robust error handling with graceful degradation

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

## 🔬 Research Process & Learning

### Human-Like Research Behavior

AI-Infant mimics human research patterns:

- **Shows Thinking**: Displays reasoning process in real-time logs
- **Identifies Gaps**: Recognizes what it still needs to know
- **Iterative Search**: Searches based on knowledge gaps, not just initial queries
- **Forms Conclusions**: Synthesizes evidence into coherent conclusions
- **Learns Continuously**: Model updates during research like human brain development

### Emergent Curiosity

The system develops genuine curiosity organically:
- **No Pre-programmed Interests**: Doesn't have predetermined research topics
- **Organic Development**: Interests emerge from research process itself
- **Natural Specialization**: Like the Jennifer Aniston neuron, develops specialized knowledge areas
- **Autonomous Decision Making**: Chooses research topics based on its own curiosity

### Learning Examples

During research, the system:
1. **Analyzes Content**: Extracts key facts and insights from web content
2. **Generates Hypotheses**: Forms testable hypotheses about the research topic
3. **Identifies Gaps**: Finds missing information and unanswered questions
4. **Updates Models**: Improves its internal models with new knowledge
5. **Builds Confidence**: Assigns confidence scores to all findings

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
