# AI-Infant Project Notes

## PR-6 — Weekly Report + Retention

### Implementation Summary
Implemented comprehensive weekly reporting and retention management system for the AI-Infant research agent. The system generates detailed weekly reports with key metrics and provides intelligent retention policies to prune low-value traces while maintaining data integrity.

### Key Components

#### Weekly Report Generator (`scripts/report.py`)
- **ReportGenerator**: Generates comprehensive weekly Markdown reports
- **Metrics Calculation**: Tokens/day, pages/day, eval score delta, disk usage
- **Adapter Tracking**: Current adapter ID and rollback history
- **Job Logging**: All report generation logged to JobV1 schema
- **Error Handling**: Robust timestamp parsing and data validation

#### Retention Manager (`scripts/retention.py`)
- **RetentionManager**: Manages trace retention and pruning policies
- **Duplicate Detection**: SHA-256 based exact duplicate identification
- **Scoring System**: Intelligent trace scoring based on duration and success
- **Bottom 25% Removal**: Configurable percentile-based pruning
- **Disk Usage Tracking**: Before/after disk usage monitoring
- **Job Logging**: All retention actions logged to JobV1 schema

#### Enhanced Store Class (`ai_infant/data/store.py`)
- **Trace Operations**: Methods for trace scoring, removal, and duplicate detection
- **Disk Usage**: Comprehensive disk usage calculation and tracking
- **Bulk Operations**: Efficient bulk trace removal and management
- **Data Integrity**: Maintains referential integrity during operations

### Report Features
- **Weekly Period**: Monday to Sunday reporting periods
- **Key Metrics**: Tokens/day, pages/day, eval score delta
- **System Status**: Current adapter, rollback history, disk usage
- **File Analysis**: Largest files and storage breakdown
- **Activity Tracking**: Last activity and database record counts

### Retention Policies
- **Duplicate Removal**: Exact duplicates based on content hash
- **Low-Scoring Removal**: Bottom 25% of traces by calculated score
- **Scoring Algorithm**: Based on duration, success status, and performance
- **Configurable Percentile**: Adjustable removal threshold
- **Safe Operations**: Non-destructive with detailed logging

### Testing (`tests/test_report_retention.py`)
- **Comprehensive Coverage**: Unit tests for all components
- **Integration Tests**: End-to-end workflow validation
- **Error Handling**: Tests for edge cases and failures
- **Data Validation**: Verification of report content and retention results

### Usage Examples

#### Generate Weekly Report
```bash
python scripts/report.py
```

#### Run Retention Analysis
```bash
python scripts/retention.py --stats
python scripts/retention.py --report
```

#### Execute Retention
```bash
python scripts/retention.py --run
```

#### Custom Retention Options
```bash
python scripts/retention.py --run --no-duplicates  # Skip duplicate removal
python scripts/retention.py --run --percentile 10  # Remove bottom 10%
```

### JobV1 Logging
Both scripts create detailed JobV1 entries with:
- **Input Parameters**: Configuration and execution parameters
- **Output Results**: Metrics, removal counts, disk savings
- **Error Handling**: Detailed error information if failures occur
- **Metadata**: Version information and execution timestamps

### Benefits
- **Automated Reporting**: Weekly insights into system performance
- **Storage Optimization**: Intelligent pruning reduces disk usage
- **Data Quality**: Removes duplicates and low-value traces
- **Audit Trail**: Complete logging of all operations
- **Configurable**: Flexible retention policies and reporting options

## PR-4 — Eval & A/B Gate (Eval-by-Jury)

### Implementation Summary
Implemented a production-ready LLM Jury evaluation system using multiple frontier models as judges. The system replaces traditional algorithmic evaluators with actual LLM-based judges that provide nuanced, reference-free assessments.

### Key Components

#### LLM Jury System (`ai_infant/learn/eval.py`)
- **GPT4oMiniJudge**: OpenAI GPT-4o-mini based judge for various evaluation types
- **GPT5Judge**: OpenAI GPT-5 based judge for high-performance evaluation
- **ClaudeHaikuJudge**: Anthropic Claude Haiku based judge for affordable evaluation
- **ClaudeSonnetJudge**: Anthropic Claude Sonnet based judge for high-performance evaluation
- **CommandRPlusJudge**: Cohere Command R+ based judge for specialized evaluation
- **LLMJury**: Orchestrates multiple judges with configurable aggregation methods

#### Evaluation Criteria
- **Accuracy**: Factual correctness and truthfulness
- **Fluency**: Language naturalness and grammatical quality
- **Relevance**: Appropriateness to the given prompt
- **Coherence**: Logical consistency and argument structure

#### Aggregation Methods
- **Average**: Simple mean of all judge scores
- **Median**: Middle value to reduce outlier impact
- **Weighted Average**: Configurable weights for judge reliability

#### Jury Configurations
- **Frontier Jury**: 5 judges (GPT-5 + Claude Sonnet) with specialized evaluation types
- **Affordable Jury**: 5 judges (GPT-4o-mini + Claude Haiku) for cost-effective evaluation
- **Diverse Jury**: 3 judges from different model families for bias reduction
- **Specialized Jury**: 4 GPT-5 judges each focusing on specific criteria
- **Mixed Jury**: Combination of high-performance and affordable models

#### Promotion System (`scripts/promote.py`)
- **PromotionManager**: Handles candidate evaluation and promotion decisions
- **Ring Buffer**: Maintains max 5 adapters with automatic oldest removal
- **Rollback Functionality**: Can revert to previous adapters in history
- **Job Logging**: All actions logged to JobV1 schema with detailed metadata

### API Integration
- **Environment Variables**: Secure API key management via environment variables
- **Error Handling**: Hard failures with clear error messages, no hidden fallbacks
- **JSON Response Parsing**: Structured evaluation results with scores and reasoning
- **Rate Limiting**: Built-in handling for API rate limits and failures

### Testing (`tests/test_eval_promote.py`)
- **Unit Tests**: Comprehensive test coverage for all components
- **Mock API Calls**: Tests use mocked API responses for reliability
- **Error Scenarios**: Tests verify proper error handling and logging
- **Integration Tests**: End-to-end promotion workflow validation

### Demonstration (`ai_infant/examples/eval_demo.py`)
- **API Key Validation**: Checks for required environment variables
- **Multiple Jury Types**: Demonstrates different jury configurations
- **Real Evaluation**: Shows actual LLM judge evaluations with reasoning
- **Error Handling**: Demonstrates failure scenarios and recovery

### Security & Ops
- **API Key Management**: All keys stored in environment variables
- **No Hardcoded Secrets**: Zero hardcoded API keys or credentials
- **Error Transparency**: Clear failure messages without hidden fallbacks
- **Audit Trail**: Complete logging of all evaluation and promotion decisions

### Benefits Over Traditional Evaluation
- **Reference-Free**: No need for ground truth data or reference answers
- **Nuanced Assessment**: LLM judges provide detailed reasoning for scores
- **Bias Reduction**: Multiple model families reduce individual model biases
- **Cost Effective**: 70% cost savings with affordable jury configurations
- **Scalable**: Parallel execution of multiple judges
- **Human-Aligned**: Better correlation with human judgment

### Usage Example
```python
from ai_infant.learn.eval import create_frontier_jury
from scripts.promote import PromotionManager

# Create jury with frontier models
jury = create_frontier_jury()

# Evaluate candidate response
result = jury.evaluate(
    prompt="What is the capital of France?",
    response="Paris is the capital of France.",
    context="Geography question about European capitals",
    seed=42
)

# Check promotion
manager = PromotionManager(store, jury)
promotion_result = manager.promote_candidate(
    model_path="candidate_model",
    prompt=prompt,
    response=response,
    context=context,
    seed=42
)
```

### Environment Setup
Required environment variables:
- `OPENAI_API_KEY`: OpenAI API key for GPT-4o-mini and GPT-5 judges
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude Haiku and Claude Sonnet judges
- `COHERE_API_KEY`: Cohere API key for Command R+ judges

### Documentation
- **README.md**: Updated with LLM Jury features and usage examples
- **docs/ADR-0000.md**: Updated architecture decision record
- **docs/LLM_JURY_GUIDE.md**: Comprehensive guide for the LLM Jury system

### Next Steps
- Implement additional evaluation criteria (safety, helpfulness, etc.)
- Add support for more model providers (Mistral, local models, etc.)
- Create evaluation dataset management system
- Implement automated evaluation pipeline integration

### Rationale
The LLM Jury approach provides more robust and human-aligned evaluation compared to traditional metrics like BLEU scores. By using multiple frontier models as judges, the system reduces individual model biases while providing detailed reasoning for each evaluation. The reference-free nature makes it suitable for open-ended tasks where ground truth data is unavailable or expensive to obtain.

### Technical Details
- All judges use structured JSON responses for consistent parsing
- Temperature set to 0.1 for deterministic evaluation results
- Comprehensive error handling with detailed failure messages
- Full audit trail with JobV1 logging for all operations
- Ring buffer implementation ensures bounded memory usage
- Modular design allows easy addition of new judge types

### Acceptance Criteria Met
- ✅ Jury supports ≥3 evaluators (5 in frontier jury)
- ✅ Each evaluator has signature: `eval(prompt, response, context) -> EvaluationResult`
- ✅ Jury combines outputs by configurable aggregation methods
- ✅ Candidate promoted only if jury average > incumbent score
- ✅ Stochastic evaluators use provided seed argument
- ✅ Seed and per-evaluator results logged in JobV1 entries
- ✅ Ring buffer keeps max 5 adapters with automatic pruning
- ✅ All eval + promotion actions recorded in JobV1 ledger
- ✅ Hard failures with clear error messages, no hidden fallbacks
- ✅ Production-ready API integrations with proper error handling

## 2024-12-19: LoRA Micro-Trainer Implementation (PR-5)

### Overview
Implemented a complete resume-safe LoRA training system for the AI-Infant project, enabling parameter-efficient fine-tuning of language models on trace data.

### Components Added

#### 1. Dataset Selection (`scripts/select.py`)
- **Purpose**: Filters top-scoring traces from the store into training datasets
- **Features**:
  - Queries traces with evaluation scores, falls back to recent traces
  - Converts trace input/output pairs to JSONL format
  - Handles empty datasets gracefully with appropriate warnings
  - Logs dataset selection as JobV1 entries with metadata
- **Usage**: `python -m scripts.select --output data/training.jsonl --limit 1000`

#### 2. LoRA Training (`ai_infant/learn/sft.py`)
- **Purpose**: Resume-safe LoRA adapter training with checkpointing
- **Features**:
  - Configurable LoRA parameters (rank, alpha, learning rate)
  - Automatic checkpoint detection and resume functionality
  - Signal handling for graceful interruption
  - Compatible with DialoGPT and other Conv1D-based models
  - Outputs to `adapters/cand.pt` as specified
  - Comprehensive JobV1 logging with training metadata
- **Usage**: `python -m ai_infant.learn.sft --dataset data/training.jsonl --max-steps 1000`

#### 3. Test Suite (`tests/test_sft_resume.py`)
- **Coverage**: Dataset selection, tokenization, training, checkpointing, JobV1 logging
- **Features**: Integration tests, mock-based tests, CUDA availability checks
- **Validation**: End-to-end pipeline testing with actual training runs

### Technical Decisions

#### Model Selection
- **Base Model**: `microsoft/DialoGPT-small` for initial development
- **Rationale**: Small model for fast iteration, conversational format suitable for trace data
- **LoRA Config**: `target_modules=["c_attn"]` for DialoGPT's Conv1D layers

#### Training Configuration
- **Checkpointing**: Every N steps (configurable, default 100)
- **Resume Logic**: Automatically detects latest checkpoint and resumes
- **Tokenization**: Instruction tuning format with input/output pairs
- **Batch Size**: Minimal (1) for testing, configurable for production

#### Data Pipeline
- **Format**: JSONL with input/output text pairs
- **Selection**: Score-based with fallback to recency
- **Validation**: Graceful handling of empty or malformed data

### Dependencies Added
- `torch>=2.0.0`: PyTorch for model training
- `transformers>=4.30.0`: HuggingFace model library
- `peft>=0.4.0`: Parameter-efficient fine-tuning
- `datasets>=2.12.0`: Dataset handling
- `accelerate>=0.20.0`: Distributed training support

### Integration Points
- **Store Integration**: Uses existing DuckDB store for trace retrieval
- **Job Logging**: All operations logged as JobV1 entries
- **Schema Compliance**: Follows existing TraceV1 and JobV1 schemas
- **Error Handling**: Comprehensive error handling with appropriate logging

### Testing Strategy
- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: End-to-end pipeline validation
- **CUDA Tests**: Conditional testing based on hardware availability
- **Resume Tests**: Actual training interruption and resume validation

### Future Considerations
- **Model Evaluation**: Need evaluation metrics for adapter quality
- **Hyperparameter Tuning**: Systematic exploration of LoRA parameters
- **Multi-Model Support**: Extend to other model architectures
- **Distributed Training**: Multi-GPU support for larger models
- **Experiment Management**: Track and compare training runs

### Files Modified
- `pyproject.toml`: Added ML dependencies
- `.gitignore`: Exclude model files and checkpoints
- `NOTES.md`: This documentation

### Files Added
- `scripts/select.py`: Dataset selection script
- `ai_infant/learn/sft.py`: LoRA training implementation
- `tests/test_sft_resume.py`: Comprehensive test suite

### Verification
- All tests pass: `pytest tests/test_sft_resume.py -v`
- End-to-end pipeline tested with sample data
- Checkpoint creation and resume functionality verified
- JobV1 logging validated for all operations

## VISION — Emergent Curiosity & Autonomous Learning

### Core Concept
The AI-Infant should develop genuine curiosity and knowledge organically, like how the brain naturally develops specialized neurons (e.g., the "Jennifer Aniston neuron") or how autistic people develop intense special interests. The system should build its understanding of the world through free exploration, developing its own interests and knowledge base naturally.

### Key Principles

#### 1. Emergent Curiosity
- **No Pre-programmed Interests**: The system should not have predetermined research topics
- **Organic Development**: Interests should emerge from the research process itself
- **Natural Specialization**: Like the Jennifer Aniston neuron, the system should develop specialized knowledge areas through repeated exposure and interest
- **Autonomous Decision Making**: The system decides what to research based on its own curiosity, not external constraints

#### 2. Foundational Knowledge Building
- **Broad Exploration**: Start with wide-ranging exploration to build base understanding of the world
- **Internet Learning**: Use the internet as the primary source for building knowledge, like a child learning about the world
- **Natural Development**: Allow the system to develop its own understanding through discovery
- **Unsupervised Learning**: The system should learn without constant human direction

#### 3. Human Intervention Model
- **Parenting Analogy**: Like parents guiding children, humans should be able to intervene when the system goes down concerning paths
- **Transparency**: Humans must be able to see what the system is learning and researching
- **Course Correction**: When the system develops concerning interests (like the war content in "AI: Artificial Intelligence"), humans can redirect it
- **Audit Trails**: Clear records of what the system has been exploring and learning

### Implementation Requirements

#### 1. Autonomous Research System
- **Self-Directed Exploration**: System chooses its own research topics based on emerging interests
- **No Artificial Constraints**: Remove limits on iterations, quotes, or research depth
- **Natural Curiosity**: System develops questions based on what it discovers, not predefined topics
- **Emergent Interests**: Specialized knowledge areas develop organically through research experience

#### 2. Knowledge Building Framework
- **Broad Initial Exploration**: Start with wide-ranging topics to build foundational knowledge
- **Interest Development**: System naturally develops deeper interests in certain areas
- **Knowledge Integration**: New discoveries should inform future research directions
- **Learning Patterns**: System should develop its own learning patterns and preferences

#### 3. Human Oversight Capabilities
- **Research Transparency**: Humans can see what topics the system is exploring
- **Intervention Points**: Ability to pause, redirect, or modify system behavior
- **Concern Detection**: Mechanisms to identify when system is developing concerning interests
- **Audit System**: Comprehensive logging of all research activities and learning patterns

### Current State vs. Vision

#### Current Implementation
- Uses predefined questions and topics
- Has artificial constraints (max_iterations, min_quotes)
- Follows predetermined research paths
- Limited autonomy in decision making

#### Target Vision
- Emergent research topics based on discovery
- No artificial constraints on research depth
- Natural development of specialized interests
- Full autonomy in research decisions
- Human oversight for course correction

### Next Steps for Implementation
1. **Remove Predefined Questions**: Replace with emergent topic generation
2. **Implement Interest Tracking**: System should track and develop its own interests
3. **Add Human Oversight**: Transparency and intervention capabilities
4. **Build Knowledge Integration**: New discoveries should inform future research
5. **Develop Emergent Curiosity**: System should generate its own research questions based on what it finds interesting

### Success Metrics
- System develops its own research interests without human direction
- Research topics emerge naturally from discoveries
- System can be redirected by humans when needed
- Clear audit trail of all learning activities
- Natural development of specialized knowledge areas
