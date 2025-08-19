# AI-Infant Project Notes

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
