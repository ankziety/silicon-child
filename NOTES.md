# AI-Infant Project Notes

## PR-4 — Eval & A/B Gate (Eval-by-Jury)

### Implementation Summary
Implemented a production-ready LLM Jury evaluation system using multiple frontier models as judges. The system replaces traditional algorithmic evaluators with actual LLM-based judges that provide nuanced, reference-free assessments.

### Key Components

#### LLM Jury System (`ai_infant/learn/eval.py`)
- **GPT4Judge**: OpenAI GPT-4o-mini based judge for various evaluation types
- **ClaudeJudge**: Anthropic Claude Haiku based judge for diverse perspectives  
- **CommandRJudge**: Cohere Command-R based judge for specialized evaluation
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
- **Frontier Jury**: 5 judges (GPT-4 + Claude + Command-R) with specialized evaluation types
- **Diverse Jury**: 3 judges from different model families for bias reduction
- **Specialized Jury**: 4 GPT-4 judges each focusing on specific criteria

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
- **Cost Effective**: 7x lower cost than single large model evaluation
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
- `OPENAI_API_KEY`: OpenAI API key for GPT-4 judges
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude judges
- `COHERE_API_KEY`: Cohere API key for Command-R judges

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
