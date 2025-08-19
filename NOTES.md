# Development Notes

## PR-1: Contracts & CI Setup (2024-01-01)

### Changes Made
- Created JSON schemas for core contracts: DocV1, TraceV1, JobV1
- Established CI pipeline with ruff, mypy, pytest, and coverage
- Added Makefile with verify, fmt, typecheck, test targets
- Created ADR-0000 documenting two-layer architecture and schema rules
- Added comprehensive test suite validating fixtures against schemas
- Configured package structure with proper initialization

### Rationale
- **Schema Stability**: Immutable contracts ensure data integrity across system evolution
- **CI Automation**: Automated validation prevents regressions and maintains code quality
- **Test Coverage**: Fixture validation ensures schemas work correctly with real data
- **Documentation**: ADR establishes architectural principles for future development

### Technical Details
- All schemas follow JSON Schema Draft-07 specification
- Strict mypy configuration ensures type safety
- 100% test coverage requirement enforced
- Ruff handles formatting and linting
- pytest with coverage reporting

### Schema Contracts
- **DocV1**: Document ingestion with metadata and content validation
- **TraceV1**: Execution tracing with operation status and timing
- **JobV1**: Job ledger for tracking all major system actions

### Next Steps
- Implement core modules (crawl, text, learn, plan)
- Add data storage layer (DuckDB + JSONL)
- Create orchestration scripts
