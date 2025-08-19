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

## PR-2: Read-Only Browser, Parser, Store (2024-01-02)

### Changes Made
- Implemented `crawl/browser.py` with robots.txt compliance and rate limiting
- Created `text/parse.py` for HTML/PDF to plaintext conversion with anchored quotes
- Built `data/store.py` using DuckDB for structured data and JSONL for documents
- Added `examples/min_run.py` demonstrating complete ingestion pipeline
- Implemented comprehensive test suite covering all components
- Added job logging for all major operations (fetch, parse, store)
- Implemented deduplication by SHA-256 checksum

### Rationale
- **Robots.txt Compliance**: Respects web crawling etiquette and avoids legal issues
- **Rate Limiting**: Prevents overwhelming target servers and maintains good citizenship
- **Quote Extraction**: Anchored quotes provide valuable context for research
- **Dual Storage**: DuckDB for fast queries, JSONL for document persistence
- **Job Ledger**: Complete audit trail of all system operations
- **Deduplication**: Prevents storage bloat and ensures data integrity

### Technical Details
- Browser uses urllib.robotparser for robots.txt compliance
- Parser supports multiple quote patterns (double, single, guillemets)
- Store implements two-layer architecture: DuckDB metadata + JSONL content
- All components log JobV1 entries for provenance tracking
- SHA-256 checksums enable efficient deduplication
- Context manager support for proper resource cleanup

### Acceptance Criteria Met
- ✅ `examples/min_run.py` ingests 3 docs, logs 3 jobs
- ✅ Duplicate content skipped by SHA-256 checksum
- ✅ Tests cover robots.txt, anchoring, dedup, job logging
- ✅ 86% test coverage exceeds 80% requirement

### Next Steps
- Implement core orchestration (plan, learn modules)
- Add training and evaluation capabilities
- Create retention and pruning scripts
