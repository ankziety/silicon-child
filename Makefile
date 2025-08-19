.PHONY: verify fmt typecheck test install clean help

# Default target
.DEFAULT_GOAL := help

# Python settings
PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin
PYTHON_VENV := $(VENV_BIN)/python
PIP_VENV := $(VENV_BIN)/pip

# Tool paths
RUFF := $(VENV_BIN)/ruff
MYPY := $(VENV_BIN)/mypy
PYTEST := $(VENV_BIN)/pytest

# Install development dependencies
install: $(VENV)/bin/activate
	$(PIP_VENV) install -e .[dev]

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP_VENV) install --upgrade pip setuptools wheel

# Format code
fmt: $(RUFF)
	$(RUFF) format .
	$(RUFF) check . --fix

# Type checking
typecheck: $(MYPY)
	$(MYPY) --strict .

# Run tests
test: $(PYTEST)
	$(PYTEST) -q tests/

# Run all verification steps
verify: fmt typecheck test
	@echo "All checks passed ✓"

# Clean build artifacts
clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Show help
help:
	@echo "Available targets:"
	@echo "  install    - Set up development environment"
	@echo "  verify     - Run all checks (fmt + typecheck + test)"
	@echo "  fmt        - Format code and fix linting issues"
	@echo "  typecheck  - Run mypy type checking"
	@echo "  test       - Run pytest test suite"
	@echo "  clean      - Clean build artifacts"
	@echo "  help       - Show this help message"
