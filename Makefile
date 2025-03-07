.PHONY: quality style test docs utils

PYTHON ?= python

check_dirs := examples src tests utils

# Check code quality of the source code
quality:
	$(PYTHON) -m ruff check $(check_dirs)
	$(PYTHON) -m ruff format --check $(check_dirs)
	$(PYTHON) utils/check_tests_in_ci.py

# Format source code automatically
style:
	$(PYTHON) -m ruff check $(check_dirs) --fix
	$(PYTHON) -m ruff format $(check_dirs)
	
# Run smolagents tests
test:
	$(PYTHON) -m pytest ./tests/
