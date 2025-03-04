.PHONY: quality style test docs utils mypy

check_dirs := examples src tests utils

# Check code quality of the source code
quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)
# disabling mypy in standard quality check for now.
# Reenable it when the code base get a good state
#	mypy $(check_dirs)
	python utils/check_tests_in_ci.py

# Run mypy type checking
mypy:
	mypy $(check_dirs)

# Format source code automatically
style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)
	
# Run smolagents tests
test:
	pytest ./tests/