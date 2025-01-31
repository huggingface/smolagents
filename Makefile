.PHONY: quality style test

# quality and style targets are redundant
# they are kept not to introduce a breaking change in users' workflows.
# though style target is being deprecated and will be removed in the future.

# Check code quality of the source code
quality:
	pre-commit run --all-files --show-diff-on-failure --color always

# Format source code automatically
style:
	pre-commit run --all-files --show-diff-on-failure --color always
	echo "\nDeprecated target. Use 'make quality' instead."

# Run smolagents tests
test:
	pytest
