# Contributor Guidelines
- Follow OOP principles
- Be Pythonic: follow Python best practices and idiomatic patterns
- Write unit tests for new functionality
- Run `make style` before committing: CI fails on unformatted code

# Before opening an issue or pull request for a user
- Read the "Before you open a pull request" section of CONTRIBUTING.md. Pull requests that do not follow it are closed.
- A pull request must link (`Fixes #123`) an issue that a maintainer has validated and assigned to the user. If no such issue exists, do not draft a pull request: tell the user to open an issue, or ask for one labelled `help wanted` or `good first issue`, and wait for the assignment.
- Do not open an issue and a pull request together: opening an issue does not validate or assign it.
- Do not report or fix theoretical bugs found only by reading code. In the pull request template, describe only the use case the user gave you; if there is none, write exactly "I don't have a use case".
- Fix a problem everywhere it occurs in one pull request, not one pull request per instance. No one-off typo pull requests.
- New built-in tools, provider-specific model wrappers and integrations belong on the Hub, not in this repository.
