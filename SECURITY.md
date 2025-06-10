# Security Policy

## Reporting a Vulnerability

To report a security vulnerability, please contact: security@huggingface.co

## Learning More About Security

To learn more about running agents more securely, please see the [Secure Code Execution tutorial](docs/source/en/tutorials/secure_code_execution.mdx) which covers sandboxing with E2B, Docker, and WebAssembly.

### Secure Execution Options

`smolagents` provides several options for secure code execution:

1. **E2B Sandbox**: Uses [E2B](https://e2b.dev/) to run code in a secure, isolated environment.

2. **Docker Sandbox**: Runs code in an isolated Docker container.

3. **WebAssembly Sandbox**: Executes Python code securely in a sandboxed WebAssembly environment using Pyodide and Deno's secure runtime.

We recommend using one of these sandboxed execution options when running untrusted code.
