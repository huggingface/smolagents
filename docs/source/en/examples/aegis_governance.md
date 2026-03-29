# Governance with Aegis

[Aegis](https://github.com/Acacian/aegis) is an open-source governance runtime that auto-instruments smolagents (and other AI frameworks) with security guardrails. It adds prompt injection detection, PII masking, policy-as-code enforcement, and audit trail -- without modifying your agent code.

## Quick Start

Install Aegis alongside smolagents:

```bash
pip install agent-aegis smolagents
```

Initialize Aegis before creating your agent:

```python
import aegis
from smolagents import CodeAgent, HfApiModel

# Auto-instrument all supported frameworks including smolagents
aegis.auto_instrument()

model = HfApiModel()
agent = CodeAgent(tools=[], model=model)

# Aegis transparently applies governance policies to all LLM calls
agent.run("What is the capital of France?")
```

## Policy Configuration

Define governance rules in a YAML policy file:

```yaml
# policy.yaml
version: "1.0"
guardrails:
  injection_detection:
    enabled: true
    action: block
  pii_masking:
    enabled: true
    mask_types: [email, phone, ssn]
```

Load the policy:

```python
import aegis

aegis.auto_instrument()
aegis.init(policy_path="policy.yaml")
```

## Key Features

| Feature | Description |
|---|---|
| **Auto-instrumentation** | One-line setup, no code changes needed |
| **Prompt Injection Detection** | 107 detection patterns across 13 categories |
| **PII Masking** | Automatic redaction of sensitive data |
| **Policy-as-Code** | YAML-based governance policies with `aegis plan/test` |
| **Audit Trail** | Complete log of all LLM interactions |

## References

- [Aegis Documentation](https://acacian.github.io/aegis/)
- [Aegis on PyPI](https://pypi.org/project/agent-aegis/)
- [Aegis GitHub](https://github.com/Acacian/aegis)
