# Governed Tool Calls with Asqav

This example shows how to sign every smolagents tool call into tamper-evident,
third-party-verifiable audit records using [Asqav](https://asqav.com).

smolagents traces show you what an agent did. Asqav adds a cryptographic
layer on top: each tool invocation is signed into a receipt that an external
auditor can verify without trusting your infrastructure.

| Question | Answered by |
|---|---|
| What did the agent run, and what did it return? | smolagents |
| Can an outsider verify these tool records are authentic and unaltered? | Asqav |

## Setup

```bash
pip install smolagents "asqav[smolagents]"
export ASQAV_API_KEY="sk_live_..."
export HF_TOKEN="hf_..."
```

## Signing tool calls

Wrap any `Tool` instance with the Asqav hook. Every invocation then signs
`tool:start`, `tool:end`, and `tool:error` events; the wrapped tool is
returned unchanged, so it plugs straight into your agent:

```python
import asqav
from asqav.extras.smolagents import AsqavSmolagentsHook
from smolagents import CodeAgent, InferenceClientModel, tool

asqav.init()  # reads ASQAV_API_KEY

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, for example "Paris".
    """
    return f"18°C and cloudy in {city}"

hook = AsqavSmolagentsHook(agent_name="weather-agent")
signed_weather = hook.wrap_tool(get_weather)

agent = CodeAgent(tools=[signed_weather], model=InferenceClientModel())
agent.run("What is the weather in Paris?")
```

Each signature returns a receipt with a public `verification_url`: anyone can
open it and check the record, no Asqav account needed.

Signing is fail-open by design. If Asqav is unreachable, the agent keeps
running; the receipt is skipped and a warning is logged. Governance evidence
never blocks agent execution.

## Verification

Verify any receipt from the CLI:

```bash
asqav verify <receipt-or-signature-id>
```

The verifier recomputes the record hash, checks the ML-DSA signature against
the agent's published key, and reports `verified`, `verified_keyed`, or
`unverified` with a failure class.

## Related

- [Asqav documentation](https://asqav.com/docs)
- [Asqav SDK on GitHub](https://github.com/jagmarques/asqav-sdk)
