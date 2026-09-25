# Detect prompt injection, tool poisoning and unsafe generated actions during
# an agent run using a step callback, and keep an audit trail (useful for
# EU AI Act Article 12 style verifiable event logging) -- without changing any
# smolagents internals.
#
# This uses ATR (Agent Threat Rules), an open, vendor-neutral detection
# standard for AI agents (MIT licensed; like Sigma, but for prompt injection,
# tool poisoning, MCP attacks and skill compromise). The rules ship bundled in
# the pip package, so no network or rule download is needed.
#
# The pattern -- a step callback that inspects each step -- is engine-agnostic:
# ATR is one concrete rule engine you can swap for any other check.
#
#   pip install smolagents pyatr

from pyatr import scan

from smolagents import CodeAgent, InferenceClientModel, WebSearchTool


# Every finding is recorded here so you keep a tamper-evident trail of what the
# agent saw and did across the whole run.
audit_log: list[dict] = []


def atr_security_callback(memory_step, agent=None):
    """Scan each agent step against ATR rules.

    Registered via ``step_callbacks``, this runs at the end of every step. It
    inspects the model output, the code the CodeAgent is about to run
    (``code_action``), and any tool observations -- which is where indirect
    prompt injection from fetched web pages or files lands.
    """
    parts = []
    for attr in ("model_output", "code_action", "observations"):
        value = getattr(memory_step, attr, None)
        if value:
            parts.append(value if isinstance(value, str) else str(value))
    text = "\n".join(parts)
    if not text.strip():
        return

    for match in scan(text):
        audit_log.append(
            {
                "step": getattr(memory_step, "step_number", None),
                "rule": match.rule_id,
                "title": match.title,
                "severity": match.severity,
            }
        )
        # Enforce lane: stop the agent on a critical finding.
        if match.severity == "critical" and agent is not None:
            print(f"[ATR] BLOCKED by {match.rule_id}: {match.title}")
            agent.interrupt()
        # Hunt lane: record lower-severity findings and let the run continue.
        else:
            print(f"[ATR] flagged {match.rule_id} ({match.severity}): {match.title}")


agent = CodeAgent(
    tools=[WebSearchTool()],
    model=InferenceClientModel(),
    step_callbacks=[atr_security_callback],
)

agent.run("Search for today's weather in Paris and summarise it.")

# Inspect everything ATR flagged during the run.
print("\nAudit trail:")
for entry in audit_log:
    print(entry)
