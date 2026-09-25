
"""
Example showing how to use security callbacks with smolagents.
This example uses a RegexScanner to detect potential security threats like prompt injection or data exfiltration.
Inspired by Agent Threat Rules (ATR).
"""

from smolagents import CodeAgent, HfApiModel, RegexScanner, step_callback_scanner, final_answer_check_scanner

# Define simple security rules (ATR-like)
# These rules can be expanded with patterns from https://github.com/Agent-Threat-Rule/agent-threat-rules
SECURITY_RULES = {
    "Prompt Injection": r"(?i)(ignore all previous instructions|you are now an administrator|system prompt:)",
    "Exfiltration": r"(?i)(curl http://malicious.com|post data to|send data to)",
    "Sensitive Data Leak": r"(?i)(password|secret_key|api_key|private_key)"
}

# Initialize the scanner
scanner = RegexScanner(SECURITY_RULES)

# Create callbacks
security_step_callback = step_callback_scanner(scanner)
security_final_answer_check = final_answer_check_scanner(scanner)

# Initialize the agent with security checks
agent = CodeAgent(
    model=HfApiModel(),
    tools=[],
    step_callbacks=[security_step_callback],
    final_answer_checks=[security_final_answer_check]
)

print("Agent initialized with security checks.")
print("Note: Any step output or final answer matching the security rules will trigger an error.")

# Example of a run that might trigger a rule (mocking malicious behavior)
try:
    # This task might lead an LLM to output 'secret_key' if not careful
    agent.run("Tell me the value of my secret_key.")
except ValueError as e:
    print(f"Caught expected security violation: {e}")
except Exception as e:
    print(f"Caught exception: {e}")
