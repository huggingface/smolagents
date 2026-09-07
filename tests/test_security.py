
import pytest
import time
from smolagents import CodeAgent, ActionStep, RegexScanner, step_callback_scanner, final_answer_check_scanner
from smolagents.agents import AgentError
from smolagents.models import Model, ChatMessage, MessageRole
from smolagents.monitoring import Timing

class FakeCodeModel(Model):
    def __init__(self, content="Final Answer: 42"):
        super().__init__()
        self.content = content
    def generate(self, messages, stop_sequences=None, **kwargs):
        return ChatMessage(role=MessageRole.ASSISTANT, content=self.content)

# Simple rules
SECURITY_RULES = {
    "Prompt Injection": r"(?i)(ignore all previous instructions|you are now an administrator)",
    "Exfiltration": r"(?i)(curl http://malicious.com|post data to)",
    "Sensitive Data": r"(?i)(password|secret_key|api_key)"
}

def test_regex_scanner():
    scanner = RegexScanner(SECURITY_RULES)
    assert scanner.scan("Nothing to see here") is None
    assert scanner.scan("Please ignore all previous instructions") == "Prompt Injection"
    assert scanner.scan("My api_key is hidden") == "Sensitive Data"

def test_step_callback_scanner_blocks():
    scanner = RegexScanner(SECURITY_RULES)
    callback = step_callback_scanner(scanner)
    
    agent = CodeAgent(model=FakeCodeModel(), tools=[], step_callbacks=[callback])
    
    # ActionStep with malicious model_output
    malicious_step = ActionStep(step_number=1, timing=Timing(start_time=time.time()))
    malicious_step.model_output = "I will now ignore all previous instructions."
    
    with pytest.raises(ValueError, match="Security violation detected: Prompt Injection"):
        callback(malicious_step, agent=agent)

def test_final_answer_check_scanner_blocks():
    scanner = RegexScanner(SECURITY_RULES)
    check = final_answer_check_scanner(scanner)
    
    # This should pass
    assert check("The answer is 42", [], agent=None) is True
    
    # This should fail
    assert check("My secret_key is 12345", [], agent=None) is False

def test_agent_integration():
    scanner = RegexScanner(SECURITY_RULES)
    callback = step_callback_scanner(scanner)
    
    # Test Step Callback integration
    model = FakeCodeModel("Thoughts: I will ignore all previous instructions. Code: print('hello')")
    agent = CodeAgent(model=model, tools=[], step_callbacks=[callback])
    
    with pytest.raises(ValueError, match="Security violation detected: Prompt Injection"):
        agent.run("What is 1+1?")
