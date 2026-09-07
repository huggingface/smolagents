
import re
from typing import Any, Callable, Dict, List, Optional

from .memory import ActionStep, MemoryStep


class RegexScanner:
    """
    A simple regex-based scanner for security checks.

    Args:
        rules (`Dict[str, str]`): A dictionary mapping rule names to regex patterns.
    """

    def __init__(self, rules: Dict[str, str]):
        self.rules = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in rules.items()}

    def scan(self, text: str) -> Optional[str]:
        """
        Scans the text for any matches in the rules.

        Args:
            text (`str`): The text to scan.

        Returns:
            `Optional[str]`: The name of the first rule that matched, or None if no match.
        """
        if not text:
            return None
        for name, pattern in self.rules.items():
            if pattern.search(text):
                return name
        return None


def step_callback_scanner(scanner: RegexScanner, raise_error: bool = True) -> Callable:
    """
    Creates a step callback that scans the step's outputs for security violations.

    Args:
        scanner (`RegexScanner`): The scanner to use.
        raise_error (`bool`, *optional*, defaults to `True`): Whether to raise a `ValueError` if a violation is detected.

    Returns:
        `Callable`: The step callback.
    """

    def callback(step: MemoryStep, agent: Any = None):
        text_to_scan = ""
        if isinstance(step, ActionStep):
            if step.model_output:
                text_to_scan += step.model_output
            if step.action_output:
                text_to_scan += str(step.action_output)
            if step.observations:
                text_to_scan += step.observations

        hit = scanner.scan(text_to_scan)
        if hit and raise_error:
            raise ValueError(f"Security violation detected: {hit}")
        return hit

    return callback


def final_answer_check_scanner(scanner: RegexScanner) -> Callable:
    """
    Creates a final answer check that scans the final answer for security violations.

    Args:
        scanner (`RegexScanner`): The scanner to use.

    Returns:
        `Callable`: The final answer check.
    """

    def check(final_answer: Any, memory: List[MemoryStep], agent: Any = None):
        hit = scanner.scan(str(final_answer))
        return hit is None

    return check
