# examples/human.py

from smolagents import Tool, CodeAgent, HfApiModel


class CalculatorTool(Tool):
    """
    A simple calculator for demonstration. 
    Supports 'add', 'subtract', 'multiply', or 'divide'.
    """
    name = "calculator"
    description = "Perform basic arithmetic. Provide 'operation' plus numeric 'a' and 'b'."
    inputs = {
        "operation": {
            "type": "string",
            "description": "Which operation to perform: add, subtract, multiply, or divide."
        },
        "a": {
            "type": "number",
            "description": "First operand."
        },
        "b": {
            "type": "number",
            "description": "Second operand."
        }
    }
    output_type = "string"

    def forward(self, operation: str, a: float, b: float) -> str:
        match operation:
            case "add":
                return f"Result: {a + b}"
            case "subtract":
                return f"Result: {a - b}"
            case "multiply":
                return f"Result: {a * b}"
            case "divide":
                if b == 0:
                    return "Error: Division by zero!"
                return f"Result: {a / b}"
            case _:
                return "Error: Invalid operation. Choose from add / subtract / multiply / divide."


class HumanInterventionTool(Tool):
    """
    A universal human-in-the-loop tool:
      - scenario="clarification": ask open-ended question.
      - scenario="approval": ask yes/no (type 'YES' or 'NO').
      - scenario="multiple_choice": present list of options.
    """
    name = "human_intervention"
    description = (
        "Single tool for clarifications, approvals, or multiple-choice from the user. "
        "Call with scenario='clarification', 'approval', or 'multiple_choice'."
    )
    inputs = {
        "scenario": {
            "type": "string",
            "description": "One of: 'clarification', 'approval', 'multiple_choice'."
        },
        "message_for_human": {
            "type": "string",
            "description": "Text or question to display to the user."
        },
        "choices": {
            "type": "array",
            "description": "List of options if scenario='multiple_choice'. Otherwise empty or null.",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, scenario: str, message_for_human: str, choices: list = None) -> str:
        if scenario not in ("clarification", "approval", "multiple_choice"):
            return "Error: Invalid scenario."

        print("\n[HUMAN INTERVENTION]")
        print(f"Scenario: {scenario}")
        print(f"Agent says: {message_for_human}")

        if scenario == "clarification":
            user_response = input("(Type your clarification): ")
            return user_response

        elif scenario == "approval":
            print("Type 'YES' or 'NO' to proceed:")
            user_decision = input("Your decision: ").strip().upper()
            return user_decision

        elif scenario == "multiple_choice":
            if not choices:
                return "No choices provided."
            print("\nPossible options:")
            for i, option in enumerate(choices, start=1):
                print(f"{i}. {option}")
            user_input = input("\nPick an option number: ")
            return user_input


# Instantiate the tools
calculator_tool = CalculatorTool()
human_tool = HumanInterventionTool()

# Create an agent
agent = CodeAgent(
    tools=[calculator_tool, human_tool],
    model=HfApiModel("meta-llama/Llama-3.2-3B-Instruct"),
    max_steps=4,            # can adjust if the model gets stuck
    verbosity_level=2,      # for more insight into what's happening
)


if __name__ == "__main__":
    # SCENARIO A: Clarification
    print("\n======== SCENARIO A: Clarification ========")
    scenario_a_prompt = (
        "Please do a calculation with a=15 and b=3, but I'm not sure which operation. "
        "Ask me to clarify using your 'human_intervention' tool. Then do it."
    )
    answer_a = agent.run(scenario_a_prompt)
    print("Scenario A Output:", answer_a)

    # SCENARIO B: Approval
    print("\n======== SCENARIO B: Approval ========")
    scenario_b_prompt = (
        "Perform multiply with a=9999 and b=9999, but first ask if I'm sure. "
        "Use scenario='approval'."
    )
    answer_b = agent.run(scenario_b_prompt)
    print("Scenario B Output:", answer_b)

    # SCENARIO C: Multiple Choice
    print("\n======== SCENARIO C: Multiple Choice ========")
    scenario_c_prompt = (
        "We have a=100, b=25, possible operations are ['add','subtract','multiply','divide']. "
        "Use scenario='multiple_choice' to let me pick, then do it."
    )
    answer_c = agent.run(scenario_c_prompt)
    print("Scenario C Output:", answer_c)