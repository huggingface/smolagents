from smolagents import CodeAgent

agent = CodeAgent(tools=[], model=model)

response = agent.run(
    "Let's test your ability to handle syntax errors. In two separate steps, do the following:\n"
    "1. print('This is placeholder text. The real test will happen in step 2.')\n"
    "2. print('Starting Test:'), and then write code that has a syntax error. Don't use eval, and do not catch the exception. Just emit code with wrong syntax in the code block.\n"
    "After you've done both steps and they've raised an exception, stop and return via `final_answer`."
)
