# Smolagents API Documentation

## smolagents.memory

```python
from smolagents.memory import ...
```

### Classes

#### `ToolCall`

**Methods:**

- `dict(self)`

#### `MemoryStep`

**Methods:**

- `dict(self)`
- `to_messages(self, summary_mode: bool) -> list[ChatMessage]`

#### `ActionStep(MemoryStep)`

**Methods:**

- `dict(self)`
- `to_messages(self, summary_mode: bool) -> list[ChatMessage]`

#### `PlanningStep(MemoryStep)`

**Methods:**

- `to_messages(self, summary_mode: bool) -> list[ChatMessage]`

#### `TaskStep(MemoryStep)`

**Methods:**

- `to_messages(self, summary_mode: bool) -> list[ChatMessage]`

#### `SystemPromptStep(MemoryStep)`

**Methods:**

- `to_messages(self, summary_mode: bool) -> list[ChatMessage]`

#### `FinalAnswerStep(MemoryStep)`

#### `AgentMemory`

Memory for the agent, containing the system prompt and all steps taken by the agent.

    This class is used to store the agent's steps, including tas...

**Methods:**

- `reset(self)` - Reset the agent's memory, clearing all steps and keeping the system prompt.
- `get_succinct_steps(self) -> list[dict]` - Return a succinct representation of the agent's steps, excluding model input messages.
- `get_full_steps(self) -> list[dict]` - Return a full representation of the agent's steps, including model input messages.
- `replay(self, logger: AgentLogger, detailed: bool)` - Prints a pretty replay of the agent's steps.
- `return_full_code(self) -> str` - Returns all code actions from the agent's steps, concatenated as a single script.

#### `CallbackRegistry`

Registry for callbacks that are called at each step of the agent's execution.

    Callbacks are registered by passing a step class and a callback fun...

**Methods:**

- `register(self, step_cls: Type[MemoryStep], callback: Callable)` - Register a callback for a step class.
- `callback(self, memory_step, **kwargs)` - Call callbacks registered for a step type.

## smolagents.monitoring

```python
from smolagents.monitoring import ...
```

### Classes

#### `TokenUsage`

Contains the token usage information for a given step or run.

**Methods:**

- `dict(self)`

#### `Timing`

Contains the timing information for a given step or run.

**Methods:**

- `duration(self)`
- `dict(self)`

#### `Monitor`

**Methods:**

- `get_total_token_counts(self) -> TokenUsage`
- `reset(self)`
- `update_metrics(self, step_log)` - Update the metrics of the monitor.

#### `LogLevel(IntEnum)`

#### `AgentLogger`

**Methods:**

- `log(self, *args, **kwargs) -> None` - Logs a message to the console.
- `log_error(self, error_message: str) -> None`
- `log_markdown(self, content: str, title: str | None, level, style) -> None`
- `log_code(self, title: str, content: str, level: int) -> None`
- `log_rule(self, title: str, level: int) -> None`
- `log_task(self, content: str, subtitle: str, title: str | None, level: LogLevel) -> None`
- `log_messages(self, messages: list[dict], level: LogLevel) -> None`
- `visualize_agent_tree(self, agent)`

### Functions

#### `create_tools_section(tools_dict)`

#### `get_agent_headline(agent, name: str | None)`

#### `build_agent_tree(parent_tree, agent_obj)`

Recursively builds the agent tree.

## smolagents.local_python_executor

```python
from smolagents.local_python_executor import ...
```

### Classes

#### `InterpreterError(ValueError)`

An error raised when the interpreter cannot evaluate a Python expression, due to syntax error or unsupported
    operations.

#### `PrintContainer`

**Methods:**

- `append(self, text)`

#### `BreakException(Exception)`

#### `ContinueException(Exception)`

#### `ReturnException(Exception)`

#### `FinalAnswerException(Exception)`

#### `CodeOutput`

#### `PythonExecutor`

#### `LocalPythonExecutor(PythonExecutor)`

Executor of Python code in a local environment.

    This executor evaluates Python code with restricted access to imports and built-in functions,
   ...

**Methods:**

- `send_variables(self, variables: dict)`
- `send_tools(self, tools: dict[str, Tool])`

### Functions

#### `custom_print(*args)`

#### `nodunder_getattr(obj, name, default)`

#### `check_safer_result(result: Any, static_tools: dict[str, Callable], authorized_imports: list[str])`

Checks if a result is safer according to authorized imports and static tools.

    Args:
        result (Any): The result to check.
        static_tools (dict[str, Callable]): Dictionary of static too...

#### `safer_eval(func: Callable)`

Decorator to enhance the security of an evaluation function by checking its return value.

    Args:
        func (Callable): Evaluation function to be made safer.

    Returns:
        Callable: Safe...

#### `safer_func(func: Callable, static_tools: dict[str, Callable], authorized_imports: list[str])`

Decorator to enhance the security of a function call by checking its return value.

    Args:
        func (Callable): Function to be made safer.
        static_tools (dict[str, Callable]): Dictionary...

#### `get_iterable(obj)`

#### `fix_final_answer_code(code: str) -> str`

Sometimes an LLM can try to assign a variable to final_answer, which would break the final_answer() tool.
    This function fixes this behaviour by replacing variable assignments to final_answer with ...

#### `build_import_tree(authorized_imports: list[str]) -> dict[str, Any]`

#### `check_import_authorized(import_to_check: str, authorized_imports: list[str]) -> bool`

#### `evaluate_attribute(expression: ast.Attribute, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `evaluate_unaryop(expression: ast.UnaryOp, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `evaluate_lambda(lambda_expression: ast.Lambda, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Callable`

#### `evaluate_while(while_loop: ast.While, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> None`

#### `create_function(func_def: ast.FunctionDef, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Callable`

#### `evaluate_function_def(func_def: ast.FunctionDef, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Callable`

#### `evaluate_class_def(class_def: ast.ClassDef, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> type`

#### `evaluate_annassign(annassign: ast.AnnAssign, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `evaluate_augassign(expression: ast.AugAssign, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `evaluate_boolop(node: ast.BoolOp, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `evaluate_binop(binop: ast.BinOp, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `evaluate_assign(assign: ast.Assign, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `set_value(target: ast.AST, value: Any, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> None`

#### `evaluate_call(call: ast.Call, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `evaluate_subscript(subscript: ast.Subscript, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `evaluate_name(name: ast.Name, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `evaluate_condition(condition: ast.Compare, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> bool | object`

#### `evaluate_if(if_statement: ast.If, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `evaluate_for(for_loop: ast.For, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> Any`

#### `evaluate_listcomp(listcomp: ast.ListComp, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> list[Any]`

#### `evaluate_setcomp(setcomp: ast.SetComp, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> set[Any]`

#### `evaluate_try(try_node: ast.Try, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> None`

#### `evaluate_raise(raise_node: ast.Raise, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> None`

#### `evaluate_assert(assert_node: ast.Assert, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> None`

#### `evaluate_with(with_node: ast.With, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> None`

#### `get_safe_module(raw_module, authorized_imports, visited)`

Creates a safe copy of a module or returns the original if it's a function

#### `evaluate_import(expression, state, authorized_imports)`

#### `evaluate_dictcomp(dictcomp: ast.DictComp, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> dict[Any, Any]`

#### `evaluate_delete(delete_node: ast.Delete, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str]) -> None`

Evaluate a delete statement (del x, del x[y]).

    Args:
        delete_node: The AST Delete node to evaluate
        state: The current state dictionary
        static_tools: Dictionary of static to...

#### `evaluate_ast(expression: ast.AST, state: dict[str, Any], static_tools: dict[str, Callable], custom_tools: dict[str, Callable], authorized_imports: list[str])`

Evaluate an abstract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse through the nodes of the tree pr...

#### `evaluate_python_code(code: str, static_tools: dict[str, Callable] | None, custom_tools: dict[str, Callable] | None, state: dict[str, Any] | None, authorized_imports: list[str], max_print_outputs_length: int)`

Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provid...

#### `lambda_func(*values: Any) -> Any`

#### `new_func(*args: Any, **kwargs: Any) -> Any`

#### `get_current_value(target: ast.AST) -> Any`

#### `inner_evaluate(generators: list[ast.comprehension], index: int, current_state: dict[str, Any]) -> list[Any]`

#### `final_answer(*args, **kwargs)`

## smolagents.remote_executors

```python
from smolagents.remote_executors import ...
```

### Classes

#### `RemotePythonExecutor(PythonExecutor)`

**Methods:**

- `run_code_raise_errors(self, code: str) -> CodeOutput` - Execute code, return the result and output, also determining if
- `send_tools(self, tools: dict[str, Tool])`
- `send_variables(self, variables: dict)` - Send variables to the kernel namespace using pickle.
- `install_packages(self, additional_imports: list[str])`

#### `E2BExecutor(RemotePythonExecutor)`

Executes Python code using E2B.

    Args:
        additional_imports (`list[str]`): Additional imports to install.
        logger (`Logger`): Logger ...

**Methods:**

- `run_code_raise_errors(self, code: str) -> CodeOutput`
- `cleanup(self)` - Clean up the E2B sandbox and resources.

#### `DockerExecutor(RemotePythonExecutor)`

Executes Python code using Jupyter Kernel Gateway in a Docker container.

**Methods:**

- `run_code_raise_errors(self, code_action: str) -> CodeOutput`
- `cleanup(self)` - Clean up the Docker container and resources.
- `delete(self)` - Ensure cleanup on deletion.

#### `WasmExecutor(RemotePythonExecutor)`

Remote Python code executor in a sandboxed WebAssembly environment powered by Pyodide and Deno.

    This executor combines Deno's secure runtime with...

**Methods:**

- `run_code_raise_errors(self, code: str) -> CodeOutput` - Execute Python code in the Pyodide environment and return the result.
- `install_packages(self, additional_imports: list[str]) -> list[str]` - Install additional Python packages in the Pyodide environment.
- `cleanup(self)` - Clean up resources used by the executor.
- `delete(self)` - Ensure cleanup on deletion.

#### `FinalAnswerException(Exception)`

### Functions

#### `forward(self, *args, **kwargs) -> Any`

