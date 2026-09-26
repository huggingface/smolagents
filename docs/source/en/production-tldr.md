# Building Reliable Agents — TL;DR

> Full details: [tutorials/building_good_agents.md](./tutorials/building_good_agents.md)

## Top 5 Tips for Production Agents

### 1. Simplify the workflow — reduce LLM calls

Every LLM call is a potential error. Minimize them:
- Merge related tools into one (e.g., one tool that calls both travel API + weather API instead of two separate tools)
- Use deterministic code for logic whenever possible; only let the LLM make decisions where genuine reasoning is needed

### 2. Make tool descriptions excellent

The LLM only knows what you tell it. Bad descriptions cause bad tool calls:
- Specify exact input formats (e.g., `date in format '%m/%d/%y %H:%M:%S'`)
- Describe what the output looks like
- Log errors and edge cases explicitly inside `forward()` using `print` statements — these appear in the agent's memory and help it self-correct

```python
@tool
def get_weather_api(location: str, date_time: str) -> str:
    """Returns the weather report.
    Args:
        location: Place name, e.g. "Anchor Point, Taghazout, Morocco".
        date_time: Date and time formatted as '%m/%d/%y %H:%M:%S'.
    """
    try:
        date_time = datetime.strptime(date_time, '%m/%d/%y %H:%M:%S')
    except Exception as e:
        raise ValueError("Bad date format. Expected '%m/%d/%y %H:%M:%S'. Error: " + str(e))
    ...
```

### 3. Use a stronger LLM when debugging

Many failures are "LLM dumb" errors, not code bugs. If an agent misbehaves, try a more powerful model (e.g., upgrade from a 7B to `Qwen2.5-72B-Instruct` or a frontier model) before debugging the code. This quickly distinguishes model errors from system errors.

### 4. Pass context via additional_args

Avoid string-stuffing objects into the prompt. Pass images, dataframes, or other objects directly:

```python
agent.run(
    "Analyze the audio and summarize the key points.",
    additional_args={"audio_file": "https://example.com/recording.mp3"}
)
```

### 5. Instrument runs with OpenTelemetry

For production observability, use `SmolagentsInstrumentor` to export traces to any OpenTelemetry-compatible platform (Phoenix, Langfuse, etc.):

```python
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
SmolagentsInstrumentor().instrument()
# Now run your agents normally — traces are captured automatically
```

## Key Reliability Patterns

- **Retry on failure:** Build retry logic at the task level, not just the tool level
- **Cap tool call depth:** Set `max_steps` on the agent to prevent runaway loops
- **Validate outputs:** Check agent outputs against expected schemas before passing to downstream systems
- **Use planning:** For complex tasks, enable the planning step (`planning_interval`) so the agent periodically re-evaluates its progress

## Debugging Checklist

1. Is the task description clear and unambiguous?
2. Are tool descriptions specific enough (formats, units, edge cases)?
3. Is the LLM powerful enough for this task?
4. Are tool errors being logged inside `forward()` with `print`?
5. Are instrumentation traces enabled for post-hoc inspection?

> For the complete guide including examples and advanced patterns: see [tutorials/building_good_agents.md](./tutorials/building_good_agents.md)
