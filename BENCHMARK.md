# KCP Benchmark Results

Comparing baseline (free exploration) vs KCP-guided navigation for answering 8 common smolagents questions.

## Results

| Query | Baseline | KCP | Saved |
|-------|----------|-----|-------|
| What is the difference between CodeAgent and ToolCallingAgent? | 22 | 2 | 20 |
| How do I create a custom tool? | 7 | 3 | 4 |
| How do I run a CodeAgent safely with sandboxed code execution? | 24 | 8 | 16 |
| How do I set up a multi-agent system with a manager and subagents? | 11 | 6 | 5 |
| Which LLM models does smolagents support? | 19 | 2 | 17 |
| How do I implement RAG with smolagents? | 6 | 6 | 0 |
| How do I debug or inspect an agent's run? | 22 | 3 | 19 |
| What are the best practices for building reliable agents in production? | 10 | 3 | 7 |
| **TOTAL** | **121** | **33** | **88** |

**Reduction: 73% fewer tool calls with KCP**

## Key Findings

The biggest wins came from direct-lookup queries where the TL;DR files were an exact match: "CodeAgent vs ToolCallingAgent" and "Which LLM models are supported?" both dropped from 19-22 tool calls to just 2 (read knowledge.yaml, read the TL;DR). The baseline agent in these cases explored the full repository tree, read multiple source files, and followed several dead ends before finding the answer.

The RAG query showed no improvement (6 vs 6), because the RAG example is spread across conceptual and example files in a way that required similar exploration regardless of guidance. This is the expected ceiling for KCP: it helps most when there is a clear, stable answer location, and least when the answer genuinely requires synthesizing multiple sources.

The sandboxed execution query remained relatively expensive with KCP (8 calls) because the topic spans both the guided tour and the dedicated secure_code_execution tutorial — the agent correctly read both after consulting knowledge.yaml, which is the right behavior.

## Methodology

- Model: `claude-haiku-4-5-20251001` (same for both conditions)
- Queries: 8 representative questions covering all major smolagents topics
- Tool count: number of tool_use blocks returned by the Anthropic API across all turns
- Baseline: agent instructed to explore the repository freely
- KCP: agent instructed to read `knowledge.yaml` first, match triggers, and prefer TL;DR summary units
- Max turns: 20 per query
- File truncation: files larger than 8000 characters truncated at that boundary
