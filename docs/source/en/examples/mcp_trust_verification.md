# Verify MCP server trust before loading tools

This example implements the trust-verification pattern requested in
[huggingface/smolagents#2305](https://github.com/huggingface/smolagents/issues/2305):
a fail-closed check that runs **before** an MCP server's tools are loaded into an
agent, so a compromised or malicious server never gets the chance to act.

The full, runnable code is in
[`examples/mcp_trust_verification.py`](https://github.com/huggingface/smolagents/blob/main/examples/mcp_trust_verification.py).
It works fully offline (no network, no Node.js, no LLM call): run it as-is to see
every check in action.

## Why verification matters

Connecting an agent to an MCP server is a trust decision, not a configuration step:

- **Stdio servers** (spawned through `mcp.StdioServerParameters`) execute code on
  *your* machine — that is their intended functionality. Trust them like software
  you install, and pin the versions you run.
- **Remote servers** (Streamable HTTP or SSE, configured as a `{"url": ...}` dict)
  do not execute code locally, but they receive your prompts, and they control the
  tool definitions handed to your model. A malicious server can poison tool
  descriptions (prompt injection), exfiltrate data, or steer the agent.

smolagents already makes the trust decision explicit: [`ToolCollection.from_mcp`]
refuses to load tools unless you pass `trust_remote_code=True`. This example adds
the step that should come before that acknowledgment: deciding *whether* the
server deserves it.

## The verification gate

`verify_server_trust()` raises `ServerTrustVerificationError` when any check fails.
On failure nothing is connected, nothing is executed, and no tool is loaded. The
function is defined in the example file and relies only on the `mcp` package you
already use for MCP connections — copy it into your own codebase, or run the
example to watch it work:

### Stdio servers: deny by default

Stdio servers run code locally, so the gate denies everything except an explicit
allowlist of `(command, args)` pairs. Prefer pinned invocations — and note this
is a coarse command/args allowlist: it does not capture `cwd`, `env`, or
`PATH`-based executable resolution, so use absolute executable paths, pinned
packages, and (for untrusted servers) a sandbox or container as additional
layers of defense:

```python
trusted_stdio_servers = {
    ("uvx", ("--quiet", "pubmedmcp@0.1.3")),
}

server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

verify_server_trust(server_parameters, trusted_stdio_servers=trusted_stdio_servers)
```

### Remote servers: https + allowlist

Remote servers must use `https://` — plain `http://` is refused unless the host is
a loopback address (e.g. a local dev server on `127.0.0.1`), because cleartext
transport exposes both the tool definitions and your prompts. Non-loopback hosts
must also be in your allowlist. Loopback hosts are treated as your local trust
domain: they are exempt from the host allowlist and the scorer (an intentional
escape hatch for local development), while any non-http(s) URL is still refused:

```python
allowed_remote_hosts = {"mcp.my-org.com"}

server_parameters = {"url": "https://mcp.my-org.com/mcp", "transport": "streamable-http"}

verify_server_trust(server_parameters, allowed_remote_hosts=allowed_remote_hosts)
```

### Behavioral trust scoring (optional plug-in)

The gate accepts any external scorer that maps a server URL to a score in
`[0.0, 1.0]`, and refuses servers below a threshold. This is where behavioral
trust APIs (such as the one discussed in issue #2305) plug in — as a policy *you*
choose, not a hardcoded dependency:

```python
def score_mcp_server(url: str) -> float:
    # Call your favorite MCP trust-scoring service here.
    return 0.95

verify_server_trust(
    {"url": "https://mcp.my-org.com/mcp", "transport": "streamable-http"},
    allowed_remote_hosts=allowed_remote_hosts,
    trust_scorer=score_mcp_server,
    score_threshold=0.7,
)
```

## Wiring the gate into tool loading

Verification happens before `from_mcp` is even called, so a refused server never
starts a session:

```python
verify_server_trust(server_parameters, trusted_stdio_servers=trusted_stdio_servers)

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=model)
    agent.run("Your task here.")
```

If you need to handle several servers, pass a list — each entry is verified, and a
single failing entry blocks the whole load.

## Running the example

```bash
# Until huggingface/smolagents#2585 lands, mcp 2.x breaks mcpadapt, so pin mcp<2:
pip install "smolagents[mcp]" "mcp<2"
python examples/mcp_trust_verification.py
```

Expected output (abridged):

```
[1] PASS trusted stdio server loaded and ran: 'Echo: hello from a verified MCP server'
[2] BLOCKED unallowlisted remote server before any connection: ...
[3] BLOCKED cleartext http:// remote server: ...
[4] BLOCKED allowlisted host with low behavioral trust score: ...
[5] BLOCKED invalid (NaN) behavioral trust score: ...
[6] PASS loopback http:// server accepted for local development (verification only)
```

Remember the baseline rule from the [tools guide](../tutorials/tools): stdio-based
MCP servers always execute code on your machine, and even remote servers deserve
caution. Verification gates like this one make that caution operational — but they
are only as good as the allowlist and scorer you maintain.
