"""Manual measurement, not a test: count requests for ONE LiteLLMModel `generate()` call.

The server answers 429 to everything, so every layer that retries shows up as extra
requests. Run it by hand: `python tests/_manual/measure_retry_requests.py`.

On smolagents 1.26.0 (before the fix in this branch) one call spends 9 requests and
872.11 s; with the fix it spends 5 requests and 424.87 s. Same server, same model id,
default RETRY_WAIT; only the code changed.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


OUT = Path.cwd()  # writes requests.jsonl / result.json next to wherever you run it
REQUESTS = OUT / "requests.jsonl"
REQUESTS.write_text("", encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        with REQUESTS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": round(time.time(), 3), "path": self.path}) + "\n")
        body = json.dumps(
            {
                "error": {
                    "message": "Rate limit reached for requests",
                    "type": "requests",
                    "code": "rate_limit_exceeded",
                }
            }
        ).encode()
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("retry-after", "0")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _replays_disable_inner_retries(model) -> bool:
    """Whether this checkout carries the retry-layer fix.

    The fix is in the closure returned by `LiteLLMModel._completion_create`: from the second
    attempt on it passes `num_retries=0`, so litellm does not retry on top of the retryer.
    It cannot be seen in `smolagents.__file__`: that path is the package `__init__.py`, and the
    method is defined in `models.py`.
    """
    factory = getattr(model, "_completion_create", None)
    if factory is None:
        return False
    return "num_retries" in factory().__code__.co_consts


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    import smolagents
    from smolagents import LiteLLMModel

    model = LiteLLMModel(
        model_id="openai/gpt-4o-mini",
        api_base=f"http://127.0.0.1:{port}/v1",
        api_key="sk-test",
    )
    started = time.time()
    error = None
    try:
        model.generate([{"role": "user", "content": "hello"}])
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {str(exc)[:140]}"
    elapsed = time.time() - started

    count = len(REQUESTS.read_text(encoding="utf-8").splitlines())
    result = {
        "smolagents": smolagents.__version__,
        "requests": count,
        "wall_s": round(elapsed, 2),
        "error": error,
        "num_retries_patched": _replays_disable_inner_retries(model),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
