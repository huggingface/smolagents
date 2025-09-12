"""Render decentralized_smolagents messages.jsonl into a readable HTML page.

Usage (library):
    from scripts.html_renderer import MessagesHtmlRenderer
    html = MessagesHtmlRenderer().render_file("path/to/messages.jsonl")

This module intentionally avoids external deps.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class MessagesHtmlRenderer:
    """Render JSONL chat-like logs produced by MessageStore into HTML.

    The renderer is defensive against schema variations:
    - Handles dict or string `content` fields
    - Handles missing optional fields
    - Escapes HTML in user-provided text
    """

    def __init__(self, title: str = "Run Messages", theme: str = "light") -> None:
        self.title = title
        self.theme = theme

    def render_file(self, jsonl_path: str | Path) -> str:
        messages = list(self._iter_messages(Path(jsonl_path)))
        return self.render(messages)

    def render(self, messages: List[Dict[str, Any]]) -> str:
        head = self._build_head()
        body = self._build_body(messages)
        return f"<!DOCTYPE html><html lang=\"en\">{head}{body}</html>"

    # ------------------------------- internals -------------------------------
    def _iter_messages(self, path: Path) -> Iterable[Dict[str, Any]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except Exception:
                    # Skip malformed lines
                    continue

    def _build_head(self) -> str:
        # Minimal CSS, no external fonts for offline use
        css = (
            "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;"
            "margin:0;padding:0;background:#0b0c10;color:#e6edf3;}"
            ".container{max-width:1100px;margin:0 auto;padding:24px;}"
            ".header{position:sticky;top:0;background:#0b0c10;border-bottom:1px solid #1f2328;padding:12px 24px;z-index:1;}"
            ".title{font-size:20px;font-weight:600;}"
            ".msg{display:grid;grid-template-columns:220px 1fr;gap:12px;padding:14px 10px;border-bottom:1px solid #1f2328;}"
            ".meta{color:#9ea7b3;font-size:12px;line-height:1.35;}"
            ".sender{font-weight:600;color:#c9d1d9;}"
            ".type{display:inline-block;border:1px solid #30363d;border-radius:999px;padding:2px 8px;font-size:11px;color:#c9d1d9;margin-left:6px;}"
            ".thread{color:#a5d6ff;margin-left:8px;}"
            ".content{white-space:pre-wrap;word-break:break-word;font-size:14px;line-height:1.5;}"
            ".pill{display:inline-block;margin-right:6px;margin-top:4px;border:1px solid #30363d;border-radius:999px;padding:2px 8px;font-size:11px;color:#9ea7b3;}"
            ".section{margin-top:18px;}"
            ".code{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px;display:block;overflow:auto;}"
        )
        return (
            "<head>"
            f"<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(self.title)}</title>"
            f"<style>{css}</style>"
            "</head>"
        )

    def _build_body(self, messages: List[Dict[str, Any]]) -> str:
        header = (
            "<div class=\"header\">"
            f"<div class=\"title\">{html.escape(self.title)}</div>"
            "</div>"
        )
        items = [self._render_message(m) for m in messages]
        return f"<body>{header}<div class=\"container\">{''.join(items)}</div></body>"

    def _render_message(self, m: Dict[str, Any]) -> str:
        ts = html.escape(str(m.get("timestamp", "")))
        sender = html.escape(str(m.get("sender", "")))
        msg_type = html.escape(str(m.get("type", "message") or "message"))
        thread = html.escape(str(m.get("thread_id", "main") or "main"))
        recipients = m.get("recipients", [])
        recipients = recipients if isinstance(recipients, list) else [recipients]
        recipients_html = "".join(f"<span class=\"pill\">{html.escape(str(r))}</span>" for r in recipients if r)

        content_html = self._render_content(m.get("content"))

        meta = (
            f"<div class=\"meta\"><span class=\"sender\">{sender}</span>"
            f"<span class=\"type\">{msg_type}</span>"
            f"<span class=\"thread\"># {thread}</span><div>{ts}</div>{recipients_html}</div>"
        )

        return f"<div class=\"msg\">{meta}<div class=\"content\">{content_html}</div></div>"

    def _render_content(self, content: Any) -> str:
        # String content
        if isinstance(content, str):
            return html.escape(content)

        # Dict content with known subtypes
        if isinstance(content, dict):
            t = content.get("type")
            if t == "poll":
                return self._render_poll(content)
            if t == "vote":
                return self._kv_block(content, keys=["poll_id", "voter", "vote", "confidence", "rationale"]) 
            if t == "final_answer":
                return self._kv_block(content, keys=["answer", "poll_id", "tally", "source_proposer"]) 
            # Generic dict fallback
            return self._kv_block(content)

        # Anything else
        try:
            return self._kv_block(json.loads(str(content)))
        except Exception:
            return f"<code class=\"code\">{html.escape(str(content))}</code>"

    def _kv_block(self, obj: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
        if not isinstance(obj, dict):
            return f"<code class=\"code\">{html.escape(str(obj))}</code>"
        items: List[str] = []
        if keys is None:
            keys = list(obj.keys())
        for k in keys:
            if k in obj:
                v = obj[k]
                pretty = self._pretty(v)
                items.append(f"<div><b>{html.escape(str(k))}:</b> {pretty}</div>")
        # Include any remaining keys not listed explicitly
        for k, v in obj.items():
            if k in keys:
                continue
            items.append(f"<div><b>{html.escape(str(k))}:</b> {self._pretty(v)}</div>")
        return "<div class=\"section\">" + "".join(items) + "</div>"

    def _pretty(self, value: Any) -> str:
        if isinstance(value, (str, int, float)) or value is None:
            return html.escape(str(value))
        try:
            dump = json.dumps(value, ensure_ascii=False, indent=2)
            return f"<code class=\"code\">{html.escape(dump)}</code>"
        except Exception:
            return f"<code class=\"code\">{html.escape(str(value))}</code>"

    def _render_poll(self, poll: Dict[str, Any]) -> str:
        # Render poll with a friendly layout emphasizing question and proposal
        question = html.escape(str(poll.get("question", "Poll")))
        proposal = poll.get("proposal")
        proposal_html = self._pretty(proposal)
        header = f"<div><b>Question:</b> {question}</div>"
        body = f"<div><b>Proposal:</b> {proposal_html}</div>"
        meta = self._kv_block(
            poll,
            keys=["poll_id", "options", "threshold", "status", "proposer", "final_answer"],
        )
        return "<div class=\"section\">" + header + body + meta + "</div>"


