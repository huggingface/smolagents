"""CLI to convert messages.jsonl runs into a readable HTML file.

Example:
    python -m examples.decentralized_smolagents_benchmark.scripts.convert_messages_to_html \
      --input examples/decentralized_smolagents_benchmark/runs/4f0079d6/messages.jsonl \
      --output examples/decentralized_smolagents_benchmark/runs/4f0079d6/index.html \
      --title "Run 4f0079d6"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .html_renderer import MessagesHtmlRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert messages.jsonl to HTML")
    parser.add_argument("--input", required=True, help="Path to messages.jsonl")
    parser.add_argument("--output", required=True, help="Path to write HTML file")
    parser.add_argument("--title", default="Run Messages", help="HTML title")
    args = parser.parse_args()

    renderer = MessagesHtmlRenderer(title=args.title)
    html = renderer.render_file(args.input)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote HTML to {out_path}")


if __name__ == "__main__":
    main()


