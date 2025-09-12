"""Batch convert all runs' messages.jsonl to HTML index files.

Example:
    python -m examples.decentralized_smolagents_benchmark.scripts.convert_all_runs_to_html \
      --runs-dir /home/ecca/GitFiles/dec_smolagents/examples/decentralized_smolagents_benchmark/runs \
      --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .html_renderer import MessagesHtmlRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert all run messages.jsonl to HTML")
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Path to runs directory containing subfolders with messages.jsonl",
    )
    parser.add_argument(
        "--title-prefix",
        default="Run",
        help="Prefix for HTML title; run folder name is appended",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing index.html files if present",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists() or not runs_dir.is_dir():
        raise SystemExit(f"Runs dir not found or not a directory: {runs_dir}")

    renderer = MessagesHtmlRenderer()
    count = 0
    for messages_file in runs_dir.glob("*/messages.jsonl"):
        run_folder = messages_file.parent
        out_file = run_folder / "index.html"
        if out_file.exists() and not args.force:
            print(f"Skipping {run_folder.name}: index.html exists (use --force to overwrite)")
            continue
        renderer.title = f"{args.title_prefix} {run_folder.name}"
        html = renderer.render_file(messages_file)
        out_file.write_text(html, encoding="utf-8")
        print(f"Wrote {out_file}")
        count += 1

    if count == 0:
        print("No messages.jsonl files found.")
    else:
        print(f"Converted {count} run(s).")


if __name__ == "__main__":
    main()


