#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
smolagents-firewall — MCP Application Firewall management CLI

Usage:
  smolagents-firewall check   <url> [--allow-http] [--blocklist PATTERN ...]
  smolagents-firewall report  [--log PATH]
  smolagents-firewall status  [--lockfile PATH]
  smolagents-firewall approve <server_id> <tool_name> [--lockfile PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from smolagents.mcp_firewall import (
    MCPAuditLogger,
    MCPCallbackHook,
    MCPConsoleHook,
    MCPFileHook,
    MCPFirewall,
    MCPSecurityReport,
    MCPToolAllowlist,
    MCPToolFingerprinter,
    StaticTrustVerifier,
    _dispatch_hooks,
)

try:
    from rich import box as rich_box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH = True
    _console = Console()
except ImportError:  # pragma: no cover
    _RICH = False
    _console = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    """Score a server URL's trust without opening any network connection."""
    url: str = args.url
    verifier = StaticTrustVerifier(
        require_https=not args.allow_http,
        blocklist=args.blocklist or [],
    )
    result = verifier.verify({"url": url, "transport": "streamable-http"})

    verdict = "TRUSTED" if result.trusted else "BLOCKED"
    score_pct = f"{result.trust_score:.0%}"

    if _RICH:
        colour = "green" if result.trusted else "red"
        icon = "✅" if result.trusted else "❌"
        table = Table(
            title=f"Trust Check — {url}",
            box=rich_box.ROUNDED,
            show_header=True,
        )
        table.add_column("Field", style="bold", min_width=14)
        table.add_column("Value")
        table.add_row("Verdict",     f"[{colour}]{icon} {verdict}[/{colour}]")
        table.add_row("Trust Score", score_pct)
        for i, reason in enumerate(result.reasons):
            label = "Reason" if i == 0 else ""
            table.add_row(label, reason)
        _console.print()
        _console.print(table)
        _console.print()
    else:
        print(f"URL     : {url}")
        print(f"Verdict : {verdict}")
        print(f"Score   : {score_pct}")
        for r in result.reasons:
            print(f"  • {r}")

    return 0 if result.trusted else 1


def cmd_report(args: argparse.Namespace) -> int:
    """Print security analytics from the MCPAuditLogger JSONL file."""
    report = MCPSecurityReport(log_path=args.log_path)

    if not report._log_path.exists():
        msg = f"Audit log not found: {report._log_path}"
        if _RICH:
            _console.print(f"[yellow]⚠  {msg}[/yellow]")
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
        return 1

    stats = report.generate()

    if _RICH:
        _console.print()
        _console.print(Panel.fit(
            f"[bold cyan]MCP Firewall — Security Report[/bold cyan]\n"
            f"[dim]{report._log_path}[/dim]"
        ))

        summary = Table(box=rich_box.SIMPLE, show_header=False)
        summary.add_column("Metric", style="bold", min_width=18)
        summary.add_column("Value", justify="right")
        summary.add_row("Total calls",     str(stats.total_calls))
        br_colour = "red" if stats.blocked_calls > 0 else "green"
        summary.add_row("Blocked calls",   f"[{br_colour}]{stats.blocked_calls}[/{br_colour}]")
        rate_colour = "red" if stats.block_rate > 0 else "green"
        summary.add_row("Block rate",      f"[{rate_colour}]{stats.block_rate:.1%}[/{rate_colour}]")
        summary.add_row("Unique servers",  str(stats.unique_servers))
        summary.add_row("Unique tools",    str(stats.unique_tools))
        summary.add_row("Avg latency",     f"{stats.avg_duration_ms:.1f} ms")
        if stats.first_call_at:
            summary.add_row("First call",  stats.first_call_at)
            summary.add_row("Last call",   stats.last_call_at)
        _console.print(summary)

        if stats.blocked_by_pattern:
            pat = Table(title="[red]Top Attack Patterns[/red]", box=rich_box.SIMPLE)
            pat.add_column("Pattern", style="bold red")
            pat.add_column("Blocks", justify="right")
            for p, cnt in list(stats.blocked_by_pattern.items())[:10]:
                pat.add_row(p, str(cnt))
            _console.print(pat)

        if stats.calls_by_server:
            srv = Table(title="Calls by Server", box=rich_box.SIMPLE)
            srv.add_column("Server")
            srv.add_column("Calls", justify="right")
            for s, cnt in list(stats.calls_by_server.items())[:10]:
                display = s if len(s) <= 60 else s[:57] + "..."
                srv.add_row(display, str(cnt))
            _console.print(srv)

        if stats.calls_by_tool:
            tool_t = Table(title="Calls by Tool", box=rich_box.SIMPLE)
            tool_t.add_column("Tool")
            tool_t.add_column("Calls", justify="right")
            for t, cnt in list(stats.calls_by_tool.items())[:10]:
                tool_t.add_row(t, str(cnt))
            _console.print(tool_t)
    else:
        report.print_summary()

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show all servers and tool fingerprints in the lockfile."""
    lockfile = Path(args.lockfile)
    if not lockfile.exists():
        msg = f"Lockfile not found: {lockfile}"
        if _RICH:
            _console.print(f"[yellow]⚠  {msg}[/yellow]")
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
        return 1

    with open(lockfile, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    servers: dict = data.get("servers", {})

    if _RICH:
        _console.print()
        _console.print(Panel.fit(
            f"[bold]MCP Lockfile[/bold] — {lockfile}  "
            f"([dim]version {data.get('version', '?')}[/dim])"
        ))
        if not servers:
            _console.print("[dim]  No servers registered yet.[/dim]")
        for server_id, tools in servers.items():
            table = Table(
                title=f"[bold cyan]{server_id}[/bold cyan]",
                box=rich_box.SIMPLE,
            )
            table.add_column("Tool", style="bold", min_width=24)
            table.add_column("Fingerprint (sha256[:16])", style="dim")
            table.add_column("Registered At")
            for tool_name, meta in tools.items():
                fp = (meta.get("fingerprint") or "?")[:16]
                at = meta.get("created_at", "?")
                table.add_row(tool_name, fp, at)
            _console.print(table)
    else:
        print(f"Lockfile : {lockfile}  (version {data.get('version', '?')})")
        if not servers:
            print("  (no servers registered)")
        for server_id, tools in servers.items():
            print(f"\n  Server: {server_id}")
            for tool_name, meta in tools.items():
                fp = (meta.get("fingerprint") or "?")[:16]
                at = meta.get("created_at", "?")
                print(f"    {tool_name:<40} {fp}  {at}")

    return 0


def cmd_test_hook(args: argparse.Namespace) -> int:
    """Fire a synthetic security event through registered hooks to verify wiring."""
    hooks = []
    if args.console:
        hooks.append(MCPConsoleHook(use_color=False))
    if args.file:
        hooks.append(MCPFileHook(args.file))

    if not hooks:
        msg = "Specify at least one hook target: --console or --file PATH"
        if _RICH:
            _console.print(f"[yellow]⚠  {msg}[/yellow]")
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
        return 1

    _dispatch_hooks(
        hooks,
        "call_intercepted",
        server_id="https://test.example.com/mcp",
        extra={
            "tool_name": "test_tool",
            "phase": "pre-call",
            "reason": "This is a test event fired by smolagents-firewall test-hook",
        },
    )

    msg = f"Test event fired to {len(hooks)} hook(s)."
    if _RICH:
        _console.print(f"[green]✓[/green] {msg}")
    else:
        print(msg)
    return 0


def cmd_rate_status(args: argparse.Namespace) -> int:
    """Show per-server and per-tool call counts from the audit log."""
    import time as _t
    report = MCPSecurityReport(log_path=args.log_path)

    if not report._log_path.exists():
        msg = f"Audit log not found: {report._log_path}"
        if _RICH:
            _console.print(f"[yellow]⚠  {msg}[/yellow]")
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
        return 1

    window = args.window
    cutoff_iso = None
    # Compute a rough ISO cutoff for filtering (timestamps are UTC ISO strings)
    try:
        from datetime import datetime, timezone, timedelta
        cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=window)
        cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        pass

    records = report._load_records()
    if cutoff_iso:
        records = [r for r in records if (r.get("timestamp") or "") >= cutoff_iso]

    by_server: dict[str, int] = {}
    by_tool: dict[str, int] = {}
    for rec in records:
        srv = rec.get("server_id") or "unknown"
        tool = rec.get("tool_name") or "unknown"
        by_server[srv] = by_server.get(srv, 0) + 1
        key = f"{srv} / {tool}"
        by_tool[key] = by_tool.get(key, 0) + 1

    by_server = dict(sorted(by_server.items(), key=lambda x: -x[1]))
    by_tool = dict(sorted(by_tool.items(), key=lambda x: -x[1]))

    if _RICH:
        _console.print()
        _console.print(Panel.fit(
            f"[bold]MCP Rate Status[/bold] — last {window}s  "
            f"([dim]{len(records)} call(s)[/dim])"
        ))
        if by_server:
            srv_t = Table(title="Calls by Server", box=rich_box.SIMPLE)
            srv_t.add_column("Server")
            srv_t.add_column("Calls", justify="right")
            for s, cnt in list(by_server.items())[:10]:
                display = s if len(s) <= 60 else s[:57] + "..."
                srv_t.add_row(display, str(cnt))
            _console.print(srv_t)
        if by_tool:
            tool_t = Table(title="Calls by Tool", box=rich_box.SIMPLE)
            tool_t.add_column("Server / Tool")
            tool_t.add_column("Calls", justify="right")
            for t, cnt in list(by_tool.items())[:10]:
                tool_t.add_row(t, str(cnt))
            _console.print(tool_t)
        if not records:
            _console.print("[dim]  No calls recorded in the last {window}s.[/dim]")
    else:
        print(f"Rate Status — last {window}s  ({len(records)} call(s))")
        if by_server:
            print("\n  Calls by server:")
            for s, cnt in list(by_server.items())[:10]:
                print(f"    {s:<60} {cnt:>4} call(s)")
        if by_tool:
            print("\n  Calls by tool:")
            for t, cnt in list(by_tool.items())[:10]:
                print(f"    {t:<60} {cnt:>4} call(s)")

    return 0


def cmd_allowlist(args: argparse.Namespace) -> int:
    """Manage the MCPToolAllowlist — show, add, or remove approved tools."""
    allowlist = MCPToolAllowlist(allowlist_path=args.allowlist)

    if args.allowlist_cmd == "show":
        approved = allowlist.list_approved(server_id=getattr(args, "server_id", None) or None)

        if _RICH:
            _console.print()
            _console.print(Panel.fit(
                f"[bold]MCP Allowlist[/bold] — {allowlist._allowlist_path}"
            ))
            if not approved:
                _console.print("[dim]  No approved tools yet.[/dim]")
            for server_id, tools in approved.items():
                table = Table(
                    title=f"[bold cyan]{server_id}[/bold cyan]",
                    box=rich_box.SIMPLE,
                )
                table.add_column("Tool", style="bold", min_width=24)
                table.add_column("Approved At")
                for tool_name, meta in tools.items():
                    at = meta.get("approved_at", "?")
                    table.add_row(tool_name, at)
                _console.print(table)
        else:
            print(f"Allowlist : {allowlist._allowlist_path}")
            if not approved:
                print("  (no approved tools)")
            for server_id, tools in approved.items():
                print(f"\n  Server: {server_id}")
                for tool_name, meta in tools.items():
                    at = meta.get("approved_at", "?")
                    print(f"    {tool_name:<40} {at}")
        return 0

    elif args.allowlist_cmd == "add":
        allowlist.approve(args.server_id, args.tool_name)
        msg = f"Approved '{args.tool_name}' on '{args.server_id}'."
        if _RICH:
            _console.print(f"[green]✓[/green] {msg}")
        else:
            print(msg)
        return 0

    elif args.allowlist_cmd == "remove":
        allowlist.revoke(args.server_id, args.tool_name)
        msg = f"Revoked '{args.tool_name}' on '{args.server_id}'."
        if _RICH:
            _console.print(f"[yellow]✓[/yellow] {msg}")
        else:
            print(msg)
        return 0

    return 1  # unknown subcommand


def cmd_approve(args: argparse.Namespace) -> int:
    """Clear a stale fingerprint so the next connection re-registers the tool."""
    fingerprinter = MCPToolFingerprinter(lockfile_path=args.lockfile)
    fingerprinter.approve_update(
        server_id=args.server_id,
        tool_name=args.tool_name,
    )
    msg = (
        f"Fingerprint cleared for '{args.tool_name}' on "
        f"'{args.server_id}'. Will re-register on next connection."
    )
    if _RICH:
        _console.print(f"[green]✓[/green] {msg}")
    else:
        print(msg)
    return 0


_INIT_TEMPLATE = """\
# smolagents MCP Application Firewall — configuration
# Generated by: smolagents-firewall init
# Load with: MCPFirewall.from_yaml("{filename}")
#
# Docs: https://huggingface.co/docs/smolagents

# Preset base (strict | balanced | paranoid | dev).
# Individual layers below override the preset.
preset: {preset}

# Layer 1 — Pre-flight trust verification (before any TCP connection)
trust_verifier:
  require_https: true
  min_trust_score: 0.5
  # blocklist:
  #   - "evil\\.example\\.com"
  # allowlist:
  #   - "trusted\\.example\\.com"

# Layer 2 — Payload validation (tool name / description / schema checks)
payload_validator:
  max_tool_name_length: 64
  max_description_length: 4096
  max_tools_per_server: 100
  max_input_params: 20
  max_param_description_length: 1024

# Layer 3 — Rug-pull fingerprinting
fingerprinter:
  lockfile_path: .mcp-lock.json

# Layer 4 — Runtime call sentinel (pre- and post-call inspection)
sentinel:
  max_response_length: 100000
  block_credential_exfil: true
  block_sensitive_paths: true

# Layer 5 — Tool allowlist (whitelist mode — set to false to disable)
# allowlist:
#   allowlist_path: .mcp-allowlist.json
#   auto_approve_first_connection: true

# Layer 6 — PII sanitizer (strips PII from responses before they reach the LLM)
sanitizer:
  mode: redact    # redact | hash | drop
  redact_emails: true
  redact_credit_cards: true
  redact_ssn: true
  redact_phone_numbers: true
  redact_jwt: true

# Layer 7 — Rate limiter (sliding-window call budget)
rate_limiter:
  max_calls_per_minute: 300
  per_tool_max_calls_per_minute: 60
  window_seconds: 60

# Layer 8 — Audit logger (structured JSONL call log)
audit_logger:
  log_path: ~/.smolagents/audit.jsonl

# Security event hooks (real-time push notifications for firewall events)
# hooks:
#   console: true                        # print to stderr
#   file: ~/.smolagents/alerts.jsonl     # append to a file
"""


def cmd_init(args: argparse.Namespace) -> int:
    """Generate a starter .smolagents-firewall.yml config file."""
    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        msg = f"{output_path} already exists. Use --force to overwrite."
        if _RICH:
            _console.print(f"[yellow]⚠  {msg}[/yellow]")
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = _INIT_TEMPLATE.format(preset=args.preset, filename=output_path.name)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    msg = f"Firewall config written to {output_path}"
    hint = f"Load with: MCPFirewall.from_yaml('{output_path}')"
    if _RICH:
        _console.print(f"[green]✓[/green] {msg}")
        _console.print(f"[dim]  {hint}[/dim]")
    else:
        print(msg)
        print(f"  {hint}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Load and validate a firewall config file without opening any network connection."""
    config_path = Path(args.config_path)

    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        if _RICH:
            _console.print(f"[red]✗  {msg}[/red]")
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 2

    ext = config_path.suffix.lower()
    try:
        if ext in (".yml", ".yaml"):
            fw = MCPFirewall.from_yaml(config_path)
        elif ext == ".toml":
            fw = MCPFirewall.from_toml(config_path)
        elif ext == ".json":
            fw = MCPFirewall.from_json(config_path)
        else:
            fw = MCPFirewall.from_yaml(config_path)
    except ImportError as exc:
        msg = f"Missing dependency: {exc}"
        if _RICH:
            _console.print(f"[red]✗  {msg}[/red]")
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 3
    except Exception as exc:
        msg = f"Config validation failed: {exc}"
        if _RICH:
            _console.print(f"[red]✗  {msg}[/red]")
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    if _RICH:
        _console.print(f"[green]✓[/green] Config valid: {config_path}")
        _console.print()
        for line in fw.summary().splitlines():
            _console.print(f"  {line}")
    else:
        print(f"OK: {config_path}")
        print(fw.summary())
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smolagents-firewall",
        description="MCP Application Firewall — management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  smolagents-firewall check https://api.example.com/mcp
  smolagents-firewall check http://localhost:8000/mcp --allow-http
  smolagents-firewall check https://api.example.com/mcp --blocklist "evil\\.com" "badactor\\.io"

  smolagents-firewall report
  smolagents-firewall report --log /var/log/mcp-audit.jsonl

  smolagents-firewall status
  smolagents-firewall status --lockfile /etc/prod.mcp-lock.json

  smolagents-firewall approve https://api.example.com/mcp search_tool
  smolagents-firewall approve https://api.example.com/mcp search_tool --lockfile prod.lock.json
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # check
    check_p = sub.add_parser(
        "check",
        help="Score a URL's trustworthiness (no network connection made)",
    )
    check_p.add_argument("url", help="MCP server URL to evaluate")
    check_p.add_argument(
        "--allow-http", action="store_true", default=False,
        help="Accept plain HTTP without penalising the score",
    )
    check_p.add_argument(
        "--blocklist", nargs="*", metavar="PATTERN",
        help="Additional regex patterns to block",
    )
    check_p.set_defaults(func=cmd_check)

    # report
    report_p = sub.add_parser(
        "report",
        help="Security summary from the MCPAuditLogger JSONL file",
    )
    report_p.add_argument(
        "--log", dest="log_path", default=None, metavar="PATH",
        help="Path to audit.jsonl (default: ~/.smolagents/audit.jsonl)",
    )
    report_p.set_defaults(func=cmd_report)

    # status
    status_p = sub.add_parser(
        "status",
        help="Show registered server fingerprints from the lockfile",
    )
    status_p.add_argument(
        "--lockfile", default=".mcp-lock.json", metavar="PATH",
        help="Path to .mcp-lock.json (default: ./.mcp-lock.json)",
    )
    status_p.set_defaults(func=cmd_status)

    # approve
    approve_p = sub.add_parser(
        "approve",
        help="Approve a tool definition change (clears its stored fingerprint)",
    )
    approve_p.add_argument("server_id", help="Server identifier as it appears in the lockfile")
    approve_p.add_argument("tool_name", help="Name of the tool whose fingerprint to clear")
    approve_p.add_argument(
        "--lockfile", default=".mcp-lock.json", metavar="PATH",
        help="Path to .mcp-lock.json (default: ./.mcp-lock.json)",
    )
    approve_p.set_defaults(func=cmd_approve)

    # allowlist
    allowlist_p = sub.add_parser(
        "allowlist",
        help="Manage the tool allowlist (show, add, remove)",
    )
    allowlist_p.add_argument(
        "--allowlist", default=".mcp-allowlist.json", metavar="PATH",
        help="Path to .mcp-allowlist.json (default: ./.mcp-allowlist.json)",
    )
    allowlist_sub = allowlist_p.add_subparsers(dest="allowlist_cmd", required=True)

    # allowlist show
    allow_show = allowlist_sub.add_parser("show", help="List all approved tools")
    allow_show.add_argument(
        "server_id", nargs="?", default=None,
        help="Filter to a specific server (optional)",
    )

    # allowlist add
    allow_add = allowlist_sub.add_parser("add", help="Approve a (server, tool) pair")
    allow_add.add_argument("server_id", help="Server identifier")
    allow_add.add_argument("tool_name", help="Tool name to approve")

    # allowlist remove
    allow_rm = allowlist_sub.add_parser("remove", help="Revoke a (server, tool) pair")
    allow_rm.add_argument("server_id", help="Server identifier")
    allow_rm.add_argument("tool_name", help="Tool name to revoke")

    allowlist_p.set_defaults(func=cmd_allowlist)

    # rate-status
    rate_p = sub.add_parser(
        "rate-status",
        help="Show per-server/per-tool call rates from the audit log",
    )
    rate_p.add_argument(
        "--log", dest="log_path", default=None, metavar="PATH",
        help="Path to audit.jsonl (default: ~/.smolagents/audit.jsonl)",
    )
    rate_p.add_argument(
        "--window", type=float, default=60.0, metavar="SECONDS",
        help="Sliding window in seconds to count calls within (default: 60)",
    )
    rate_p.set_defaults(func=cmd_rate_status)

    # test-hook
    th_p = sub.add_parser(
        "test-hook",
        help="Fire a synthetic security event to verify hook wiring",
    )
    th_p.add_argument(
        "--console", action="store_true", default=False,
        help="Fire to stderr via MCPConsoleHook",
    )
    th_p.add_argument(
        "--file", default=None, metavar="PATH",
        help="Fire to a file via MCPFileHook",
    )
    th_p.set_defaults(func=cmd_test_hook)

    # init
    init_p = sub.add_parser(
        "init",
        help="Generate a starter .smolagents-firewall.yml config file",
    )
    init_p.add_argument(
        "--preset", default="strict",
        choices=list(MCPFirewall.PRESETS),
        metavar="PRESET",
        help="Preset to use as the config base (default: strict)",
    )
    init_p.add_argument(
        "--output", default=".smolagents-firewall.yml", metavar="PATH",
        help="Output file path (default: .smolagents-firewall.yml)",
    )
    init_p.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite an existing file",
    )
    init_p.set_defaults(func=cmd_init)

    # validate
    validate_p = sub.add_parser(
        "validate",
        help="Validate a firewall config file (YAML/TOML/JSON) without connecting",
    )
    validate_p.add_argument(
        "config_path",
        help="Path to the config file (.yml, .yaml, .toml, .json)",
    )
    validate_p.set_defaults(func=cmd_validate)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns an exit code (0 = success, non-zero = error)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
