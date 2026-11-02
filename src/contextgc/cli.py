"""Command line: ``contextgc prune trace.json --budget 2000``.

Input is either a ContextGC trace (``{"blocks": [...], "steps": [...]}``, as written by
``ContextGraph.save``) or a plain chat transcript (a JSON list of ``{"role", "content"}``
messages; system messages are pinned).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

from . import __version__
from .attribution import LexicalSupportScorer, TransformersLogProbScorer
from .errors import BudgetExceeded
from .graph import ContextGraph
from .policy import Policy


def load_trace(path: Path) -> ContextGraph:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "blocks" in data:
        return ContextGraph.from_dict(data)
    if isinstance(data, list):
        g = ContextGraph()
        for i, msg in enumerate(data):
            role = msg.get("role")
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content)
            if role in ("system", "user", "assistant"):
                g.add_message(role, content, id=msg.get("id") or f"{role}#{i}")
            elif role == "tool":
                g.add(content, "tool_result", id=msg.get("id") or f"tool_result#{i}")
            else:
                raise ValueError(f"message {i}: unsupported role {role!r}")
        return g
    raise ValueError("input must be a ContextGC trace object or a list of chat messages")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="contextgc", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"contextgc {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("prune", help="collect a trace down to a token budget")
    pr.add_argument("input", type=Path)
    pr.add_argument("--budget", type=int, required=True, help="token budget")
    pr.add_argument("--keep-recent", type=int, default=4)
    pr.add_argument("--sweep", choices=["lazy", "eager"], default="lazy")
    pr.add_argument("--infer-edges", action="store_true", help="infer dependencies from the query")
    pr.add_argument("--infer-threshold", type=float, default=0.2)
    pr.add_argument("--attribution", choices=["none", "ablation"], default="none")
    pr.add_argument("--scorer", default="lexical", help="'lexical' or 'hf:<model name>'")
    pr.add_argument("--min-attribution", type=float, default=0.05)
    pr.add_argument("--query", help="text the next step depends on (default: latest user message)")
    pr.add_argument("--target", help="text whose support is attributed (default: --query)")
    pr.add_argument("--style", choices=["openai", "anthropic", "text"], default="openai")
    pr.add_argument("--no-tombstones", action="store_true")
    pr.add_argument("--warn-only", action="store_true", help="do not fail when the budget is not met")
    pr.add_argument("--out", type=Path, help="write {report, rendered, trace} JSON here")
    pr.add_argument("--audit", type=Path, help="write the audit log as JSONL here")
    return p


def _scorer(spec: str):
    if spec == "lexical":
        return LexicalSupportScorer()
    if spec.startswith("hf:"):
        return TransformersLogProbScorer(spec[3:])
    raise ValueError("--scorer must be 'lexical' or 'hf:<model name>'")


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    graph = load_trace(args.input)
    policy = Policy(
        budget_tokens=args.budget,
        keep_recent=args.keep_recent,
        sweep=args.sweep,
        attribution=args.attribution,
        min_attribution=args.min_attribution,
        infer_edges=args.infer_edges,
        infer_threshold=args.infer_threshold,
        tombstones=not args.no_tombstones,
        on_over_budget="warn" if args.warn_only else "raise",
        name="cli",
    )
    scorer = _scorer(args.scorer) if args.attribution == "ablation" else None
    exit_code = 0
    try:
        report = graph.collect(policy, scorer=scorer, query=args.query, target=args.target)
    except BudgetExceeded as exc:
        report = exc.report
        exit_code = 2
    print(report.summary(), file=sys.stderr)
    if exit_code:
        print("error: budget not met without removing protected context", file=sys.stderr)
    rendered = graph.render(args.style, tombstones=policy.tombstones)
    if args.audit:
        graph.audit.to_jsonl(args.audit)
    if args.out:
        args.out.write_text(
            json.dumps({"report": report.to_dict(), "rendered": rendered, "trace": graph.to_dict()}, indent=2),
            encoding="utf-8",
        )
    else:
        print(rendered if isinstance(rendered, str) else json.dumps(rendered, indent=2))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
