"""Synthetic needle-retention benchmark.

Each session is an agent transcript: a pinned system prompt, a task, N search calls whose
results all look alike ("The launch code for Project X is ..." plus filler), and a final
question about two specific projects. Every strategy must fit the context into a token
budget. A session passes if both needed results survive.

This is a synthetic sanity check of the pipeline, not a real-world result: the question is
lexically linked to the needed results by construction, which favours similarity-based
inference. Use it to compare strategies and catch regressions, not to claim accuracy on
real agent traffic.

Usage:
    python benchmarks/needle_retention.py --sessions 200 --out benchmarks/RESULTS.md
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from typing import Callable, Dict, List, Tuple

from contextgc import ContextGraph, LexicalSupportScorer, Policy

PROJECTS = """Aurora Borealis Cobalt Delta Ember Falcon Garnet Harbor Indigo Juniper Kestrel
Lantern Meridian Nimbus Onyx Pioneer Quartz Raven Sierra Tundra Umber Vertex Willow Xenon
Yarrow Zephyr Atlas Beacon Cascade Dynamo Echo Fjord Glacier Horizon Ion Jade Krypton Lotus
Mosaic Nova""".split()

FILLER = [
    "The project team met weekly to review the launch checklist and open risks.",
    "Budget approvals were routed through the regional finance office.",
    "Several vendors were evaluated before the final hardware order was placed.",
    "Documentation for the project was moved to the shared engineering wiki.",
    "The steering committee asked for an updated timeline before the next review.",
    "Field tests were repeated after a firmware update changed sensor calibration.",
    "Security review flagged two low-risk findings that were fixed the same week.",
    "Stakeholders requested a short summary of launch readiness for leadership.",
    "The integration environment was rebuilt to match production more closely.",
    "A training session covered the new deployment runbook for operators.",
    "Spare parts inventory was reconciled against the procurement ledger.",
    "The quality team sampled units from each batch for destructive testing.",
]

LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def make_session(rng: random.Random, n_results: int, n_needles: int) -> Tuple[ContextGraph, List[str], str]:
    g = ContextGraph()
    g.add("You are a research agent. Answer only from tool results.", "system", pinned=True, id="sys")
    g.add_message("user", "Collect launch facts about every project in the portfolio.", id="task")
    names = rng.sample(PROJECTS, n_results)
    needles = rng.sample(names, n_needles)
    for name in names:
        call = g.add_tool_call("search", {"q": f"{name} project launch facts"}, id=f"call_{name}")
        code = f"{rng.choice(LETTERS)}{rng.randint(1, 9)}-{rng.randint(10, 99)}"
        body = rng.sample(FILLER, 6)
        body.insert(rng.randint(1, 5), f"The launch code for Project {name} is {code}.")
        g.add_tool_result(call, f"Search results for Project {name}. " + " ".join(body), id=f"res_{name}")
    question = "What are the launch codes for " + " and ".join(f"Project {n}" for n in needles) + "?"
    g.add_message("user", question, id="question")
    return g, [f"res_{n}" for n in needles], question


def truncate_oldest(g: ContextGraph, budget: int, question: str) -> None:
    """Baseline: drop the oldest unpinned units (never the final question) until under budget."""
    for _unit, blocks in g.units().items():
        if g.tokens <= budget:
            break
        if any(b.pinned for b in blocks) or any(b.id == "question" for b in blocks):
            continue
        g._move_to_cold([b.id for b in blocks], action="pruned", reason="truncate-oldest")


def contextgc_infer(threshold: float) -> Callable[[ContextGraph, int, str], None]:
    def run(g: ContextGraph, budget: int, question: str) -> None:
        g.collect(
            Policy(
                budget_tokens=budget, keep_recent=1, infer_edges=True, infer_threshold=threshold, on_over_budget="warn"
            )
        )

    return run


def contextgc_ablation(threshold: float) -> Callable[[ContextGraph, int, str], None]:
    def run(g: ContextGraph, budget: int, question: str) -> None:
        g.collect(
            Policy(
                budget_tokens=budget,
                keep_recent=1,
                infer_edges=True,
                infer_threshold=threshold,
                attribution="ablation",
                min_attribution=0.01,
                on_over_budget="warn",
            ),
            scorer=LexicalSupportScorer(),
            target=question,
        )

    return run


def evaluate(
    strategy, sessions: int, n_results: int, n_needles: int, budget_frac: float, seed: int
) -> Dict[str, float]:
    rng = random.Random(seed)
    passed, budget_met, reductions, millis = 0, 0, [], []
    for _ in range(sessions):
        g, needles, question = make_session(rng, n_results, n_needles)
        before = g.tokens
        budget = int(before * budget_frac)
        t0 = time.perf_counter()
        strategy(g, budget, question)
        millis.append((time.perf_counter() - t0) * 1000)
        passed += all(n in g for n in needles)
        budget_met += g.tokens <= budget
        reductions.append(1 - g.tokens / before)
    return {
        "retention": passed / sessions,
        "budget_met": budget_met / sessions,
        "reduction": statistics.mean(reductions),
        "ms_p50": statistics.median(millis),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--results", type=int, default=24, help="tool results per session")
    ap.add_argument("--needles", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", help="write a Markdown table here")
    args = ap.parse_args()

    strategies = {
        "truncate oldest (baseline)": truncate_oldest,
        "ContextGC: inferred edges": contextgc_infer(args.threshold),
        "ContextGC: inferred edges + ablation": contextgc_ablation(args.threshold),
    }
    rows = []
    for frac in (0.15, 0.25, 0.40):
        for name, fn in strategies.items():
            r = evaluate(fn, args.sessions, args.results, args.needles, frac, args.seed)
            rows.append((frac, name, r))
            print(
                f"budget {frac:.0%} | {name:38s} | retention {r['retention']:.1%} | "
                f"budget met {r['budget_met']:.1%} | reduction {r['reduction']:.1%} | p50 {r['ms_p50']:.2f} ms"
            )

    if args.out:
        lines = [
            "# Needle-retention benchmark (synthetic)",
            "",
            f"Generated by `python benchmarks/needle_retention.py --sessions {args.sessions} "
            f"--results {args.results} --needles {args.needles} --threshold {args.threshold} --seed {args.seed}`.",
            "",
            f"{args.sessions} synthetic sessions per row; each has {args.results} look-alike search results, "
            f"of which {args.needles} are needed to answer the final question. A session passes when every "
            "needed result survives collection. Token counts use the default 4-characters-per-token estimate.",
            "",
            "| Budget (share of session tokens) | Strategy | Needed results kept "
            "| Budget met | Tokens removed | p50 latency |",
            "|---|---|---|---|---|---|",
        ]
        for frac, name, r in rows:
            lines.append(
                f"| {frac:.0%} | {name} | {r['retention']:.1%} | {r['budget_met']:.1%} | "
                f"{r['reduction']:.1%} | {r['ms_p50']:.2f} ms |"
            )
        lines += [
            "",
            "**Caveat:** synthetic data where the question names the needed projects, so lexical "
            "inference is favoured by construction. Treat this as a regression and sanity check of "
            "the pipeline, not as evidence of accuracy on real agent traces.",
            "",
        ]
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
