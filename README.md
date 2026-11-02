# ContextGC

[![ci](https://github.com/vamsireddy0807/contextgc/actions/workflows/ci.yml/badge.svg)](https://github.com/vamsireddy0807/contextgc/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**Dependency-aware context runtime for LLM agents: semantic garbage collection, dependency tracking and causal attribution for context pruning.**

Long-running agents pile up system prompts, user turns, tool calls, tool results and retrieved chunks until the context window overflows. The usual fixes are blunt:

| Approach | What goes wrong |
|---|---|
| Drop the oldest messages | Deletes the tool result that step 14 still needs |
| Summarise old turns | Lossy, unverifiable, silently changes facts |
| Token-level compression | Saves tokens with no idea which blocks later steps depend on |

ContextGC treats context like a heap. It tracks **what depends on what**, keeps everything that a live step depends on, removes what nothing needs, and when it has to choose, it **measures** each block's contribution by ablation before pruning it. Every removal is audited and reversible, and protected context is never silently truncated.

```
pip install "git+https://github.com/vamsireddy0807/contextgc"          # core: zero dependencies
pip install "contextgc[hf] @ git+https://github.com/vamsireddy0807/contextgc"   # + local model scorer
```

## Quickstart

```python
from contextgc import ContextGraph, LexicalSupportScorer, Policy

g = ContextGraph(budget_tokens=150)
g.add_message("system", "You are a careful research agent.")          # system messages are pinned
call = g.add_tool_call("search", {"q": "aurora"})
res = g.add_tool_result(call, "Project Aurora launches 14 March; launch code K7-29.")
other = g.add_tool_result(g.add_tool_call("search", {"q": "cobalt"}), "Cobalt budget is 4M.")

g.link("draft_section_1", [res])        # explicit dependency: protected while the step is live
g.add_message("user", "Now add the Aurora risks.")

report = g.collect(
    Policy(keep_recent=2, infer_edges=True, attribution="ablation"),
    scorer=LexicalSupportScorer(),
)
print(report.summary())                 # tokens before/after, swept, pruned, policy version
messages = g.render("openai")           # or "anthropic" / "text"; tool call/result pairs stay intact
g.restore(other)                        # anything removed can be brought back
```

`python examples/research_agent.py` runs a fuller session: 227 tokens collected to 129 under a 150-token budget, with the explicitly cited Aurora facts protected and the unrelated searches swept.

## How it works

**Blocks** are units of context: `system`, `user`, `assistant`, `tool_call`, `tool_result`, `retrieved`, `note`. A tool call and its result share a **removal unit**, so they are always kept or removed together (model APIs reject orphaned tool results).

**Steps** are anything that consumes context, such as a model call or an answer being drafted. `g.link(step, uses=[...])` records **explicit** edges; `infer_edges=True` adds **inferred** edges from the current query by TF-IDF cosine similarity. An explicit edge is never downgraded to an inferred one.

`collect()` runs three phases:

1. **Mark.** Roots are pinned blocks, the newest `keep_recent` blocks, and every block a live step uses through an explicit edge. Their closure over `depends_on` and removal units is **hard-reachable** and is never removed. Blocks reached only through inferred edges are **soft-reachable**.
2. **Sweep.** Unreachable units move to the cold store, oldest first (`sweep="lazy"` stops as soon as the budget is met; `"eager"` sweeps them all).
3. **Prune.** If still over budget, soft-reachable units are pruned:
   * `attribution="ablation"`: each unit is scored by leave-one-unit-out ablation, `u(context) - u(context without unit)`. Units are pruned lowest first and only below `min_attribution`, and the **marginal** score is re-checked against the current context before each prune, so two redundant copies of the same evidence are never both pruned on stale scores.
   * `attribution="none"`: pruned by lowest inferred confidence, oldest first.

## Explicit failure boundaries

* Pinned blocks, recent blocks and explicit dependencies of live steps are never removed.
* If the budget cannot be met without breaking that rule, `collect()` raises `BudgetExceeded` (with the partial report attached) instead of truncating. Use `on_over_budget="warn"` to get a report with `budget_met=False` instead.
* If a scorer fails, times out or returns NaN, the affected units are **kept** and an audit entry records why (fail closed). Units beyond `max_attribution_units` are kept, not guessed.
* Adding a block that depends on removed context restores that context first; restoring a block restores its whole unit and its dependencies.
* Every sweep, prune, restore and error-driven keep is written to an append-only audit log stamped with the policy's version hash, optionally mirrored to JSONL as it happens.

## Scorers

| Scorer | What it measures | Needs |
|---|---|---|
| `LexicalSupportScorer` | IDF-weighted share of the target's terms present in the context; deterministic | nothing |
| `TransformersLogProbScorer` | Mean log-probability of the target under a local causal LM given the context | `contextgc[hf]` |
| `CallableScorer(fn)` | Anything you wrap, such as a hosted model's log-probs or a judge | your function |

The attribution target defaults to the latest user message; pass `target=` (for example a drafted answer) to attribute against that instead.

## Rendering

* `render("openai")`: chat messages; consecutive tool calls are merged into one assistant message and results become `role: tool` messages.
* `render("anthropic")`: `{"system", "messages"}` with `tool_use` and `tool_result` blocks, merged same-role turns, and tool results placed first in a user turn.
* `render("text")`: a transcript with inline tombstones, for logs.

In the API styles, removed context is summarised in a single system note listing the restorable ids. Notes are never placed inline, so they cannot split a tool call from its result.

## CLI

```
contextgc prune chat.json --budget 2000 --keep-recent 4 --infer-edges \
    --attribution ablation --scorer lexical --style openai --out pruned.json --audit audit.jsonl
```

Input is a saved trace (`ContextGraph.save`) or a plain list of `{"role", "content"}` messages. The command exits with code 2 when the budget cannot be met without removing protected context.

## Benchmark (synthetic)

`python benchmarks/needle_retention.py --sessions 200 --out benchmarks/RESULTS.md` builds sessions of 24 look-alike search results in which 2 are needed for the final question, then fits each session into a budget.

| Budget | Strategy | Needed results kept | Tokens removed |
|---|---|---|---|
| 25% of session | truncate oldest (baseline) | 0.5% | 78.6% |
| 25% of session | ContextGC: inferred edges | 100.0% | 78.6% |
| 40% of session | truncate oldest (baseline) | 12.0% | 62.1% |
| 40% of session | ContextGC: inferred edges | 100.0% | 62.1% |

Full table in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). **This is a synthetic sanity check.** The question names the needed projects, so similarity-based inference is favoured by construction. It shows the pipeline behaves as designed and catches regressions; it is not evidence of accuracy on real agent traffic. A benchmark on recorded agent traces is on the roadmap.

## Project layout

```
src/contextgc/
  graph.py         blocks, units, steps, edges, restore, save/load
  gc.py            mark, sweep, prune; CollectReport
  attribution.py   ablation attribution and scorers
  infer.py         TF-IDF edge inference from the query
  policy.py        versioned policies
  render.py        openai / anthropic / text rendering with tombstones
  audit.py         append-only audit log (JSONL)
  store.py         cold store
  cli.py           `contextgc prune`
tests/             pytest suite (50 tests)
benchmarks/        synthetic needle-retention benchmark
examples/          end-to-end research-agent session
```

## Roadmap

* **v0.2:** batched ablation for model scorers; group-testing to cut scorer calls on large candidate sets.
* **v0.3:** benchmark on recorded agent traces (multi-hop QA and tool-heavy tasks) against truncation, summarisation and LLMLingua, measuring answer accuracy at fixed budgets.
* **v0.4:** adapters for agent frameworks; OpenTelemetry spans for every collection.

## Related work

* ContextCite (NeurIPS 2024), attributing model generations to context by ablation: https://papers.nips.cc/paper_files/paper/2024/hash/adbea136219b64db96a9941e4249a857-Abstract-Conference.html
* LLMLingua (Microsoft), prompt compression: https://github.com/microsoft/LLMLingua
* MemGPT and Letta, memory management for agents: https://letta.com/blog/memgpt-and-letta
* OpenTelemetry GenAI semantic conventions for agent spans: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans

## License

MIT
