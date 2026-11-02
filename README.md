# ContextGC

**Dependency-aware context runtime for LLM agents: semantic garbage collection, dependency tracking and causal attribution for context pruning.**

> Status: in development (v0.1). Every number in this README is a target or a plan, not a result. Results go in `benchmarks/RESULTS.md` only after they are measured.

---

## Why

Long-running agents pile up context: system prompts, user turns, tool outputs, retrieved chunks, scratch notes. When the window fills, today's options are blunt:

| Approach | What goes wrong |
|---|---|
| Drop the oldest messages | Deletes the one tool result step 14 still needs |
| Summarise old turns | Lossy, unverifiable, and silently changes facts |
| Token-level compression | Saves tokens but has no idea which blocks later steps depend on |

None of them know **what depends on what**, and none of them can explain **why** a block was removed. Pruning becomes a guess, and failures are silent.

## Core idea: treat context like a heap

1. **Nodes.** Every context block is a node: `id`, `kind` (system, user, assistant, tool_result, retrieved_chunk, note), `tokens`, `content_hash`, `created_at`.
2. **Edges ("uses").** When a step consumes a block, record an edge `step -> block`.
   - *Explicit*: a tool call references a result id; an answer cites a chunk id.
   - *Inferred*: lexical or semantic overlap between a step's output and a block (lower confidence, marked as inferred).
3. **Roots.** System prompt, pinned facts, the current task, open tool calls. Roots are never collected.
4. **Mark and sweep.** Mark everything reachable from roots and live steps; unreachable blocks move to cold storage.
5. **Causal attribution before pruning reachable blocks.** If the budget still isn't met, score reachable, unpinned blocks by ablation: remove the block, re-score the current answer (log-probability or answer similarity), and measure the change. Only blocks whose attribution falls below a threshold are pruned.
6. **Audit trail.** Every prune records *what*, *why* (unreachable or low attribution plus score), *when* and *policy version*. Every prune is reversible from cold storage.

```
 messages / tool results / retrieved chunks
                 |
                 v
        +-----------------+      uses-edges       +-----------+
        |  ContextGraph   | <-------------------- |  Steps    |
        +-----------------+                       +-----------+
                 |
        mark from roots + live steps
                 |
     +-----------+------------+
     | unreachable            | reachable, over budget
     v                        v
  sweep to cold store    ablation attribution -> prune if score < threshold
                 \             /
                  v           v
             audit log (reason, score, policy version)
                 |
                 v
          render() -> prompt within budget
```

## Explicit failure boundaries

- Pinned and root blocks are **never** pruned.
- If attribution cannot be computed (API error, timeout), the block is **kept** (fail closed).
- If the budget cannot be met without dropping a high-attribution reachable block, `collect()` raises `BudgetExceeded`; it never silently truncates.
- Inferred edges never override explicit ones.

## API sketch (planned)

```python
from contextgc import ContextGraph, Policy

g = ContextGraph(budget_tokens=16_000)

sys_id = g.add("You are a careful research agent.", kind="system", pinned=True)
r1 = g.add(search_results, kind="tool_result", produced_by="search#1")
r2 = g.add(old_scratchpad, kind="note")

g.link(step="answer#7", uses=[r1])           # explicit dependency

report = g.collect(policy=Policy(
    attribution="ablation",                  # or "none" for pure mark-and-sweep
    min_attribution=0.05,
    scorer="logprob",                        # pluggable: logprob | embedding_similarity
))

prompt = g.render()                          # within budget, order preserved
for entry in g.audit_log():
    print(entry.block_id, entry.reason, entry.score)
```

## Roadmap

- **v0.1:** graph model, explicit edges, mark and sweep, token budget, cold store, audit log, unit tests.
- **v0.2:** ablation attribution with a small open model (log-probabilities) plus a pluggable API scorer; batching to keep attribution cost bounded.
- **v0.3:** benchmark harness comparing truncation, summarisation, LLMLingua and ContextGC on long multi-hop QA and recorded agent traces.
- **v0.4:** adapters for OpenAI-compatible and Anthropic message lists; OpenTelemetry spans for every prune.

## Evaluation plan (what to measure before claiming anything)

| Metric | Definition |
|---|---|
| Token reduction | % tokens removed at equal answer accuracy |
| Accuracy drop | percentage-point change at a fixed token budget |
| Attribution cost | extra model calls and latency per `collect()` |
| Safety | number of times a block later needed was pruned (should be 0 with explicit edges) |

Rule: only numbers in `benchmarks/RESULTS.md`, produced by a script in `benchmarks/`, go on a resume.

## Repository layout (planned)

```
contextgc/
  graph.py         # nodes, edges, roots
  gc.py            # mark and sweep, cold store
  attribution.py   # ablation scorers
  policy.py        # thresholds, budgets, versioned policies
  render.py        # prompt assembly within budget
  audit.py         # append-only prune log
tests/
benchmarks/
examples/
```

## Related work to study and build on

- ContextCite (NeurIPS 2024): attributing model generations to context by ablation. https://papers.nips.cc/paper_files/paper/2024/hash/adbea136219b64db96a9941e4249a857-Abstract-Conference.html
- LLMLingua (Microsoft): prompt compression, a natural baseline. https://github.com/microsoft/LLMLingua
- MemGPT / Letta: memory management for agents. https://letta.com/blog/memgpt-and-letta
- OpenTelemetry GenAI semantic conventions for agent spans. https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans

## License

MIT (proposed).
