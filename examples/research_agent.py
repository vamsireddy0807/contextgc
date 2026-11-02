"""A research agent session pruned to a budget, end to end.

Run:  python examples/research_agent.py
"""

from contextgc import ContextGraph, LexicalSupportScorer, Policy

g = ContextGraph(budget_tokens=150)
g.add_message("system", "You are a careful research agent. Cite tool results.")
g.add_message("user", "Prepare a launch brief for Project Aurora.")

facts = {
    "aurora": "Project Aurora launches on 14 March from the Pune facility; launch code K7-29.",
    "cobalt": "Project Cobalt is a payments revamp with a 4 million budget, owned by Meera.",
    "delta": "Project Delta migrates the data warehouse to a new region next quarter.",
    "aurora_risks": "Aurora risks: supplier delay on sensor boards; mitigation is a second vendor.",
}
results = {}
for key, text in facts.items():
    call = g.add_tool_call("search", {"q": key.replace("_", " ")}, id=f"call_{key}")
    results[key] = g.add_tool_result(call, (text + " ") * 2, id=f"res_{key}")

# The agent already drafted a section that explicitly cites the launch facts.
g.link("draft_section_1", [results["aurora"]])
g.add_message("assistant", "Draft: Aurora launches 14 March (see res_aurora).")
g.add_message("user", "Now add the Aurora risks and mitigation.")

print(f"before: {g.tokens} tokens, {len(g)} blocks, budget {g.budget_tokens}")
report = g.collect(
    Policy(keep_recent=2, infer_edges=True, attribution="ablation", min_attribution=0.02),
    scorer=LexicalSupportScorer(),
)
print(report.summary())
print("kept:", [b.id for b in g.blocks()])
print("\naudit:")
for e in g.audit_log():
    print(f"  {e.action:8s} {e.block_id:16s} {e.reason}")
print("\nrendered transcript:\n")
print(g.render("text"))

# Anything removed can be brought back on demand:
g.restore("res_cobalt")
assert "res_cobalt" in g and "call_cobalt" in g
