import json

import pytest

from contextgc import (
    AuditLog,
    CallableScorer,
    ContextGraph,
    LexicalSupportScorer,
    Policy,
    Scorer,
    attribute_units,
    infer_edges,
)
from contextgc.cli import main


# ----------------------------------------------------------------- attribution
def test_lexical_scorer_is_bounded_and_monotone(session):
    s = LexicalSupportScorer()
    blocks = session.blocks()
    target = "Aurora launch code K7-29"
    s.prepare(blocks, target)
    full = s.score(blocks, target)
    assert 0.0 <= full <= 1.0
    assert s.score([], target) == 0.0
    assert s.score([b for b in blocks if b.id != "res_aurora"], target) < full


def test_attribute_units_ranks_the_supporting_unit_highest(session):
    units = {u: bl for u, bl in session.units().items() if u.startswith("call_")}
    results = attribute_units(session.blocks(), units, LexicalSupportScorer(), "Aurora launch code K7-29")
    best = max(results, key=lambda r: r.score)
    assert best.unit == "call_aurora"
    assert set(best.block_ids) == {"call_aurora", "res_aurora"}


def test_attribute_units_reports_nan_as_error(session):
    units = {"call_aurora": session.units()["call_aurora"]}
    results = attribute_units(session.blocks(), units, CallableScorer(lambda c, t, q: float("nan")), "x")
    assert results[0].score is None and "base score failed" in results[0].error


def test_scorer_protocol_runtime_check():
    assert isinstance(LexicalSupportScorer(), Scorer)


def test_transformers_scorer_is_optional():
    from contextgc import TransformersLogProbScorer

    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            TransformersLogProbScorer("any")


# ------------------------------------------------------------------- infer
def test_infer_edges_links_relevant_units_only(session):
    sims = infer_edges(session, "What is the Aurora launch code?", threshold=0.2)
    step = session.step("__query__")
    assert {"res_aurora", "call_aurora"} <= set(step.edges)
    assert "res_cobalt" not in step.edges
    assert sims["res_aurora"] > sims["res_cobalt"]


def test_infer_edges_replaces_previous_inference(session):
    infer_edges(session, "Aurora launch code", threshold=0.2)
    infer_edges(session, "Cobalt budget March", threshold=0.2)
    edges = session.step("__query__").edges
    assert "res_cobalt" in edges and "res_aurora" not in edges


# ------------------------------------------------------------------ render
def test_openai_render_merges_parallel_tool_calls_and_pairs_results():
    g = ContextGraph()
    g.add_message("system", "rules")
    c1 = g.add_tool_call("search", {"q": "a"})
    c2 = g.add_tool_call("search", {"q": "b"})
    g.add_tool_result(c1, "A")
    g.add_tool_result(c2, "B")
    msgs = g.render("openai")
    assert msgs[0] == {"role": "system", "content": "rules"}
    assert msgs[1]["role"] == "assistant" and len(msgs[1]["tool_calls"]) == 2
    assert [m["tool_call_id"] for m in msgs[2:]] == [c1, c2]


def test_openai_render_puts_tombstones_in_one_system_note(session):
    session.collect(Policy(keep_recent=1, sweep="eager"))
    msgs = session.render("openai")
    notes = [m for m in msgs if m["role"] == "system" and "[contextgc]" in m["content"]]
    assert len(notes) == 1
    assert "call_cobalt, res_cobalt" in notes[0]["content"]
    assert not any(m["role"] == "tool" for m in msgs)
    assert session.render("openai", tombstones=False)[1]["role"] != "system"


def test_anthropic_render_alternates_and_orders_tool_results_first():
    g = ContextGraph()
    g.add_message("system", "rules")
    g.add_message("user", "go")
    c = g.add_tool_call("search", {"q": "a"})
    g.add("chunk", "retrieved")
    g.add_tool_result(c, "A")
    out = g.render("anthropic")
    assert out["system"] == "rules"
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "user"]
    last = out["messages"][-1]["content"]
    assert last[0]["type"] == "tool_result" and last[0]["tool_use_id"] == c


def test_text_render_shows_inline_tombstones(session):
    session.collect(Policy(keep_recent=1, sweep="eager"))
    text = session.render("text")
    assert text.count("[contextgc tombstone]") == 4  # task + three tool units
    assert text.index("[system sys]") < text.index("[contextgc tombstone]")


def test_render_rejects_unknown_style(session):
    with pytest.raises(ValueError):
        session.render("xml")


# ------------------------------------------------------------------- audit
def test_audit_log_mirrors_to_jsonl(tmp_path, session):
    path = tmp_path / "audit.jsonl"
    g = ContextGraph(audit=AuditLog(path))
    g.add("old note", id="n")
    g.add_message("user", "q")
    g.collect(Policy(keep_recent=1, sweep="eager"))
    entries = AuditLog.read_jsonl(path)
    assert [e.block_id for e in entries] == ["n"]
    session.collect(Policy(keep_recent=1, sweep="eager"))
    out = tmp_path / "full.jsonl"
    session.audit.to_jsonl(out)
    assert len(AuditLog.read_jsonl(out)) == len(session.audit)


def test_policy_version_is_stable_and_sensitive():
    assert Policy().version_id == Policy().version_id
    assert Policy(keep_recent=3).version_id != Policy(keep_recent=4).version_id
    with pytest.raises(ValueError):
        Policy(infer_threshold=2)
    with pytest.raises(ValueError):
        Policy(sweep="sometimes")


# --------------------------------------------------------------------- cli
def test_cli_prunes_a_chat_transcript(tmp_path, capsys):
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Tell me about Cobalt budget and timelines in detail. " * 10},
        {"role": "assistant", "content": "Cobalt has a 4 million budget. " * 10},
        {"role": "user", "content": "What is the Aurora launch code?"},
    ]
    src = tmp_path / "chat.json"
    src.write_text(json.dumps(messages))
    out = tmp_path / "out.json"
    audit = tmp_path / "audit.jsonl"
    code = main(["prune", str(src), "--budget", "40", "--keep-recent", "1", "--out", str(out), "--audit", str(audit)])
    assert code == 0
    data = json.loads(out.read_text())
    assert data["report"]["budget_met"] is True
    assert data["rendered"][-1]["content"] == "What is the Aurora launch code?"
    assert audit.read_text().strip()
    assert "removed" in capsys.readouterr().err


def test_cli_exit_code_2_when_budget_cannot_be_met(tmp_path):
    src = tmp_path / "chat.json"
    src.write_text(json.dumps([{"role": "system", "content": "x" * 400}, {"role": "user", "content": "hi"}]))
    assert main(["prune", str(src), "--budget", "5", "--out", str(tmp_path / "o.json")]) == 2


def test_cli_round_trips_a_saved_trace(tmp_path, session):
    path = tmp_path / "trace.json"
    session.save(path)
    code = main(
        [
            "prune",
            str(path),
            "--budget",
            "80",
            "--keep-recent",
            "1",
            "--infer-edges",
            "--attribution",
            "ablation",
            "--style",
            "text",
            "--warn-only",
            "--out",
            str(tmp_path / "o.json"),
        ]
    )
    assert code == 0
    data = json.loads((tmp_path / "o.json").read_text())
    assert "res_aurora" in {b["id"] for b in data["trace"]["blocks"]}
