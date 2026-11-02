import pytest

from contextgc import BlockKind, ContextGraph, DuplicateBlock, EdgeKind, UnknownBlock


def test_add_assigns_ids_tokens_and_order():
    g = ContextGraph()
    a = g.add("hello world", "user")
    b = g.add("second", "assistant")
    assert [x.id for x in g.blocks()] == [a, b]
    assert g.get(a).kind is BlockKind.USER
    assert g.get(a).tokens == 3  # ceil(11 / 4)
    assert g.tokens == g.get(a).tokens + g.get(b).tokens


def test_custom_token_counter_and_explicit_tokens():
    g = ContextGraph(token_counter=lambda s: len(s.split()))
    a = g.add("one two three")
    b = g.add("ignored", tokens=10)
    assert g.get(a).tokens == 3
    assert g.get(b).tokens == 10


def test_duplicate_and_unknown_ids_are_rejected():
    g = ContextGraph()
    g.add("x", id="a")
    with pytest.raises(DuplicateBlock):
        g.add("y", id="a")
    with pytest.raises(UnknownBlock):
        g.add("z", depends_on=["missing"])
    with pytest.raises(UnknownBlock):
        g.get("missing")


def test_system_messages_are_pinned_by_default():
    g = ContextGraph()
    s = g.add_message("system", "rules")
    assert g.get(s).pinned


def test_add_message_rejects_non_chat_roles():
    with pytest.raises(ValueError):
        ContextGraph().add_message("tool_result", "x")


def test_tool_call_and_result_share_a_unit():
    g = ContextGraph()
    call = g.add_tool_call("search", {"q": "x"})
    res = g.add_tool_result(call, "found x")
    assert g.get(call).unit == g.get(res).unit == call
    assert g.get(res).depends_on == (call,)
    assert g.get(res).meta["tool_call_id"] == call
    assert list(g.units()) == [call]


def test_tool_result_requires_a_tool_call():
    g = ContextGraph()
    u = g.add("not a call", "user")
    with pytest.raises(ValueError):
        g.add_tool_result(u, "x")


def test_explicit_edge_is_never_downgraded():
    g = ContextGraph()
    a = g.add("a")
    g.link("s", [a])
    g.link("s", [a], inferred=True, confidence=0.3)
    edge = g.step("s").edges[a]
    assert edge.kind is EdgeKind.EXPLICIT and edge.confidence == 1.0


def test_link_validates_confidence():
    g = ContextGraph()
    a = g.add("a")
    with pytest.raises(ValueError):
        g.link("s", [a], inferred=True, confidence=1.5)


def test_complete_marks_step_not_live():
    g = ContextGraph()
    a = g.add("a")
    g.link("s", [a])
    g.complete("s")
    assert g.live_steps() == []


def test_restore_brings_back_whole_unit_and_audits(session):
    session._move_to_cold(["call_cobalt", "res_cobalt"], action="swept", reason="test")
    assert "res_cobalt" not in session
    restored = session.restore("res_cobalt")
    assert set(restored) == {"call_cobalt", "res_cobalt"}
    assert [e.action for e in session.audit_log()][-2:] == ["restored", "restored"]


def test_new_block_depending_on_cold_block_restores_it(session):
    session._move_to_cold(["call_delta", "res_delta"], action="swept", reason="test")
    session.add("Delta summary", depends_on=["res_delta"])
    assert "res_delta" in session and "call_delta" in session


def test_serialisation_round_trip(session, tmp_path):
    session.link("answer", ["res_aurora"])
    session._move_to_cold(["call_delta", "res_delta"], action="swept", reason="test")
    path = tmp_path / "trace.json"
    session.save(path)
    loaded = ContextGraph.load(path)
    assert [b.id for b in loaded.blocks()] == [b.id for b in session.blocks()]
    assert loaded.cold.ids() == ["call_delta", "res_delta"]
    assert loaded.step("answer").edges["res_aurora"].kind is EdgeKind.EXPLICIT
    new_id = loaded.add("after load")
    assert loaded.get(new_id).seq > max(b.seq for b in session.blocks())


def test_latest_returns_newest_of_kind(session):
    assert session.latest("user").id == "question"
    assert ContextGraph().latest("user") is None
