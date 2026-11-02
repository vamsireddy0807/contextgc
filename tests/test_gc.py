import pytest

from contextgc import BudgetExceeded, LexicalSupportScorer, Policy, reachability


def test_no_budget_lazy_sweep_removes_nothing(session):
    before = session.tokens
    report = session.collect(Policy(keep_recent=1))
    assert report.swept == [] and report.pruned == []
    assert session.tokens == before and report.budget_met


def test_eager_sweep_removes_every_unreachable_unit(session):
    report = session.collect(Policy(keep_recent=1, sweep="eager"))
    remaining = {b.id for b in session.blocks()}
    assert remaining == {"sys", "question"}
    assert set(report.swept) == {
        "task",
        "call_aurora",
        "res_aurora",
        "call_cobalt",
        "res_cobalt",
        "call_delta",
        "res_delta",
    }


def test_explicit_dependencies_are_protected(session):
    session.link("answer", ["res_aurora"])
    session.collect(Policy(keep_recent=1, sweep="eager"))
    assert "res_aurora" in session and "call_aurora" in session  # whole unit kept


def test_completed_step_no_longer_protects(session):
    session.link("answer", ["res_aurora"])
    session.complete("answer")
    session.collect(Policy(keep_recent=1, sweep="eager"))
    assert "res_aurora" not in session


def test_pinned_blocks_are_never_removed(session):
    session.pin("res_cobalt")
    session.collect(Policy(keep_recent=0, sweep="eager"))
    assert "res_cobalt" in session and "sys" in session


def test_lazy_sweep_is_oldest_first_and_stops_at_budget(session):
    budget = session.tokens - session.get("res_aurora").tokens - session.get("call_aurora").tokens
    report = session.collect(Policy(keep_recent=1, budget_tokens=budget))
    assert report.swept[:1] == ["task"]
    assert report.budget_met
    assert session.tokens <= budget


def test_inferred_edges_keep_relevant_context_over_irrelevant(session):
    budget = session.tokens // 2
    report = session.collect(Policy(keep_recent=1, budget_tokens=budget, infer_edges=True))
    assert "res_aurora" in session
    assert "res_cobalt" not in session and "res_delta" not in session
    assert report.budget_met


def test_budget_exceeded_raises_with_report(session):
    session.link("answer", ["res_aurora", "res_cobalt", "res_delta"])
    with pytest.raises(BudgetExceeded) as exc:
        session.collect(Policy(keep_recent=1, budget_tokens=10))
    assert exc.value.report.budget_met is False
    assert "res_aurora" in session  # protected context untouched


def test_warn_mode_returns_report_instead_of_raising(session):
    session.link("answer", ["res_aurora", "res_cobalt", "res_delta"])
    report = session.collect(Policy(keep_recent=1, budget_tokens=10, on_over_budget="warn"))
    assert report.budget_met is False


def test_soft_pruning_without_scorer_uses_lowest_confidence_first(session):
    session.link("q", ["res_aurora"], inferred=True, confidence=0.9)
    session.link("q", ["res_cobalt"], inferred=True, confidence=0.1)
    session.link("q", ["res_delta"], inferred=True, confidence=0.5)
    unit = session.get("res_cobalt").tokens + session.get("call_cobalt").tokens
    budget = session.tokens - session.get("task").tokens - unit
    report = session.collect(Policy(keep_recent=1, budget_tokens=budget))
    assert report.pruned == ["call_cobalt", "res_cobalt"]
    assert "res_aurora" in session and "res_delta" in session


def test_ablation_prunes_unsupportive_units_and_keeps_supportive(session):
    for name in ("aurora", "cobalt", "delta"):
        session.link("q", [f"res_{name}"], inferred=True, confidence=0.5)
    budget = session.get("res_aurora").tokens + 60
    report = session.collect(
        Policy(keep_recent=1, budget_tokens=budget, attribution="ablation", on_over_budget="warn"),
        scorer=LexicalSupportScorer(),
        target="The Aurora launch code is K7-29",
    )
    assert "res_aurora" in session
    assert set(report.pruned) >= {"res_cobalt", "res_delta"}
    assert report.scores["res_aurora"] > report.scores["res_cobalt"]


def test_ablation_never_prunes_both_copies_of_redundant_evidence():
    from contextgc import ContextGraph

    g = ContextGraph()
    g.add("system", "system", pinned=True)
    a = g.add("The Aurora launch code is K7-29.", "retrieved", id="copy_a")
    b = g.add("Again: the Aurora launch code is K7-29.", "retrieved", id="copy_b")
    c = g.add("Cobalt ships in March with a large budget.", "retrieved", id="noise")
    g.add_message("user", "Aurora launch code?", id="q")
    for x in (a, b, c):
        g.link("q", [x], inferred=True, confidence=0.5)
    g.collect(
        Policy(keep_recent=1, budget_tokens=1, attribution="ablation", on_over_budget="warn"),
        scorer=LexicalSupportScorer(),
        target="Aurora launch code K7-29",
    )
    assert ("copy_a" in g) or ("copy_b" in g)
    assert "noise" not in g


def test_ablation_fails_closed_when_scorer_errors(session):
    from contextgc import CallableScorer

    def broken(context, target, query):
        raise RuntimeError("model unavailable")

    for name in ("aurora", "cobalt", "delta"):
        session.link("q", [f"res_{name}"], inferred=True, confidence=0.5)
    report = session.collect(
        Policy(keep_recent=1, budget_tokens=1, attribution="ablation", on_over_budget="warn"),
        scorer=CallableScorer(broken),
    )
    assert report.pruned == []
    assert {"res_aurora", "res_cobalt", "res_delta"} <= set(report.kept_on_error)
    assert all(x in session for x in ("res_aurora", "res_cobalt", "res_delta"))
    assert any(e.action == "kept" for e in session.audit_log())


def test_ablation_requires_scorer(session):
    with pytest.raises(ValueError):
        session.collect(Policy(attribution="ablation"))


def test_max_attribution_units_keeps_unscored_units(session):
    for name in ("aurora", "cobalt", "delta"):
        session.link("q", [f"res_{name}"], inferred=True, confidence=0.5)
    report = session.collect(
        Policy(
            keep_recent=1,
            budget_tokens=1,
            attribution="ablation",
            max_attribution_units=1,
            on_over_budget="warn",
        ),
        scorer=LexicalSupportScorer(),
        target="Aurora launch code K7-29",
    )
    assert len(report.kept_on_error) == 4  # two units of call+result not evaluated


def test_reachability_split(session):
    session.link("answer", ["res_aurora"])
    session.link("q", ["res_cobalt"], inferred=True, confidence=0.4)
    hard, soft = reachability(session, Policy(keep_recent=1))
    assert {"sys", "question", "res_aurora", "call_aurora"} <= hard
    assert soft == {"res_cobalt", "call_cobalt"}


def test_every_removal_is_audited_with_policy_version(session):
    policy = Policy(keep_recent=1, sweep="eager")
    report = session.collect(policy)
    entries = session.audit_log()
    assert len(entries) == len(report.swept)
    assert {e.policy_version for e in entries} == {policy.version_id}
    assert all(e.reason == "unreachable" for e in entries)


def test_report_fields(session):
    report = session.collect(Policy(keep_recent=1, sweep="eager"))
    d = report.to_dict()
    assert d["tokens_removed"] == report.tokens_before - report.tokens_after
    assert 0 < report.reduction < 1
    assert "removed" in report.summary()
