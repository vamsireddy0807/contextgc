import pytest

from contextgc import ContextGraph


@pytest.fixture
def session():
    """A small agent session: pinned system prompt, a task, three searches, a question."""
    g = ContextGraph()
    g.add("You are a careful research agent.", "system", pinned=True, id="sys")
    g.add_message("user", "Collect facts about the projects.", id="task")
    for name, fact in [
        ("aurora", "The Aurora launch code is K7-29 and its owner is Meera."),
        ("cobalt", "The Cobalt budget is 4 million and it ships in March."),
        ("delta", "The Delta region is Pune and its lead is Arjun."),
    ]:
        call = g.add_tool_call("search", {"q": name}, id=f"call_{name}")
        g.add_tool_result(call, fact * 4, id=f"res_{name}")
    g.add_message("user", "What is the Aurora launch code?", id="question")
    return g
