"""Smoke test: the package imports and its entry point runs.

Replaced by real tests in phase 1.
"""

from agent_harness import main


def test_main_runs(capsys):
    main()
    assert "agent-harness" in capsys.readouterr().out
