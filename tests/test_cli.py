from __future__ import annotations

import pytest

from langchain_human_in_the_loop import cli


def test_cli_timeout_override(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured: dict[str, float] = {}

    class FakeHumanInTheLoop:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured["timeout"] = kwargs["timeout"]
            captured["max_credits"] = kwargs["max_credits"]

        def invoke(self, prompt: str) -> dict[str, str]:
            assert prompt == "Review this function."
            return {"status": "approved", "output": "ok"}

    monkeypatch.setattr(cli, "HumanInTheLoop", FakeHumanInTheLoop)

    exit_code = cli.main(
        [
            "--project-id",
            "1",
            "--max-credit",
            "40",
            "--timeout",
            "1200",
            "Review this function.",
        ]
    )

    assert exit_code == 0
    assert captured["timeout"] == 1200.0
    assert captured["max_credits"] == 40
    assert '"status": "approved"' in capsys.readouterr().out


def test_cli_timeout_minus_one(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float] = {}

    class FakeHumanInTheLoop:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured["timeout"] = kwargs["timeout"]

        def invoke(self, prompt: str) -> dict[str, str]:
            return {"status": "approved", "output": prompt}

    monkeypatch.setattr(cli, "HumanInTheLoop", FakeHumanInTheLoop)

    exit_code = cli.main(
        [
            "--project-id",
            "1",
            "--max-credit",
            "10",
            "--timeout",
            "-1",
            "Never timeout.",
        ]
    )

    assert exit_code == 0
    assert captured["timeout"] == -1


def test_cli_rejects_timeout_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--project-id", "1", "--max-credit", "10", "--timeout", "0", "Review."])

    assert exc.value.code == 2


def test_cli_returns_non_zero_on_timeout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeHumanInTheLoop:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def invoke(self, prompt: str) -> dict[str, str]:
            raise TimeoutError("Invoke timed out. Increase --timeout.")

    monkeypatch.setattr(cli, "HumanInTheLoop", FakeHumanInTheLoop)

    exit_code = cli.main(["--project-id", "1", "--max-credit", "10", "Review."])

    assert exit_code == 1
    assert "Invoke timed out. Increase --timeout." in capsys.readouterr().err
