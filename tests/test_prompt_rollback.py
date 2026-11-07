"""Runtime prompt version activation and production rollback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.manage_prompts import main as manage_prompts
from src.config import settings
from src.generation.generator import DEFAULT_SYSTEM_PROMPT, Generator
from src.prompt_catalog import DEFAULT_SYSTEM, PROMPT_FIELDS, sync_builtin_prompts
from src.prompt_management import PromptStore, PromptValidationError, prompt_hash


def test_new_version_is_immutable_and_does_not_activate_itself(tmp_path: Path) -> None:
    store = PromptStore(tmp_path / "prompts.db")
    first = "First prompt.\n{context}"
    second = "Second prompt.\n{context}"

    first_hash = store.register(DEFAULT_SYSTEM, first, ("context",), actor="deploy-1")
    second_hash = store.register(DEFAULT_SYSTEM, second, ("context",), actor="deploy-2")

    assert first_hash != second_hash
    assert store.current(DEFAULT_SYSTEM).prompt == first  # type: ignore[union-attr]
    assert {version.prompt for version in store.versions(DEFAULT_SYSTEM)} == {first, second}


def test_activate_and_rollback_are_immediate_and_audited(tmp_path: Path) -> None:
    path = tmp_path / "prompts.db"
    store = PromptStore(path)
    first_hash = store.register(DEFAULT_SYSTEM, "Good {context}", ("context",), actor="deploy")
    bad_hash = store.register(DEFAULT_SYSTEM, "Bad {context}", ("context",), actor="deploy")

    store.activate(DEFAULT_SYSTEM, bad_hash[:12], ("context",), actor="alice")
    # A separate store simulates another API process observing the active pointer.
    assert PromptStore(path).current(DEFAULT_SYSTEM).hash == bad_hash  # type: ignore[union-attr]

    restored = store.rollback(DEFAULT_SYSTEM, ("context",), actor="bob")
    assert restored == first_hash
    assert PromptStore(path).current(DEFAULT_SYSTEM).prompt == "Good {context}"  # type: ignore[union-attr]
    history = store.history(DEFAULT_SYSTEM)
    assert [(event.action, event.actor) for event in history[:2]] == [
        ("rollback", "bob"),
        ("activate", "alice"),
    ]


def test_activation_rejects_a_changed_placeholder_contract(tmp_path: Path) -> None:
    store = PromptStore(tmp_path / "prompts.db")
    with pytest.raises(PromptValidationError, match="missing context"):
        store.register(DEFAULT_SYSTEM, "No template field", ("context",))
    with pytest.raises(PromptValidationError, match="unexpected question"):
        store.register(DEFAULT_SYSTEM, "{context} {question}", ("context",))


def test_all_builtin_runtime_prompts_sync_and_validate(tmp_path: Path) -> None:
    store = PromptStore(tmp_path / "prompts.db")
    hashes = sync_builtin_prompts(actor="test", store=store)
    assert set(hashes) == set(PROMPT_FIELDS)
    assert len(store.versions()) == len(PROMPT_FIELDS)
    assert all(version.active for version in store.versions())


def test_generator_uses_activated_prompt_and_rolls_back_without_restart(
    tmp_path: Path,
) -> None:
    store = PromptStore(settings.prompt_db_file)
    built_in_hash = store.register(
        DEFAULT_SYSTEM,
        DEFAULT_SYSTEM_PROMPT,
        PROMPT_FIELDS[DEFAULT_SYSTEM],
        actor="startup",
    )
    replacement = "Production prompt. Evidence:\n{context}"
    replacement_hash = store.register(
        DEFAULT_SYSTEM, replacement, PROMPT_FIELDS[DEFAULT_SYSTEM], actor="alice"
    )
    store.activate(
        DEFAULT_SYSTEM,
        replacement_hash,
        PROMPT_FIELDS[DEFAULT_SYSTEM],
        actor="alice",
    )

    generator = Generator()
    generator._client = MagicMock()
    generator._client.complete.return_value = "answer"
    generator.generate("question", [{"document": "evidence", "metadata": {}}])
    first_system = generator._client.complete.call_args.kwargs["system"]
    assert first_system.startswith("Production prompt.")
    assert "evidence" in first_system

    store.rollback(DEFAULT_SYSTEM, PROMPT_FIELDS[DEFAULT_SYSTEM], actor="bob")
    generator.generate("question", [{"document": "evidence", "metadata": {}}])
    second_system = generator._client.complete.call_args.kwargs["system"]
    assert second_system.startswith("You are a helpful research assistant.")
    assert store.current(DEFAULT_SYSTEM).hash == built_in_hash  # type: ignore[union-attr]


def test_management_cli_registers_activates_and_rolls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "prompts.db"
    candidate = tmp_path / "prompt.txt"
    candidate.write_text("CLI prompt\n{context}", encoding="utf-8")

    manage_prompts(
        [
            "--db",
            str(db),
            "--actor",
            "release-bot",
            "register",
            DEFAULT_SYSTEM,
            str(candidate),
            "--activate",
        ]
    )
    assert "Activated" in capsys.readouterr().out
    assert PromptStore(db).current(DEFAULT_SYSTEM).hash == prompt_hash(candidate.read_text())  # type: ignore[union-attr]

    manage_prompts(["--db", str(db), "--actor", "on-call", "rollback", DEFAULT_SYSTEM])
    assert "Rolled back" in capsys.readouterr().out
    assert PromptStore(db).current(DEFAULT_SYSTEM).prompt == DEFAULT_SYSTEM_PROMPT  # type: ignore[union-attr]
