#!/usr/bin/env python3
"""Register, activate, inspect, and roll back production prompts.

Examples:
    python -m scripts.manage_prompts list
    python -m scripts.manage_prompts register generation.default_system prompt.txt --activate
    python -m scripts.manage_prompts rollback generation.default_system
    python -m scripts.manage_prompts rollback generation.default_system --to 8d0c2f1a
    python -m scripts.manage_prompts history generation.default_system
"""

from __future__ import annotations

import argparse
import getpass
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from src.config import settings
from src.prompt_catalog import PROMPT_FIELDS, builtin_prompt_specs, sync_builtin_prompts
from src.prompt_management import PromptError, PromptStore


def _short(value: str | None) -> str:
    return value[:12] if value else "-"


def _time(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat(timespec="seconds")


def _actor(value: str | None) -> str:
    return value or getpass.getuser() or "operator"


def _spec(name: str):
    try:
        return builtin_prompt_specs()[name]
    except KeyError:
        raise PromptError(f"Unknown runtime prompt {name!r}.") from None


def _print_versions(store: PromptStore, name: str | None) -> None:
    versions = store.versions(name)
    if not versions:
        print("No prompt versions registered.")
        return
    print("ACTIVE  NAME                           VERSION       CREATED (UTC)                 BY")
    for version in versions:
        print(
            f"{'*' if version.active else ' ':<7} "
            f"{version.name:<30} {_short(version.hash):<13} "
            f"{_time(version.created_at):<29} {version.created_by}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage versioned runtime prompts")
    parser.add_argument(
        "--db",
        type=Path,
        default=settings.prompt_db_file,
        help="Prompt SQLite database (default: RAG_PROMPT_DB_PATH or <data>/prompts.db)",
    )
    parser.add_argument("--actor", help="Name recorded in the activation audit trail")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("sync", help="Register prompts shipped with this application")
    list_parser = commands.add_parser("list", help="List versions and their active pointers")
    list_parser.add_argument("name", nargs="?", choices=sorted(PROMPT_FIELDS))

    show = commands.add_parser("show", help="Print one prompt version")
    show.add_argument("name", choices=sorted(PROMPT_FIELDS))
    show.add_argument("--version", help="Full hash or unambiguous hash prefix; defaults to active")

    register = commands.add_parser("register", help="Register a prompt from a UTF-8 text file")
    register.add_argument("name", choices=sorted(PROMPT_FIELDS))
    register.add_argument("file", type=Path)
    register.add_argument("--activate", action="store_true")

    activate = commands.add_parser("activate", help="Activate a registered prompt version")
    activate.add_argument("name", choices=sorted(PROMPT_FIELDS))
    activate.add_argument("version", help="Full hash or unambiguous hash prefix")

    rollback = commands.add_parser("rollback", help="Restore the previously active prompt")
    rollback.add_argument("name", choices=sorted(PROMPT_FIELDS))
    rollback.add_argument("--to", dest="to_version", help="Rollback to a specific hash/prefix")

    history = commands.add_parser("history", help="Show activation and rollback events")
    history.add_argument("name", choices=sorted(PROMPT_FIELDS))

    args = parser.parse_args(argv)
    actor = _actor(args.actor)
    store = PromptStore(args.db)

    try:
        # This makes the CLI useful on a fresh deployment and makes new code versions visible,
        # but deliberately does not move an existing active pointer.
        sync_builtin_prompts(actor="cli-sync", store=store)

        if args.command == "sync":
            print(f"Registered built-in prompt versions in {store.path}")
        elif args.command == "list":
            _print_versions(store, args.name)
        elif args.command == "show":
            if args.version:
                # Activation validates and mutates, so resolve a prefix from the read-only list.
                matches = [
                    version
                    for version in store.versions(args.name)
                    if version.hash.startswith(args.version)
                ]
                if not matches:
                    raise PromptError(f"No version {args.version!r} exists for {args.name}.")
                if len(matches) > 1:
                    raise PromptError(f"Version prefix {args.version!r} is ambiguous.")
                version = matches[0]
            else:
                active_version = store.current(args.name)
                if active_version is None:
                    raise PromptError(f"No active prompt named {args.name}.")
                version = active_version
            print(version.prompt)
        elif args.command == "register":
            prompt = args.file.read_text(encoding="utf-8")
            digest = store.register(args.name, prompt, PROMPT_FIELDS[args.name], actor=actor)
            if args.activate:
                store.activate(args.name, digest, PROMPT_FIELDS[args.name], actor=actor)
            print(f"{'Activated' if args.activate else 'Registered'} {args.name} {_short(digest)}")
        elif args.command == "activate":
            digest = store.activate(args.name, args.version, PROMPT_FIELDS[args.name], actor=actor)
            print(f"Activated {args.name} {_short(digest)}")
        elif args.command == "rollback":
            digest = store.rollback(
                args.name,
                PROMPT_FIELDS[args.name],
                actor=actor,
                to_version=args.to_version,
            )
            print(f"Rolled back {args.name} to {_short(digest)}")
        elif args.command == "history":
            print("TIME (UTC)                    ACTION      FROM          TO            ACTOR")
            for event in store.history(args.name):
                print(
                    f"{_time(event.created_at):<29} {event.action:<11} "
                    f"{_short(event.from_hash):<13} {_short(event.to_hash):<13} {event.actor}"
                )
    except (OSError, UnicodeError, PromptError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
