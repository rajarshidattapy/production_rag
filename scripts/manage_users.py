#!/usr/bin/env python3
"""Manage user accounts on the server, including when sign-up is turned off.

Usage (from the project root, or /app in the container):
    python -m scripts.manage_users create alice
    python -m scripts.manage_users set-password alice
    python -m scripts.manage_users list
    python -m scripts.manage_users delete alice
"""

from __future__ import annotations

import argparse
import getpass
import sys
from collections.abc import Sequence

from src import audit
from src.chat.store import ChatStore, UsernameTakenError
from src.config import settings


def _read_password() -> str:
    password = getpass.getpass("New password: ")
    if len(password) < settings.min_password_length:
        sys.exit(f"Password must be at least {settings.min_password_length} characters.")
    if password != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    return password


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage user accounts")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("create", "Create an account"),
        ("set-password", "Reset a password, clear any lockout, and end the user's sessions"),
        ("delete", "Delete an account and all of its chat history"),
    ):
        commands.add_parser(name, help=description).add_argument("username")
    commands.add_parser("list", help="List usernames")
    args = parser.parse_args(argv)

    store = ChatStore(settings.chat_db_file)

    def done(succeeded: bool) -> None:
        audit.record(
            f"admin.user_{args.command.replace('-', '_')}",
            outcome="success" if succeeded else "failure",
            actor=audit.cli_actor(),
            username=args.username,
        )

    if args.command == "list":
        print("\n".join(store.list_usernames()))
    elif args.command == "create":
        try:
            created = store.create_user(args.username, _read_password())
        except UsernameTakenError:
            done(False)
            sys.exit(f"{args.username} already exists.")
        done(True)
        print(f"Created {created.username}")
    elif args.command == "set-password":
        changed = store.set_password(args.username, _read_password())
        done(changed)
        if not changed:
            sys.exit(f"No account named {args.username}.")
        print(f"Password changed for {args.username}")
    elif args.command == "delete":
        deleted = store.delete_user(args.username)
        done(deleted)
        if not deleted:
            sys.exit(f"No account named {args.username}.")
        print(f"Deleted {args.username} and their chat history")


if __name__ == "__main__":
    main()
