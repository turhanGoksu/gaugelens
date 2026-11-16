"""Secrets from the environment or the project's ``.env``, which is never committed."""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def read(names: tuple[str, ...], env_file: Path = ENV_FILE) -> dict[str, str]:
    """Each name from the environment, else from ``env_file``.

    A missing name stops the program with a hint; values are never printed.
    """
    values = {name: os.environ.get(name, "") for name in names}
    if not all(values.values()) and env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            key = key.strip()
            if sep and key in values and not values[key]:
                values[key] = value.strip().strip("'\"")
    if missing := [name for name in names if not values[name]]:
        raise SystemExit(f"Set {' and '.join(missing)} (see .env.example).")
    return values
