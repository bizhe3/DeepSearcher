"""Checkpoint persistence helpers for trajectory crash recovery."""

from __future__ import annotations

import os
from typing import Optional

from deepresearch.agent.types import Trajectory


def save_checkpoint(trajectory: Trajectory, path: str) -> None:
    """Atomically save a trajectory checkpoint as formatted JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        file.write(trajectory.model_dump_json(indent=2))
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> Optional[Trajectory]:
    """Load a trajectory checkpoint, returning None for missing/invalid files."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()
        return Trajectory.model_validate_json(text)
    except (FileNotFoundError, OSError, ValueError):
        return None
