"""Shared test helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402


@pytest.fixture()
def override_settings():
    """
    Settings is a frozen dataclass (so nothing changes it by accident at runtime).
    Tests that need a different value go through this, and everything is put back
    afterwards.
    """
    saved = {}

    def set_(**values):
        for name, value in values.items():
            saved.setdefault(name, getattr(settings, name))
            object.__setattr__(settings, name, value)

    yield set_
    for name, value in saved.items():
        object.__setattr__(settings, name, value)
