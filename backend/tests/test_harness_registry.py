"""Harness registry (no ML imports)."""
from __future__ import annotations

from app.services import harness_registry


def test_put_get_release() -> None:
    harness_registry.release("nonexistent")
    harness_registry.put("s1", object())
    assert harness_registry.get("s1") is not None
    harness_registry.touch("s1")
    harness_registry.release("s1")
    assert harness_registry.get("s1") is None
