"""Shared pytest fixtures.

Unit tests must be hermetic: their expected defaults (endpoint paths, whether a
backend delegates, whether an API key is present) are all resolved from
environment variables at construction time. Any ambient value -- exported in a
developer's shell, or pushed into ``os.environ`` by the ADK agent loader when
the integration suite reads ``examples/.env`` -- would otherwise silently change
what the unit tests assert.
"""

from __future__ import annotations

from collections.abc import Iterator
import os

import pytest

# Every environment variable that changes how a backend configures itself.
_BACKEND_ENV_PREFIXES: tuple[str, ...] = (
    "DIFFUSIONGEMMA_",
    "OPENJEV_",
    "TYPESAFE_",
)


@pytest.fixture(autouse=True)
def isolate_backend_env(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Strip backend configuration from the environment for unit tests.

    Integration tests are exempt: they deliberately load ``examples/.env`` to
    exercise the same configuration path a real ``adk web`` session uses.
    """
    if "tests/unit" in request.node.nodeid or "tests\\unit" in request.node.nodeid:
        for name in list(os.environ):
            if name.startswith(_BACKEND_ENV_PREFIXES):
                monkeypatch.delenv(name, raising=False)
    yield
