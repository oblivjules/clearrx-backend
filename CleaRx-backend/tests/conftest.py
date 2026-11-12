import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def isolate_database_connections(monkeypatch, request):
    """Unit tests must mock connections; integration tests explicitly opt in."""
    import psycopg2

    integration_enabled = (
        request.module.__name__ == "test_capture_attempts"
        and os.getenv("TEST_DATABASE_URL")
        and os.getenv("TEST_PROFILE_ID")
    )
    if not integration_enabled:
        def reject_connection(*args, **kwargs):
            raise AssertionError("Database access must be mocked outside opt-in integration tests")

        monkeypatch.setattr(psycopg2, "connect", reject_connection)
