import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

psycopg2 = pytest.importorskip("psycopg2")

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to run database integration tests",
)


@pytest.fixture
def capture_session(monkeypatch):
    from db import DatabaseService

    profile_id = os.getenv("TEST_PROFILE_ID")
    if not profile_id:
        pytest.skip("Set TEST_PROFILE_ID to an existing isolated test profile UUID")
    uuid.UUID(profile_id)
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    service = DatabaseService()
    session_id = service.create_capture_session(
        user_id=profile_id,
        image_source="test",
    )
    try:
        yield service, session_id
    finally:
        with service._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'DELETE FROM "PRESCRIPTION_IMAGE" WHERE session_id = %s',
                    (session_id,),
                )
                cur.execute(
                    'DELETE FROM "CAPTURE_SESSION" WHERE session_id = %s',
                    (session_id,),
                )


def test_three_captures_allocate_attempts_and_total(capture_session):
    service, session_id = capture_session

    attempts = [
        service.create_prescription_image(
            session_id=session_id,
            image_bytes=f"image-{index}".encode(),
            width=100,
            height=100,
        )[1]
        for index in range(3)
    ]

    assert attempts == [1, 2, 3]
    with service._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT total_captures FROM "CAPTURE_SESSION" WHERE session_id = %s',
                (session_id,),
            )
            assert cur.fetchone()[0] == 3


def test_concurrent_captures_allocate_unique_attempts(capture_session):
    service, session_id = capture_session

    def create(index):
        return service.create_prescription_image(
            session_id=session_id,
            image_bytes=f"concurrent-{index}".encode(),
            width=100,
            height=100,
        )[1]

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = sorted(executor.map(create, range(2)))

    assert attempts == [1, 2]
