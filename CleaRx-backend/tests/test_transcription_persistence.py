import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import db


class _Cursor:
    def __init__(self):
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, query, params):
        self.executed.append((query, params))

    def fetchone(self):
        return ("output-1",)


class _Connection:
    def __init__(self):
        self.cursor_instance = _Cursor()
        self.commits = 0
        self.rollbacks = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


def test_output_and_fields_share_one_transaction(monkeypatch):
    connection = _Connection()
    execute_values_calls = []

    monkeypatch.setattr(db.psycopg2, "connect", lambda *args, **kwargs: connection)

    def execute_values(cursor, query, rows, fetch=False):
        execute_values_calls.append((cursor, query, rows, fetch))
        return [("field-1", 0)]

    monkeypatch.setattr(db, "execute_values", execute_values)

    output_id, field_ids = db.DatabaseService().create_transcription_output_with_fields(
        job_id="job-1",
        raw_text="raw",
        overall_confidence=0.9,
        token_count=1,
        output_status="completed",
        fields=[("drug_name", "Example", 0, 0.9, 0, 7)],
    )

    assert output_id == "output-1"
    assert field_ids == {0: "field-1"}
    assert len(execute_values_calls) == 1
    assert execute_values_calls[0][0] is connection.cursor_instance
    assert execute_values_calls[0][3] is True
    assert "RETURNING field_id, sequence_order" in execute_values_calls[0][1]
    assert connection.commits == 1
