# Refactoring migration changelog

Changes are grouped under the MVP Validation Findings & Refactoring Priorities.
Passing regression tests does not establish improved recognition accuracy or
achievement of post-refactoring outcome targets.

## B1 — Isolated backend regression setup

- Priorities: shared verification infrastructure supporting #1–#5.
- Added development requirements and a shared test import path.
- Blocked real PostgreSQL connections in unit tests unless explicitly mocked.
- Capture integration tests require TEST_DATABASE_URL and a valid TEST_PROFILE_ID
  from a disposable test database. The fixture restores environment changes and
  removes its images before removing the capture session.
- Kept the original non-pooled DatabaseService API; pooling belongs to B8.
- Verification: `python -m pytest tests -q` — 15 passed, 2 skipped.
  The skips are the opt-in database capture tests. The suite also passed with
  an unreachable application DATABASE_URL and both integration opt-ins unset
  (15 passed, 2 skipped). `git diff --check` passed.
- Commit: pending explicit user approval.

Run from CleaRx-backend with `python -m pytest tests -q`. Without explicit test
database configuration, capture integration tests skip. Never configure these
write tests against an application or production database.
