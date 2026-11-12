# Local/test processing reliability — B8

`APP_ENV` defaults to `local`; `local` and `test` are the supported values. Other
values fail startup until authentication, ownership and HTTPS deployment work is
completed. This environment guard is not authentication: keep the demo on a
trusted network. No deployment or public-access configuration is performed here.

## Configuration

- `DATABASE_URL`: backend PostgreSQL connection; keep credentials private.
- `DB_POOL_SIZE`: maximum connections per worker, default 8, minimum 4.
- `CORS_ORIGINS`: comma-separated browser origins; defaults to
  `http://localhost:8000`. CORS is not access control or authentication.
- `FILE_STORAGE_DIR` and `DRUG_CATALOG_IMPORT_ID`: see IMAGE_UPLOAD.md and
  scripts/README_catalog.md. No new SQL migration is required for B8; catalog
  integration still requires B2's existing schema, not a rerun over live tables.

Connection pools are lazy, bounded and closed on application shutdown. Processing
uses two expensive-request slots per worker. Each registered image is serialized
across workers by a PostgreSQL transaction advisory lock. The lock holds one pool
connection until processing finishes. The database role must support ordinary
application writes and PostgreSQL advisory-lock functions. Pool exhaustion is not
an unbounded queue; database failures may return a retryable error.

## Request and persistence behavior

`/process-image` requires an image ID when database logging is enabled. Submitted
bytes must match the registered checksum. A mismatch returns 400. An in-flight
duplicate or an already-saved result returns 409 before another OCR request.
Durable response replay is not implemented: capture/upload a new image for a new
attempt. Cancellation waits for active work or lock acquisition to finish before
releasing its database lock and processing slot.

Image processing, OCR, parsing, matching and persistence run in worker threads
where blocking work would otherwise stall the event loop. Successful responses
include `stage_timings_ms` for preprocessing, OCR, parsing, matching, persistence
and total duration. These timings are diagnostics, not benchmark results.

`status=success` means a reading was produced. `persistence_status=saved` means
required result writes and the final image-status update succeeded.
`persistence_status=failed` preserves the reading but indicates incomplete saving;
`disabled` means no database persistence was available. An image becomes
`processed` only after required writes complete; unsuccessful writes use
`save_failed`. Confirm that the deployed schema permits those status values.
The result graph still spans multiple transactions, so partial rows can remain.
This migration does not claim a single atomic transaction or automatic repair.

## Verification

From `CleaRx-backend`:

```powershell
python -m pytest tests -q
python main.py
```

Database capture tests require `TEST_DATABASE_URL` and an existing isolated
`TEST_PROFILE_ID` in a disposable database. They create/delete records and must
not target the application or production database. Other tests block unmocked
PostgreSQL connections and use generated images/mocked providers.

Migration checks cover pooling, lock release, cancellation, concurrency limits,
checksum/duplicate rejection, partial writes and completion ordering. No live
database/advisory-lock smoke test or live provider call was performed. Real
schema constraints, multi-worker exclusion and device-to-backend flows still
require integration verification. Earlier MVP measurements are not replaced;
no new accuracy or confidence-calibration claims are made.

Remaining work includes authenticated ownership, production HTTPS, explicit
retention/deletion, durable replay, atomic result persistence and nullable
confidence/provenance columns. See OCR.md for recognition names/confidence and
IMAGE_UPLOAD.md for unchanged storage/privacy limitations. Frontend build,
notice and review UI changes are handled in the separate frontend migration.
