# CleaRx Backend

FastAPI backend for CleaRx, a prescription-reading Android application. It checks
image quality, reads handwriting, extracts medicine instructions and compares
recognized names with a pharmaceutical catalog. Results include field review
flags and recognition provenance for comparison with the prescription.

CleaRx assists reading; it does not establish medication safety or make dispensing
decisions.

## Companion application

- [Android frontend repository](https://github.com/oblivjules/clearrx-frontend)
- Configured Android release endpoint: `https://clearx-refactored.onrender.com/`

This README describes the code in this checkout. A deployed service may run a
different revision; inspect its `/health` and `/openapi.json` before integration.

## Features

- Camera and uploaded JPEG, PNG and WebP images with shared quality checks.
- Bounded image decoding: maximum 10 MB and 24 megapixels.
- Perspective correction, shadow correction and readable comparison previews.
- Genuine local TrOCR as the primary recognition option and an explicitly selected
  external recognition option.
- Deterministic prescription parsing, including dose/frequency extraction,
  uncertainty markers and stated-versus-derived duration handling.
- Versioned product catalogs, brand/generic matching and strength/form/route checks.
- Review flags for uncertain, unmatched or conflicting fields.
- PostgreSQL persistence, bounded connection pooling, per-image processing locks,
  checksum validation and explicit saving outcomes.

## Requirements

- Python with compatible wheels for `requirements.txt`. The recorded local
  regression run used Python 3.13.
- PyTorch/torchvision appropriate for the host's CPU or CUDA environment.
- An existing CleaRx PostgreSQL/Supabase application schema and database role.
- TrOCR model artifacts available locally or downloadable on first use.
- A server-side `ANTHROPIC_API_KEY` only when using the external option.

The repository does **not** contain a complete bootstrap migration for the
application database. `migrations/001_drug_catalog.sql` adds the catalog to an
existing schema; `capture_metrics.sql` contains reporting queries.

## Recognition configuration

| Public option | API `model` value | Processing |
| --- | --- | --- |
| CleaRx TrOCR | `clearx-trocr` (default) |  Local TrOCR on the backend |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | External provider; requires a backend API key |


## API workflow

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | API health/version response |
| POST | `/capture-session` | Create a session with a valid existing profile `user_id` |
| POST | `/preview-check` | Lightweight live-preview feedback |
| POST | `/capture-image` | Register and store a camera image |
| POST | `/validate-capture` | Validate camera-image quality |
| POST | `/validate-upload` | Check crop/upload quality without registration |
| POST | `/upload-file` | Register, store and validate an uploaded image |
| POST | `/process-image` | Read, parse, match and persist results |
| GET | `/files/{image_id}` | Retrieve a registered image file |
| POST | `/capture-session/close` | Close a session as completed or abandoned |

1. Create a capture session with an existing database profile UUID.
2. Register an image through `/capture-image` or `/upload-file`. Camera images
   additionally use `/validate-capture`; `/upload-file` validates on registration.
3. Submit the registered `image_id`, **the same image bytes**, and the selected
   `model` to `/process-image` as multipart form data.
4. Read the response and close the session when finished.

With database persistence enabled, processing requires an image ID. Changed bytes
return 400; an in-flight or already-saved duplicate returns 409. `status=success`
means a reading was produced, while `persistence_status` separately reports
`saved`, `failed` or `disabled`. A successful reading does not guarantee saving.

See [IMAGE_UPLOAD.md](IMAGE_UPLOAD.md) for request fields and shared acceptance
rules. Framing is advisory for camera and upload; lighting, blur and readability
still affect acceptance. Ground truth is optional and intended for QA/CER checks.

## Pharmaceutical catalog

The catalog uses `drug_catalog_imports`, `drug_products` and
`drug_product_aliases`, with nullable product references in result tables.
`DRUG_CATALOG_IMPORT_ID` pins a snapshot; otherwise startup selects the latest.
If loading fails, legacy name matching remains available.

Follow [scripts/README_catalog.md](scripts/README_catalog.md) for schema review,
CSV validation, explicit imports and record-count verification. Do not rerun the
CREATE/ALTER migration over existing catalog tables. The saved report's 49,943
records are not an independently verified live database count; the source CSV is
not bundled here. Stored OCR aliases are not searched by the current matcher,
and ingredient-order normalization remains outstanding.

## Storage and operations

Image files are written to backend storage; PostgreSQL stores image metadata,
file paths, recognized text and structured prescription results rather than image
file blobs. Session closure does not implement automatic file/record deletion.
See [IMAGE_UPLOAD.md](IMAGE_UPLOAD.md) for storage behavior and
[OPERATIONS.md](OPERATIONS.md) for concurrency, persistence and deployment limits.


## Project map

| Path | Responsibility |
| --- | --- |
| `main.py` | FastAPI routes and processing orchestration |
| `image_processor.py`, `image_upload.py` | Quality checks, preprocessing and bounded input handling |
| `ocr_service.py`, `recognition_config.py` | Local TrOCR inference and configuration |
| `trocr_service.py` | External provider adapter and public option resolution |
| `parser.py`, `result_policy.py` | Field extraction and review policy |
| `product_catalog.py`, `drug_validator.py` | Product matching and legacy matching |
| `db.py` | Database access, pooling and persistence |
| `scripts/`, `migrations/` | Catalog import tooling and catalog schema |
| `tests/` | Regression tests |
