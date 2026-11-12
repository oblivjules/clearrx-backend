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
- Commit: `82ad417` — `test: establish isolated backend regression setup`.

## B2 — Versioned product catalog imports

- Priority: #5 (updated catalog infrastructure and provenance).
- Added the catalog migration, CSV validator/importer, source report, operational
  instructions and importer regression tests. Existing application code is unchanged.
- Retained snapshot hashing, shared aliases, veterinary flags, raw source records,
  transaction boundaries and nullable product references from the refactored code.
- Added an import-entry validation guard so direct calls cannot apply a report
  containing validation errors.
- The preserved report lists 49,943 source records. The CSV and database
  configuration were not found in the three workspaces or Downloads; actual
  source/database counts are unverified. No real database changes were made.
- Subsequent user-supplied evidence: product schema matches all 15 migration
  columns; 100 distinct exported rows share import
  `f669f862-76ff-4ee2-878a-746cf57d4e76`. They corroborate populated catalog
  storage but not the total count, constraints, RLS or alias-table contents.
  All sampled rows are brand-only with missing formulation fields appropriately
  flagged; the sample is not representative evidence for the whole catalog.
- Reviewed local remote-tracking branches improved-lexicon (`381a0d1`) and
  priority-5/optimize (`03c158a`): catalog/import work originates in `f4a91e1`,
  matching integration in `29eb7a7`. Both preserve identical schema/importer
  blobs. Optimization adds caching and conflict handling; aliases remain unused
  for matching and combination ingredient order is not canonicalized.
- Ingredient-order normalization is not part of the importer. Matching integration
  remains B3; this limitation does not negate the catalog infrastructure changes.
- Verification: `python -m pytest tests -q` — 26 passed, 2 skipped
  (opt-in capture database tests). Import tests cover CSV validation, dry run,
  mocked transaction success/error handling and duplicate-source no-op behavior.
  `git diff --check` passed. No PostgreSQL migration or real import was run.
- Commit: `9918b46` — `feat(catalog): add versioned product catalog imports`.

## B3 — Product matching and formulation checks

- Priorities: #5 catalog integration, supporting #2 field reliability and #4
  review metadata. B2 commit: `9918b46`.
- Migrated product matching/cache, startup snapshot selection, brand/generic
  indexes and parser integration. Catalog failures retain legacy name matching.
- Propagated catalog field conflicts, uncertain fields, name similarity and
  nullable resolved-product references to API/persistence. Reuse the same match
  result for response and storage; keep metadata out of transcribed-field rows.
- Preserved numeric brands and separate unmatched medicines with their own sig.
  Retained unmatched OCR names for review and removed private helper fields from
  public medicine objects.
- Moved route provenance into B3 as a dependency of formulation checks: inferred
  routes must not be treated as observed catalog evidence. Remaining parser
  dose/uncertainty changes stay in B4. Pooling and persistence outcomes stay in B8.
- Known matching limits: unused OCR aliases, no ingredient-order canonicalization,
  no complex combination-strength parsing. Catalog counts and live database
  compatibility remain unverified beyond the supplied schema/sample evidence.
- Verification: `python -m pytest tests -q` — 48 passed, 2 skipped
  (opt-in database capture tests). Includes product matching, parser integration,
  pinned/latest snapshot loading, startup fallback, API conflict metadata and
  mocked nullable-product persistence. Backend import and `git diff --check`
  passed. No real database/model calls, schema changes or accuracy benchmarks.
- Commit: `ef3d658` — `feat(catalog): integrate product matching and formulation checks`.

## B4 — Critical-field uncertainty and dose provenance

- Priorities: #2 recognition/critical-field reliability and #4 review guidance.
- Migrated rational fractional/decimal dose arithmetic, invalid-dose review flags,
  stated-duration precedence and restrictions on quantity-derived durations.
  Variable schedules and drop quantities are not extrapolated into durations.
- Preserve uncertainty markers through extraction, normalize marker case/spacing,
  and flag cutoff text even when truncation hides the exact missing field.
- Any uncertain field now marks the medicine for review while preserving field
  scope. Retained the existing parser-test name; only its changed review-policy
  assertion was migrated. Route provenance and shared result policy landed in B3.
- Added parser/review safety regressions and mocked persistence checks. Upload
  limits and runtime policy tests remain in their later migration steps.
- Verification: `python -m pytest tests -q` — 67 passed, 2 skipped
  (opt-in capture database tests). Covers fractional/decimal/invalid doses,
  stated-duration precedence, variable schedules, cutoff/uncertainty markers
  and review flags reaching mocked persistence. `git diff --check` passed.
  No real database/model calls or accuracy benchmarks were performed.
- Commit: `9eacb0c` — `fix(parser): preserve critical-field uncertainty and dose provenance`.

## B5 — Shared image validation and persistent uploads

- Priority: #3 capture/upload alternatives and shared quality acceptance.
- Migrated bounded JPEG/PNG/WebP decoding, upload/crop-validation/file-retrieval
  endpoints, camera file storage and database file-path access. Added the default
  storage directory to the repository-root ignore file.
- Extracted one camera/upload quality policy. As approved, framing stays advisory
  for both sources; the refactored upload-only framing gate was not migrated.
  Lighting, blur, readability and handwriting thresholds remain unchanged.
- Retained attempt/metric recording for registered uploads, including rejected
  attempts. Crop checks have no storage or database side effects. Registration
  failures remove the newly written file; retrieval enforces storage containment.
- Added IMAGE_UPLOAD.md documenting endpoints and real retention limitations:
  no automatic deletion/masking, no ownership authentication, and no atomic disk/
  database transaction. Production deployment and privacy notice work are not
  claimed complete. No new SQL migration or database writes were performed.
- Verification: `python -m pytest tests -q` — 98 passed, 2 skipped
  (opt-in capture database tests). Includes camera/upload parity under controlled
  conditions and real checks on generated images, all six image endpoint byte
  limits, supported formats, MIME/pixel limits, storage cleanup and retrieval
  containment. `git diff --check` passed. No real database or OCR calls were made.
- Commit: pending explicit user approval.

Run from CleaRx-backend with `python -m pytest tests -q`. Without explicit test
database configuration, capture integration tests skip. Never configure these
write tests against an application or production database.
