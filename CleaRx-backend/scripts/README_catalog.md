# Versioned drug catalog imports — Priority #5

Run commands from the backend repository root. Install
`CleaRx-backend/requirements-dev.txt` for tests.

## Validate a source file

```powershell
python CleaRx-backend/scripts/import_drug_catalog.py 'C:\path\to\complete_fda_drugs.csv'
```

This is a dry run: it neither loads `.env` nor connects to a database.
Use `--report path/to/report.json` to save a fresh report. Required CSV headers,
in this order, are:

```text
drug_id,brand_name,generic_name,dosage_strength,dosage_form,route_of_administration,therapeutic_class,rx_status,ph_fda_number,common_ocr_variants
```

Resolve validation errors before importing. Missing values remain unknown.
Veterinary flags are heuristic and require review. Aliases may identify multiple
products; this is not inherently an import error. Normalization preserves
ingredient order and combination separators; order-independent combination
matching is not implemented by this importer.

## Schema and explicit import

Back up the intended database and review
`CleaRx-backend/migrations/001_drug_catalog.sql` against its existing schema.
It expects `public.drug_match_results` and `public.prescription_line_items`.
Apply this numbered migration once to a disposable database first. It adds
catalog snapshot, product and alias tables, indexes, row-level security, and
nullable product references. It does not replace legacy lexicon IDs or history.
RLS has no client policies; import through the backend database role.

If the catalog tables already exist, do not rerun this CREATE/ALTER migration.
Compare existing constraints, indexes, RLS and product-reference columns with
the migration before recording it as an existing schema. A product-column
export alone does not establish all those properties.

```powershell
python CleaRx-backend/scripts/import_drug_catalog.py 'C:\path\to\complete_fda_drugs.csv' --apply
```

Only `--apply` loads `CleaRx-backend/.env`. An existing `DATABASE_URL` takes
precedence. Confirm the intended database privately before running it.
Each import uses one transaction. An error rolls back the snapshot, products
and aliases together. Reimporting the same file hash is a no-op. A different
file creates a new snapshot without deleting earlier imports. Snapshots are
append-only by importer behavior; the schema does not prohibit privileged updates.

Verify actual records rather than relying on a saved report:

```sql
SELECT i.import_id, i.source_name, i.source_sha256,
       i.validation_summary->>'record_count' AS reported_records,
       count(p.product_id) AS actual_products
FROM public.drug_catalog_imports i
LEFT JOIN public.drug_products p USING (import_id)
GROUP BY i.import_id;

SELECT p.import_id, count(*) AS actual_aliases
FROM public.drug_product_aliases a
JOIN public.drug_products p USING (product_id)
GROUP BY p.import_id;
```

## Verification status for B2

`catalog_validation_report.json` is preserved from the refactored implementation.
It reports 49,943 records for `complete_fda_drugs.csv`, with SHA-256
`41a0c0d06d2d27757066c992857410ab8568063e3ac9efb0f9189cc59d002202`.
Neither that source CSV nor database configuration was available in the three
workspaces or Downloads during B2. The full count has not been independently
verified. The report does not establish current registration validity or clinical
correctness. No migration or import has been executed against a real database
during this migration work.

The subsequently supplied column export matches the migration's 15 product
columns, nullability and defaults. Array metadata says ARRAY; element types,
constraints, indexes and RLS still need database inspection. The supplied content
export has 100 distinct product/source IDs under import
`f669f862-76ff-4ee2-878a-746cf57d4e76`, valid UUIDs and source IDs matching their
raw records. All 100 have brand names; none has a generic name, strength, form,
route, therapeutic class, prescription status or registration number. Missing-field
flags match those omissions. This corroborates populated catalog storage in the
supplied export, not the full 49,943-record count or alias/import table contents.
It is a sample, not evidence that all catalog rows lack those fields. The sample
export is not copied into the repository.

Additional evidence from locally available remote-tracking branches:

- `origin/refactor/improved-lexicon` at `381a0d1` includes `f4a91e1`
  (versioned storage/imports) and `29eb7a7` (matching/formulation validation).
- `origin/refactor/priority-5/optimize` at `03c158a` preserves that architecture
  and adds bounded caching, more candidate names and conflict propagation.
- Migration and importer blobs are identical on both branches and the inspected
  refactored checkout. Neither reviewed branch tracks the source CSV.
- Matching searches main brand/generic names; stored OCR aliases are not searched.
  Ingredient-order canonicalization remains absent. These are specific matching
  limitations, not an absence of updated catalog infrastructure.

The review did not fetch, switch branches or modify the refactored repository.

Local regression tests cover validation and mocked transaction success,
duplicate import and rollback behavior. PostgreSQL execution, foreign keys,
RLS and a full-size source import still require disposable-database verification.
No accuracy benchmarks or calibration reports were added.

B2 prepares import infrastructure; B3 integrates the running backend as below.
Rerunning the migration is not part of that integration.

## Backend integration — B3

Startup loads human product records from `drug_products`. Set
`DRUG_CATALOG_IMPORT_ID` to pin an import UUID; otherwise it selects the newest
snapshot by import timestamp. For the supplied export, the import UUID is
`f669f862-76ff-4ee2-878a-746cf57d4e76`. Restart after changing this setting or
importing a catalog. Startup logs the loaded snapshot ID. If catalog loading
fails, legacy matching remains available without product field checks.

Matching uses exact main brand/generic names before fuzzy candidates, caches
lookups, excludes veterinary records and checks observed strength, form and
route against candidate formulations. Unknown values remain unknown. Inferred
oral routes are identified as inferred and excluded from observed route evidence.
Neither catalog data nor missing values supply prescription instructions.

A matched name does not guarantee a resolved formulation. `product_id` is
nullable and populated only for a uniquely resolved formulation with matching
strength and no conflicts. Both drug-match and prescription-line records retain
this reference; existing legacy lexicon IDs remain separate. Per-medicine API
metadata includes `catalog_import_id`, `name_similarity`, `name_matched`,
`field_validation`, `formulation_status`, `review_required` and review reasons.
Internal candidate/matching helpers are removed from public medicine objects.
Unmatched OCR names remain visible for review rather than being discarded.

Known limits remain: main-name matching does not use stored OCR aliases or
canonicalize ingredient order, complex combination strengths are unknown, and
unresolved matches do not have a dedicated persisted catalog-import reference.
No recognition accuracy or clinical validation is implied by regression tests.
