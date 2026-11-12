-- Run with the Supabase SQL editor after backing up the database.
-- Catalog snapshots are immutable; no existing lexicon IDs are replaced.
BEGIN;

CREATE TABLE public.drug_catalog_imports (
    import_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name text NOT NULL,
    source_sha256 text NOT NULL UNIQUE,
    imported_at timestamptz NOT NULL DEFAULT now(),
    normalization_version integer NOT NULL DEFAULT 1,
    validation_summary jsonb NOT NULL
);

CREATE TABLE public.drug_products (
    product_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    import_id uuid NOT NULL REFERENCES public.drug_catalog_imports(import_id),
    source_drug_id text NOT NULL,
    brand_name text,
    generic_name text,
    dosage_strength text,
    dosage_form text,
    routes text[] NOT NULL DEFAULT '{}',
    therapeutic_class text,
    rx_status text,
    ph_fda_number text,
    is_veterinary boolean NOT NULL DEFAULT false,
    quality_flags text[] NOT NULL DEFAULT '{}',
    raw_record jsonb NOT NULL,
    UNIQUE (import_id, source_drug_id)
);

CREATE TABLE public.drug_product_aliases (
    product_id uuid NOT NULL REFERENCES public.drug_products(product_id),
    normalized_alias text NOT NULL CHECK (btrim(normalized_alias) <> ''),
    PRIMARY KEY (product_id, normalized_alias)
);
CREATE INDEX drug_product_alias_lookup ON public.drug_product_aliases(normalized_alias);
CREATE INDEX drug_products_import_lookup ON public.drug_products(import_id);

ALTER TABLE public.drug_match_results
    ADD COLUMN product_id uuid REFERENCES public.drug_products(product_id);
ALTER TABLE public.prescription_line_items
    ADD COLUMN product_id uuid REFERENCES public.drug_products(product_id);

-- Catalog access is through the server's database connection.
ALTER TABLE public.drug_catalog_imports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.drug_products ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.drug_product_aliases ENABLE ROW LEVEL SECURITY;

COMMIT;
