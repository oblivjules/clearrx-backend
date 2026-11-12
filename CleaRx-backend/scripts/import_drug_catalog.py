"""Validate locally by default. --apply imports an immutable snapshot in one transaction."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import unicodedata

FIELDS = ('drug_id brand_name generic_name dosage_strength dosage_form '
          'route_of_administration therapeutic_class rx_status ph_fda_number '
          'common_ocr_variants').split()


def clean(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', html.unescape(value)).split())


def normalize(value: str) -> str:
    # Preserve digits, combination separators, and release suffixes.
    return clean(value).casefold()


def placeholder(value: str) -> bool:
    return 'see reverse' in normalize(value)


def read_catalog(path: Path):
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
    if reader.fieldnames != FIELDS:
        raise ValueError('CSV headers must match the documented catalog schema exactly')
    products, seen = [], set()
    populated, flag_counts = collections.Counter(), collections.Counter()
    alias_products = collections.defaultdict(set)
    brands = collections.Counter()
    errors = []
    for line, raw in enumerate(reader, 2):
        if None in raw or any(v is None for v in raw.values()):
            errors.append(f'Line {line}: malformed column count')
            continue
        row = {k: clean(v) for k, v in raw.items()}
        source_id = row['drug_id']
        if not source_id or source_id in seen:
            errors.append(f'Line {line}: missing or duplicate drug_id {source_id!r}')
        seen.add(source_id)
        if not row['brand_name'] and not row['generic_name']:
            errors.append(f'Line {line}: both names missing')
        populated.update(k for k, v in row.items() if v)
        flags = []
        if placeholder(row['generic_name']):
            flags.append('placeholder_generic')
        for key in ('generic_name', 'dosage_strength', 'dosage_form',
                    'route_of_administration', 'ph_fda_number'):
            if not row[key]:
                flags.append('missing_' + key)
        vet = bool(re.search(r'\(vet\b', row['generic_name'], re.I)
                   or row['ph_fda_number'].upper().startswith('VR'))
        if vet:
            flags.append('veterinary_indicator')
        aliases = {normalize(a) for a in [row['brand_name'], row['generic_name'],
                   *row['common_ocr_variants'].split(';')]
                   if a.strip() and not placeholder(a)}
        for alias in aliases:
            alias_products[alias].add(source_id)
        if row['brand_name']:
            brands[normalize(row['brand_name'])] += 1
        flag_counts.update(flags)
        products.append((row, raw, flags, vet, sorted(aliases)))
    summary = {
        'source_name': path.name,
        'source_sha256': hashlib.sha256(payload).hexdigest(),
        'normalization_version': 1,
        'record_count': len(products),
        'populated_fields': dict(populated),
        'quality_flags': dict(flag_counts),
        'unique_aliases': len(alias_products),
        'aliases_shared_by_multiple_records': sum(len(v) > 1 for v in alias_products.values()),
        'brand_names_with_multiple_records': sum(n > 1 for n in brands.values()),
        'errors': errors,
        'notes': ['Shared aliases are candidate sets, not necessarily data errors.',
                  'Veterinary indicators are heuristic; records are retained and flagged.',
                  'No source registration validity or clinical correctness is verified.'],
    }
    return products, summary


def apply_import(products, summary):
    if summary['errors']:
        raise ValueError('Validation failed; nothing imported.')
    import psycopg2
    from psycopg2.extras import Json, execute_values
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError('DATABASE_URL is required for --apply')
    with psycopg2.connect(database_url, sslmode='require') as conn:
        with conn.cursor() as cur:
            cur.execute('''INSERT INTO public.drug_catalog_imports
                (source_name, source_sha256, validation_summary)
                VALUES (%s, %s, %s) ON CONFLICT (source_sha256) DO NOTHING
                RETURNING import_id''', (summary['source_name'], summary['source_sha256'], Json(summary)))
            result = cur.fetchone()
            if result is None:
                return 'Identical source already imported; no changes made.'
            import_id = result[0]
            values = []
            for row, raw, flags, vet, aliases in products:
                values.append((import_id, row['drug_id'], row['brand_name'] or None,
                    None if placeholder(row['generic_name']) else row['generic_name'] or None,
                    row['dosage_strength'] or None, row['dosage_form'] or None,
                    [clean(r) for r in row['route_of_administration'].split(';') if r.strip()],
                    row['therapeutic_class'] or None, row['rx_status'] or None,
                    row['ph_fda_number'] or None, vet, flags, Json(raw)))
            execute_values(cur, '''INSERT INTO public.drug_products
                (import_id, source_drug_id, brand_name, generic_name, dosage_strength,
                 dosage_form, routes, therapeutic_class, rx_status, ph_fda_number,
                 is_veterinary, quality_flags, raw_record) VALUES %s''', values, page_size=1000)
            cur.execute('SELECT source_drug_id, product_id FROM public.drug_products WHERE import_id = %s', (import_id,))
            ids = dict(cur.fetchall())
            aliases = [(ids[row['drug_id']], alias)
                       for row, _, _, _, names in products for alias in names]
            execute_values(cur, '''INSERT INTO public.drug_product_aliases
                (product_id, normalized_alias) VALUES %s''', aliases, page_size=2000)
    return f'Imported {len(products)} records. Import ID: {import_id}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    products, summary = read_catalog(args.csv_path)
    output = json.dumps(summary, indent=2)
    if args.report:
        args.report.write_text(output + '\n', encoding='utf-8')
    print(output)
    if summary['errors']:
        raise SystemExit('Validation failed; nothing imported.')
    if args.apply:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
        print(apply_import(products, summary))
    else:
        print('Dry run only. No database connection or writes performed.')


if __name__ == '__main__':
    main()
