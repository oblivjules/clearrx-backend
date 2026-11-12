"""Shared output fields and review policy; similarity is not OCR confidence."""

PRESCRIPTION_FIELDS = (
    'drug_name', 'brand_name', 'strength', 'dosage_form', 'dose_amount',
    'frequency', 'route_of_administration', 'quantity', 'duration', 'timing',
    'meal_relation', 'administration_instructions',
)


def extracted_fields(medicine):
    # Explicit schema avoids persisting booleans, lists, and diagnostics as text.
    return [(name, medicine[name]) for name in PRESCRIPTION_FIELDS
            if medicine.get(name) not in (None, '')]


def apply_match_review(medicine, result):
    medicine['name_matched'] = bool(result.get('matched'))
    medicine['name_similarity'] = result.get('raw_score')
    medicine['product_id'] = result.get('product_id')
    medicine['catalog_import_id'] = result.get('catalog_import_id')
    medicine['field_validation'] = result.get('field_validation') or {}
    medicine['formulation_status'] = result.get('formulation_status', 'unresolved')
    reasons = set(medicine.get('review_reasons') or [])
    reasons.update(result.get('review_reasons') or [])
    fields = set(medicine.get('uncertain_fields') or [])
    fields.update(f for f, status in medicine['field_validation'].items() if status == 'conflict')
    if not result.get('matched'):
        reasons.add('unmatched_drug_name')
        fields.add('drug_name')
    medicine['review_reasons'] = sorted(reasons)
    medicine['uncertain_fields'] = sorted(fields)
    medicine['uncertain'] = bool(medicine.get('uncertain') or reasons or fields)
    medicine['review_required'] = medicine['uncertain']


def public_medicine(medicine):
    return {name: value for name, value in medicine.items() if not name.startswith('_')}
