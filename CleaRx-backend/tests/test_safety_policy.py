import pytest
from unittest.mock import Mock

import main
from parser import parse_prescription
from result_policy import extracted_fields, apply_match_review, public_medicine


def test_fractional_dose_duration():
    medicine = parse_prescription('Amoxicillin 500mg tablet #10\nTake 1/2 tablet twice daily')[0]
    assert medicine['duration'] == '10 days'
    assert medicine['duration_source'] == 'derived'


def test_decimal_dose_is_not_misread_as_five_tablets():
    medicine = parse_prescription('Amoxicillin 500mg tablet #10\nTake 0.5 tablet twice daily')[0]
    assert medicine['dose_amount'] == '0.5 tablet'
    assert medicine['duration'] == '10 days'


@pytest.mark.parametrize('dose', ['0', '1/0'])
def test_invalid_doses_are_reviewed_without_duration(dose):
    medicine = parse_prescription(f'Amoxicillin 500mg tablet #10\nTake {dose} tablet twice daily')[0]
    assert medicine['duration'] is None
    assert 'invalid_dose_amount' in medicine['review_reasons']
    assert 'dose_amount' in medicine['uncertain_fields']


def test_stated_duration_is_not_replaced_by_quantity_arithmetic():
    medicine = parse_prescription('Amoxicillin 500mg tablet #10\nTake 1 tablet twice daily for 3 days')[0]
    assert medicine['duration'] == '3 days'
    assert medicine['duration_source'] == 'stated'


def test_uncertain_drug_name_retains_a_field_flag():
    medicine = parse_prescription('Amoxicillin [UNCERTAIN] 500mg tablet')[0]
    assert 'drug_name' in medicine['uncertain_fields']


@pytest.mark.parametrize('schedule', ['as needed', 'PRN', 'then taper'])
def test_no_derived_duration_for_variable_schedule(schedule):
    medicine = parse_prescription(f'Amoxicillin 500mg tablet #10\nTake 1 tablet twice daily {schedule}')[0]
    assert medicine['duration'] is None


def test_drops_do_not_derive_days_from_ambiguous_quantity():
    medicine = parse_prescription('Amoxicillin 500mg drops #10\nUse 2 drops twice daily')[0]
    assert medicine['duration'] is None


@pytest.mark.parametrize('marker', ['[CUTOFF]', '[cutoff]', '[ Cutoff ]'])
def test_cutoff_survives_as_review_metadata(marker):
    medicine = parse_prescription(f'Amoxicillin 500mg tablet\nTake 1 tablet twice daily {marker}')[0]
    assert medicine['uncertain']
    assert 'cutoff_text' in medicine['review_reasons']
    assert 'administration_instructions' in medicine['uncertain_fields']


def test_unmatched_name_is_flagged_even_when_ocr_text_is_preserved():
    medicine = {'drug_name': 'Unknown name', 'uncertain_fields': []}
    apply_match_review(medicine, {'matched': False, 'review_reasons': []})
    assert medicine['review_required']
    assert 'drug_name' in medicine['uncertain_fields']
    assert not medicine['name_matched']


def test_field_conflict_does_not_claim_name_is_wrong():
    medicine = {'drug_name': 'Example'}
    apply_match_review(medicine, {'matched': True, 'raw_score': 100,
        'field_validation': {'strength': 'conflict'}, 'review_reasons': ['strength_conflict']})
    assert medicine['name_similarity'] == 100
    assert medicine['review_required']
    assert medicine['uncertain_fields'] == ['strength']


def test_metadata_is_not_persisted_as_a_prescription_field():
    medicine = {'drug_name': 'Example', 'review_reasons': ['cutoff_text'],
                'uncertain': True, '_drug_validation': {}, 'route_source': 'inferred'}
    assert extracted_fields(medicine) == [('drug_name', 'Example')]
    assert '_drug_validation' not in public_medicine(medicine)


@pytest.mark.parametrize('text, field', [
    ('Amoxicillin 500mg tablet #10\nTake 0 tablet twice daily', 'dose_amount'),
    ('Amoxicillin [ uncertain ] 500mg tablet', 'drug_name'),
    ('Amoxicillin 500mg tablet\nTake 1 tablet twice daily [ Cutoff ]', 'administration_instructions'),
])
def test_parser_review_flags_reach_persistence(monkeypatch, text, field):
    medicine = parse_prescription(text, ocr_confidence=.95)[0]
    match = dict(matched=True, corrected='Amoxicillin', confidence=1., raw_score=100,
                 review_reasons=[], field_validation={})
    apply_match_review(medicine, match)
    medicine['_drug_validation'] = match
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_persist_drug_validation', Mock())
    matches, save = Mock(), Mock(return_value='prescription-1')
    monkeypatch.setattr(main, '_db_persist_drug_matches', matches)
    monkeypatch.setattr(main, '_db_persist_structured_prescription', save)
    main._persist_modules_3_4(image_id='image-1', output_id='output-1', field_ids={},
        medicines=[medicine], raw_medicines=[medicine], ocr_confidence=.95)
    assert medicine['review_required']
    assert field in medicine['uncertain_fields']
    item = save.call_args.kwargs['line_items'][0]
    assert item['line_item_status'] == main.LINE_LOW_CONFIDENCE
    scores = {entry['field_name']: entry for entry in item['confidence_fields']}
    # A cutoff may omit the field entirely; do not invent a value to persist it.
    if field in scores:
        assert scores[field]['is_flagged']
        assert scores[field]['composite_score'] <= main.UNCERTAIN_FIELD_SCORE_CAP
    assert 'review_required' not in dict(extracted_fields(medicine))
