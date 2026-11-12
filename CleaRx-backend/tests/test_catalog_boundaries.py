import asyncio
from io import BytesIO
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from PIL import Image
from starlette.datastructures import Headers, UploadFile
from starlette.requests import Request

import db
import main
from drug_validator import DrugValidator
from parser import parse_prescription, refresh_drug_catalog
from product_catalog import ProductCatalog
from result_policy import apply_match_review, extracted_fields


@pytest.fixture
def catalog_state():
    original = {name: value for name, value in vars(DrugValidator).items()
                if name.isupper() or name == '_product_catalog'}
    yield
    for name, value in original.items():
        setattr(DrugValidator, name, value)
    refresh_drug_catalog()


def product():
    return dict(product_id='product-1', brand_name='Example', generic_name='Amoxicillin',
                dosage_strength='500mg', dosage_form='Tablet', routes=['Oral'],
                is_veterinary=False)


@pytest.mark.parametrize('requested', [None, 'import-1'])
def test_loader_selects_latest_or_pinned_snapshot(monkeypatch, requested):
    if requested:
        monkeypatch.setenv('DRUG_CATALOG_IMPORT_ID', requested)
    else:
        monkeypatch.delenv('DRUG_CATALOG_IMPORT_ID', raising=False)
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = {'import_id': 'import-1'}
    cursor.fetchall.return_value = [product()]
    context = MagicMock()
    context.__enter__.return_value = connection
    service = db.DatabaseService()
    monkeypatch.setattr(service, '_get_conn', lambda: context)
    import_id, products = service.fetch_product_catalog()
    assert import_id == 'import-1'
    assert products == [product()]
    selection = cursor.execute.call_args_list[0]
    assert ('WHERE import_id = %s' in selection.args[0]) if requested else (
        'ORDER BY imported_at DESC' in selection.args[0])
    assert cursor.execute.call_args_list[1].args[1] == ('import-1',)
    assert 'AND NOT is_veterinary' in cursor.execute.call_args_list[1].args[0]


@pytest.mark.parametrize('available', [True, False])
def test_startup_uses_catalog_or_preserves_legacy_matching(monkeypatch, catalog_state, available):
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_LEXICON_ID_BY_NAME', {})
    monkeypatch.setattr(main, '_check_database_connection', lambda: None)
    monkeypatch.setattr(main, '_fetch_drug_catalog', lambda: [('legacy', 'LegacyMed', 'generic', False)])
    fetch = Mock(return_value=('import-1', [product()])) if available else Mock(side_effect=ValueError('no catalog'))
    monkeypatch.setattr(db._db_service, 'fetch_product_catalog', fetch)

    async def run():
        async with main.lifespan(main.app):
            if available:
                assert DrugValidator().validate_drug_name('Example')['catalog_import_id'] == 'import-1'
            else:
                assert DrugValidator._product_catalog is None
                assert DrugValidator().validate_drug_name('LegacyMed')['matched']
    asyncio.run(run())


def test_parser_preserves_separate_unmatched_medicine_and_route_provenance(catalog_state):
    DrugValidator.use_product_catalog(ProductCatalog([product()], 'import-1'))
    refresh_drug_catalog()
    medicines = parse_prescription('Example 500mg tablet\nTake 1 tablet twice daily\n'
                                   'Zzzmed 10mg tablet\nTake 2 tablets once daily')
    assert len(medicines) == 2
    assert medicines[0]['dose_amount'] == '1 tablet'
    assert medicines[1]['drug_name_raw'] == 'Zzzmed'
    assert medicines[1]['dose_amount'] == '2 tablet'
    assert medicines[0]['route_source'] == 'inferred'


def test_process_emits_catalog_conflicts_without_private_metadata(monkeypatch, catalog_state):
    DrugValidator.use_product_catalog(ProductCatalog([product()], 'import-1'))
    refresh_drug_catalog()
    monkeypatch.setattr(main, '_DB_AVAILABLE', False)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for name, result in [('check_lighting', 'good'), ('check_lighting_cnr', 'good'),
                         ('check_blur', 'sharp'), ('perspective_correct', (image, False)),
                         ('enhance_for_handwriting', (image, 80, 1))]:
        monkeypatch.setattr(main._img_proc, name, lambda *args, result=result: result)
    monkeypatch.setattr(main, 'run_trocr_ocr', lambda *args, **kwargs:
                        ('Example 750mg tablet\nTake 1 tablet twice daily', .95))
    stream = BytesIO()
    Image.new('RGB', (8, 8)).save(stream, 'PNG')
    stream.seek(0)
    result = asyncio.run(main.process_image(
        Request({'type': 'http', 'headers': []}),
        UploadFile(stream, filename='test.png', headers=Headers({'content-type': 'image/png'})),
        None, None, model=main.PRIMARY_OPTION))
    medicine = result['medicines'][0]
    assert result['status'] == 'success'
    assert result['review_required']
    assert medicine['name_matched']
    assert medicine['catalog_import_id'] == 'import-1'
    assert medicine['product_id'] is None
    assert medicine['field_validation']['strength'] == 'conflict'
    assert medicine['field_validation']['route_of_administration'] == 'unknown'
    assert 'strength' in medicine['uncertain_fields']
    assert all(not key.startswith('_') for key in medicine)


def test_persistence_reuses_match_and_keeps_product_references(monkeypatch):
    match = ProductCatalog([product()], 'import-1').match('Example', strength_value='500mg')
    medicine = dict(drug_name='Amoxicillin', brand_name='Example', strength='500mg',
                    _drug_name_raw='Example', _product_id='product-1', _drug_validation=match)
    apply_match_review(medicine, match)
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_persist_drug_validation', Mock())
    matches, prescriptions = Mock(), Mock(return_value='prescription-1')
    monkeypatch.setattr(main, '_db_persist_drug_matches', matches)
    monkeypatch.setattr(main, '_db_persist_structured_prescription', prescriptions)
    monkeypatch.setattr(main, 'validate_drug_name', Mock(side_effect=AssertionError('must reuse match')))
    main._persist_modules_3_4(image_id='image-1', output_id='output-1', field_ids={0: 'field-1'},
                             medicines=[medicine], raw_medicines=[medicine], ocr_confidence=.95)
    row = matches.call_args.kwargs['matches'][0]
    item = prescriptions.call_args.kwargs['line_items'][0]
    assert row['field_id'] == 'field-1'
    assert row['product_id'] == item['product_id'] == 'product-1'
    assert [field['field_name'] for field in item['confidence_fields']] == [
        name for name, _ in extracted_fields(medicine)]
    assert 'review_required' not in dict(extracted_fields(medicine))


@pytest.mark.parametrize('product_id', [None, 'product-1'])
def test_database_writes_nullable_product_references(monkeypatch, product_id):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = ('prescription-1',)
    insert = Mock(side_effect=[[('match-1', 'field-1')], None, [('line-1', 1)]])
    monkeypatch.setattr(db, 'execute_values', insert)
    service = db.DatabaseService(connection_factory=lambda: connection)
    service.persist_drug_matches(matches=[dict(
        raw_input='Example', field_id='field-1', is_flagged=False, product_id=product_id,
        original_token='Example', corrected_token='Amoxicillin', token_sort_ratio=None,
        partial_ratio=None, algorithm_used='exact', score_cutoff_used=None, correction_applied=True)])
    service.persist_structured_prescription(output_id='output-1', prescription_date=None,
        overall_status='extracted', line_items=[dict(item_order=1, line_item_status='matched', product_id=product_id)])
    assert insert.call_args_list[0].args[2][0][-1] == product_id
    assert insert.call_args_list[2].args[2][0][-1] == product_id
