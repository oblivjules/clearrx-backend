from unittest.mock import Mock

import main


def test_partial_result_save_cannot_report_success(monkeypatch):
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_persist_drug_validation', lambda **kwargs: None)
    def fail(**kwargs):
        raise RuntimeError('simulated write failure')
    monkeypatch.setattr(main, '_db_persist_drug_matches', fail)
    monkeypatch.setattr(main, '_db_persist_structured_prescription', lambda **kwargs: 'prescription')
    medicine = {'drug_name': 'Example', 'uncertain': False, '_drug_validation': {
        'matched': True, 'corrected': 'Example', 'confidence': 1.0}}
    saved = main._persist_modules_3_4(image_id='image', output_id='output',
        field_ids={0: 'field'}, medicines=[medicine], raw_medicines=[medicine], ocr_confidence=0.5)
    assert saved is False


def test_field_rows_exclude_review_and_inference_metadata(monkeypatch):
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_db_get_or_create_trocr_model', lambda **kwargs: 'model')
    monkeypatch.setattr(main, '_db_create_inference_job', lambda **kwargs: 'job')
    monkeypatch.setattr(main, '_db_finish_inference_job', lambda **kwargs: None)
    create = Mock(return_value=('output', {0: 'field'}))
    monkeypatch.setattr(main, '_db_create_transcription_output_with_fields', create)
    medicine = dict(drug_name='Example', uncertain=True, uncertain_fields=['drug_name'],
                    review_reasons=['cutoff_text'], route_source='inferred')
    main._persist_module_2(image_id='image', preprocessing_id='pre', provider='claude-vision',
        model='test-model', raw_text='Example', confidence=0.5, image_shape=(8, 8, 3),
        medicines=[medicine], raw_medicines=[medicine], inference_time_ms=10, failure_reason=None)
    rows = create.call_args.kwargs['fields']
    assert len(rows) == 1 and rows[0][0] == 'drug_name'


def test_database_lock_is_transaction_scoped_and_released():
    from db import DatabaseService
    service = DatabaseService()
    cursor = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=False)
    cursor.fetchone.return_value = (True,)
    conn = Mock(closed=0)
    conn.cursor.return_value = cursor
    pool = Mock()
    pool.getconn.return_value = conn
    service._pool = pool
    token = service.acquire_processing_lock('image')
    assert token is conn
    assert 'pg_try_advisory_xact_lock' in cursor.execute.call_args.args[0]
    conn.commit.assert_not_called()
    service.release_processing_lock(conn, 'image')
    conn.rollback.assert_called_once()
    pool.putconn.assert_called_once_with(conn, close=False)
