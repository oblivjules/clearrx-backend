from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(main, '_DB_AVAILABLE', False)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for name, result in [('check_lighting', 'good'), ('check_lighting_cnr', 'good'),
                         ('check_blur', 'sharp'), ('perspective_correct', (image, False)),
                         ('enhance_for_handwriting', (image, 80, 1))]:
        monkeypatch.setattr(main._img_proc, name, lambda *args, result=result: result)
    stream = BytesIO()
    Image.new('RGB', (8, 8)).save(stream, 'PNG')
    files = {'file': ('test.png', stream.getvalue(), 'image/png')}
    client = TestClient(main.app)
    yield client, files
    client.close()


@pytest.mark.parametrize('option,engine,label', [
    ('clearx-trocr', 'claude-sonnet-4-6', 'CleaRx TrOCR'),
    ('claude-sonnet-4-6', 'claude-sonnet-4-6', 'CleaRx TrOCR'),
    ('claude-sonnet-5-5', 'claude-sonnet-5-5', 'Claude Sonnet 5.5'),
])
def test_sources_are_public_but_logs_record_actual_provider(monkeypatch, pipeline, caplog, option, engine, label):
    recognize = Mock(return_value=('Amoxicillin 500mg tablet\nTake 1 tablet twice daily [CUTOFF]', None))
    monkeypatch.setattr(main, 'run_trocr_ocr', recognize)
    caplog.set_level('INFO', logger='main')
    client, files = pipeline
    response = client.post('/process-image', files=files, data={'model': option})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['status'] == 'success'
    assert body['ocr_source'] == label
    assert body['ocr_confidence'] is None
    assert body['ocr_confidence_basis'] == 'unavailable'
    assert body['trocr_confidence'] == 0
    assert body['review_required']
    assert 'cutoff_text' in body['medicines'][0]['review_reasons']
    assert recognize.call_args.kwargs['model'] == engine
    assert 'provider=anthropic' in caplog.text and engine in caplog.text
    assert 'claude-sonnet-4-6' not in response.text


def test_unknown_option_fails_before_provider(monkeypatch, pipeline):
    recognize = Mock()
    monkeypatch.setattr(main, 'run_trocr_ocr', recognize)
    client, files = pipeline
    response = client.post('/process-image', files=files, data={'model': 'private-invalid-model'})
    assert response.status_code == 422
    assert response.json()['detail'] == 'Unsupported recognition option'
    recognize.assert_not_called()


@pytest.mark.parametrize('fails', [False, True])
def test_provider_outage_uses_genuine_local_entrypoint(monkeypatch, pipeline, fails):
    monkeypatch.setattr(main, 'run_trocr_ocr', Mock(return_value=None))
    local = Mock(side_effect=RuntimeError('private model details')) if fails else Mock(
        return_value=('Amoxicillin 500mg tablet\nTake 1 tablet twice daily', .8))
    monkeypatch.setitem(sys.modules, 'ocr_service', SimpleNamespace(run_trocr=local))
    client, files = pipeline
    response = client.post('/process-image', files=files)
    assert response.status_code == 200
    body = response.json()
    local.assert_called_once()
    assert body['ocr_source'] == ('Unavailable' if fails else 'Local TrOCR')
    assert body['status'] == ('rejected' if fails else 'success')
    assert body['ocr_confidence'] == (None if fails else .8)
    assert 'private model details' not in response.text


@pytest.mark.parametrize('provider,model,base', [
    ('claude-vision', 'claude-sonnet-4-6', 'n/a'),
    ('claude-vision', 'claude-sonnet-5-5', 'n/a'),
    ('trocr-finetuned', 'local-trocr', 'microsoft/trocr-base-handwritten'),
])
def test_registry_records_actual_engine_without_fabricated_training(monkeypatch, provider, model, base):
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    registry = Mock(return_value='model-1')
    monkeypatch.setattr(main, '_db_get_or_create_trocr_model', registry)
    monkeypatch.setattr(main, '_db_create_inference_job', Mock(return_value='job-1'))
    monkeypatch.setattr(main, '_db_finish_inference_job', Mock())
    monkeypatch.setattr(main, '_db_create_transcription_output_with_fields', Mock(return_value=('out-1', {})))
    main._persist_module_2(image_id='image-1', preprocessing_id='preprocess-1', provider=provider,
        model=model, raw_text='Example', confidence=.5, image_shape=(8, 8, 3),
        medicines=[], inference_time_ms=10, failure_reason=None)
    assert registry.call_args.kwargs['model_name'] == model
    assert registry.call_args.kwargs['base_model'] == base
    if provider == 'claude-vision':
        assert registry.call_args.kwargs['fine_tuned_on'] == 'n/a'
        assert registry.call_args.kwargs['weights_path'] == 'n/a'
