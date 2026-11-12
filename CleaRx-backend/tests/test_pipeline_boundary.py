import asyncio
from io import BytesIO

import numpy as np
from PIL import Image
from starlette.datastructures import Headers, UploadFile
from starlette.requests import Request

import main


def test_process_returns_unknown_ocr_confidence_and_cutoff_review(monkeypatch):
    monkeypatch.setattr(main, '_DB_AVAILABLE', False)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for method, result in [('check_lighting', 'good'), ('check_lighting_cnr', 'good'),
                           ('check_blur', 'sharp'), ('perspective_correct', (image, False)),
                           ('remove_shadows', image), ('enhance_for_handwriting', (image, 80, 1)),
                           ('create_document_preview', image)]:
        monkeypatch.setattr(main._img_proc, method, lambda *args, result=result, **kwargs: result)
    monkeypatch.setattr(main, 'run_trocr_ocr', lambda *args, **kwargs:
        ('Amoxicillin 500mg tablet\nTake 1 tablet twice daily [CUTOFF]', None))
    stream = BytesIO()
    Image.new('RGB', (8, 8)).save(stream, 'PNG')
    stream.seek(0)
    upload = UploadFile(stream, filename='test.png', headers=Headers({'content-type': 'image/png'}))
    request = Request({'type': 'http', 'headers': []})
    result = asyncio.run(main.process_image(request, upload, None, None, main.PRIMARY_OPTION))
    assert result['status'] == 'success'
    assert result['ocr_confidence'] is None
    assert result['ocr_confidence_basis'] == 'unavailable'
    assert result['trocr_confidence'] == 0
    assert result['persistence_status'] == 'disabled'
    assert result['review_required']
    assert result['medicines'][0]['review_required']
    assert 'cutoff_text' in result['medicines'][0]['review_reasons']
    assert all(not name.startswith('_') for name in result['medicines'][0])
    assert result['stage_timings_ms']['total_ms'] >= result['stage_timings_ms']['ocr_ms']


def test_inflight_image_request_is_rejected_before_ocr(monkeypatch):
    from db import _db_service
    from fastapi import HTTPException
    import pytest
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(_db_service, 'acquire_processing_lock', lambda image_id: None)
    request = Request({'type': 'http', 'headers': []})
    with pytest.raises(HTTPException) as error:
        asyncio.run(main.process_image(request, None, '00000000-0000-0000-0000-000000000001',
                                       None, main.PRIMARY_OPTION))
    assert error.value.status_code == 409


def test_processing_lock_released_on_pipeline_exception(monkeypatch):
    from db import _db_service
    from unittest.mock import Mock
    import pytest
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    token = object()
    release = Mock()
    monkeypatch.setattr(_db_service, 'acquire_processing_lock', lambda image_id: token)
    monkeypatch.setattr(_db_service, 'release_processing_lock', release)
    async def fail(*args, **kwargs):
        raise RuntimeError('pipeline failed')
    monkeypatch.setattr(main, '_process_image', fail)
    request = Request({'type': 'http', 'headers': []})
    with pytest.raises(RuntimeError):
        asyncio.run(main.process_image(request, None, '00000000-0000-0000-0000-000000000001',
                                       None, main.PRIMARY_OPTION))
    release.assert_called_once()
