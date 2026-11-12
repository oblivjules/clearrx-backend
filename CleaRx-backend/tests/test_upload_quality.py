from io import BytesIO
import asyncio
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main


@pytest.mark.parametrize('lighting,blur,score,framing,reason', [
    ('too_dark', 'sharp', 90, 'aligned', 'lighting'),
    ('too_bright', 'sharp', 90, 'aligned', 'lighting'),
    ('good', 'blurry', 90, 'aligned', 'blur'),
    ('good', 'sharp', 30, 'aligned', 'readability'),
    ('good', 'sharp', 90, 'not_found', None),
    ('good', 'sharp', 90, 'too_small', None),
    ('good', 'sharp', 90, 'aligned', None),
])
def test_camera_and_upload_share_acceptance_and_quality(
    monkeypatch, tmp_path, lighting, blur, score, framing, reason,
):
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_STORAGE_ROOT', tmp_path)
    monkeypatch.setattr(main, '_db_prescription_image_exists', lambda **kw: True)
    image_id = '00000000-0000-0000-0000-000000000001'
    register = Mock(return_value=(image_id, 2))
    metrics = Mock()
    monkeypatch.setattr(main, '_db_create_prescription_image', register)
    monkeypatch.setattr(main, '_db_log_validate_capture', metrics)
    quad = {
        'aligned': np.array([[2, 2], [29, 2], [29, 29], [2, 29]], dtype=np.float32),
        'too_small': np.array([[2, 2], [11, 2], [11, 11], [2, 11]], dtype=np.float32),
        'not_found': None,
    }[framing]
    for method, value in [
        ('check_lighting', lighting), ('check_lighting_cnr', lighting),
        ('check_blur', blur), ('find_paper_quad', quad),
        ('_fast_readability_score', score),
        ('detect_handwriting', {'is_handwritten': False, 'confidence': 0.0}),
    ]:
        monkeypatch.setattr(main._img_proc, method, lambda *a, value=value, **kw: value)
    stream = BytesIO()
    Image.new('RGB', (32, 32), color=(120, 120, 120)).save(stream, 'PNG')
    files = {'file': ('image.png', stream.getvalue(), 'image/png')}
    client = TestClient(main.app)  # No lifespan or real database.
    try:
        camera = client.post('/validate-capture', files=files, data={'image_id': image_id})
        preview = client.post('/validate-upload', files=files)
        assert camera.status_code == preview.status_code == 200
        assert camera.json() == preview.json()
        body = camera.json()
        assert body['is_usable'] == (reason is None)
        assert body['rejection_reason'] == reason
        assert body['feedback']['framing'] == framing
        register.assert_not_called()
        assert not list(tmp_path.iterdir())
        metrics.assert_called_once()  # Crop preview creates no attempt/metric row.
        metrics.reset_mock()
        uploaded = client.post('/upload-file', files=files, data={'session_id': 'session-1'})
        assert uploaded.status_code == (422 if reason else 200)
        register.assert_called_once()
        assert register.call_args.kwargs['session_id'] == 'session-1'
        assert len(list(tmp_path.iterdir())) == 1
        metrics.assert_called_once()
        assert metrics.call_args.kwargs['is_usable'] == body['is_usable']
        assert metrics.call_args.kwargs['quality_score'] == body['quality_score']
        saved = uploaded.json()['detail'] if reason else uploaded.json()
        assert saved['attempt_number'] == 2
        if reason:
            for key in body:
                assert saved[key] == body[key]
    finally:
        client.close()


@pytest.mark.parametrize('handwritten,confidence,score,expected', [
    (True, .8, 42, True), (True, .8, 41, False),
    (True, .6, 55, True), (True, .6, 54, False),
    (False, .0, 65, True), (False, .0, 64, False),
])
def test_shared_policy_retains_readability_thresholds(monkeypatch, handwritten, confidence, score, expected):
    for method, value in [
        ('check_lighting', 'good'), ('check_lighting_cnr', 'good'),
        ('check_blur', 'sharp'), ('find_paper_quad', None),
        ('_fast_readability_score', score),
        ('detect_handwriting', {'is_handwritten': handwritten, 'confidence': confidence}),
    ]:
        monkeypatch.setattr(main._img_proc, method, lambda *a, value=value, **kw: value)
    result = asyncio.run(main._validate_image_quality(np.zeros((32, 32, 3), dtype=np.uint8)))
    assert result['is_usable'] == expected


@pytest.mark.parametrize('kind', ['blank', 'noise', 'document'])
def test_camera_upload_parity_with_real_quality_checks(monkeypatch, kind):
    monkeypatch.setattr(main, '_DB_AVAILABLE', False)
    pixels = np.full((256, 256, 3), 220, dtype=np.uint8)
    if kind == 'noise':
        pixels = np.random.default_rng(7).integers(0, 256, pixels.shape, dtype=np.uint8)
    elif kind == 'document':
        pixels[:20] = pixels[-20:] = 60
        pixels[:, :20] = pixels[:, -20:] = 60
        for y in range(50, 200, 20):
            pixels[y:y+2, 50:190] = 30
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, 'PNG')
    files = {'file': ('test.png', stream.getvalue(), 'image/png')}
    client = TestClient(main.app)
    try:
        camera = client.post('/validate-capture', files=files)
        upload = client.post('/validate-upload', files=files)
    finally:
        client.close()
    assert camera.status_code == upload.status_code == 200
    assert camera.json() == upload.json()
