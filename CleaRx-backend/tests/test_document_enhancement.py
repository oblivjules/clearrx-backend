import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from image_processor import ImageProcessor


@pytest.mark.parametrize('method', ['remove_shadows', 'create_document_preview'])
@pytest.mark.parametrize('invalid', [None, np.empty((0, 0, 3), dtype=np.uint8)])
def test_empty_images_are_rejected(method, invalid):
    with pytest.raises(ValueError, match='non-empty'):
        getattr(ImageProcessor(), method)(invalid)


def test_uniform_lighting_needs_no_shadow_correction():
    image = np.full((96, 128, 3), 210, dtype=np.uint8)
    result = ImageProcessor().remove_shadows(image)
    np.testing.assert_array_equal(result, image)
    assert not np.shares_memory(result, image)


@pytest.mark.parametrize('method', ['remove_shadows', 'create_document_preview'])
def test_shadowed_strokes_retain_geometry_and_visible_contrast(method):
    # Generated scanner-like input only, not a prescription accuracy benchmark.
    paper = np.tile(np.linspace(110, 235, 256, dtype=np.uint8), (192, 1))
    paper[50:54, 40:210] = 30
    paper[100:103, 40:210] = 80
    image = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)
    before = image.copy()
    result = getattr(ImageProcessor(), method)(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(image, before)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # Both strokes remain darker at their original positions in shadow and light.
    for start, end in [(50, 54), (100, 103)]:
        for x in [60, 190]:
            assert float(gray[start-8:start-4, x].mean()) - float(gray[start:end, x].mean()) > 10
    if method == 'create_document_preview':
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])


@pytest.mark.parametrize('text', ['Amoxicillin 500mg tablet\nTake 1 tablet twice daily', ''])
@pytest.mark.parametrize('fallback', [False, True])
def test_pipeline_uses_corrected_pixels_but_returns_separate_comparison_images(monkeypatch, text, fallback):
    monkeypatch.setattr(main, '_DB_AVAILABLE', False)
    original = np.full((16, 16, 3), 60, dtype=np.uint8)
    corrected = np.full_like(original, 120)
    enhanced = np.full_like(original, 255)
    preview = np.full_like(original, 200)
    for name, value in [('check_lighting', 'good'), ('check_lighting_cnr', 'good'),
                        ('check_blur', 'sharp'), ('perspective_correct', (original, False))]:
        monkeypatch.setattr(main._img_proc, name, lambda *args, value=value: value)
    shadow = Mock(return_value=corrected)
    multi_pass = Mock(return_value=(enhanced, 80, 1))
    make_preview = Mock(return_value=preview)
    recognize = Mock(return_value=None if fallback else (text, None))
    local = Mock(return_value=(text, .8))
    monkeypatch.setattr(main._img_proc, 'remove_shadows', shadow)
    monkeypatch.setattr(main._img_proc, 'enhance_for_handwriting', multi_pass)
    monkeypatch.setattr(main._img_proc, 'create_document_preview', make_preview)
    monkeypatch.setattr(main, 'run_trocr_ocr', recognize)
    monkeypatch.setitem(sys.modules, 'ocr_service', SimpleNamespace(run_trocr=local))
    stream = BytesIO()
    Image.new('RGB', (16, 16), (60, 60, 60)).save(stream, 'PNG')
    client = TestClient(main.app)
    try:
        response = client.post('/process-image', files={'file': ('image.png', stream.getvalue(), 'image/png')})
    finally:
        client.close()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['status'] == ('success' if text else 'rejected')
    assert shadow.call_args.args[0] is original
    assert multi_pass.call_args.args[0] is corrected
    assert recognize.call_args.args[0] is corrected
    assert make_preview.call_args.args[0] is corrected
    if fallback:
        assert local.call_args.args[0] is corrected
    else:
        local.assert_not_called()
    output_preview = cv2.imdecode(np.frombuffer(base64.b64decode(body['enhanced_image']), np.uint8), cv2.IMREAD_COLOR)
    output_original = cv2.imdecode(np.frombuffer(base64.b64decode(body['original_image']), np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(output_preview, preview)
    np.testing.assert_array_equal(output_original, original)
