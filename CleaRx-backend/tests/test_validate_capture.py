from io import BytesIO
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main


@pytest.mark.parametrize('document_detected', [False, True])
def test_validate_capture_awaits_image_before_computing_variance(monkeypatch, document_detected):
    # Exercise the actual upload/decode, Laplacian, and HTTP serialization path.
    # Database writes and heuristic quality decisions are isolated from this test.
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_db_prescription_image_exists', lambda **kwargs: True)
    save_metrics = Mock(return_value='metrics-id')
    monkeypatch.setattr(main, '_db_log_validate_capture', save_metrics)
    image = np.random.default_rng(7).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    quad = np.array([[2, 2], [29, 2], [29, 29], [2, 29]], dtype=np.float32)
    for method, value in [
        ('check_lighting', 'good'), ('check_lighting_cnr', 'good'),
        ('check_blur', 'sharp'), ('find_paper_quad', quad if document_detected else None),
        ('_fast_readability_score', 90),
        ('detect_handwriting', {'is_handwritten': False, 'confidence': 0.0}),
    ]:
        monkeypatch.setattr(main._img_proc, method, lambda *args, value=value, **kwargs: value)
    stream = BytesIO()
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(stream, 'PNG')
    # No lifespan context: this test never opens the configured real database.
    client = TestClient(main.app, raise_server_exceptions=False)
    try:
        response = client.post('/validate-capture',
            files={'file': ('capture.png', stream.getvalue(), 'image/png')},
            data={'image_id': '00000000-0000-0000-0000-000000000001'})
    finally:
        client.close()
    assert response.status_code == 200, response.text
    assert response.json()['is_usable'] is True
    assert (response.json()['document_quad'] is not None) == document_detected
    save_metrics.assert_called_once()
    saved = save_metrics.call_args.kwargs
    expected = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    assert saved['blur_score'] == pytest.approx(expected)
    assert saved['document_detected'] == document_detected
