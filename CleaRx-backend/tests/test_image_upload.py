from io import BytesIO
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import image_upload
import main


def payload(format='PNG'):
    stream = BytesIO()
    Image.new('RGB', (32, 32), (120, 120, 120)).save(stream, format)
    return stream.getvalue()


@pytest.mark.parametrize('format,mime,suffix', [
    ('JPEG', 'image/jpeg', '.jpg'), ('PNG', 'image/png', '.png'), ('WEBP', 'image/webp', '.webp'),
])
def test_supported_decoding(format, mime, suffix):
    image, extension = image_upload.decode_image(payload(format), mime)
    assert image.shape == (32, 32, 3)
    assert extension == suffix


def test_invalid_mime_corruption_and_limits(monkeypatch):
    for raw, mime, status in [(payload(), 'image/jpeg', 415), (b'broken', 'image/png', 400),
                              (payload(), 'application/pdf', 415)]:
        with pytest.raises(HTTPException) as error:
            image_upload.decode_image(raw, mime)
        assert error.value.status_code == status
    with monkeypatch.context() as patch:
        patch.setattr(image_upload, 'MAX_BYTES', 2)
        with pytest.raises(HTTPException) as error:
            image_upload.decode_image(b'abc', 'image/png')
        assert error.value.status_code == 413
    monkeypatch.setattr(image_upload, 'MAX_PIXELS', 20)
    decoder = Mock(side_effect=AssertionError('must reject before OpenCV allocation'))
    monkeypatch.setattr(image_upload.cv2, 'imdecode', decoder)
    with pytest.raises(HTTPException) as error:
        image_upload.decode_image(payload(), 'image/png')
    assert error.value.status_code == 413
    decoder.assert_not_called()


@pytest.mark.parametrize('endpoint', ['/preview-check', '/capture-image', '/upload-file',
                                      '/validate-capture', '/validate-upload', '/process-image'])
def test_all_image_endpoints_use_bounded_decoding(monkeypatch, endpoint):
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_db_prescription_image_exists', lambda **kw: True)
    monkeypatch.setattr(main, '_db_has_completed_transcription_result', lambda **kw: False)
    monkeypatch.setattr(image_upload, 'MAX_BYTES', 2)
    # The existing processing status update is mocked; persistence changes are B8.
    monkeypatch.setattr(main, '_db_update_image_status', Mock())
    client = TestClient(main.app)
    try:
        response = client.post(endpoint, files={'file': ('test.png', b'abc', 'image/png')},
            data={'session_id': 'session-1', 'image_id': '00000000-0000-0000-0000-000000000001'})
    finally:
        client.close()
    assert response.status_code == 413, response.text


@pytest.mark.parametrize('endpoint', ['/capture-image', '/upload-file'])
def test_registration_failure_removes_file(monkeypatch, tmp_path, endpoint):
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_STORAGE_ROOT', tmp_path)
    monkeypatch.setattr(main, '_db_create_prescription_image', Mock(side_effect=RuntimeError('database unavailable')))
    client = TestClient(main.app)
    try:
        response = client.post(endpoint, files={'file': ('test.png', payload(), 'image/png')},
                               data={'session_id': 'session-1'})
    finally:
        client.close()
    assert response.status_code == 503
    assert not list(tmp_path.iterdir())


def test_camera_storage_and_retrieval_are_contained(monkeypatch, tmp_path):
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_STORAGE_ROOT', tmp_path / 'storage')
    image_id = '00000000-0000-0000-0000-000000000001'
    register = Mock(return_value=(image_id, 1))
    monkeypatch.setattr(main, '_db_create_prescription_image', register)
    client = TestClient(main.app)
    try:
        raw = payload()
        response = client.post('/capture-image', files={'file': ('test.png', raw, 'image/png')},
                               data={'session_id': 'session-1'})
        assert response.status_code == 200
        path = register.call_args.kwargs['file_path']
        assert register.call_args.kwargs['image_bytes'] == raw
        monkeypatch.setattr(main, '_db_get_prescription_image_path', lambda **kw: path)
        response = client.get('/files/' + image_id)
        assert response.status_code == 200
        assert response.content == raw
        outside = tmp_path / 'outside.png'
        outside.write_bytes(raw)
        monkeypatch.setattr(main, '_db_get_prescription_image_path', lambda **kw: str(outside))
        assert client.get('/files/' + image_id).status_code == 500
        monkeypatch.setattr(main, '_db_get_prescription_image_path', lambda **kw: None)
        assert client.get('/files/' + image_id).status_code == 404
        assert client.get('/files/not-a-uuid').status_code == 400
    finally:
        client.close()
