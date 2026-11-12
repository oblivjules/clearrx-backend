import asyncio
from io import BytesIO
from threading import Event
from unittest.mock import Mock, MagicMock

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

import db
import main
from runtime_policy import validate_runtime

IMAGE_ID = '00000000-0000-0000-0000-000000000001'


@pytest.mark.parametrize('mode', ['production', 'staging'])
def test_runtime_blocks_unsupported_deployment(monkeypatch, mode):
    monkeypatch.setenv('APP_ENV', mode)
    with pytest.raises(RuntimeError, match='authentication'):
        validate_runtime()


@pytest.mark.parametrize('completed,checksum,expected', [(True, True, 409), (False, False, 400)])
def test_duplicates_and_checksum_mismatches_do_not_spend_ocr(monkeypatch, completed, checksum, expected):
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_db_prescription_image_exists', lambda **kw: True)
    monkeypatch.setattr(main, '_db_has_completed_transcription_result', lambda **kw: completed)
    monkeypatch.setattr(db._db_service, 'acquire_processing_lock', Mock(return_value=object()))
    release = Mock()
    monkeypatch.setattr(db._db_service, 'release_processing_lock', release)
    monkeypatch.setattr(db._db_service, 'prescription_image_matches', lambda **kw: checksum)
    recognize = Mock(side_effect=AssertionError('must not call OCR'))
    monkeypatch.setattr(main, 'run_trocr_ocr', recognize)
    stream = BytesIO()
    Image.new('RGB', (8, 8)).save(stream, 'PNG')
    client = TestClient(main.app)
    try:
        response = client.post('/process-image', files={'file': ('test.png', stream.getvalue(), 'image/png')},
                               data={'image_id': IMAGE_ID, 'model': 'claude-sonnet-5-5'})
    finally:
        client.close()
    assert response.status_code == expected
    recognize.assert_not_called()
    release.assert_called_once()


@pytest.mark.parametrize('saved,status_failure', [(True, False), (False, False), (True, True)])
def test_completion_status_follows_required_writes(monkeypatch, saved, status_failure):
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    monkeypatch.setattr(main, '_db_prescription_image_exists', lambda **kw: True)
    monkeypatch.setattr(main, '_db_has_completed_transcription_result', lambda **kw: False)
    monkeypatch.setattr(db._db_service, 'acquire_processing_lock', lambda *args: object())
    monkeypatch.setattr(db._db_service, 'release_processing_lock', Mock())
    monkeypatch.setattr(db._db_service, 'prescription_image_matches', lambda **kw: True)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for name, value in [('check_lighting', 'good'), ('check_lighting_cnr', 'good'),
                        ('check_blur', 'sharp'), ('perspective_correct', (image, False)),
                        ('remove_shadows', image), ('enhance_for_handwriting', (image, 80, 1)),
                        ('create_document_preview', image)]:
        monkeypatch.setattr(main._img_proc, name, lambda *args, value=value: value)
    monkeypatch.setattr(main, 'run_trocr_ocr', lambda *args, **kw:
                        ('Amoxicillin 500mg tablet\nTake 1 tablet twice daily', None))
    monkeypatch.setattr(main, '_db_create_preprocessing_log', lambda **kw: 'pre-1')
    monkeypatch.setattr(main, '_persist_module_2', lambda **kw: ('output-1', {}))
    events = []
    def persist(**kw):
        events.append('required-writes')
        return saved
    def update(**kw):
        events.append(kw['status'])
        if status_failure and kw['status'] == 'processed':
            raise RuntimeError('status write failed')
    monkeypatch.setattr(main, '_persist_modules_3_4', persist)
    monkeypatch.setattr(main, '_db_update_image_status', update)
    stream = BytesIO()
    Image.new('RGB', (8, 8)).save(stream, 'PNG')
    client = TestClient(main.app)
    try:
        response = client.post('/process-image', files={'file': ('test.png', stream.getvalue(), 'image/png')},
                               data={'image_id': IMAGE_ID, 'model': 'claude-sonnet-5-5'})
    finally:
        client.close()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['status'] == 'success'  # Reading survives a failed save.
    assert body['persistence_status'] == ('saved' if saved and not status_failure else 'failed')
    assert events == ['processing', 'required-writes', 'processed' if saved else 'save_failed']


def test_processing_concurrency_is_bounded(monkeypatch):
    monkeypatch.setattr(main, '_DB_AVAILABLE', False)
    async def run():
        monkeypatch.setattr(main, '_PROCESSING_LIMIT', asyncio.Semaphore(2))
        two_started, finish = asyncio.Event(), asyncio.Event()
        active = peak = started = 0
        async def process(*args, **kw):
            nonlocal active, peak, started
            active += 1
            started += 1
            peak = max(peak, active)
            if started == 2:
                two_started.set()
            await finish.wait()
            active -= 1
            return {}
        monkeypatch.setattr(main, '_process_image', process)
        request = Request({'type': 'http', 'headers': []})
        tasks = [asyncio.create_task(main.process_image(request, None, None, None, main.PRIMARY_OPTION)) for _ in range(3)]
        await asyncio.wait_for(two_started.wait(), 2)
        assert started == peak == 2
        finish.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert peak == 2 and started == 3
    asyncio.run(run())


@pytest.mark.parametrize('cancel_during_acquire', [False, True])
def test_cancellation_waits_for_worker_before_releasing_lock(monkeypatch, cancel_during_acquire):
    monkeypatch.setattr(main, '_DB_AVAILABLE', True)
    entered, finish = Event(), Event()
    token = object()
    release = Mock()
    def acquire(*args):
        if cancel_during_acquire:
            entered.set()
            assert finish.wait(5)
        return token
    monkeypatch.setattr(db._db_service, 'acquire_processing_lock', acquire)
    monkeypatch.setattr(db._db_service, 'release_processing_lock', release)
    async def process(*args, **kw):
        entered.set()
        assert await asyncio.to_thread(finish.wait, 5)
        return {}
    monkeypatch.setattr(main, '_process_image', process)
    async def run():
        monkeypatch.setattr(main, '_PROCESSING_LIMIT', asyncio.Semaphore(2))
        task = asyncio.create_task(main.process_image(Request({'type': 'http', 'headers': []}),
            None, IMAGE_ID, None, main.PRIMARY_OPTION))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        release.assert_not_called()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        release.assert_called_once_with(token, IMAGE_ID)
    asyncio.run(run())


def test_pool_recycles_connections_and_rolls_back_failures():
    service = db.DatabaseService()
    pool, conn = Mock(), MagicMock(closed=0)
    pool.getconn.return_value = conn
    service._pool = pool
    with service._get_conn() as borrowed:
        assert borrowed is conn
    conn.commit.assert_called_once()
    with pytest.raises(RuntimeError):
        with service._get_conn():
            raise RuntimeError('write failed')
    conn.rollback.assert_called_once()
    assert pool.putconn.call_count == 2
    service.close()
    pool.closeall.assert_called_once()
    assert service._pool is None
